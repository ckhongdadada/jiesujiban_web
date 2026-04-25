from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import sys

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.unit_classifier.acceptance import build_acceptance_report, format_acceptance_report, write_acceptance_report

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPLY_MIN_LEN = 50
REPLY_MAX_LEN = 600
LOW_QUALITY_PATTERNS = [
    r"正在.*?研究",
    r"正在.*?推进",
    r"请.*?耐心等待",
    r"已转.*?部门",
    r"已转办",
    r"请.*?关注.*?进展",
]
ONSITE_NEGATIVE_PATTERNS = [
    r"经.*?核查.*?未发现",
    r"经.*?核实.*?未发现",
    r"经.*?现场.*?未.*?发现",
    r"现场.*?查看.*?未.*?发现",
    r"经.*?核查.*?无.*?问题",
    r"经.*?核实.*?符合.*?标准",
    r"未见.*?异常",
    r"无明显.*?问题",
]
REQUIRED_COLUMNS = ["留言标签", "留言标题", "留言正文", "官方回复单位", "官方回复正文"]


def clean_training_data(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^[\s\S]*?(?:关于.*?回复信|关于.*?的函)\s*\n", "", text, flags=re.DOTALL)
    text = re.sub(r"您于[\d年月日]+.*?(?:现答复如下|回复如下)[：:。]?\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"您的留言.*?[，,。！!\n]\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^(?:尊敬的.*?[，,！!\n]|您好\s*[！!，,：:\n]|你好\s*[！!，,：:\n])\s*", "", text, flags=re.DOTALL)

    tail_patterns = [
        r"感谢您对.*?(?:理解[与和]?支持|关心[与和]?支持|关注[与和]?支持).*$",
        r"感谢您的.*?(?:理解|支持|关注|关心).*$",
        r"特此回复.*$",
        r"祝您.*?愉快.*$",
        r"请.*?谅解.*$",
        r"如有.*?疑问.*?联系.*$",
        r"欢迎.*?再次.*?留言.*$",
        r"\n\s*[\u4e00-\u9fa5]{2,20}(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|局|处|科)\s*\n[\s\S]*$",
        r"\n\s*\d{4}年\d{1,2}月\d{1,2}日\s*$",
    ]
    for pattern in tail_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)

    text = re.sub(r"^[\s：:,，。！!\n]+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_low_quality(text: str) -> bool:
    return any(re.search(pattern, text) and len(text) < 120 for pattern in LOW_QUALITY_PATTERNS)


def is_onsite_negative(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in ONSITE_NEGATIVE_PATTERNS)


def load_and_filter_data(data_file: str) -> pd.DataFrame:
    print(f"\n读取数据: {data_file}")
    df = pd.read_excel(data_file, engine="openpyxl") if data_file.endswith(".xlsx") else pd.read_csv(data_file)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"生成数据缺少必要列: {missing}")

    df.fillna("", inplace=True)
    total_raw = len(df)
    print(f"原始样本数: {total_raw:,}")

    df = df[df["官方回复正文"].str.len() > 5]
    df["answer"] = df["官方回复正文"].apply(clean_training_data)
    df = df[df["answer"].str.len() >= REPLY_MIN_LEN]
    df = df[df["answer"].str.len() <= REPLY_MAX_LEN]
    df = df[~df["answer"].apply(is_low_quality)]
    df = df[~df["answer"].apply(is_onsite_negative)]
    df = df.drop_duplicates(subset=["answer"], keep="first")
    df = df.reset_index(drop=True)

    print(f"最终可用训练样本: {len(df):,}  （过滤率: {(1 - len(df)/total_raw):.1%}）")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Qwen LoRA 回复生成模型")
    parser.add_argument("--model-name", default=r"C:\Users\28414\PycharmProjects\接诉即办\qwen_models\Qwen\Qwen2___5-1___5B-Instruct")
    parser.add_argument("--data-file", default=r"C:\Users\28414\Desktop\合并后数据 - 副本.xlsx")
    parser.add_argument("--output-dir", default=r"C:\Users\28414\Desktop\qwen_reply_model")
    parser.add_argument("--report-dir", default=r"C:\Users\28414\PycharmProjects\接诉即办项目\data\training_reports")
    parser.add_argument("--train-from-scratch", action="store_true")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    last_checkpoint = None

    if not os.path.exists(args.model_name):
        raise FileNotFoundError(f"未找到生成基础模型目录: {args.model_name}")
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"未找到生成训练数据文件: {args.data_file}")

    if args.train_from_scratch:
        if os.path.exists(args.output_dir):
            print(f"TRAIN_FROM_SCRATCH=True，正在清空旧目录: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        print("从头开始训练")
    else:
        if os.path.exists(args.output_dir):
            last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint:
            print(f"检测到检查点: {last_checkpoint}")
            print("将从该检查点继续训练")
        else:
            print("未检测到检查点，将从头开始训练")
            os.makedirs(args.output_dir, exist_ok=True)

    torch.cuda.empty_cache()
    print(f"正在加载模型: {args.model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    df = load_and_filter_data(args.data_file)

    def build_prompt(row):
        user_content = f"标签: {row['留言标签']}\n标题: {row['留言标题']}\n正文: {row['留言正文']}"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的政务业务处理引擎。"
                    "请直接输出核实情况和办理结果，"
                    "不需要任何问候语、引导语和结束语，"
                    "不需要署名或日期，控制在200字以内。"
                ),
            },
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    df["prompt"] = df.apply(build_prompt, axis=1)
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.05, seed=42)

    def process_func(example):
        instruction_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(example["answer"], add_special_tokens=False)["input_ids"]
        response_ids.append(tokenizer.eos_token_id)

        input_ids = instruction_ids + response_ids
        labels = [-100] * len(instruction_ids) + response_ids

        if len(input_ids) > args.max_length:
            allowed_response_len = args.max_length - len(instruction_ids) - 1
            if allowed_response_len > 0:
                response_ids = response_ids[:allowed_response_len] + [tokenizer.eos_token_id]
                input_ids = instruction_ids + response_ids
                labels = [-100] * len(instruction_ids) + response_ids
            else:
                input_ids = input_ids[: args.max_length - 1] + [tokenizer.eos_token_id]
                labels = labels[: args.max_length - 1] + [tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    tokenized_datasets = dataset.map(process_func, remove_columns=dataset["train"].column_names)
    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=200,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print("启动训练...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report = build_acceptance_report(
        target="generator",
        generator_base_model=args.model_name,
        generator_lora_dir=args.output_dir,
    )
    paths = write_acceptance_report(report, report_dir=args.report_dir, filename_prefix="generator_acceptance")
    print("\n=== 训练产物验收 ===")
    print(format_acceptance_report(report))
    print(f"验收报告: {paths['latest']}")
    print(f"归档报告: {paths['archived']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断，但检查点应已自动保存。")
        raise
