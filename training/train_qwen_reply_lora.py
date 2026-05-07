from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "checkpoints" / "generator" / "base_models" / "qwen_models" / "Qwen" / "Qwen2___5-1___5B-Instruct"
DEFAULT_DATA = Path(r"C:\Users\28414\Desktop\合并后数据 - 副本.xlsx")
DEFAULT_OUTPUT = PROJECT_ROOT / "checkpoints" / "generator" / "lora" / "qwen_reply_model_qlora"


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"不支持的数据格式: {path}")


def normalize_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize reply-generation data to tag/title/body/answer columns."""
    candidate_sets = [
        {
            "tag": "留言标签",
            "title": "留言标题",
            "body": "留言正文",
            "answer": "官方回复正文",
        },
        {
            "tag": "message_tag_level1",
            "title": "message_title",
            "body": "message_body",
            "answer": "reply_body_clean",
        },
        {
            "tag": "tag",
            "title": "title",
            "body": "body",
            "answer": "answer",
        },
    ]
    for mapping in candidate_sets:
        if all(column in df.columns for column in mapping.values()):
            out = pd.DataFrame({key: df[value] for key, value in mapping.items()})
            out = out.fillna("").astype(str)
            out = out[
                out["title"].str.strip().ne("")
                & out["body"].str.strip().ne("")
                & out["answer"].str.strip().ne("")
            ].copy()
            return out
    raise ValueError("生成训练数据缺少必要字段，需包含留言标签/留言标题/留言正文/官方回复正文或等价字段。")


def build_prompt(tokenizer: AutoTokenizer, row: dict[str, Any]) -> str:
    user_content = (
        f"留言标签: {row.get('tag', '')}\n"
        f"留言标题: {row.get('title', '')}\n"
        f"留言正文: {row.get('body', '')}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是政务留言办理辅助生成引擎。"
                "请直接输出核实情况和办理结果，不要问候语、落款和日期，控制在220字以内。"
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_example(example: dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> dict[str, Any]:
    prompt = example["prompt"]
    answer = str(example["answer"]).strip()
    instruction_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

    input_ids = instruction_ids + response_ids
    labels = [-100] * len(instruction_ids) + response_ids

    if len(input_ids) > max_length:
        allowed_response_len = max_length - len(instruction_ids)
        if allowed_response_len > 1:
            response_ids = response_ids[: allowed_response_len - 1] + [tokenizer.eos_token_id]
            input_ids = instruction_ids + response_ids
            labels = [-100] * len(instruction_ids) + response_ids
        else:
            input_ids = input_ids[: max_length - 1] + [tokenizer.eos_token_id]
            labels = labels[: max_length - 1] + [tokenizer.eos_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_quantization_config(args: argparse.Namespace) -> BitsAndBytesConfig | None:
    if args.quantization != "qlora_4bit":
        return None
    compute_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen 政务回复 QLoRA/LoRA 微调脚本")
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quantization", choices=["qlora_4bit", "lora_16bit"], default="qlora_4bit")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true", default=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--bf16", action="store_true", default=torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"训练数据不存在: {args.data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = build_quantization_config(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    df = normalize_training_frame(read_table(args.data_path))
    df["prompt"] = [build_prompt(tokenizer, row) for row in df.to_dict(orient="records")]
    dataset = Dataset.from_pandas(df[["prompt", "answer"]], preserve_index=False).train_test_split(test_size=0.05, seed=42)
    tokenized = dataset.map(
        lambda example: tokenize_example(example, tokenizer, args.max_length),
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.quantization == "qlora_4bit" else "adamw_torch",
        bf16=bool(args.bf16),
        fp16=not bool(args.bf16),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"QLoRA/LoRA 训练完成，适配器已保存到: {output_dir}")


if __name__ == "__main__":
    main()
