from __future__ import annotations

import os
import re
import json
import threading
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def _get_base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_gen_tokenizer = None
_gen_model = None
_gen_lock = threading.Lock()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))


def get_latest_lora_checkpoint(base_dir: str) -> str:
    if not os.path.isdir(base_dir):
        return base_dir
    if os.path.exists(os.path.join(base_dir, "adapter_config.json")):
        return base_dir
    import glob
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not checkpoints:
        return base_dir
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)
    return checkpoints[-1]


def load_generator(base_model_path: str = None, lora_path: str = None) -> bool:
    global _gen_tokenizer, _gen_model

    with _gen_lock:
        if _gen_model is not None:
            return True

        if base_model_path is None:
            base_model_path = os.path.join(_get_base_dir(), "qwen_models", "Qwen", "Qwen2___5-1___5B-Instruct")
        if lora_path is None:
            lora_path = os.path.join(_get_base_dir(), "qwen_reply_model")

        print("[生成模型] 正在加载（FP16精度）...")
        try:
            actual_lora_dir = get_latest_lora_checkpoint(lora_path)
            print(f"[生成模型] LoRA 实际加载路径: {actual_lora_dir}")

            _gen_tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, trust_remote_code=True, local_files_only=True
            )
            if _gen_tokenizer.pad_token is None:
                _gen_tokenizer.pad_token = _gen_tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            )
            _gen_model = PeftModel.from_pretrained(base_model, actual_lora_dir)
            _gen_model.eval()
            print("[生成模型] 加载完成")
            return True

        except Exception as e:
            print(f"[生成模型] 加载失败: {e}")
            return False


def fallback_generate_reply(
    tag: str,
    title: str,
    body: str,
    unit: str,
    location_result: dict[str, Any],
    retrieval_hits: list[dict[str, Any]],
) -> str:
    district = location_result.get("district") or "相关区域"
    issue = pick_issue_keyword(title, body)
    evidence = retrieval_hits[0]["title"] if retrieval_hits else "相关政策和类似案例"
    return (
        f"经核实，您反映的{district}{issue}问题已转请{unit}结合现场情况进一步核查处理。"
        f"我们将参考{evidence}中的办理口径，重点核实问题成因、责任主体和整改安排，"
        f"并督促相关单位及时反馈办理进展。"
    )


def pick_issue_keyword(title: str, body: str) -> str:
    text = f"{title} {body}"
    keyword_rules = [
        ("垃圾清运", "垃圾清运"),
        ("异味", "异味扰民"),
        ("噪声", "噪声扰民"),
        ("停车", "停车秩序"),
        ("施工", "施工管理"),
        ("积水", "道路积水"),
        ("物业", "物业服务"),
    ]
    for token, label in keyword_rules:
        if token in text:
            return label
    return "诉求"


def generate_reply_with_context(
    tag: str,
    title: str,
    body: str,
    unit: str,
    location_result: dict[str, Any],
    retrieval_hits: list[dict[str, Any]],
    base_model_path: str = None,
    lora_path: str = None,
) -> str:
    if not load_generator(base_model_path, lora_path):
        return fallback_generate_reply(tag, title, body, unit, location_result, retrieval_hits)

    district = location_result.get("district") or "未识别"
    place_lines = []
    for hit in location_result.get("places", [])[:5]:
        place_lines.append(f"- {hit['matched_text']} -> {hit['district']} ({hit['source']})")
    location_block = "\n".join(place_lines) if place_lines else "- 未提取到稳定地名"

    rag_lines = []
    for idx, hit in enumerate(retrieval_hits, start=1):
        rag_lines.append(
            f"[{idx}] {hit['title']} | 类型:{hit['doc_type']} | 区域:{hit['district'] or '全市'} | 摘要:{hit['snippet']}"
        )
    rag_block = "\n".join(rag_lines) if rag_lines else "无可用检索结果"

    user_content = (
        f"回复单位: {unit}\n"
        f"留言标签: {tag}\n"
        f"留言标题: {title}\n"
        f"留言正文: {body}\n\n"
        f"地名识别结果:\n"
        f"- 行政区: {district}\n"
        f"- 识别明细:\n{location_block}\n\n"
        f"检索到的政策/案例上下文:\n{rag_block}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是政务留言办理辅助生成引擎。"
                "请优先参考提供的政策或案例上下文，生成简洁、稳妥、可落地的回复正文。"
                "严禁在回复中使用除'行政区'字段以外的其他行政区划名称。"
                "只能使用'行政区'字段提供的具体区名，严禁编造或使用其他区名。"
                "不要编造未提供的事实、法规条文或办理结果。"
                "如果检索上下文不足，就给出审慎的通用办理口径。"
                "直接输出回复正文，不要称呼、不要落款、不要日期，控制在220字以内。"
            ),
        },
        {"role": "user", "content": user_content},
    ]

    prompt = _gen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _gen_tokenizer(prompt, return_tensors="pt").to(_gen_model.device)

    with torch.no_grad():
        output_ids = _gen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.08,
            pad_token_id=_gen_tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = _gen_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    reply = re.sub(r"^(您好|你好)[，,：:\s]+", "", reply)
    reply = re.sub(r"感谢您.*$", "", reply, flags=re.DOTALL)
    cleaned = reply.strip()
    return cleaned or fallback_generate_reply(tag, title, body, unit, location_result, retrieval_hits)


def generate_simple_reply(
    tag: str,
    title: str,
    body: str,
    unit: str,
    location_result: dict[str, Any] = None,
    base_model_path: str = None,
    lora_path: str = None,
) -> str:
    if not load_generator(base_model_path, lora_path):
        return "生成模型未就绪，请稍后重试。"

    # 提取地区信息
    district = location_result.get("district") if location_result else "未识别"
    
    user_content = (
        f"回复单位: {unit}\n"
        f"所属地区: {district}\n"
        f"标签: {tag}\n"
        f"标题: {title}\n"
        f"正文: {body}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个专业的政务业务处理引擎。"
                "请直接输出核实情况和办理结果，"
                "严禁在回复中使用除'所属地区'字段以外的其他行政区划名称。"
                "只能使用'所属地区'字段提供的具体区名，严禁编造或使用其他区名。"
                "不需要任何问候语、引导语和结束语，"
                "不需要署名或日期，控制在200字以内。"
            ),
        },
        {"role": "user", "content": user_content},
    ]

    prompt = _gen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _gen_tokenizer(prompt, return_tensors="pt").to(_gen_model.device)

    with torch.no_grad():
        output_ids = _gen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=_gen_tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = _gen_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    reply = re.sub(r"^(您好|你好)[！!，,：:\s]+", "", reply)
    reply = re.sub(r"感谢您.*?(支持|关注|理解).*?$", "", reply, flags=re.DOTALL)
    return reply.strip()
