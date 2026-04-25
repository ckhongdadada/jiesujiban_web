from __future__ import annotations

import os
import re
import threading
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.jsjb.reply_generation.verification import verify_generated_reply
from src.jsjb.core.artifacts import inspect_generator_artifacts
from src.jsjb.core.logging import StructuredLogger


def _get_base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_gen_tokenizer = None
_gen_model = None
_draft_model = None
_gen_lock = threading.Lock()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))
ENABLE_FACT_VERIFICATION = os.getenv("ENABLE_FACT_VERIFICATION", "true").lower() == "true"
ENABLE_ASSISTED_DECODING = os.getenv("ENABLE_ASSISTED_DECODING", "true").lower() == "true"
_logger = StructuredLogger()

def _effective_assisted_decoding(enable_assisted_decoding: bool | None) -> bool:
    if enable_assisted_decoding is None:
        return ENABLE_ASSISTED_DECODING
    return bool(enable_assisted_decoding)


def _resolve_paths(
    base_model_path: str | None = None,
    lora_path: str | None = None,
    draft_model_path: str | None = None,
) -> tuple[str, str, str]:
    base_model = base_model_path or os.path.join(_get_base_dir(), "qwen_models", "Qwen", "Qwen2___5-1___5B-Instruct")
    lora_dir = lora_path or os.path.join(_get_base_dir(), "qwen_reply_model")
    draft_model = draft_model_path or os.path.join(_get_base_dir(), "qwen_models", "Qwen", "Qwen2.5-0.5B-Instruct")
    return base_model, lora_dir, draft_model


def get_latest_lora_checkpoint(base_dir: str) -> str:
    if not os.path.isdir(base_dir):
        return base_dir
    if os.path.exists(os.path.join(base_dir, "adapter_config.json")):
        return base_dir

    checkpoints: list[str] = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            checkpoints.append(path)
    if not checkpoints:
        return base_dir

    checkpoints.sort(key=lambda path: int(path.rsplit("-", 1)[-1]) if path.rsplit("-", 1)[-1].isdigit() else 0)
    return checkpoints[-1]


def generator_loaded() -> bool:
    return _gen_model is not None and _gen_tokenizer is not None


def generator_status(
    base_model_path: str | None = None,
    lora_path: str | None = None,
    draft_model_path: str | None = None,
    enable_assisted_decoding: bool | None = None,
) -> dict[str, Any]:
    base_model, lora_dir, draft_model = _resolve_paths(base_model_path, lora_path, draft_model_path)
    assisted_enabled = _effective_assisted_decoding(enable_assisted_decoding)
    status = inspect_generator_artifacts(base_model, lora_dir)
    status["loaded"] = generator_loaded()
    status["device"] = str(_device)
    status["actual_lora_dir"] = get_latest_lora_checkpoint(lora_dir)
    status["draft_model_path"] = draft_model
    status["draft_model_exists"] = bool(draft_model and os.path.exists(draft_model))
    status["assisted_decoding_configured"] = assisted_enabled
    status["assisted_decoding_enabled"] = bool(assisted_enabled and _draft_model is not None)
    return status


def load_generator(
    base_model_path: str | None = None,
    lora_path: str | None = None,
    draft_model_path: str | None = None,
    enable_assisted_decoding: bool | None = None,
) -> bool:
    global _gen_tokenizer, _gen_model, _draft_model

    with _gen_lock:
        if generator_loaded():
            return True

        base_model, lora_dir, draft_model = _resolve_paths(base_model_path, lora_path, draft_model_path)
        status = inspect_generator_artifacts(base_model, lora_dir)
        if not status["runtime_ready"]:
            print(f"[生成模型] 产物未就绪 missing={status['missing']}")
            for warning in status["warnings"]:
                print(f"[生成模型] 提示: {warning}")
            return False

        actual_lora_dir = get_latest_lora_checkpoint(lora_dir)
        assisted_enabled = _effective_assisted_decoding(enable_assisted_decoding)
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        print("[生成模型] 正在加载基础模型 + LoRA 适配器...")
        print(f"[生成模型] 基础模型目录: {base_model}")
        print(f"[生成模型] LoRA 目录: {actual_lora_dir}")

        try:
            _gen_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
            if _gen_tokenizer.pad_token is None:
                _gen_tokenizer.pad_token = _gen_tokenizer.eos_token

            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=True,
            )
            if device_map is None:
                base_model_obj = base_model_obj.to(_device)

            _gen_model = PeftModel.from_pretrained(base_model_obj, actual_lora_dir)
            _gen_model.eval()

            _draft_model = None
            if assisted_enabled and draft_model and os.path.exists(draft_model):
                try:
                    draft_model_obj = AutoModelForCausalLM.from_pretrained(
                        draft_model,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    if device_map is None:
                        draft_model_obj = draft_model_obj.to(_device)
                    draft_model_obj.eval()
                    _draft_model = draft_model_obj
                    print(f"[生成模型] Draft 小模型已启用: {draft_model}")
                except Exception as draft_exc:
                    _draft_model = None
                    print(f"[生成模型] Draft 小模型加载失败，回退标准生成: {draft_exc}")

            print("[生成模型] 加载完成")
            return True
        except Exception as exc:
            _gen_tokenizer = None
            _gen_model = None
            _draft_model = None
            print(f"[生成模型] 加载失败: {exc}")
            return False


def pick_issue_keyword(title: str, body: str) -> str:
    text = f"{title} {body}"
    keyword_rules = [
        ("垃圾", "环境卫生"),
        ("清运", "垃圾清运"),
        ("异味", "异味扰民"),
        ("噪声", "噪声扰民"),
        ("停车", "停车秩序"),
        ("违停", "停车秩序"),
        ("施工", "施工管理"),
        ("积水", "道路积水"),
        ("物业", "物业服务"),
        ("占道", "占道经营"),
        ("路灯", "照明设施"),
        ("排水", "排水设施"),
    ]
    for token, label in keyword_rules:
        if token in text:
            return label
    return "相关诉求"


def _clean_generated_reply(reply: str) -> str:
    cleaned = reply or ""
    cleaned = re.sub(r"^(您好|你好|尊敬的.*?)[,，:：\s]+", "", cleaned)
    cleaned = re.sub(r"(感谢您.*?|欢迎您.*?|如有疑问.*?|特此回复.*)$", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = cleaned.strip(" \n\t,.;:，。！？!?")
    return cleaned.strip()


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
    evidence = retrieval_hits[0].get("title", "相关政策和类似案例") if retrieval_hits else "相关政策和类似案例"
    return (
        f"经核实，您反映的{district}{issue}问题已转请{unit}进一步核查处理。"
        f"后续将参考{evidence}的办理口径，核实问题成因和责任主体，"
        f"并根据核查结果依法依规推进办理。"
    )


def _build_location_block(location_result: dict[str, Any]) -> tuple[str, str]:
    district = location_result.get("district") or "未识别"
    place_lines = []
    for hit in location_result.get("places", [])[:5]:
        matched_text = hit.get("matched_text", "")
        hit_district = hit.get("district", "")
        source = hit.get("source", "")
        category = hit.get("category", "")
        place_lines.append(f"- {matched_text} -> {hit_district} | 来源:{source} | 类别:{category}")
    location_block = "\n".join(place_lines) if place_lines else "- 未提取到稳定地名"
    return district, location_block


def _build_rag_block(retrieval_hits: list[dict[str, Any]]) -> str:
    if not retrieval_hits:
        return "无可用检索结果"

    rag_lines = []
    for idx, hit in enumerate(retrieval_hits, start=1):
        status_parts = []
        if hit.get("project_status"):
            status_parts.append(f"项目状态:{hit['project_status']}")
        if hit.get("demolition_status"):
            status_parts.append(f"拆迁状态:{hit['demolition_status']}")
        if hit.get("resource_status"):
            status_parts.append(f"资源状态:{hit['resource_status']}")
        if hit.get("responsible_unit"):
            status_parts.append(f"责任单位:{hit['responsible_unit']}")

        status_text = f" | {' | '.join(status_parts)}" if status_parts else ""
        rag_lines.append(
            f"[{idx}] {hit.get('title', '未命名材料')} | 类型:{hit.get('doc_type', '参考材料')} | "
            f"区域:{hit.get('district') or '全市'}{status_text} | 摘要:{hit.get('snippet', '')}"
        )
    return "\n".join(rag_lines)


def _run_generation(
    messages: list[dict[str, str]],
    use_assisted_decoding: bool = True,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    prompt = _gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _gen_tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(_gen_model.device) for key, value in inputs.items()}

    effective_temperature = 1.0 if temperature is None else float(temperature)
    do_sample = effective_temperature > 0 and effective_temperature != 1.0
    generate_kwargs = {
        "max_new_tokens": int(max_new_tokens or MAX_NEW_TOKENS),
        "do_sample": do_sample,
        "repetition_penalty": 1.08,
        "pad_token_id": _gen_tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = effective_temperature
    if use_assisted_decoding and ENABLE_ASSISTED_DECODING and _draft_model is not None:
        generate_kwargs["assistant_model"] = _draft_model

    with torch.no_grad():
        output_ids = _gen_model.generate(**inputs, **generate_kwargs)

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = _gen_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return _clean_generated_reply(reply)


def generate_reply_with_context(
    tag: str,
    title: str,
    body: str,
    unit: str,
    location_result: dict[str, Any],
    retrieval_hits: list[dict[str, Any]],
    base_model_path: str | None = None,
    lora_path: str | None = None,
    draft_model_path: str | None = None,
    enable_verification: bool | None = None,
    enable_assisted_decoding: bool | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    return_dict: bool = False,
) -> dict[str, Any] | str:
    if enable_verification is None:
        enable_verification = ENABLE_FACT_VERIFICATION

    if not load_generator(
        base_model_path,
        lora_path,
        draft_model_path,
        enable_assisted_decoding=enable_assisted_decoding,
    ):
        result = {
            "reply": fallback_generate_reply(tag, title, body, unit, location_result, retrieval_hits),
            "verification": None,
            "fallback": True,
        }
        return result if return_dict else result["reply"]

    district, location_block = _build_location_block(location_result)
    rag_block = _build_rag_block(retrieval_hits)
    street = location_result.get("street") or location_result.get("subdistrict") or ""

    user_content = (
        f"回复单位: {unit}\n"
        f"行政区: {district}\n"
        f"承办街道: {street}\n"
        f"留言标签: {tag}\n"
        f"留言标题: {title}\n"
        f"留言正文: {body}\n\n"
        f"地名识别结果:\n"
        f"- 行政区: {district}\n"
        f"- 识别明细:\n{location_block}\n\n"
        f"检索到的政策/案例上下文:\n{rag_block}"
    )

    system_prompt = (
        "你是政务留言办理辅助生成引擎。"
        "请优先参考提供的政策、案例和地名识别结果，生成简洁、稳妥、可落地的回复正文。\n\n"
        "【事实准确性约束】\n"
        "1. 如果检索材料显示项目已完工，不要写成正在施工。\n"
        "2. 如果检索材料显示资源已运营，不要写成尚未建设。\n"
        "3. 责任单位和承办街道必须与检索材料一致，不得编造。\n"
        "4. 行政区只能使用输入中识别到的行政区。\n\n"
        "【输出要求】\n"
        "只输出回复正文，不要称呼、落款和日期，控制在220字以内。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    reply = _run_generation(
        messages,
        use_assisted_decoding=_effective_assisted_decoding(enable_assisted_decoding),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    if not reply:
        reply = fallback_generate_reply(tag, title, body, unit, location_result, retrieval_hits)
        result = {"reply": reply, "verification": None, "fallback": True}
        return result if return_dict else result["reply"]

    verification_result = None
    if enable_verification:
        try:
            verification_result = verify_generated_reply(
                generated_reply=reply,
                retrieval_hits=retrieval_hits,
                location_result=location_result,
            )
            if verification_result.get("needs_review"):
                _logger.warning(
                    "事实验证发现问题",
                    extra={
                        "summary": verification_result.get("summary"),
                        "warnings": verification_result.get("warnings"),
                        "high_risk": verification_result.get("high_risk"),
                    },
                )
        except Exception as exc:
            _logger.error(f"[事实验证] 验证失败: {exc}")
            verification_result = {"error": str(exc)}

    result = {"reply": reply, "verification": verification_result, "fallback": False}
    return result if return_dict else result["reply"]


def generate_simple_reply(
    tag: str,
    title: str,
    body: str,
    unit: str,
    location_result: dict[str, Any] | None = None,
    base_model_path: str | None = None,
    lora_path: str | None = None,
    draft_model_path: str | None = None,
    enable_assisted_decoding: bool | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    retrieval_hits: list[dict[str, Any]] = []
    if not load_generator(
        base_model_path,
        lora_path,
        draft_model_path,
        enable_assisted_decoding=enable_assisted_decoding,
    ):
        return fallback_generate_reply(tag, title, body, unit, location_result or {}, retrieval_hits)

    district = location_result.get("district") if location_result else "未识别"
    issue = pick_issue_keyword(title, body)
    user_content = (
        f"回复单位: {unit}\n"
        f"所属地区: {district}\n"
        f"诉求焦点: {issue}\n"
        f"留言标签: {tag}\n"
        f"留言标题: {title}\n"
        f"留言正文: {body}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是专业的政务业务处理辅助引擎。"
                "请直接输出核实情况和办理方向，不要问候语、结束语、落款或日期。"
                "只能使用输入中的地区与单位，不得编造，控制在200字以内。"
            ),
        },
        {"role": "user", "content": user_content},
    ]

    reply = _run_generation(
        messages,
        use_assisted_decoding=_effective_assisted_decoding(enable_assisted_decoding),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return reply or fallback_generate_reply(tag, title, body, unit, location_result or {}, retrieval_hits)
