from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Generator
from collections import deque

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from enhancements.model_artifacts import inspect_generator_artifacts
from enhancements.fact_verifier import verify_generated_reply


def _get_base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_gen_tokenizer = None
_gen_model = None
_gen_lock = threading.Lock()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))


class KVCacheManager:
    """KV Cache管理器"""
    
    def __init__(self, max_cache_size: int = 10):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.lock = threading.Lock()
        self.stats = {'hits': 0, 'misses': 0}
    
    def _get_cache_key(self, prompt: str) -> str:
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> tuple | None:
        key = self._get_cache_key(prompt)
        
        with self.lock:
            if key in self.cache:
                self.stats['hits'] += 1
                return self.cache[key]
            self.stats['misses'] += 1
            return None
    
    def set(self, prompt: str, past_key_values: tuple, cached_prompt: str):
        key = self._get_cache_key(prompt)
        
        with self.lock:
            if len(self.cache) >= self.max_cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = (past_key_values, cached_prompt)
    
    def get_stats(self) -> dict:
        with self.lock:
            total = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_cache_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate
            }
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.stats = {'hits': 0, 'misses': 0}


class GenerationCache:
    """生成结果缓存"""
    
    def __init__(self, max_size: int = 500):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def _get_key(self, tag: str, title: str, body: str, unit: str) -> str:
        import hashlib
        key_str = f"{tag}|{title}|{body}|{unit}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, tag: str, title: str, body: str, unit: str) -> str | None:
        key = self._get_key(tag, title, body, unit)
        
        with self.lock:
            return self.cache.get(key)
    
    def set(self, tag: str, title: str, body: str, unit: str, reply: str):
        key = self._get_key(tag, title, body, unit)
        
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = reply


kv_cache_manager = KVCacheManager(max_cache_size=10)
generation_cache = GenerationCache(max_size=500)


def _resolve_paths(base_model_path: str | None = None, lora_path: str | None = None) -> tuple[str, str]:
    base_model = base_model_path or os.path.join(
        _get_base_dir(), "qwen_models", "Qwen", "Qwen2___5-1___5B-Instruct"
    )
    lora_dir = lora_path or os.path.join(_get_base_dir(), "qwen_reply_model")
    return base_model, lora_dir


def get_latest_lora_checkpoint(base_dir: str) -> str:
    if not os.path.isdir(base_dir):
        return base_dir
    if os.path.exists(os.path.join(base_dir, "adapter_config.json")):
        return base_dir

    checkpoints = []
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


def generator_status(base_model_path: str | None = None, lora_path: str | None = None) -> dict[str, Any]:
    base_model, lora_dir = _resolve_paths(base_model_path, lora_path)
    status = inspect_generator_artifacts(base_model, lora_dir)
    status["loaded"] = generator_loaded()
    status["device"] = str(_device)
    status["actual_lora_dir"] = get_latest_lora_checkpoint(lora_dir)
    status["kv_cache_stats"] = kv_cache_manager.get_stats()
    return status


def load_generator(base_model_path: str | None = None, lora_path: str | None = None) -> bool:
    global _gen_tokenizer, _gen_model

    with _gen_lock:
        if generator_loaded():
            return True

        base_model, lora_dir = _resolve_paths(base_model_path, lora_path)
        status = inspect_generator_artifacts(base_model, lora_dir)
        if not status["runtime_ready"]:
            print(f"[生成模型] 产物未就绪: missing={status['missing']}")
            for warning in status["warnings"]:
                print(f"[生成模型] 提示: {warning}")
            return False

        actual_lora_dir = get_latest_lora_checkpoint(lora_dir)
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        print("[生成模型] 正在加载基础模型与 LoRA 适配器...")
        print(f"[生成模型] 基础模型目录: {base_model}")
        print(f"[生成模型] LoRA 目录: {actual_lora_dir}")
        try:
            _gen_tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
                local_files_only=True,
            )
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
            
            print("[生成模型] 开始预热...")
            _warmup_model()
            print("[生成模型] 加载完成")
            return True
        except Exception as exc:
            _gen_tokenizer = None
            _gen_model = None
            print(f"[生成模型] 加载失败: {exc}")
            return False


def _warmup_model():
    """模型预热"""
    dummy_messages = [
        {"role": "system", "content": "你是政务留言办理辅助生成引擎。"},
        {"role": "user", "content": "测试内容"}
    ]
    
    for i in range(3):
        try:
            _ = _run_generation(dummy_messages, max_new_tokens=50)
        except Exception:
            pass
    
    print("[生成模型] 预热完成")


def pick_issue_keyword(title: str, body: str) -> str:
    text = f"{title} {body}"
    keyword_rules = [
        ("垃圾", "环境卫生"), ("清运", "垃圾清运"), ("异味", "异味扰民"),
        ("噪声", "噪声扰民"), ("停车", "停车秩序"), ("违停", "停车秩序"),
        ("施工", "施工管理"), ("积水", "道路积水"), ("物业", "物业服务"),
        ("占道", "占道经营"), ("路灯", "照明设施"), ("排水", "排水设施"),
    ]
    for token, label in keyword_rules:
        if token in text:
            return label
    return "相关诉求"


def _clean_generated_reply(reply: str) -> str:
    cleaned = reply or ""
    cleaned = re.sub(r"^(您好|你好|尊敬的.*?)[，,：:\s]+", "", cleaned)
    cleaned = re.sub(r"(感谢您.*?|欢迎您.*?|如有疑问.*?|特此回复.*)$", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = cleaned.strip(" \n\t，,。")
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
    evidence = retrieval_hits[0]["title"] if retrieval_hits else "相关政策和类似案例"
    return (
        f"经核实，您反映的{district}{issue}问题已转请{unit}结合现场情况进一步核查处理。"
        f"后续将参考{evidence}中的办理口径，重点核实问题成因、责任主体和整改安排，"
        f"并督促相关单位及时反馈办理进展。"
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
        rag_lines.append(
            f"[{idx}] {hit.get('title', '未命名材料')} | 类型:{hit.get('doc_type', '参考材料')} | "
            f"区域:{hit.get('district') or '全市'} | 摘要:{hit.get('snippet', '')}"
        )
    return "\n".join(rag_lines)


def _run_generation(messages: list[dict[str, str]], max_new_tokens: int = MAX_NEW_TOKENS,
                    use_cache: bool = True) -> str:
    prompt = _gen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _gen_tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(_gen_model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = _gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.08,
            pad_token_id=_gen_tokenizer.eos_token_id,
            use_cache=use_cache,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = _gen_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return _clean_generated_reply(reply)


def _run_generation_with_kv_cache(messages: list[dict[str, str]], 
                                   max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """使用KV Cache优化生成"""
    prompt = _gen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    cached = kv_cache_manager.get(prompt)
    
    if cached:
        past_key_values, cached_prompt = cached
        if prompt.startswith(cached_prompt):
            new_text = prompt[len(cached_prompt):]
            inputs = _gen_tokenizer(new_text, return_tensors="pt")
            inputs = {key: value.to(_gen_model.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                output_ids = _gen_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    do_sample=False,
                    repetition_penalty=1.08,
                    pad_token_id=_gen_tokenizer.eos_token_id,
                )
            
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            reply = _gen_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            return _clean_generated_reply(reply)
    
    inputs = _gen_tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(_gen_model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = _gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=_gen_tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
    
    if hasattr(output_ids, 'past_key_values') and output_ids.past_key_values is not None:
        kv_cache_manager.set(prompt, output_ids.past_key_values, prompt)
    
    new_ids = output_ids.sequences[0][inputs["input_ids"].shape[1]:]
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
    enable_verification: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    生成回复并进行事实验证
    
    Returns:
        包含 reply 和 verification 结果的字典
    """
    if not load_generator(base_model_path, lora_path):
        return {
            "reply": fallback_generate_reply(tag, title, body, unit, location_result, retrieval_hits),
            "verification": None,
            "fallback": True
        }
    
    if use_cache:
        cached_reply = generation_cache.get(tag, title, body, unit)
        if cached_reply:
            return {
                "reply": cached_reply,
                "verification": None,
                "fallback": False,
                "cached": True
            }

    district, location_block = _build_location_block(location_result)
    rag_block = _build_rag_block(retrieval_hits)

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

    system_prompt = (
        "你是政务留言办理辅助生成引擎。"
        "请优先参考提供的政策、案例和地名识别结果，生成简洁、稳妥、可落地的回复正文。\n\n"
        "【事实准确性约束】\n"
        "1. 状态约束: 如果检索结果标注\"拆迁状态: 已完成\"，严禁回复\"因拆迁影响\"; 如果检索结果标注\"项目状态: 已完工\"，严禁回复\"正在施工\"。\n"
        "2. 资源约束: 如果检索结果标注\"资源状态: 已运营\"，严禁回复\"暂未建设\"; 必须优先引用检索结果中的具体设施名称和联系方式。\n"
        "3. 主体约束: 只能使用检索结果中明确标注的责任单位，严禁编造或替换其他责任单位。\n"
        "4. 街道约束: 只能使用\"承办街道\"字段中给出的街道名称，严禁替换或编造其他街道。\n\n"
        "【基本要求】\n"
        "只能使用\"行政区\"字段中给出的区名，严禁编造、替换或扩展其他行政区划名称。"
        "不要编造未提供的事实、法规条文、办理结论、完成时点或承诺。"
        "如果检索上下文不足，请给出审慎的通用办理口径，并明确后续核查处理方向。"
        "只输出回复正文，不要称呼，不要落款，不要日期，控制在220字以内。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    reply = _run_generation_with_kv_cache(messages) if use_cache else _run_generation(messages)
    
    if not reply:
        reply = fallback_generate_reply(tag, title, body, unit, location_result, retrieval_hits)
        return {
            "reply": reply,
            "verification": None,
            "fallback": True
        }
    
    if use_cache:
        generation_cache.set(tag, title, body, unit, reply)
    
    verification_result = None
    if enable_verification:
        try:
            verification_result = verify_generated_reply(
                generated_reply=reply,
                retrieval_hits=retrieval_hits,
                location_result=location_result
            )
            
            if verification_result.get("needs_review"):
                print(f"[事实验证] 发现问题: {verification_result.get('summary')}")
                for warning in verification_result.get("warnings", []):
                    print(f"  - [{warning['severity']}] {warning['message']}")
        except Exception as e:
            print(f"[事实验证] 验证失败: {e}")
            verification_result = {"error": str(e)}
    
    return {
        "reply": reply,
        "verification": verification_result,
        "fallback": False
    }


def batch_generate(
    requests: list[dict[str, Any]],
    batch_size: int = 4,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> list[dict[str, Any]]:
    """批量生成回复"""
    if not load_generator():
        return [
            {
                "reply": fallback_generate_reply(
                    req.get("tag", ""),
                    req.get("title", ""),
                    req.get("body", ""),
                    req.get("unit", ""),
                    req.get("location_result", {}),
                    req.get("retrieval_hits", [])
                ),
                "verification": None,
                "fallback": True
            }
            for req in requests
        ]
    
    all_results = []
    
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]
        
        prompts = []
        for req in batch:
            district, location_block = _build_location_block(req.get("location_result", {}))
            rag_block = _build_rag_block(req.get("retrieval_hits", []))
            
            user_content = (
                f"回复单位: {req.get('unit', '')}\n"
                f"留言标签: {req.get('tag', '')}\n"
                f"留言标题: {req.get('title', '')}\n"
                f"留言正文: {req.get('body', '')}\n\n"
                f"地名识别结果:\n"
                f"- 行政区: {district}\n"
                f"- 识别明细:\n{location_block}\n\n"
                f"检索到的政策/案例上下文:\n{rag_block}"
            )
            
            system_prompt = (
                "你是政务留言办理辅助生成引擎。"
                "请优先参考提供的政策、案例和地名识别结果，生成简洁、稳妥、可落地的回复正文。\n\n"
                "【事实准确性约束】\n"
                "1. 状态约束: 如果检索结果标注\"拆迁状态: 已完成\"，严禁回复\"因拆迁影响\"; 如果检索结果标注\"项目状态: 已完工\"，严禁回复\"正在施工\"。\n"
                "2. 资源约束: 如果检索结果标注\"资源状态: 已运营\"，严禁回复\"暂未建设\"; 必须优先引用检索结果中的具体设施名称和联系方式。\n"
                "3. 主体约束: 只能使用检索结果中明确标注的责任单位，严禁编造或替换其他责任单位。\n"
                "4. 街道约束: 只能使用\"承办街道\"字段中给出的街道名称，严禁替换或编造其他街道。\n\n"
                "【基本要求】\n"
                "只能使用\"行政区\"字段中给出的区名，严禁编造、替换或扩展其他行政区划名称。"
                "不要编造未提供的事实、法规条文、办理结论、完成时点或承诺。"
                "如果检索上下文不足，请给出审慎的通用办理口径，并明确后续核查处理方向。"
                "只输出回复正文，不要称呼，不要落款，不要日期，控制在220字以内。"
            )
            
            prompt = _gen_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
        
        encodings = _gen_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        encodings = {k: v.to(_gen_model.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = _gen_model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=_gen_tokenizer.eos_token_id
            )
        
        for j, output in enumerate(outputs):
            reply = _gen_tokenizer.decode(
                output[encodings["input_ids"][j].shape[0]:],
                skip_special_tokens=True
            )
            reply = _clean_generated_reply(reply)
            
            if not reply:
                reply = fallback_generate_reply(
                    batch[j].get("tag", ""),
                    batch[j].get("title", ""),
                    batch[j].get("body", ""),
                    batch[j].get("unit", ""),
                    batch[j].get("location_result", {}),
                    batch[j].get("retrieval_hits", [])
                )
                all_results.append({"reply": reply, "verification": None, "fallback": True})
            else:
                all_results.append({"reply": reply, "verification": None, "fallback": False})
    
    return all_results


def generate_stream(
    tag: str,
    title: str,
    body: str,
    unit: str,
    location_result: dict[str, Any] | None = None,
    retrieval_hits: list[dict[str, Any]] | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Generator[str, None, None]:
    """流式生成回复"""
    if not load_generator():
        yield fallback_generate_reply(tag, title, body, unit, location_result or {}, retrieval_hits or [])
        return

    district, location_block = _build_location_block(location_result or {})
    rag_block = _build_rag_block(retrieval_hits or [])

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

    system_prompt = (
        "你是政务留言办理辅助生成引擎。"
        "请优先参考提供的政策、案例和地名识别结果，生成简洁、稳妥、可落地的回复正文。\n\n"
        "【事实准确性约束】\n"
        "1. 状态约束: 如果检索结果标注\"拆迁状态: 已完成\"，严禁回复\"因拆迁影响\"; 如果检索结果标注\"项目状态: 已完工\"，严禁回复\"正在施工\"。\n"
        "2. 资源约束: 如果检索结果标注\"资源状态: 已运营\"，严禁回复\"暂未建设\"; 必须优先引用检索结果中的具体设施名称和联系方式。\n"
        "3. 主体约束: 只能使用检索结果中明确标注的责任单位，严禁编造或替换其他责任单位。\n"
        "4. 街道约束: 只能使用\"承办街道\"字段中给出的街道名称，严禁替换或编造其他街道。\n\n"
        "【基本要求】\n"
        "只能使用\"行政区\"字段中给出的区名，严禁编造、替换或扩展其他行政区划名称。"
        "不要编造未提供的事实、法规条文、办理结论、完成时点或承诺。"
        "如果检索上下文不足，请给出审慎的通用办理口径，并明确后续核查处理方向。"
        "只输出回复正文，不要称呼，不要落款，不要日期，控制在220字以内。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    prompt = _gen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _gen_tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(_gen_model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        _gen_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    import threading
    thread = threading.Thread(
        target=_gen_model.generate,
        kwargs={
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            do_sample=False,
            repetition_penalty=1.08
        }
    )
    thread.start()

    for text in streamer:
        yield text

    thread.join()


def generate_simple_reply(
    tag: str,
    title: str,
    body: str,
    unit: str,
    location_result: dict[str, Any] | None = None,
    base_model_path: str | None = None,
    lora_path: str | None = None,
) -> str:
    retrieval_hits: list[dict[str, Any]] = []
    if not load_generator(base_model_path, lora_path):
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
                "请直接输出核实情况和办理方向，只能使用"所属地区"字段中已有的区名。"
                "严禁编造其他行政区、承办单位、法规条文、调查结果或完成时间。"
                "不需要问候语、引导语、结束语、署名或日期，控制在200字以内。"
            ),
        },
        {"role": "user", "content": user_content},
    ]

    reply = _run_generation(messages)
    return reply or fallback_generate_reply(tag, title, body, unit, location_result or {}, retrieval_hits)


def get_generation_stats() -> dict[str, Any]:
    """获取生成模型统计信息"""
    return {
        "loaded": generator_loaded(),
        "device": str(_device),
        "kv_cache_stats": kv_cache_manager.get_stats(),
        "generation_cache_size": len(generation_cache.cache)
    }


def clear_generation_cache():
    """清空生成缓存"""
    generation_cache.clear()
    kv_cache_manager.clear()
