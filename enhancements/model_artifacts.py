from __future__ import annotations

import glob
import os
from typing import Any


CLASSIFIER_TOKENIZER_CANDIDATES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
]

GENERATOR_ADAPTER_CANDIDATES = [
    "adapter_model.safetensors",
    "adapter_model.bin",
]

GENERATOR_TOKENIZER_CANDIDATES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
]


def _file_state(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "exists": os.path.exists(path),
    }


def _first_existing(directory: str, filenames: list[str]) -> str | None:
    for name in filenames:
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    return None


def _collect_existing(directory: str, filenames: list[str]) -> list[str]:
    return [os.path.join(directory, name) for name in filenames if os.path.exists(os.path.join(directory, name))]


def _checkpoint_dirs(directory: str) -> list[str]:
    pattern = os.path.join(directory, "checkpoint-*")
    return sorted(path for path in glob.glob(pattern) if os.path.isdir(path))


def inspect_classifier_artifacts(model_dir: str, base_model_dir: str = "") -> dict[str, Any]:
    label_map_path = os.path.join(model_dir, "label_map.json")
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    config_path = os.path.join(model_dir, "config.json")

    tokenizer_files = _collect_existing(model_dir, CLASSIFIER_TOKENIZER_CANDIDATES)
    tokenizer_ready = bool(tokenizer_files)

    base_model_config = os.path.join(base_model_dir, "config.json") if base_model_dir else ""
    base_model_exists = bool(base_model_dir) and os.path.isdir(base_model_dir)
    base_model_config_exists = bool(base_model_dir) and os.path.exists(base_model_config)

    script_output_ready = (
        os.path.exists(label_map_path)
        and os.path.exists(weights_path)
        and tokenizer_ready
    )
    strict_runtime_ready = script_output_ready and os.path.exists(config_path)
    compatible_runtime_ready = strict_runtime_ready or (
        script_output_ready and base_model_exists and base_model_config_exists
    )

    tokenizer_source = None
    if tokenizer_ready:
        tokenizer_source = model_dir
    elif base_model_exists:
        tokenizer_source = base_model_dir

    warnings: list[str] = []
    missing: list[str] = []

    if not os.path.exists(label_map_path):
        missing.append("label_map.json")
    if not os.path.exists(weights_path):
        missing.append("pytorch_model.bin")
    if not tokenizer_ready:
        missing.append("classifier tokenizer files")
    if script_output_ready and not os.path.exists(config_path):
        warnings.append(
            "当前分类训练脚本不会自动写出 config.json；原始 app.py 的严格加载方式会失败。"
        )
    if script_output_ready and not strict_runtime_ready and not (base_model_exists and base_model_config_exists):
        warnings.append(
            "如需让增强版兼容加载当前训练产物，需要可访问的 CLASSIFIER_BASE_MODEL。"
        )
    if base_model_dir and not base_model_exists:
        warnings.append("CLASSIFIER_BASE_MODEL 已配置，但目录不存在。")

    return {
        "model_dir": model_dir,
        "base_model_dir": base_model_dir,
        "label_map": _file_state(label_map_path),
        "weights": _file_state(weights_path),
        "config": _file_state(config_path),
        "tokenizer_files": tokenizer_files,
        "tokenizer_ready": tokenizer_ready,
        "base_model_exists": base_model_exists,
        "base_model_config_exists": base_model_config_exists,
        "tokenizer_source": tokenizer_source,
        "script_output_ready": script_output_ready,
        "strict_runtime_ready": strict_runtime_ready,
        "compatible_runtime_ready": compatible_runtime_ready,
        "missing": missing,
        "warnings": warnings,
    }


def inspect_generator_artifacts(base_model_dir: str, lora_dir: str) -> dict[str, Any]:
    adapter_config_path = os.path.join(lora_dir, "adapter_config.json")
    adapter_weight_path = _first_existing(lora_dir, GENERATOR_ADAPTER_CANDIDATES)
    tokenizer_files = _collect_existing(lora_dir, GENERATOR_TOKENIZER_CANDIDATES)
    checkpoint_dirs = _checkpoint_dirs(lora_dir) if os.path.isdir(lora_dir) else []

    base_config_path = os.path.join(base_model_dir, "config.json")
    base_model_exists = os.path.isdir(base_model_dir)
    base_model_config_exists = os.path.exists(base_config_path)
    lora_dir_exists = os.path.isdir(lora_dir)
    adapter_ready = os.path.exists(adapter_config_path) and adapter_weight_path is not None
    runtime_ready = base_model_exists and base_model_config_exists and adapter_ready
    training_output_complete = adapter_ready

    warnings: list[str] = []
    missing: list[str] = []

    if not lora_dir_exists:
        missing.append("generator lora directory")
    if lora_dir_exists and not os.path.exists(adapter_config_path):
        missing.append("adapter_config.json")
    if lora_dir_exists and adapter_weight_path is None:
        missing.append("adapter_model.safetensors / adapter_model.bin")
    if checkpoint_dirs and not adapter_ready:
        warnings.append(
            "检测到 checkpoint-* 子目录，但根目录还没有最终 LoRA 产物；训练可能中断或尚未执行最终 save_model。"
        )
    if tokenizer_files:
        warnings.append("LoRA 目录中已包含 tokenizer 文件，可用于复现实验，但服务运行时主要读取基础模型 tokenizer。")
    if not base_model_exists:
        warnings.append("GENERATOR_BASE_MODEL 目录不存在，无法加载 LoRA。")

    return {
        "base_model_dir": base_model_dir,
        "lora_dir": lora_dir,
        "base_model_exists": base_model_exists,
        "base_model_config_exists": base_model_config_exists,
        "lora_dir_exists": lora_dir_exists,
        "adapter_config": _file_state(adapter_config_path),
        "adapter_weights": {
            "path": adapter_weight_path,
            "exists": adapter_weight_path is not None,
        },
        "tokenizer_files": tokenizer_files,
        "checkpoint_dirs": checkpoint_dirs,
        "training_output_complete": training_output_complete,
        "runtime_ready": runtime_ready,
        "missing": missing,
        "warnings": warnings,
    }
