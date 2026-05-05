"""
Model artifact inspection utilities.

Provides functions to verify that classifier and generator model files
are present and compatible before attempting to load them.
"""

from __future__ import annotations

import os
from typing import Any


_CLASSIFIER_REQUIRED_FILES = [
    "label_map.json",
]

_CLASSIFIER_OPTIONAL_FILES = [
    "model_meta.json",
    "tfidf_vectorizer.joblib",
]

_GENERATOR_REQUIRED_PATTERNS = [
    "config.json",
    "tokenizer_config.json",
]

_GENERATOR_LORA_REQUIRED_FILES = [
    "adapter_config.json",
]


def _check_files(directory: str, filenames: list[str]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    found: list[str] = []
    for name in filenames:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            found.append(name)
        else:
            missing.append(name)
    return found, missing


def inspect_classifier_artifacts(
    model_dir: str,
    base_model_dir: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model_dir": model_dir,
        "base_model_dir": base_model_dir,
        "exists": os.path.isdir(model_dir) if model_dir else False,
        "compatible_runtime_ready": False,
        "missing": [],
        "warnings": [],
        "weights": {"path": "", "exists": False},
        "tokenizer_source": "",
    }

    if not model_dir or not os.path.isdir(model_dir):
        result["missing"] = _CLASSIFIER_REQUIRED_FILES[:]
        return result

    found_req, missing_req = _check_files(model_dir, _CLASSIFIER_REQUIRED_FILES)
    result["missing"] = missing_req

    weights_candidates = [
        os.path.join(model_dir, "pytorch_model.bin"),
        os.path.join(model_dir, "model.safetensors"),
    ]
    for wc in weights_candidates:
        if os.path.exists(wc):
            result["weights"] = {"path": wc, "exists": True}
            break

    if not result["weights"]["exists"]:
        result["missing"].append("pytorch_model.bin / model.safetensors")

    tokenizer_source = ""
    if base_model_dir and os.path.isdir(base_model_dir):
        tokenizer_source = base_model_dir
    else:
        tokenizer_source = model_dir
    result["tokenizer_source"] = tokenizer_source

    if os.path.isdir(tokenizer_source):
        tok_found, tok_missing = _check_files(tokenizer_source, ["tokenizer_config.json", "vocab.txt"])
        if tok_missing:
            result["warnings"].append(f"tokenizer missing: {', '.join(tok_missing)}")

    result["compatible_runtime_ready"] = len(result["missing"]) == 0 and result["weights"]["exists"]

    return result


def inspect_generator_artifacts(
    base_model_path: str,
    lora_dir: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "base_model_path": base_model_path,
        "lora_dir": lora_dir,
        "exists": os.path.isdir(base_model_path) if base_model_path else False,
        "runtime_ready": False,
        "missing": [],
        "warnings": [],
    }

    if not base_model_path:
        result["missing"] = ["base_model_path not specified"]
        return result

    if not os.path.isdir(base_model_path):
        result["missing"].append(f"base model directory not found: {base_model_path}")
        return result

    found_req, missing_req = _check_files(base_model_path, _GENERATOR_REQUIRED_PATTERNS)
    result["missing"].extend(missing_req)

    if lora_dir and os.path.isdir(lora_dir):
        found_lora, missing_lora = _check_files(lora_dir, _GENERATOR_LORA_REQUIRED_FILES)
        if missing_lora:
            result["warnings"].append(f"LoRA adapter incomplete: missing {', '.join(missing_lora)}")
    elif lora_dir:
        result["warnings"].append(f"LoRA directory not found: {lora_dir}")

    result["runtime_ready"] = len(result["missing"]) == 0

    return result
