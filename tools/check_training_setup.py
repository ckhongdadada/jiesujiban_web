from __future__ import annotations

import importlib
import json
import os
import pathlib
import py_compile
import sys
from datetime import datetime
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhancements.runtime_config import load_runtime_config
from enhancements.training_acceptance import build_acceptance_report, write_acceptance_report

TRAINING_DIR = PROJECT_ROOT / "training"
REQUIRED_CLASSIFIER_COLUMNS = ["留言标签", "留言标题", "留言正文", "官方回复单位"]
REQUIRED_GENERATOR_COLUMNS = ["留言标签", "留言标题", "留言正文", "官方回复单位", "官方回复正文"]


def safe_import_version(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {"installed": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"installed": False, "error": str(exc)}


def inspect_excel(path: str, required_columns: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path, "exists": os.path.exists(path)}
    if not result["exists"]:
        result["missing_columns"] = required_columns
        return result
    try:
        df = pd.read_excel(path, nrows=5)
        result["columns"] = list(df.columns)
        result["missing_columns"] = [column for column in required_columns if column not in df.columns]
        result["row_sample_readable"] = True
    except Exception as exc:
        result["row_sample_readable"] = False
        result["error"] = str(exc)
    return result


def inspect_script(path: pathlib.Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        py_compile.compile(str(path), doraise=True)
        result["syntax_ok"] = True
    except Exception as exc:
        result["syntax_ok"] = False
        result["error"] = str(exc)
    return result


def inspect_directory(path: str) -> dict[str, Any]:
    result = {"path": path, "exists": os.path.exists(path)}
    parent = os.path.dirname(path) or path
    result["parent_exists"] = os.path.exists(parent)
    result["writable"] = os.access(path if os.path.exists(path) else parent, os.W_OK)
    return result


def main() -> int:
    config = load_runtime_config()
    classifier_data_path = r"C:\Users\28414\Desktop\留言板合并数据.xlsx"
    generator_data_path = r"C:\Users\28414\Desktop\合并后数据 - 副本.xlsx"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "environment": {
            "torch": safe_import_version("torch"),
            "transformers": safe_import_version("transformers"),
            "pandas": safe_import_version("pandas"),
            "datasets": safe_import_version("datasets"),
            "peft": safe_import_version("peft"),
            "bitsandbytes": safe_import_version("bitsandbytes"),
            "openpyxl": safe_import_version("openpyxl"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "paths": {
            "classifier_base_model": inspect_directory(r"C:\python\接诉即办\.venv\local_roberta_model"),
            "classifier_output_dir": inspect_directory(r"C:\Users\28414\PycharmProjects\接诉即办项目\final_model_fgm"),
            "generator_base_model": inspect_directory(config.generator_base_model),
            "generator_output_dir": inspect_directory(config.generator_lora_dir),
            "report_dir": inspect_directory(str(PROJECT_ROOT / "data" / "training_reports")),
        },
        "data": {
            "classifier": inspect_excel(classifier_data_path, REQUIRED_CLASSIFIER_COLUMNS),
            "generator": inspect_excel(generator_data_path, REQUIRED_GENERATOR_COLUMNS),
        },
        "scripts": {
            "train_unit_classifier_fgm": inspect_script(TRAINING_DIR / "train_unit_classifier_fgm.py"),
            "train_qwen_reply_lora": inspect_script(TRAINING_DIR / "train_qwen_reply_lora.py"),
            "post_training_acceptance": inspect_script(PROJECT_ROOT / "tools" / "post_training_acceptance.py"),
            "check_training_setup": inspect_script(PROJECT_ROOT / "tools" / "check_training_setup.py"),
        },
        "artifacts": build_acceptance_report(
            target="all",
            classifier_model_dir=config.classifier_model_dir,
            classifier_base_model=config.classifier_base_model,
            generator_base_model=config.generator_base_model,
            generator_lora_dir=config.generator_lora_dir,
        ),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    paths = write_acceptance_report({"target": "setup_check", **report}, filename_prefix="setup_check")
    print(f"\n检查报告: {paths['latest']}")
    print(f"归档报告: {paths['archived']}")

    classifier_ok = not report["data"]["classifier"].get("missing_columns") and report["data"]["classifier"].get("exists")
    generator_ok = not report["data"]["generator"].get("missing_columns") and report["data"]["generator"].get("exists")
    scripts_ok = all(item.get("syntax_ok") for item in report["scripts"].values())
    return 0 if classifier_ok and generator_ok and scripts_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
