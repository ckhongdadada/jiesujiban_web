from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhancements.data_paths import get_training_reports_dir


PYTHON_EXE = r"C:\Users\28414\anaconda3\envs\qwen_env\python.exe"
TRAIN_SCRIPT = PROJECT_ROOT / "training" / "train_unit_classifier_fgm.py"
REPORT_DIR = get_training_reports_dir() / "loss_compare"
DATA_PATH = r"C:\Users\28414\Desktop\留言板合并数据 - 副本.xlsx"
SAMPLE_SIZE = 2000
EPOCHS = 2
BATCH_SIZE = 16
GRAD_ACCUM = 2
SEED = 42
TOP_K = 3

METRIC_PATTERNS = {
    "top1_acc": re.compile(r"Top-1 Acc:\s*([0-9.]+)"),
    "macro_f1": re.compile(r"Macro-F1:\s*([0-9.]+)"),
    "weighted_f1": re.compile(r"Weighted-F1:\s*([0-9.]+)"),
}
TOPK_PATTERN = re.compile(r"Val Top-(\d+) Acc:\s*([0-9.]+)")


def _run_one(loss_type: str) -> dict[str, object]:
    save_dir = PROJECT_ROOT / f"final_model_compare_{loss_type}"
    log_path = REPORT_DIR / f"{loss_type}.log"
    save_dir.mkdir(parents=True, exist_ok=True)

    command = [
        PYTHON_EXE,
        str(TRAIN_SCRIPT),
        "--data-path",
        DATA_PATH,
        "--use-small-sample",
        "--sample-size",
        str(SAMPLE_SIZE),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--grad-accum-steps",
        str(GRAD_ACCUM),
        "--seed",
        str(SEED),
        "--top-k-eval",
        str(TOP_K),
        "--loss-type",
        loss_type,
        "--save-dir",
        str(save_dir),
        "--report-dir",
        str(REPORT_DIR),
        "--no-resume",
        "--train-from-scratch",
    ]

    started = time.time()
    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - started
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8")

    result: dict[str, object] = {
        "loss_type": loss_type,
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "log_path": str(log_path),
        "save_dir": str(save_dir),
    }

    for key, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(combined)
        if matches:
            result[key] = float(matches[-1])

    topk_matches = TOPK_PATTERN.findall(combined)
    if topk_matches:
        k, value = topk_matches[-1]
        result["topk_name"] = f"top{k}_acc"
        result["topk_acc"] = float(value)

    if proc.returncode != 0:
        tail_lines = [line for line in combined.strip().splitlines() if line.strip()]
        result["error_tail"] = "\n".join(tail_lines[-20:])

    return result


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    results = []
    for loss_type in ["cross_entropy", "focal", "rank_aware"]:
        print(f"\n=== 开始对比: {loss_type} ===")
        results.append(_run_one(loss_type))

    results.sort(
        key=lambda item: (
            item.get("returncode", 1),
            -(item.get("macro_f1", -1.0) if isinstance(item.get("macro_f1"), float) else -1.0),
            -(item.get("top1_acc", -1.0) if isinstance(item.get("top1_acc"), float) else -1.0),
        )
    )

    summary = {
        "dataset_path": DATA_PATH,
        "sample_size": SAMPLE_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM,
        "seed": SEED,
        "total_elapsed_seconds": round(time.time() - started, 2),
        "results": results,
        "recommended_loss": next((item["loss_type"] for item in results if item.get("returncode") == 0), ""),
    }

    summary_path = REPORT_DIR / "loss_compare_latest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_lines = [
        "# 分类损失函数小规模对比",
        "",
        f"- 数据集: `{DATA_PATH}`",
        f"- 样本规模: `{SAMPLE_SIZE}`",
        f"- Epoch: `{EPOCHS}`",
        f"- Batch Size: `{BATCH_SIZE}`",
        f"- Grad Accum: `{GRAD_ACCUM}`",
        "",
        "| Loss | Return | Top-1 | Top-3 | Macro-F1 | Weighted-F1 | 耗时(s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        markdown_lines.append(
            "| {loss} | {ret} | {top1} | {topk} | {macro} | {weighted} | {elapsed} |".format(
                loss=item["loss_type"],
                ret=item.get("returncode", ""),
                top1=item.get("top1_acc", ""),
                topk=item.get("topk_acc", ""),
                macro=item.get("macro_f1", ""),
                weighted=item.get("weighted_f1", ""),
                elapsed=item.get("elapsed_seconds", ""),
            )
        )
    markdown_lines.append("")
    markdown_lines.append(f"- 推荐损失: `{summary['recommended_loss']}`")
    markdown_lines.append("")
    for item in results:
        markdown_lines.append(f"- `{item['loss_type']}` 日志: `{item['log_path']}`")
    (REPORT_DIR / "LOSS_COMPARE.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"REPORT={summary_path}")


if __name__ == "__main__":
    main()
