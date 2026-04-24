from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from enhancements.model_artifacts import inspect_classifier_artifacts, inspect_generator_artifacts
from enhancements.data_paths import get_training_reports_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = get_training_reports_dir()


def build_acceptance_report(
    target: str = "all",
    classifier_model_dir: str = "",
    classifier_base_model: str = "",
    generator_base_model: str = "",
    generator_lora_dir: str = "",
) -> dict[str, Any]:
    target = target.lower()
    if target not in {"all", "classifier", "generator"}:
        raise ValueError("target must be one of: all, classifier, generator")

    checks: dict[str, Any] = {}
    passed = True

    if target in {"all", "classifier"}:
        classifier = inspect_classifier_artifacts(classifier_model_dir, classifier_base_model)
        checks["classifier"] = classifier
        passed = passed and classifier["compatible_runtime_ready"]

    if target in {"all", "generator"}:
        generator = inspect_generator_artifacts(generator_base_model, generator_lora_dir)
        checks["generator"] = generator
        passed = passed and generator["runtime_ready"]

    return {
        "target": target,
        "passed": passed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checks": checks,
    }


def format_acceptance_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)


def write_acceptance_report(
    report: dict[str, Any],
    report_dir: str | Path | None = None,
    filename_prefix: str | None = None,
) -> dict[str, str]:
    target = str(report.get("target") or "all")
    prefix = filename_prefix or target
    output_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = output_dir / f"{prefix}_latest.json"
    archived_path = output_dir / f"{prefix}_{timestamp}.json"
    payload = format_acceptance_report(report)

    latest_path.write_text(payload, encoding="utf-8")
    archived_path.write_text(payload, encoding="utf-8")

    return {
        "latest": str(latest_path),
        "archived": str(archived_path),
    }
