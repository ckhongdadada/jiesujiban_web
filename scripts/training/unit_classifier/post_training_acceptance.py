from __future__ import annotations

import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.core.config import load_runtime_config
from src.jsjb.unit_classifier.acceptance import (
    build_acceptance_report,
    format_acceptance_report,
    write_acceptance_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练完成后的模型接入验收")
    parser.add_argument("--target", choices=["all", "classifier", "generator"], default="all")
    parser.add_argument("--classifier-model-dir", default=None)
    parser.add_argument("--classifier-base-model", default=None)
    parser.add_argument("--generator-base-model", default=None)
    parser.add_argument("--generator-lora-dir", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runtime_config()

    report = build_acceptance_report(
        target=args.target,
        classifier_model_dir=args.classifier_model_dir or config.classifier_model_dir,
        classifier_base_model=args.classifier_base_model or config.classifier_base_model,
        generator_base_model=args.generator_base_model or config.generator_base_model,
        generator_lora_dir=args.generator_lora_dir or config.generator_lora_dir,
    )

    text = format_acceptance_report(report)
    print(text)

    if args.report_path:
        report_path = pathlib.Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
        print(f"\n报告已写入: {report_path}")
    elif not args.no_write_report:
        paths = write_acceptance_report(report, report_dir=args.report_dir)
        print(f"\n报告已写入: {paths['latest']}")
        print(f"归档报告: {paths['archived']}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
