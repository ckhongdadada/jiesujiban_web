from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.evaluation.reply_quality import (
    ReplyQualityEvaluator,
    load_reply_quality_config_file,
    read_quality_rows,
    write_quality_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated reply quality with standardized metrics.")
    parser.add_argument("--input-file", required=True, help="Input .xlsx/.csv/.json/.jsonl with generated/reference replies.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to outputs/evaluation/reply_quality/<timestamp>.",
    )
    parser.add_argument("--use-bge", action="store_true", help="Use BGE/SentenceTransformer cosine similarity if available.")
    parser.add_argument("--bge-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=260)
    parser.add_argument(
        "--config",
        default="",
        help="Optional evaluation config JSON. Defaults to configs/evaluation/reply_quality.json.",
    )
    parser.add_argument(
        "--weights-json",
        default="",
        help="Optional JSON object for quality score weights.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_quality_rows(args.input_file)
    config = load_reply_quality_config_file(args.config or None)
    if args.weights_json:
        config.weights = json.loads(args.weights_json)
    if args.use_bge:
        config.use_bge_similarity = True
    if args.bge_model:
        config.bge_model_name = args.bge_model
    if args.min_length != 40:
        config.min_length = args.min_length
    if args.max_length != 260:
        config.max_length = args.max_length
    evaluator = ReplyQualityEvaluator(config=config)
    result = evaluator.evaluate_rows(rows)
    output_dir = args.output_dir
    if not output_dir:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = str(PROJECT_ROOT / "outputs" / "evaluation" / "reply_quality" / timestamp)
    written = write_quality_outputs(result, output_dir)

    print("Reply quality evaluation completed")
    print(f"  samples: {result['metrics'].get('sample_count', 0)}")
    print(f"  quality_score: {result['metrics'].get('metric_averages', {}).get('quality_score', 0)}")
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
