from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.evaluation.batch_quality import BatchQualityEvaluator, read_input_rows, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate /api/analyze and generated-reply quality attribution.")
    parser.add_argument("--input", required=True, help="Input .xlsx/.csv/.json/.jsonl file.")
    parser.add_argument("--output", default="data/reports/batch_api_analyze_eval.json", help="Output JSON report path.")
    parser.add_argument("--api-url", default="http://127.0.0.1:5000/api/analyze", help="Analyze API URL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max sample count.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_input_rows(args.input)
    evaluator = BatchQualityEvaluator(api_url=args.api_url, timeout_seconds=args.timeout)
    result = evaluator.evaluate_rows(rows, limit=args.limit)
    written = write_outputs(result, args.output)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("\n输出文件:")
    for kind, path in written.items():
        print(f"  {kind}: {path}")
    return 0 if result["summary"]["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
