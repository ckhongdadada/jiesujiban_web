from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.core.paths import get_evaluation_data_dir, get_evaluation_outputs_dir
from src.jsjb.evaluation.rag_generation import (
    Phase1EvaluationConfig,
    Phase1RagGenerationEvaluator,
    read_phase1_cases,
    write_phase1_outputs,
)


def post_json(url: str, payload: dict[str, Any], timeout_seconds: int = 180) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {message}") from exc


def collect_api_records(cases: list[dict[str, Any]], api_url: str, timeout_seconds: int, limit: int | None = None) -> list[dict[str, Any]]:
    records = []
    for index, case in enumerate(cases):
        if limit is not None and index >= limit:
            break
        payload = {
            "tag": case.get("tag", ""),
            "title": case.get("title", ""),
            "body": case.get("body", ""),
            "_debug": True,
        }
        started = time.time()
        try:
            response = post_json(api_url, payload, timeout_seconds=timeout_seconds)
            records.append({**case, "response": response, "api_elapsed_seconds": round(time.time() - started, 4)})
            print(f"[ok] {case.get('id', index)} elapsed={records[-1]['api_elapsed_seconds']}s")
        except Exception as exc:
            records.append({**case, "response": {"status": "error", "reply": "", "retrieval": []}, "api_error": str(exc)})
            print(f"[error] {case.get('id', index)} {exc}")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1 RAG + generation evaluation.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "evaluation" / "phase1_rag_generation.json"))
    parser.add_argument("--input", default=str(get_evaluation_data_dir() / "phase1_fixed_cases.jsonl"))
    parser.add_argument("--output-dir", default=str(get_evaluation_outputs_dir()))
    parser.add_argument("--api-url", default="http://127.0.0.1:5000/api/analyze")
    parser.add_argument("--offline", action="store_true", help="Evaluate records already containing response/retrieval/reply fields.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_payload = {}
    config_path = Path(args.config)
    if config_path.exists():
        config_payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if args.input == str(get_evaluation_data_dir() / "phase1_fixed_cases.jsonl"):
            args.input = str(PROJECT_ROOT / config_payload.get("fixed_eval_set", args.input))
        if args.output_dir == str(get_evaluation_outputs_dir()):
            args.output_dir = str(PROJECT_ROOT / config_payload.get("output_dir", args.output_dir))
        if args.api_url == "http://127.0.0.1:5000/api/analyze":
            args.api_url = config_payload.get("default_api_url", args.api_url)
        if args.top_k == 3:
            args.top_k = int(config_payload.get("retrieval_top_k", args.top_k))
    cases = read_phase1_cases(args.input)
    if args.limit is not None:
        cases = cases[: args.limit]

    if args.offline:
        records = cases
    else:
        records = collect_api_records(cases, args.api_url, args.timeout, limit=args.limit)

    thresholds = config_payload.get("thresholds", {}) if isinstance(config_payload, dict) else {}
    evaluator = Phase1RagGenerationEvaluator(
        Phase1EvaluationConfig(
            retrieval_top_k=args.top_k,
            min_evidence_coverage=float(thresholds.get("min_evidence_coverage", 0.35)),
            min_retrieval_precision=float(thresholds.get("min_retrieval_precision", 0.34)),
            min_format_score=float(thresholds.get("min_format_score", 0.85)),
        )
    )
    result = evaluator.evaluate_records(records)
    result["source_input"] = str(args.input)
    result["api_url"] = "offline" if args.offline else args.api_url
    written = write_phase1_outputs(result, args.output_dir)
    print("\n=== Phase 1 Evaluation Summary ===")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("\nWritten outputs:")
    for key, value in written.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
