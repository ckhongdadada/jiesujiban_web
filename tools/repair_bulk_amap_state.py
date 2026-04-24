from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhancements.data_paths import (
    ensure_standard_data_layout,
    get_bulk_state_path,
    get_bulk_records_path,
    get_bulk_progress_path,
    get_bulk_repair_report_path,
)

BASELINE_USAGE = 80
TARGET_USAGE = 4500
DISTRICTS = [
    "东城区",
    "西城区",
    "朝阳区",
    "丰台区",
    "石景山区",
    "海淀区",
    "门头沟区",
    "房山区",
    "通州区",
    "顺义区",
    "昌平区",
    "大兴区",
    "怀柔区",
    "平谷区",
    "密云区",
    "延庆区",
]


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_keyword_expr(keyword_expr: str, fallback_district: str = "") -> tuple[str, str]:
    keyword_expr = (keyword_expr or "").strip()
    for district in DISTRICTS:
        if keyword_expr.startswith(district):
            return district, keyword_expr[len(district):].strip()
    return (fallback_district or "").strip(), keyword_expr.strip()


def iter_bulk_records(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("source") != "amap_community":
                continue
            yield row


def build_state_from_records(raw_output_path: Path) -> dict[str, dict]:
    state: dict[str, dict] = {}
    for row in iter_bulk_records(raw_output_path):
        source_url = row.get("source_url", "")
        parsed = urlparse(source_url)
        params = parse_qs(parsed.query)
        keyword_expr = unquote(params.get("keywords", [""])[0]).strip()
        page = int(params.get("page", ["1"])[0])
        fallback_district = (row.get("district") or "").strip()
        district, keyword = split_keyword_expr(keyword_expr, fallback_district)
        if not district or not keyword:
            continue
        key = f"{district}|{keyword}"
        entry = state.setdefault(
            key,
            {
                "district": district,
                "keyword": keyword,
                "count": None,
                "total_pages": None,
                "crawled_pages": set(),
                "empty_pages": set(),
            },
        )
        entry["crawled_pages"].add(page)
    return state


def merge_persisted_state(state: dict[str, dict], persisted_state: dict[str, dict]) -> dict[str, dict]:
    for key, value in persisted_state.items():
        district = value.get("district") or key.split("|", 1)[0]
        keyword = value.get("keyword") or key.split("|", 1)[1]
        entry = state.setdefault(
            key,
            {
                "district": district,
                "keyword": keyword,
                "count": None,
                "total_pages": None,
                "crawled_pages": set(),
                "empty_pages": set(),
            },
        )
        if value.get("count") is not None:
            entry["count"] = value.get("count")
        if value.get("total_pages") is not None:
            entry["total_pages"] = int(value.get("total_pages"))
        for page in value.get("crawled_pages", []):
            entry["crawled_pages"].add(int(page))
        for page in value.get("empty_pages", []):
            entry["empty_pages"].add(int(page))
    for entry in state.values():
        entry["crawled_pages"] = sorted(entry["crawled_pages"])
        entry["empty_pages"] = sorted(entry["empty_pages"])
    return state


def compute_remaining_pages(state: dict[str, dict]) -> dict[str, int]:
    known_remaining = 0
    unknown_queries = 0
    known_total_pages = 0
    crawled_pages_in_universe = 0
    for entry in state.values():
        crawled_count = len(set(entry.get("crawled_pages", [])))
        crawled_pages_in_universe += crawled_count
        total_pages = entry.get("total_pages")
        if total_pages is None:
            unknown_queries += 1
            continue
        known_total_pages += int(total_pages)
        known_remaining += max(int(total_pages) - crawled_count, 0)
    return {
        "known_remaining_pages": known_remaining,
        "unknown_queries": unknown_queries,
        "known_total_pages": known_total_pages,
        "crawled_pages_in_universe": crawled_pages_in_universe,
    }


def collect_overcrawled(state: dict[str, dict]) -> list[dict]:
    over = []
    for key, entry in state.items():
        total_pages = entry.get("total_pages")
        crawled = len(set(entry.get("crawled_pages", [])))
        if total_pages is not None and crawled > int(total_pages):
            over.append(
                {
                    "query": key,
                    "district": entry.get("district"),
                    "keyword": entry.get("keyword"),
                    "crawled_pages": crawled,
                    "total_pages": int(total_pages),
                }
            )
    over.sort(key=lambda item: (item["crawled_pages"] - item["total_pages"], item["query"]), reverse=True)
    return over


def main() -> None:
    ensure_standard_data_layout()
    state_path = get_bulk_state_path()
    raw_output_path = get_bulk_records_path()
    progress_path = get_bulk_progress_path()
    report_path = get_bulk_repair_report_path()

    rebuilt_state = build_state_from_records(raw_output_path)
    persisted_state = load_json(state_path, {})
    state = merge_persisted_state(rebuilt_state, persisted_state)
    remaining_summary = compute_remaining_pages(state)
    overcrawled = collect_overcrawled(state)
    total_requests = sum(len(set(entry.get("crawled_pages", []))) for entry in state.values())

    save_json(state_path, state)

    progress_payload = {
        "baseline_usage": BASELINE_USAGE,
        "target_usage": TARGET_USAGE,
        "additional_target_requests": max(TARGET_USAGE - BASELINE_USAGE, 0),
        "total_requests_made_in_universe": total_requests,
        "newly_written_records": 0,
        "elapsed_seconds": 0.0,
        "remaining_pages_summary": remaining_summary,
        "state_path": str(state_path),
        "raw_output_path": str(raw_output_path),
        "progress_path": str(progress_path),
        "repair_status": {
            "repaired_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "overcrawled_queries": len(overcrawled),
            "state_rebuilt_from_raw_records": True,
        },
    }
    save_json(progress_path, progress_payload)

    report_payload = {
        "queries": len(state),
        "remaining_pages_summary": remaining_summary,
        "overcrawled_queries": len(overcrawled),
        "overcrawled_sample": overcrawled[:20],
        "paths": {
            "state_path": str(state_path),
            "raw_output_path": str(raw_output_path),
            "progress_path": str(progress_path),
        },
        "conclusion": "current_universe_complete" if remaining_summary["known_remaining_pages"] == 0 and remaining_summary["unknown_queries"] == 0 else "needs_expansion_or_continuation",
    }
    save_json(report_path, report_payload)
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
