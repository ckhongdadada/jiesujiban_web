from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

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

COMMUNITY_KEYWORDS = [
    "小区",
    "家园",
    "佳苑",
    "花园",
    "公寓",
    "社区",
    "新村",
    "嘉园",
    "名苑",
    "西里",
    "东里",
    "南里",
    "北里",
    "一区",
    "二区",
    "三区",
    "四区",
    "五区",
    "六区",
    "七区",
    "八区",
    "九区",
    "十区",
    "一号院",
    "二号院",
    "三号院",
    "生活区",
    "住宅区",
    "家属院",
    "宿舍",
    "公馆",
    "华府",
    "府邸",
    "大院",
    "别墅",
    "名居",
    "雅苑",
    "锦园",
    "丽景",
    "绿洲",
    "都市",
    "城",
    "湾",
    "苑",
    "园",
    "里",
    "院",
    "庄",
    "台",
    "轩",
]

ENDPOINT = "https://restapi.amap.com/v3/place/text"
OFFSET = 20
MAX_PAGES_PER_QUERY = 50
REQUEST_DELAY_SECONDS = 0.18
TARGET_USAGE = 4500
BASELINE_USAGE = 80

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


@dataclass
class PlaceRecord:
    name: str
    district: str
    category: str
    source: str
    source_url: str
    address: str = ""
    location: str = ""
    adcode: str = ""


def normalize_text(text: str) -> str:
    return "".join((text or "").split()).strip()


def sanitize_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query.pop("key", None)
    query_items = []
    for key, values in query.items():
        for value in values:
            query_items.append((key, value))
    from urllib.parse import urlencode, urlunparse

    return urlunparse(parsed._replace(query=urlencode(query_items)))


def query_key(district: str, keyword: str) -> str:
    return f"{district}|{keyword}"


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_existing_state(records_paths: list[str]) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for path in records_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("source") != "amap_community":
                    continue
                source_url = row.get("source_url", "")
                parsed = urlparse(source_url)
                params = parse_qs(parsed.query)
                keyword_expr = params.get("keywords", [""])[0]
                page_text = params.get("page", ["1"])[0]
                district = ""
                keyword = ""
                for candidate_district in DISTRICTS:
                    if keyword_expr.startswith(candidate_district):
                        district = candidate_district
                        keyword = keyword_expr[len(candidate_district) :]
                        break
                if not district or not keyword:
                    continue
                key = query_key(district, keyword)
                entry = state.setdefault(
                    key,
                    {
                        "district": district,
                        "keyword": keyword,
                        "count": None,
                        "total_pages": None,
                        "crawled_pages": [],
                        "empty_pages": [],
                    },
                )
                page = int(page_text)
                if page not in entry["crawled_pages"]:
                    entry["crawled_pages"].append(page)
    for entry in state.values():
        entry["crawled_pages"].sort()
    return state


def fetch_page(
    session: requests.Session,
    api_key: str,
    district: str,
    keyword: str,
    page: int,
) -> tuple[list[dict[str, Any]], int, str]:
    params = {
        "key": api_key,
        "keywords": f"{district}{keyword}",
        "city": "北京",
        "citylimit": "true",
        "offset": OFFSET,
        "page": page,
        "extensions": "base",
        "types": "",
    }
    response = session.get(ENDPOINT, params=params, headers=HEADERS, timeout=20)
    response.raise_for_status()
    payload = response.json()
    pois = payload.get("pois") or []
    count = int(payload.get("count") or 0)
    records = []
    for poi in pois:
        poi_district = normalize_text(poi.get("adname") or "")
        name = normalize_text(poi.get("name") or "")
        if not name:
            continue
        records.append(
            asdict(
                PlaceRecord(
                    name=name,
                    district=poi_district,
                    category="community",
                    source="amap_community",
                    source_url=sanitize_url(response.url),
                    address=normalize_text(poi.get("address") or ""),
                    location=normalize_text(poi.get("location") or ""),
                    adcode=normalize_text(poi.get("adcode") or ""),
                )
            )
        )
    return records, count, sanitize_url(response.url)


def next_missing_page(entry: dict[str, Any]) -> int | None:
    total_pages = entry.get("total_pages")
    crawled_pages = set(entry.get("crawled_pages", []))
    if total_pages is None:
        return 1 if 1 not in crawled_pages else 2
    for page in range(1, total_pages + 1):
        if page not in crawled_pages:
            return page
    return None


def build_universe_state(existing_state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for district in DISTRICTS:
        for keyword in COMMUNITY_KEYWORDS:
            key = query_key(district, keyword)
            seeded = existing_state.get(key, {})
            state[key] = {
                "district": district,
                "keyword": keyword,
                "count": seeded.get("count"),
                "total_pages": seeded.get("total_pages"),
                "crawled_pages": seeded.get("crawled_pages", []),
                "empty_pages": seeded.get("empty_pages", []),
            }
    return state


def compute_remaining_pages(state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    known_remaining = 0
    unknown_queries = 0
    total_known_pages = 0
    crawled_pages = 0
    for entry in state.values():
        crawled = len(set(entry["crawled_pages"]))
        crawled_pages += crawled
        total_pages = entry.get("total_pages")
        if total_pages is None:
            unknown_queries += 1
            continue
        total_known_pages += total_pages
        known_remaining += max(total_pages - crawled, 0)
    return {
        "known_remaining_pages": known_remaining,
        "unknown_queries": unknown_queries,
        "known_total_pages": total_known_pages,
        "crawled_pages_in_universe": crawled_pages,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "raw" / "amap"
    api_key = os.getenv("AMAP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("AMAP_API_KEY is required")

    state_path = str(data_dir / "amap_community_bulk_state.json")
    raw_output_path = str(data_dir / "amap_community_bulk_records.jsonl")
    progress_path = str(data_dir / "amap_community_bulk_progress.json")

    existing_state = extract_existing_state(
        [
            str(project_root / "data" / "beijing_place_records.jsonl"),
            str(project_root / "data" / "beijing_place_records_communities.jsonl"),
            str(data_dir / "beijing_place_records.jsonl"),
            str(data_dir / "beijing_place_records_communities.jsonl"),
            raw_output_path,
        ]
    )
    state = build_universe_state(existing_state)

    persisted_state = load_json(state_path, {})
    for key, value in persisted_state.items():
        if key in state:
            state[key].update(value)

    additional_target = max(TARGET_USAGE - BASELINE_USAGE, 0)
    session = requests.Session()

    total_requests_made = sum(len(set(entry["crawled_pages"])) for entry in state.values())
    requests_remaining = max(additional_target - total_requests_made, 0)

    discovery_queue = deque()
    crawl_queue = deque()
    for district in DISTRICTS:
        for keyword in COMMUNITY_KEYWORDS:
            key = query_key(district, keyword)
            if 1 not in state[key]["crawled_pages"]:
                discovery_queue.append(key)
            else:
                crawl_queue.append(key)

    newly_written_records = 0
    start = time.time()

    while requests_remaining > 0 and (discovery_queue or crawl_queue):
        if discovery_queue:
            key = discovery_queue.popleft()
        else:
            key = crawl_queue.popleft()

        entry = state[key]
        page = next_missing_page(entry)
        if page is None:
            continue

        records, count, source_url = fetch_page(
            session,
            api_key,
            entry["district"],
            entry["keyword"],
            page,
        )
        total_pages = min(math.ceil(count / OFFSET) if count else page, MAX_PAGES_PER_QUERY)
        entry["count"] = count
        entry["total_pages"] = total_pages
        if page not in entry["crawled_pages"]:
            entry["crawled_pages"].append(page)
            entry["crawled_pages"].sort()
        if not records and page not in entry["empty_pages"]:
            entry["empty_pages"].append(page)

        if records:
            append_jsonl(raw_output_path, records)
            newly_written_records += len(records)

        state_snapshot = {k: v for k, v in state.items() if v["crawled_pages"]}
        save_json(state_path, state_snapshot)

        requests_remaining -= 1

        next_page = next_missing_page(entry)
        if next_page is not None and next_page <= (entry["total_pages"] or MAX_PAGES_PER_QUERY):
            crawl_queue.append(key)

        progress_payload = {
            "baseline_usage": BASELINE_USAGE,
            "target_usage": TARGET_USAGE,
            "additional_target_requests": additional_target,
            "total_requests_made_in_universe": sum(len(set(item["crawled_pages"])) for item in state.values()),
            "requests_remaining_to_target": requests_remaining,
            "newly_written_records": newly_written_records,
            "last_request": {
                "district": entry["district"],
                "keyword": entry["keyword"],
                "page": page,
                "count": count,
                "source_url": source_url,
            },
            "elapsed_seconds": round(time.time() - start, 1),
            "remaining_pages_summary": compute_remaining_pages(state),
        }
        save_json(progress_path, progress_payload)
        time.sleep(REQUEST_DELAY_SECONDS)

    final_payload = {
        "baseline_usage": BASELINE_USAGE,
        "target_usage": TARGET_USAGE,
        "additional_target_requests": additional_target,
        "total_requests_made_in_universe": sum(len(set(item["crawled_pages"])) for item in state.values()),
        "newly_written_records": newly_written_records,
        "elapsed_seconds": round(time.time() - start, 1),
        "remaining_pages_summary": compute_remaining_pages(state),
        "state_path": state_path,
        "raw_output_path": raw_output_path,
        "progress_path": progress_path,
    }
    save_json(progress_path, final_payload)
    print(json.dumps(final_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
