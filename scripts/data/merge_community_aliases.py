from __future__ import annotations

import json
import os
import re
from collections import defaultdict

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

COMMUNITY_DROP_KEYWORDS = [
    "社区服务",
    "居委会",
    "委员会",
    "工作站",
    "服务中心",
    "医院",
    "酒店",
    "停车场",
    "公交站",
    "大堂",
    "警务",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def load_json(path: str) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_community_name(name: str) -> str:
    text = normalize_text(name)
    text = re.sub(r"[（(].*?[）)]", "", text)
    if any(keyword in text for keyword in COMMUNITY_DROP_KEYWORDS):
        return ""
    if len(text) < 2 or len(text) > 24:
        return ""

    patterns = [
        r"[\u4e00-\u9fa5A-Za-z0-9]{2,24}(?:小区|家园|花园|华庭|苑|园|城|湾|府|郡|阁|居|国际|新村|嘉园|雅苑|公寓|里|庄|巷)$",
        r"[\u4e00-\u9fa5A-Za-z0-9]{2,24}[ABCDEF]区$",
        r"[\u4e00-\u9fa5A-Za-z0-9]{2,24}(?:东区|西区|南区|北区|一区|二区|三区|四区)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    base_curated_path = os.path.join(project_root, "data", "beijing_districts_curated.json")
    community_records_path = os.path.join(project_root, "data", "beijing_place_records_communities.jsonl")
    bulk_records_path = os.path.join(project_root, "data", "amap_community_bulk_records.jsonl")
    community_curated_path = os.path.join(project_root, "data", "beijing_communities_curated.json")
    merged_path = os.path.join(project_root, "data", "beijing_districts_merged.json")
    report_path = os.path.join(project_root, "data", "community_merge_report.json")

    base_curated = load_json(base_curated_path)
    rows = []
    if os.path.exists(community_records_path):
        rows.extend(load_jsonl(community_records_path))
    if os.path.exists(bulk_records_path):
        rows.extend(load_jsonl(bulk_records_path))

    community_grouped: dict[str, set[str]] = defaultdict(set)
    kept = 0
    dropped = 0
    for row in rows:
        district = normalize_text(row.get("district", ""))
        if district not in DISTRICTS:
            dropped += 1
            continue
        alias = clean_community_name(row.get("name", ""))
        if not alias:
            dropped += 1
            continue
        community_grouped[district].add(alias)
        kept += 1

    community_curated = {
        district: sorted(community_grouped.get(district, set()))
        for district in DISTRICTS
    }
    merged = {}
    for district in DISTRICTS:
        merged[district] = sorted(
            set(base_curated.get(district, [])) | set(community_curated.get(district, []))
        )

    with open(community_curated_path, "w", encoding="utf-8") as f:
        json.dump(community_curated, f, ensure_ascii=False, indent=2)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    report = {
        "community_input_records": len(rows),
        "community_kept_aliases": kept,
        "community_dropped_records": dropped,
        "input_sources": {
            "community_records_path": os.path.exists(community_records_path),
            "bulk_records_path": os.path.exists(bulk_records_path),
        },
        "community_counts": {district: len(community_curated[district]) for district in DISTRICTS},
        "merged_counts": {district: len(merged[district]) for district in DISTRICTS},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
