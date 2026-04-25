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

DROP_KEYWORDS = [
    "委员会",
    "工作站",
    "党",
    "保障",
    "执法",
    "事务所",
    "服务中心",
    "办事大厅",
    "医院",
    "警务",
    "检察",
    "法官",
    "劳动",
    "居委会",
    "社区居",
    "公交站",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def curate_name(name: str, category: str) -> str:
    text = normalize_text(name)
    if not text:
        return ""

    if category == "street":
        matches = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,12}街道", text)
        return matches[0] if matches else ""

    if category == "subway_station":
        text = re.sub(r"[（(].*?[）)]", "", text)
        text = text.replace("地铁站", "")
        if len(text) < 2 or len(text) > 10:
            return ""
        return text

    if category == "community":
        patterns = [
            r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}(?:小区|家园|花园|华庭|苑|园|里|城|湾|府|郡|阁|居)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""

    return ""


def should_drop(name: str) -> bool:
    if not name:
        return True
    if name in {"北京"}:
        return True
    return any(keyword in name for keyword in DROP_KEYWORDS)


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(project_root, "data", "beijing_place_records.jsonl")
    output_path = os.path.join(project_root, "data", "beijing_districts_curated.json")
    report_path = os.path.join(project_root, "data", "curation_report.json")

    records = load_records(input_path)
    grouped: dict[str, set[str]] = defaultdict(set)
    kept = 0
    dropped = 0

    for record in records:
        district = normalize_text(record.get("district", ""))
        category = normalize_text(record.get("category", ""))
        if district not in DISTRICTS:
            dropped += 1
            continue
        alias = curate_name(record.get("name", ""), category)
        if should_drop(alias):
            dropped += 1
            continue
        grouped[district].add(alias)
        kept += 1

    curated = {district: sorted(grouped.get(district, set())) for district in DISTRICTS}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(curated, f, ensure_ascii=False, indent=2)

    report = {
        "input_records": len(records),
        "kept_aliases": kept,
        "dropped_records": dropped,
        "district_counts": {district: len(curated[district]) for district in DISTRICTS},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
