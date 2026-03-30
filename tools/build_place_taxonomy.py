from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass

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

CATEGORY_RULES = [
    ("bus_stop", ["公交站", "客运站", "枢纽站"]),
    ("subway_station", ["地铁站"]),
    ("mall", ["商场", "购物中心", "奥特莱斯", "大悦城", "银泰", "万达"]),
    ("park", ["公园", "湿地公园", "森林公园"]),
    ("cinema", ["影院", "影城", "电影院"]),
    ("school", ["幼儿园", "小学", "中学", "学院", "大学", "学校"]),
    ("hospital", ["医院", "卫生院", "中医院"]),
    ("hotel", ["酒店", "宾馆"]),
]

CATEGORY_PRIORITY = {
    "dictionary_core": 0.88,
    "community": 0.92,
    "street": 0.90,
    "subway_station": 0.88,
    "road": 0.84,
    "bus_stop": 0.8,
    "hospital": 0.72,
    "school": 0.7,
    "mall": 0.68,
    "park": 0.66,
    "cinema": 0.62,
    "hotel": 0.56,
    "poi_misc": 0.52,
}


@dataclass
class CatalogEntry:
    alias: str
    district: str
    category: str
    confidence: float
    source: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_category(name: str, source_category: str) -> str:
    text = normalize_text(name)
    if source_category in {"street", "subway_station"}:
        return source_category
    for category, keywords in CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    if re.search(r"[\u4e00-\u9fa5A-Za-z0-9]{2,24}(?:小区|家园|花园|华庭|苑|园|城|湾|府|郡|阁|居|国际|新村|嘉园|雅苑|公寓|里|庄)$", text):
        return "community"
    if text.endswith(("路", "街", "胡同", "大道", "巷")):
        return "road"
    return "poi_misc"


def clean_alias(name: str, category: str) -> str:
    text = normalize_text(name)
    text = re.sub(r"[（(].*?[）)]", "", text)
    if category == "subway_station":
        text = text.replace("地铁站", "")
    if category == "bus_stop":
        text = text.replace("公交站", "")
    if category == "street":
        match = re.search(r"[\u4e00-\u9fa5A-Za-z0-9]{2,18}街道", text)
        return match.group(0) if match else ""
    if len(text) < 2 or len(text) > 30:
        return ""
    return text


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    merged_path = os.path.join(data_dir, "beijing_districts_merged.json")
    curated_communities_path = os.path.join(data_dir, "beijing_communities_curated.json")
    bulk_path = os.path.join(data_dir, "amap_community_bulk_records.jsonl")
    seed_path = os.path.join(data_dir, "beijing_place_records.jsonl")

    merged = load_json(merged_path)
    curated_communities = load_json(curated_communities_path) if os.path.exists(curated_communities_path) else {}
    rows = load_jsonl(seed_path) + load_jsonl(bulk_path)

    category_dicts: dict[str, dict[str, set[str]]] = defaultdict(lambda: {district: set() for district in DISTRICTS})
    catalog_entries: dict[tuple[str, str], CatalogEntry] = {}

    for district, aliases in merged.items():
        if district not in DISTRICTS:
            continue
        for alias in aliases:
            if alias in {"北京", "北京市"}:
                continue
            category = "community" if alias in curated_communities.get(district, []) else "dictionary_core"
            category_dicts[category][district].add(alias)
            catalog_entries[(alias, district)] = CatalogEntry(
                alias=alias,
                district=district,
                category=category,
                confidence=CATEGORY_PRIORITY.get(category, 0.8),
                source="merged_dictionary",
            )

    for row in rows:
        district = normalize_text(row.get("district", ""))
        if district not in DISTRICTS:
            continue
        category = infer_category(row.get("name", ""), normalize_text(row.get("category", "")))
        alias = clean_alias(row.get("name", ""), category)
        if not alias or alias in {"北京", "北京市"}:
            continue
        category_dicts[category][district].add(alias)
        key = (alias, district)
        confidence = CATEGORY_PRIORITY.get(category, 0.52)
        existing = catalog_entries.get(key)
        if existing and existing.confidence >= confidence:
            continue
        catalog_entries[key] = CatalogEntry(
            alias=alias,
            district=district,
            category=category,
            confidence=confidence,
            source=row.get("source", "raw_poi"),
        )

    taxonomy_dir = os.path.join(data_dir, "place_taxonomy")
    os.makedirs(taxonomy_dir, exist_ok=True)
    taxonomy_summary: dict[str, dict[str, int]] = {}
    for category, district_map in category_dicts.items():
        output = {district: sorted(values) for district, values in district_map.items() if values}
        if not output:
            continue
        with open(os.path.join(taxonomy_dir, f"{category}.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        taxonomy_summary[category] = {district: len(values) for district, values in output.items()}

    catalog_path = os.path.join(data_dir, "place_alias_catalog.jsonl")
    with open(catalog_path, "w", encoding="utf-8") as f:
        for entry in sorted(catalog_entries.values(), key=lambda item: (item.category, item.district, item.alias)):
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    report = {
        "catalog_entries": len(catalog_entries),
        "categories": {category: sum(district_map.values()) for category, district_map in taxonomy_summary.items()},
        "taxonomy_dir": taxonomy_dir,
        "catalog_path": catalog_path,
    }
    with open(os.path.join(data_dir, "place_taxonomy_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
