from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enhancements.data_paths import (
    get_data_root,
    get_place_taxonomy_dir,
    get_runtime_district_file,
    get_place_records_path,
    get_bulk_records_path,
    get_universe_v2_dir,
    get_runtime_catalog_path,
    get_reports_dir,
)

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

GENERIC_STOP_ALIASES = {"北京", "北京市"}
COMMUNITY_SUFFIX_PATTERN = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9]{2,30}(?:小区|家园|花园|家苑|佳苑|嘉园|佳园|名苑|雅苑|公寓|华庭|豪庭|公馆|新村|社区|生活区|家属院|家属楼|宿舍|庄园|名城|别墅)$"
)
ROAD_SUFFIXES = ("路", "街", "胡同", "巷", "大街", "大道", "辅路", "环路", "街道")

CATEGORY_RULES = [
    ("bus_stop", ["公交站", "公交场站", "公交枢纽", "首末站", "客运站", "公交站点"]),
    ("subway_station", ["地铁站"]),
    ("mall", ["商场", "购物中心", "奥特莱斯", "大悦城", "银泰", "万达", "广场"]),
    ("park", ["公园", "湿地公园", "森林公园"]),
    ("cinema", ["影院", "影城", "电影城"]),
    ("school", ["幼儿园", "小学", "中学", "学院", "大学", "学校"]),
    ("hospital", ["医院", "卫生院", "中医院"]),
    ("hotel", ["酒店", "宾馆", "旅社"]),
]

CATEGORY_PRIORITY = {
    "dictionary_core": 0.88,
    "community": 0.92,
    "street": 0.90,
    "subway_station": 0.88,
    "road": 0.84,
    "bus_stop": 0.80,
    "hospital": 0.72,
    "school": 0.70,
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


def load_existing_taxonomy(taxonomy_dir: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(taxonomy_dir):
        return rows
    for name in os.listdir(taxonomy_dir):
        if not name.endswith(".json"):
            continue
        category = os.path.splitext(name)[0]
        path = os.path.join(taxonomy_dir, name)
        data = load_json(path)
        for district, aliases in data.items():
            if district not in DISTRICTS:
                continue
            for alias in aliases:
                normalized_alias = normalize_text(alias)
                if normalized_alias:
                    rows.append(
                        {
                            "alias": normalized_alias,
                            "district": district,
                            "category": category,
                            "source": f"existing_taxonomy:{category}",
                        }
                    )
    return rows


def infer_category(name: str, source_category: str) -> str:
    text = normalize_text(name)
    source_category = normalize_text(source_category)

    if source_category in {"street", "subway_station", "bus_stop", "road", "community"}:
        return source_category

    for category, keywords in CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return category

    if COMMUNITY_SUFFIX_PATTERN.search(text):
        return "community"
    if text.endswith(ROAD_SUFFIXES):
        return "road"
    return "poi_misc"


def clean_alias(name: str, category: str) -> str:
    text = normalize_text(name)
    text = re.sub(r"[（(].*?[）)]", "", text)
    text = text.replace("地铁站", "") if category == "subway_station" else text
    text = text.replace("公交站", "") if category == "bus_stop" else text
    text = normalize_text(text)
    if category == "street":
        if "街道办事处" in text:
            text = text.replace("街道办事处", "街道")
        elif text.endswith("办事处"):
            text = text[:-3]
    if len(text) < 2 or len(text) > 40 or text in GENERIC_STOP_ALIASES:
        return ""
    return text


def build_empty_district_map() -> dict[str, set[str]]:
    return {district: set() for district in DISTRICTS}


def add_catalog_entry(
    category_dicts: dict[str, dict[str, set[str]]],
    catalog_entries: dict[tuple[str, str], CatalogEntry],
    alias: str,
    district: str,
    category: str,
    source: str,
    confidence: float,
) -> None:
    if district not in DISTRICTS or not alias:
        return
    category_dicts[category][district].add(alias)
    key = (alias, district)
    existing = catalog_entries.get(key)
    if existing and existing.confidence >= confidence:
        return
    catalog_entries[key] = CatalogEntry(
        alias=alias,
        district=district,
        category=category,
        confidence=confidence,
        source=source,
    )


def load_road_rows(project_root: str, data_dir: str) -> list[dict]:
    candidate_paths = [
        str(get_universe_v2_dir() / "road_township_beijing.jsonl"),
        os.path.join(project_root, "tools", "road_township_dictionary.jsonl"),
    ]
    for path in candidate_paths:
        rows = load_jsonl(path)
        if rows:
            return rows
    return []


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = str(get_data_root())
    taxonomy_dir = str(get_place_taxonomy_dir())
    merged_path = str(get_runtime_district_file("beijing_districts_merged.json"))
    curated_communities_path = str(get_runtime_district_file("beijing_communities_curated.json"))
    bulk_path = str(get_bulk_records_path())
    seed_path = str(get_place_records_path("beijing_place_records.jsonl"))

    merged = load_json(merged_path)
    curated_communities = load_json(curated_communities_path) if os.path.exists(curated_communities_path) else {}
    poi_rows = load_jsonl(seed_path) + load_jsonl(bulk_path)
    road_rows = load_road_rows(project_root, data_dir)
    existing_taxonomy_rows = load_existing_taxonomy(taxonomy_dir)

    category_dicts: dict[str, dict[str, set[str]]] = defaultdict(build_empty_district_map)
    catalog_entries: dict[tuple[str, str], CatalogEntry] = {}

    for row in existing_taxonomy_rows:
        category = row["category"]
        add_catalog_entry(
            category_dicts,
            catalog_entries,
            row["alias"],
            row["district"],
            category,
            row["source"],
            CATEGORY_PRIORITY.get(category, 0.52),
        )

    for district, aliases in merged.items():
        if district not in DISTRICTS:
            continue
        for alias in aliases:
            normalized_alias = normalize_text(alias)
            if not normalized_alias or normalized_alias in GENERIC_STOP_ALIASES:
                continue
            category = "community" if normalized_alias in curated_communities.get(district, []) else "dictionary_core"
            add_catalog_entry(
                category_dicts,
                catalog_entries,
                normalized_alias,
                district,
                category,
                "merged_dictionary",
                CATEGORY_PRIORITY.get(category, 0.8),
            )

    for row in poi_rows:
        district = normalize_text(row.get("district", ""))
        if district not in DISTRICTS:
            continue
        category = infer_category(row.get("name", ""), row.get("category", ""))
        alias = clean_alias(row.get("name", ""), category)
        if not alias:
            continue
        add_catalog_entry(
            category_dicts,
            catalog_entries,
            alias,
            district,
            category,
            row.get("source", "raw_poi"),
            CATEGORY_PRIORITY.get(category, 0.52),
        )

    for row in road_rows:
        district = normalize_text(row.get("district", ""))
        road = clean_alias(row.get("road", ""), "road")
        if district not in DISTRICTS or not road:
            continue
        add_catalog_entry(
            category_dicts,
            catalog_entries,
            road,
            district,
            "road",
            row.get("source", "amap_regeo_road"),
            CATEGORY_PRIORITY["road"],
        )

    os.makedirs(taxonomy_dir, exist_ok=True)
    taxonomy_summary: dict[str, dict[str, int]] = {}
    for category, district_map in category_dicts.items():
        output = {district: sorted(values) for district, values in district_map.items() if values}
        if not output:
            continue
        with open(os.path.join(taxonomy_dir, f"{category}.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        taxonomy_summary[category] = {district: len(values) for district, values in output.items()}

    catalog_path = str(get_runtime_catalog_path())
    with open(catalog_path, "w", encoding="utf-8") as f:
        for entry in sorted(catalog_entries.values(), key=lambda item: (item.category, item.district, item.alias)):
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    report = {
        "catalog_entries": len(catalog_entries),
        "categories": {category: sum(district_map.values()) for category, district_map in taxonomy_summary.items()},
        "taxonomy_dir": taxonomy_dir,
        "catalog_path": catalog_path,
        "road_source_rows": len(road_rows),
    }
    report_path = get_reports_dir() / "crawl" / "place_taxonomy_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
