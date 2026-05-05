#!/usr/bin/env python3
"""
将 data/processed/purified/ 下的政府机关、商场、医院、学校等数据
导入 place_alias_catalog.jsonl，补充非小区类地名。
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.jsjb.location.rule_based import DISTRICT_NAMES

PURIFIED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "purified")
CATALOG_PATH = os.path.join(PROJECT_ROOT, "data", "runtime", "place_alias_catalog.jsonl")

CATEGORY_MAP = {
    "government.jsonl": "government",
    "commercial.jsonl": "mall",
    "hospital.jsonl": "hospital",
    "school.jsonl": "school",
    "transit.jsonl": "transit_facility",
    "community_clean.jsonl": None,
}

GOVERNMENT_KEYWORDS = re.compile(
    r"政府|街道办|居委会|派出所|城管|执法局|信访|政务|民政局|住建委|房管|"
    r"交通委|水务局|环保局|卫健委|教委|市场监管局|应急|消防|公安|检察院|法院"
)

PARK_KEYWORDS = re.compile(r"公园|绿地|森林公园|湿地公园|郊野公园|体育公园")

RIVER_LAKE_KEYWORDS = re.compile(r"河|湖|水库|渠|塘|淀|潭|池|泉|溪")

BRIDGE_KEYWORDS = re.compile(r"桥|大桥|立交桥|天桥|高架")

MARKET_KEYWORDS = re.compile(r"市场|批发|菜市场|农贸市场|集贸市场")

CULTURAL_KEYWORDS = re.compile(r"博物馆|图书馆|文化馆|美术馆|剧院|音乐厅|体育馆|体育场|展览馆|科技馆")

MALL_KEYWORDS = re.compile(r"商场|购物中心|百货|商城|广场|天街|大悦城|万达|银泰|SKP|合生汇|荟聚")

SCHOOL_KEYWORDS = re.compile(r"小学|中学|高中|幼儿园|学校|大学|学院|附中|附小|实验学校")

HOSPITAL_KEYWORDS = re.compile(r"医院|诊所|卫生服务中心|卫生服务站|门诊|急救|疾控")

TRANSIT_NOISE = re.compile(
    r"停车场|充电站|加油站|洗车|驾校|租车|维修|4S店|检测场|"
    r"出入口|入口|出口|地上|地下|车位|车位锁"
)

WEAK_GENERIC = {
    "北京", "北京市", "中国", "小区", "社区", "公寓", "大厦", "广场",
    "家园", "花园", "酒店", "宾馆", "饭店", "餐厅", "超市", "便利店",
}


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip())


def extract_district(record: dict) -> str:
    district = record.get("district", "")
    if district in DISTRICT_NAMES:
        return district
    for field in ["address", "name"]:
        text = record.get(field, "")
        for d in DISTRICT_NAMES:
            if d in text or d.replace("区", "") in text:
                return d
    return district


def classify_record(name: str, default_category: str) -> str | None:
    if GOVERNMENT_KEYWORDS.search(name):
        return "government"
    if PARK_KEYWORDS.search(name):
        return "park"
    if BRIDGE_KEYWORDS.search(name):
        return "bridge"
    if RIVER_LAKE_KEYWORDS.search(name):
        return "river_lake"
    if SCHOOL_KEYWORDS.search(name):
        return "school"
    if HOSPITAL_KEYWORDS.search(name):
        return "hospital"
    if MALL_KEYWORDS.search(name):
        return "mall"
    if MARKET_KEYWORDS.search(name):
        return "market"
    if CULTURAL_KEYWORDS.search(name):
        return "cultural_site"
    return default_category if default_category else None


def is_noise(name: str) -> bool:
    if len(name) < 2 or len(name) > 30:
        return True
    if name in WEAK_GENERIC:
        return True
    if TRANSIT_NOISE.search(name):
        return True
    if re.match(r"^[A-Za-z0-9\-–—]+$", name):
        return True
    return False


def load_existing_catalog() -> set[tuple[str, str, str]]:
    existing: set[tuple[str, str, str]] = set()
    if not os.path.exists(CATALOG_PATH):
        return existing
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            existing.add((d.get("alias", ""), d.get("district", ""), d.get("category", "")))
    return existing


def main() -> None:
    existing_keys = load_existing_catalog()
    new_records: list[dict] = []
    cat_counts: dict[str, int] = defaultdict(int)
    skipped_noise = 0
    skipped_dup = 0
    skipped_no_cat = 0

    for filename, default_cat in CATEGORY_MAP.items():
        filepath = os.path.join(PURIFIED_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [跳过] {filename} 不存在")
            continue

        print(f"[处理] {filename} (默认类别: {default_cat or '自动分类'})")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                name = normalize_name(record.get("name", ""))
                if not name or is_noise(name):
                    skipped_noise += 1
                    continue

                district = extract_district(record)
                category = classify_record(name, default_cat)
                if not category:
                    skipped_no_cat += 1
                    continue

                key = (name, district, category)
                if key in existing_keys:
                    skipped_dup += 1
                    continue

                source = record.get("source", f"purified:{filename.replace('.jsonl','')}")
                entry = {
                    "alias": name,
                    "district": district,
                    "category": category,
                    "confidence": 0.88,
                    "source": source,
                }
                new_records.append(entry)
                existing_keys.add(key)
                cat_counts[category] += 1

    print("\n" + "=" * 60)
    print("导入统计")
    print("=" * 60)
    print(f"新增记录: {len(new_records)}")
    print(f"跳过(噪声): {skipped_noise}")
    print(f"跳过(重复): {skipped_dup}")
    print(f"跳过(无类别): {skipped_no_cat}")
    print("\n按类别:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    if not new_records:
        print("\n无新记录需要写入。")
        return

    with open(CATALOG_PATH, "a", encoding="utf-8") as f:
        for entry in new_records:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n已追加 {len(new_records)} 条到 {CATALOG_PATH}")


if __name__ == "__main__":
    main()
