from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.core.paths import ensure_standard_data_layout, get_runtime_file  # noqa: E402


MASTER_TABLE_PATH = Path(r"C:\Users\28414\Documents\New project\raw_data_analysis\master_table_v1.csv")
ROAD_TOWNSHIP_SOURCES = [
    PROJECT_ROOT / "data" / "universe" / "v2" / "road_township_beijing.jsonl",
    PROJECT_ROOT / "tools" / "road_township_dictionary.jsonl",
]
OUTPUT_PATH = get_runtime_file("beijing_location_seed_aliases.json")

DISTRICT_NAMES = {
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
}

GENERIC_BLOCKLIST = {
    "北京",
    "北京市",
    "公园",
    "广场",
    "社区",
    "小区",
    "花园",
    "家园",
    "大厦",
    "公寓",
    "天街",
}
PREFIX_BLOCKLIST = ("请", "建议", "增加", "关于", "推进", "本人", "我家", "如下图")

LOCATION_PATTERN = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9]{2,24}(?:镇|乡|街道|胡同|巷|路|街|大街|大道|社区|小区|家园|花园|公园|广场|天街|站|园区)"
)


def _clean_fragment(text: str, district: str) -> list[str]:
    text = str(text or "").strip()
    if not text or text == "nan":
        return []
    text = re.sub(r"北京市", "", text)
    text = text.replace(district, "").replace(district.replace("区", ""), "")
    text = re.sub(r"[（(].*?[)）]", " ", text)
    text = re.sub(r"[，,；;。/\s]+", " ", text)
    aliases = []
    for match in LOCATION_PATTERN.findall(text):
        alias = match.strip()
        if len(alias) < 2 or alias in GENERIC_BLOCKLIST or alias.startswith(PREFIX_BLOCKLIST):
            continue
        aliases.append(alias)
    return aliases


def _load_master_seeds() -> dict[str, set[str]]:
    if not MASTER_TABLE_PATH.exists():
        return {}
    df = pd.read_csv(MASTER_TABLE_PATH, low_memory=False)
    seeds: dict[str, set[str]] = defaultdict(set)
    sub = df[["district_from_file", "message_location_raw"]].dropna()
    for _, row in sub.iterrows():
        district = str(row["district_from_file"]).strip()
        if district not in DISTRICT_NAMES:
            continue
        aliases = _clean_fragment(row["message_location_raw"], district)
        for alias in aliases:
            seeds[district].add(alias)
    text_candidates: dict[tuple[str, str], int] = defaultdict(int)
    text_df = df[["district_from_file", "message_title", "message_body"]].fillna("")
    for _, row in text_df.iterrows():
        district = str(row["district_from_file"]).strip()
        if district not in DISTRICT_NAMES:
            continue
        text = f"{row['message_title']} {row['message_body']}"
        for alias in LOCATION_PATTERN.findall(text):
            alias = alias.strip()
            if alias in GENERIC_BLOCKLIST or len(alias) < 3 or alias.startswith(PREFIX_BLOCKLIST):
                continue
            text_candidates[(district, alias)] += 1

    for (district, alias), count in text_candidates.items():
        keep = count >= 2 or alias.endswith(("镇", "乡", "街道", "天街", "站"))
        if keep:
            seeds[district].add(alias)
    return seeds


def _load_road_township_seeds() -> dict[str, set[str]]:
    seeds: dict[str, set[str]] = defaultdict(set)
    for path in ROAD_TOWNSHIP_SOURCES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                item = json.loads(line)
                district = str(item.get("district", "")).strip()
                if district not in DISTRICT_NAMES:
                    continue
                for key in ("township", "high_precision_zone"):
                    alias = str(item.get(key, "")).strip()
                    if alias and alias not in GENERIC_BLOCKLIST:
                        seeds[district].add(alias)
                road = str(item.get("road", "")).strip()
                if road and len(road) >= 2:
                    seeds[district].add(road)
    return seeds


def main() -> None:
    ensure_standard_data_layout()
    merged: dict[str, set[str]] = defaultdict(set)

    for source in (_load_master_seeds(), _load_road_township_seeds()):
        for district, aliases in source.items():
            merged[district].update(aliases)

    output = {district: sorted(merged.get(district, set())) for district in sorted(DISTRICT_NAMES)}
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "output_path": str(OUTPUT_PATH),
        "district_count": len(output),
        "alias_count": sum(len(v) for v in output.values()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
