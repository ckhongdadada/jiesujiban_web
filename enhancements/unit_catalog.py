from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from enhancements.data_paths import get_runtime_unit_catalog_path

DISTRICT_NAMES = [
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

GENERIC_BAD_UNITS = {
    "",
    "认领交办",
    "@北京12345",
    "@北京市12345",
    "北京12345",
}


def normalize_unit_text(value: str | None) -> str:
    text = str(value or "").replace("\u3000", " ").strip()
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    text = text.lstrip("@")
    text = re.sub(r"[，,。；;、：:]+$", "", text)

    # 去掉常见尾缀噪声，避免把“回复正文中的单位+附注”当成新类。
    text = re.sub(r"(?:回复单位|承办单位|办理单位)[:：]?", "", text)
    text = re.sub(r"[（(](?:回复|反馈|办理|转办|承办|热线转办)[^）)]*[）)]$", "", text)

    # 统一高频机构长短名（尽量只做语义等价替换）。
    replacements = [
        ("住房和城市建设委员会", "住建委"),
        ("住房和城乡建设委员会", "住建委"),
        ("住房和城乡建设局", "住建局"),
        ("卫生健康委员会", "卫健委"),
        ("市场监督管理局", "市场监管局"),
        ("城市管理委员会", "城管委"),
        ("城市管理综合行政执法局", "城管执法局"),
        ("城市管理行政执法局", "城管执法局"),
        ("城市管理执法局", "城管执法局"),
        ("人民政府办公室", "政府办"),
        ("街道办事处", "街道办"),
        ("地区办事处", "地区办"),
        ("政务服务便民热线", "12345"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)

    # 统一热线写法，便于在后续被 GENERIC_BAD_UNITS 识别过滤。
    if "12345" in text and ("北京" in text or "热线" in text):
        return "北京12345"
    return text.strip()


def _split_pipe_values(value: str | None) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [normalize_unit_text(part) for part in text.split(" | ") if normalize_unit_text(part)]


@lru_cache(maxsize=1)
def load_unit_catalog(path: str | None = None) -> dict[str, Any]:
    catalog_path = Path(path) if path else get_runtime_unit_catalog_path()
    if not catalog_path.exists():
        return {"alias_to_unit": {}, "unit_meta": {}}
    try:
        return json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"alias_to_unit": {}, "unit_meta": {}}


def canonicalize_unit(unit: str | None, catalog: dict[str, Any] | None = None) -> str:
    normalized = normalize_unit_text(unit)
    if not normalized:
        return ""
    catalog = catalog or load_unit_catalog()
    alias_to_unit = catalog.get("alias_to_unit", {})
    return normalize_unit_text(alias_to_unit.get(normalized, normalized))


def infer_unit_district(unit: str | None, catalog: dict[str, Any] | None = None) -> str:
    canonical = canonicalize_unit(unit, catalog=catalog)
    if not canonical:
        return ""
    catalog = catalog or load_unit_catalog()
    meta = catalog.get("unit_meta", {}).get(canonical, {})
    districts = meta.get("districts", []) or []
    if districts:
        return districts[0]
    for district in DISTRICT_NAMES:
        if district in canonical:
            return district
    return ""


def score_unit_candidate(
    unit: str | None,
    district: str | None = None,
    tag: str | None = None,
    catalog: dict[str, Any] | None = None,
) -> float:
    canonical = canonicalize_unit(unit, catalog=catalog)
    if not canonical or canonical in GENERIC_BAD_UNITS:
        return -1.0

    catalog = catalog or load_unit_catalog()
    meta = catalog.get("unit_meta", {}).get(canonical, {})
    score = 0.0

    districts = meta.get("districts", []) or []
    category = meta.get("category", "")
    top_tags = meta.get("top_tags", []) or []

    if district:
        if district in districts:
            score += 0.22
        elif districts:
            score -= 0.14
        elif canonical.startswith(("市", "北京市")):
            score += 0.04

    if tag and tag in top_tags:
        score += 0.08

    if category == "pseudo":
        score -= 0.4
    elif category == "hotline_center":
        score -= 0.08
    elif category == "department":
        score += 0.03
    elif category == "town_or_subdistrict":
        score += 0.05

    if "回复单位" in canonical or canonical.endswith("回复"):
        score -= 0.2

    return score


def build_catalog_from_frames(mapping_df, master_df) -> dict[str, Any]:
    alias_to_unit: dict[str, str] = {}
    unit_meta: dict[str, dict[str, Any]] = {}

    def ensure_meta(unit: str) -> dict[str, Any]:
        if unit not in unit_meta:
            unit_meta[unit] = {
                "districts": [],
                "tag_counts": {},
                "top_tags": [],
                "category": "",
                "source_types": [],
                "aliases": [],
                "sample_count": 0,
            }
        return unit_meta[unit]

    if mapping_df is not None:
        for _, row in mapping_df.iterrows():
            canonical = canonicalize_unit(row.get("normalized_unit"))
            if not canonical or canonical in GENERIC_BAD_UNITS:
                continue
            meta = ensure_meta(canonical)
            meta["category"] = normalize_unit_text(row.get("unit_category"))
            meta["sample_count"] = max(meta["sample_count"], int(row.get("sample_count") or 0))
            for source_type in _split_pipe_values(row.get("source_type")):
                if source_type not in meta["source_types"]:
                    meta["source_types"].append(source_type)
            for district in _split_pipe_values(row.get("district_examples")):
                if district in DISTRICT_NAMES and district not in meta["districts"]:
                    meta["districts"].append(district)
            for alias in _split_pipe_values(row.get("raw_unit_examples")):
                if alias and alias not in GENERIC_BAD_UNITS:
                    alias_to_unit[alias] = canonical
                    if alias not in meta["aliases"]:
                        meta["aliases"].append(alias)

    if master_df is not None:
        for _, row in master_df.iterrows():
            canonical = canonicalize_unit(row.get("reply_unit_norm"))
            raw = normalize_unit_text(row.get("reply_unit_raw"))
            district = normalize_unit_text(row.get("district_from_file"))
            tag = normalize_unit_text(row.get("message_tag_level1"))
            if not canonical or canonical in GENERIC_BAD_UNITS:
                continue
            meta = ensure_meta(canonical)
            meta["sample_count"] += 1
            if district in DISTRICT_NAMES and district not in meta["districts"]:
                meta["districts"].append(district)
            if raw and raw not in GENERIC_BAD_UNITS:
                alias_to_unit[raw] = canonical
                if raw not in meta["aliases"]:
                    meta["aliases"].append(raw)
            alias_to_unit[canonical] = canonical
            if tag:
                meta["tag_counts"][tag] = int(meta["tag_counts"].get(tag, 0)) + 1

    for canonical, meta in unit_meta.items():
        top_tags = sorted(meta["tag_counts"].items(), key=lambda item: (-item[1], item[0]))
        meta["top_tags"] = [tag for tag, _ in top_tags[:5]]
        meta["districts"] = sorted(meta["districts"])
        meta["source_types"] = sorted(set(meta["source_types"]))
        meta["aliases"] = sorted(set(meta["aliases"] + [canonical]))
        meta.pop("tag_counts", None)
        alias_to_unit[canonical] = canonical

    return {
        "alias_to_unit": dict(sorted(alias_to_unit.items())),
        "unit_meta": dict(sorted(unit_meta.items())),
    }
