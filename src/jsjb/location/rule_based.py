from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from src.jsjb.core.paths import get_runtime_catalog_path, get_runtime_district_file

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

GENERIC_STOP_ALIASES = {"北京", "北京市"}
WEAK_GENERIC_ALIASES = {
    "一条",
    "二条",
    "三条",
    "四条",
    "五条",
    "六环",
    "公寓",
    "天街",
    "公园",
    "广场",
    "运动公园",
    "家园",
    "花园",
    "社区",
    "小区",
    "大厦",
    "大道",
    "大街",
    "胡同",
    "街道",
    "一号楼",
    "二号楼",
    "三号楼",
}
GENERIC_ALIAS_PATTERNS = [
    re.compile(r"^[一二三四五六七八九十0-9]+条$"),
    re.compile(r"^[一二三四五六七八九十0-9]+号楼$"),
    re.compile(r"^[A-Za-z0-9]+区$"),
]
DISTRICT_SHORT_NAMES = {district.replace("区", ""): district for district in DISTRICT_NAMES}
DIRECT_DISTRICT_PATTERN = re.compile("|".join(map(re.escape, DISTRICT_NAMES)))
ADDRESS_SNIPPET_PATTERN = re.compile(
    r"(?:北京市)?[^，。；;\n]{0,24}(?:区|镇|乡|街道|胡同|巷|路|街|大街|大道|社区|小区|家园|花园|公寓|大厦|广场)"
)
GEOCODE_CANDIDATE_SUFFIXES = (
    "区",
    "镇",
    "乡",
    "街道",
    "胡同",
    "巷",
    "路",
    "街",
    "大街",
    "大道",
    "社区",
    "小区",
    "家园",
    "花园",
    "公寓",
    "大厦",
    "广场",
    "公园",
    "站",
    "医院",
    "学校",
)

CATEGORY_SCORE_MULTIPLIERS = {
    "district": 1.30,
    "administrative_seed": 1.20,
    "road": 1.10,
    "bus_stop": 1.02,
    "subway_station": 1.00,
    "community": 0.95,
    "dictionary_core": 0.92,
    "dictionary": 0.88,
    "park": 0.84,
    "hospital": 0.84,
    "school": 0.82,
    "mall": 0.80,
    "hotel": 0.76,
    "lac": 0.72,
    "amap_geocode": 0.96,
    "poi_misc": 0.58,
    "unknown": 0.60,
}
SOURCE_SCORE_MULTIPLIERS = {
    "direct_district": 1.35,
    "administrative_seed": 1.18,
    "dictionary_seed": 1.10,
    "dictionary": 1.00,
    "dictionary_core": 1.04,
    "road_source": 1.08,
    "amap_geocode": 0.96,
    "amap_geocode_lac": 0.98,
    "amap_geocode_snippet": 0.94,
    "lac": 0.82,
}
CATEGORY_PRIORITY = {
    "district": 0,
    "administrative_seed": 1,
    "road": 2,
    "bus_stop": 3,
    "subway_station": 4,
    "community": 5,
    "dictionary_core": 6,
    "dictionary": 7,
    "park": 8,
    "hospital": 9,
    "school": 10,
    "mall": 11,
    "hotel": 12,
    "lac": 13,
    "amap_geocode": 14,
    "poi_misc": 15,
    "unknown": 16,
}


@dataclass
class LocationHit:
    district: str
    matched_text: str
    source: str
    confidence: float
    category: str = "unknown"


class BeijingDistrictResolver:
    def __init__(self, alias_path: str | None = None, enable_lac: bool | None = None) -> None:
        base_dir = os.path.dirname(__file__)
        merged_path = str(get_runtime_district_file("beijing_districts_merged.json"))
        curated_path = str(get_runtime_district_file("beijing_districts_curated.json"))
        extra_path = str(get_runtime_district_file("beijing_districts_extra.json"))
        seed_path = str(get_runtime_district_file("beijing_location_seed_aliases.json"))
        catalog_path = str(get_runtime_catalog_path())

        self.alias_paths = [alias_path or os.path.join(base_dir, "beijing_districts.json")]
        if os.path.exists(merged_path):
            self.alias_paths.append(merged_path)
        elif os.path.exists(curated_path):
            self.alias_paths.append(curated_path)
        elif os.path.exists(extra_path):
            self.alias_paths.append(extra_path)
        if os.path.exists(seed_path):
            self.alias_paths.append(seed_path)

        self.alias_to_district = self._load_aliases(self.alias_paths)
        self.alias_catalog = self._load_catalog(catalog_path)
        self.alias_catalog_items = sorted(self.alias_catalog.items(), key=lambda item: len(item[0]), reverse=True)
        self.alias_items = sorted(self.alias_to_district.items(), key=lambda item: len(item[0]), reverse=True)
        self.enable_lac = (
            enable_lac if enable_lac is not None else os.getenv("ENABLE_LAC", "false").lower() == "true"
        )
        self.amap_api_key = os.getenv("AMAP_API_KEY", "").strip()
        self._lac_model = None
        self._amap_cache: dict[str, str | None] = {}

    def resolve(self, text: str) -> dict[str, Any]:
        normalized = self._normalize(text)
        hits: list[LocationHit] = []
        hits.extend(self._match_direct_districts(normalized))
        hits.extend(self._match_aliases(normalized))
        lac_loc_entities = self._extract_lac_loc_entities(normalized)
        hits.extend(self._match_lac_entities(lac_loc_entities))

        if self.amap_api_key:
            hits.extend(self._match_amap(normalized, lac_loc_entities, hits))

        hits = self._deduplicate_hits(hits)

        district, confidence, source = self._pick_best_district(hits)
        places = [asdict(hit) for hit in hits]
        snippets = self._extract_address_snippets(normalized)
        return {
            "district": district,
            "confidence": confidence,
            "method": source,
            "places": places,
            "address_snippets": snippets,
            "districts": sorted({hit.district for hit in hits}),
            "categories": sorted({hit.category for hit in hits}),
        }

    def _load_aliases(self, paths: list[str]) -> dict[str, str]:
        alias_to_district: dict[str, str] = {}
        for path in paths:
            if not path or not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8-sig") as f:
                district_map = json.load(f)

            for district, aliases in district_map.items():
                if district not in DISTRICT_NAMES:
                    continue
                alias_to_district[district] = district
                alias_to_district[district.replace("区", "")] = district
                for alias in aliases:
                    normalized_alias = self._normalize(alias)
                    if not self._is_valid_alias(normalized_alias):
                        continue
                    alias_to_district[normalized_alias] = district
        return alias_to_district

    def _load_catalog(self, path: str) -> dict[str, list[dict[str, Any]]]:
        if not os.path.exists(path):
            return {}
        catalog: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                alias = self._normalize(item.get("alias", ""))
                district = item.get("district", "")
                if not alias or district not in DISTRICT_NAMES or not self._is_valid_alias(alias):
                    continue
                catalog[alias].append(item)
        filtered: dict[str, list[dict[str, Any]]] = {}
        for alias, items in catalog.items():
            districts = {item.get("district", "") for item in items}
            if len(districts) > 1 and self._is_ambiguous_generic_alias(alias):
                continue
            filtered[alias] = items
        return filtered

    def _is_valid_alias(self, alias: str) -> bool:
        if not alias or alias in GENERIC_STOP_ALIASES or alias in WEAK_GENERIC_ALIASES:
            return False
        if len(alias) < 2:
            return False
        if alias.startswith(("请", "建议", "增加", "关于", "推进", "本人", "我家", "如下图")):
            return False
        for pattern in GENERIC_ALIAS_PATTERNS:
            if pattern.match(alias):
                return False
        return True

    def _is_ambiguous_generic_alias(self, alias: str) -> bool:
        if alias in WEAK_GENERIC_ALIASES:
            return True
        if len(alias) <= 3:
            return True
        return alias.endswith(("公寓", "花园", "家园", "社区", "小区", "广场", "大厦", "天街", "公园"))

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text or "")

    def _match_direct_districts(self, text: str) -> list[LocationHit]:
        return [
            LocationHit(
                district=match.group(0),
                matched_text=match.group(0),
                source="direct_district",
                confidence=0.99,
                category="district",
            )
            for match in DIRECT_DISTRICT_PATTERN.finditer(text)
        ]

    def _match_aliases(self, text: str) -> list[LocationHit]:
        hits: list[LocationHit] = []
        matched_aliases: set[str] = set()
        for alias, payload in self.alias_catalog_items:
            if not self._is_valid_alias(alias) or alias not in text:
                continue
            matched_aliases.add(alias)
            for item in payload:
                hits.append(
                    LocationHit(
                        district=item["district"],
                        matched_text=alias,
                        source=item.get("source", "dictionary"),
                        confidence=float(item.get("confidence", 0.84)),
                        category=item.get("category", "dictionary"),
                    )
                )

        for alias, district in self.alias_items:
            if alias in matched_aliases or not self._is_valid_alias(alias) or alias not in text:
                continue
            hits.append(
                LocationHit(
                    district=district,
                    matched_text=alias,
                    source="administrative_seed" if alias.endswith(("镇", "乡", "街道")) else "dictionary_seed",
                    confidence=0.87 if district == alias else 0.83,
                    category="administrative_seed"
                    if alias.endswith(("镇", "乡", "街道"))
                    else ("dictionary_core" if alias.endswith(("站",)) else "dictionary"),
                )
            )
        return hits

    def _get_lac_model(self):
        if self._lac_model is not None:
            return self._lac_model
        if not self.enable_lac:
            return None
        try:
            from LAC import LAC
        except Exception:
            return None
        self._lac_model = LAC(mode="lac")
        return self._lac_model

    def _extract_lac_loc_entities(self, text: str) -> list[str]:
        lac = self._get_lac_model()
        if lac is None:
            return []
        try:
            lac_result = lac.run(text)
        except Exception:
            return []
        if not lac_result or len(lac_result) != 2:
            return []

        words, tags = lac_result
        seen: set[str] = set()
        entities: list[str] = []
        for word, tag in zip(words, tags):
            if tag != "LOC":
                continue
            normalized_word = self._normalize(word)
            if not normalized_word or normalized_word in seen:
                continue
            seen.add(normalized_word)
            entities.append(normalized_word)
        return entities

    def _match_lac_entities(self, loc_entities: list[str]) -> list[LocationHit]:
        if not self.enable_lac:
            return []
        hits: list[LocationHit] = []
        for normalized_word in loc_entities:
            district = self.alias_to_district.get(normalized_word) or DISTRICT_SHORT_NAMES.get(normalized_word)
            if district:
                hits.append(
                    LocationHit(
                        district=district,
                        matched_text=normalized_word,
                        source="lac",
                        confidence=0.80,
                        category="lac",
                    )
                )
        return hits

    def _is_geocode_candidate(self, text: str) -> bool:
        token = self._normalize(text)
        if not token or not (2 <= len(token) <= 30):
            return False
        if token in DISTRICT_NAMES or token in DISTRICT_SHORT_NAMES:
            return True
        if token in WEAK_GENERIC_ALIASES:
            return False
        if token in self.alias_to_district or token in self.alias_catalog:
            return True
        return token.endswith(GEOCODE_CANDIDATE_SUFFIXES)

    def _match_amap(
        self,
        text: str,
        lac_loc_entities: list[str] | None = None,
        existing_hits: list[LocationHit] | None = None,
    ) -> list[LocationHit]:
        # 已明确到行政区时不再调用 geocode，节省配额
        if existing_hits and any(hit.source == "direct_district" for hit in existing_hits):
            return []

        hits: list[LocationHit] = []
        candidates: list[tuple[str, str, float, str]] = []
        seen_candidates: set[str] = set()

        # 先用 LAC 的 LOC 实体做 geocode（优先）
        for entity in lac_loc_entities or []:
            token = self._normalize(entity)
            if not self._is_geocode_candidate(token) or token in seen_candidates:
                continue
            seen_candidates.add(token)
            candidates.append((token, token, 0.84, "amap_geocode_lac"))

        # 再回退到正则地址片段
        for snippet in self._extract_address_snippets(text):
            token = self._normalize(snippet)
            if not self._is_geocode_candidate(token) or token in seen_candidates:
                continue
            seen_candidates.add(token)
            candidates.append((snippet, token, 0.76, "amap_geocode_snippet"))

        for raw_text, query_token, confidence, source_name in candidates[:6]:
            district = self._geocode_by_amap(query_token)
            if district:
                hits.append(
                    LocationHit(
                        district=district,
                        matched_text=raw_text,
                        source=source_name,
                        confidence=confidence,
                        category="amap_geocode",
                    )
                )
        return hits

    def _extract_address_snippets(self, text: str) -> list[str]:
        seen: set[str] = set()
        snippets: list[str] = []
        for match in ADDRESS_SNIPPET_PATTERN.finditer(text):
            snippet = match.group(0)
            if 2 <= len(snippet) <= 30 and snippet not in seen:
                seen.add(snippet)
                snippets.append(snippet)
        return snippets

    def _geocode_by_amap(self, address: str) -> str | None:
        normalized_address = self._normalize(address)
        if not self.amap_api_key or not normalized_address:
            return None
        if normalized_address in self._amap_cache:
            return self._amap_cache[normalized_address]

        params = urlencode({"key": self.amap_api_key, "address": normalized_address, "city": "北京"})
        url = f"https://restapi.amap.com/v3/geocode/geo?{params}"
        try:
            with urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            self._amap_cache[normalized_address] = None
            return None

        geocodes = payload.get("geocodes") or []
        if not geocodes:
            self._amap_cache[normalized_address] = None
            return None

        district = self._normalize(geocodes[0].get("district", ""))
        if district in DISTRICT_NAMES:
            self._amap_cache[normalized_address] = district
            return district
        if district in DISTRICT_SHORT_NAMES:
            mapped = DISTRICT_SHORT_NAMES[district]
            self._amap_cache[normalized_address] = mapped
            return mapped
        self._amap_cache[normalized_address] = None
        return None

    def _deduplicate_hits(self, hits: list[LocationHit]) -> list[LocationHit]:
        best_by_key: dict[tuple[str, str, str, str], LocationHit] = {}
        for hit in hits:
            key = (hit.district, hit.matched_text, hit.source, hit.category)
            prev = best_by_key.get(key)
            if prev is None or hit.confidence > prev.confidence:
                best_by_key[key] = hit
        return list(best_by_key.values())

    def _pick_best_district(self, hits: list[LocationHit]) -> tuple[str | None, float, str]:
        if not hits:
            return None, 0.0, "none"
        direct_hits = [hit for hit in hits if hit.source == "direct_district" or hit.category == "district"]
        if direct_hits:
            direct_scores: dict[str, float] = defaultdict(float)
            for hit in direct_hits:
                direct_scores[hit.district] += hit.confidence
            district, score = max(direct_scores.items(), key=lambda item: (item[1], item[0]))
            confidence = round(min(0.99, score / max(len(direct_hits), 1)), 3)
            return district, confidence, "direct_district"

        district_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "score": 0.0,
                "count": 0,
                "best_source": "dictionary",
                "best_category": "unknown",
                "best_confidence": 0.0,
                "longest_match": 0,
            }
        )

        for hit in hits:
            source_weight = SOURCE_SCORE_MULTIPLIERS.get(hit.source, 1.0)
            category_weight = CATEGORY_SCORE_MULTIPLIERS.get(hit.category, CATEGORY_SCORE_MULTIPLIERS["unknown"])
            length_bonus = min(len(hit.matched_text), 12) * 0.01
            weighted_score = hit.confidence * source_weight * category_weight + length_bonus

            stats = district_stats[hit.district]
            stats["score"] += weighted_score
            stats["count"] += 1
            stats["longest_match"] = max(stats["longest_match"], len(hit.matched_text))

            current_priority = CATEGORY_PRIORITY.get(stats["best_category"], 99)
            hit_priority = CATEGORY_PRIORITY.get(hit.category, 99)
            if hit.confidence > stats["best_confidence"] or hit_priority < current_priority:
                stats["best_source"] = hit.source
                stats["best_category"] = hit.category
                stats["best_confidence"] = hit.confidence

        district, stats = max(
            district_stats.items(),
            key=lambda item: (
                item[1]["score"],
                item[1]["count"],
                -CATEGORY_PRIORITY.get(item[1]["best_category"], 99),
                item[1]["longest_match"],
            ),
        )
        confidence = round(min(0.99, stats["score"] / max(stats["count"], 1)), 3)
        return district, confidence, stats["best_source"]


class LocationNER(BeijingDistrictResolver):
    def __init__(self, data_dir: str | None = None, **kwargs):
        if data_dir:
            alias_path = os.path.join(data_dir, "beijing_districts_merged.json")
            if os.path.exists(alias_path):
                kwargs["alias_path"] = alias_path
        super().__init__(**kwargs)

    def extract_district(self, text: str) -> dict[str, Any]:
        return self.resolve(text)
