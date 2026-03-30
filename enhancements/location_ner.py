from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

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

DIRECT_DISTRICT_PATTERN = re.compile("|".join(map(re.escape, DISTRICT_NAMES)))
ADDRESS_SNIPPET_PATTERN = re.compile(
    r"(?:北京市)?[^，。；；、\n]{0,20}(?:区|镇|乡|街道|路|街|村|社区|园|大厦|广场)",
)


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
        project_root = os.path.dirname(base_dir)
        merged_path = os.path.join(project_root, "data", "beijing_districts_merged.json")
        curated_path = os.path.join(project_root, "data", "beijing_districts_curated.json")
        extra_path = os.path.join(project_root, "data", "beijing_districts_extra.json")
        catalog_path = os.path.join(project_root, "data", "place_alias_catalog.jsonl")
        self.alias_paths = [alias_path or os.path.join(base_dir, "beijing_districts.json")]
        if os.path.exists(merged_path):
            self.alias_paths.append(merged_path)
        elif os.path.exists(curated_path):
            self.alias_paths.append(curated_path)
        else:
            self.alias_paths.append(extra_path)
        self.alias_to_district = self._load_aliases(self.alias_paths)
        self.alias_catalog = self._load_catalog(catalog_path)
        self.enable_lac = (
            enable_lac
            if enable_lac is not None
            else os.getenv("ENABLE_LAC", "false").lower() == "true"
        )
        self.amap_api_key = os.getenv("AMAP_API_KEY", "").strip()

    def resolve(self, text: str) -> dict[str, Any]:
        normalized = self._normalize(text)
        hits: list[LocationHit] = []
        hits.extend(self._match_direct_districts(normalized))
        hits.extend(self._match_aliases(normalized))
        hits.extend(self._match_lac_entities(normalized))

        if not hits and self.amap_api_key:
            hits.extend(self._match_amap(normalized))

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
                alias_to_district[district] = district
                alias_to_district[district.replace("区", "")] = district
                for alias in aliases:
                    alias_to_district[alias] = district
        return alias_to_district

    def _normalize(self, text: str) -> str:
        text = text or ""
        return re.sub(r"\s+", "", text)

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
                alias = item.get("alias", "")
                if alias:
                    catalog[alias].append(item)
        return dict(catalog)

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
        alias_items = sorted(
            self.alias_catalog.items() if self.alias_catalog else self.alias_to_district.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for alias, payload in alias_items:
            if alias in GENERIC_STOP_ALIASES:
                continue
            if alias and alias in text:
                if self.alias_catalog:
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
                else:
                    district = payload
                    hits.append(
                        LocationHit(
                            district=district,
                            matched_text=alias,
                            source="dictionary",
                            confidence=0.90 if district == alias else 0.84,
                            category="dictionary",
                        )
                    )
        return hits

    def _match_lac_entities(self, text: str) -> list[LocationHit]:
        if not self.enable_lac:
            return []

        try:
            from LAC import LAC
        except Exception:
            return []

        lac = LAC(mode="lac")
        lac_result = lac.run(text)
        if not lac_result or len(lac_result) != 2:
            return []

        words, tags = lac_result
        hits: list[LocationHit] = []
        for word, tag in zip(words, tags):
            if tag != "LOC":
                continue
            district = self.alias_to_district.get(word)
            if district:
                hits.append(
                    LocationHit(
                        district=district,
                        matched_text=word,
                        source="lac",
                        confidence=0.80,
                        category="lac",
                    )
                )
        return hits

    def _match_amap(self, text: str) -> list[LocationHit]:
        hits: list[LocationHit] = []
        for snippet in self._extract_address_snippets(text)[:3]:
            district = self._geocode_by_amap(snippet)
            if district:
                hits.append(
                    LocationHit(
                        district=district,
                        matched_text=snippet,
                        source="amap_geocode",
                        confidence=0.76,
                        category="amap_geocode",
                    )
                )
        return hits

    def _extract_address_snippets(self, text: str) -> list[str]:
        seen: set[str] = set()
        snippets: list[str] = []
        for match in ADDRESS_SNIPPET_PATTERN.finditer(text):
            snippet = match.group(0)
            if snippet not in seen:
                seen.add(snippet)
                snippets.append(snippet)
        return snippets

    def _geocode_by_amap(self, address: str) -> str | None:
        if not self.amap_api_key or not address:
            return None

        params = urlencode({"key": self.amap_api_key, "address": address, "city": "北京"})
        url = f"https://restapi.amap.com/v3/geocode/geo?{params}"
        try:
            with urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

        geocodes = payload.get("geocodes") or []
        if not geocodes:
            return None

        district = (geocodes[0].get("district") or "").strip()
        return district if district in DISTRICT_NAMES else None

    def _pick_best_district(self, hits: list[LocationHit]) -> tuple[str | None, float, str]:
        if not hits:
            return None, 0.0, "none"

        score_by_district: dict[str, float] = defaultdict(float)
        best_source_by_district: dict[str, str] = {}
        for hit in hits:
            score_by_district[hit.district] += hit.confidence
            if hit.district not in best_source_by_district or hit.confidence > 0.9:
                best_source_by_district[hit.district] = hit.source

        district, score = max(score_by_district.items(), key=lambda item: item[1])
        confidence = round(min(score / max(len(hits), 1), 0.99), 3)
        return district, confidence, best_source_by_district[district]


class LocationNER(BeijingDistrictResolver):
    def __init__(self, data_dir: str | None = None, **kwargs):
        if data_dir:
            alias_path = os.path.join(data_dir, "beijing_districts_merged.json")
            if os.path.exists(alias_path):
                kwargs["alias_path"] = alias_path
        super().__init__(**kwargs)

    def extract_district(self, text: str) -> dict[str, Any]:
        return self.resolve(text)
