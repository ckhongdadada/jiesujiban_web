from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

DISTRICT_NAMES = [
    "???",
    "???",
    "???",
    "???",
    "????",
    "???",
    "????",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
    "???",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
COMMON_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
GEOCODE_ENDPOINT = "https://restapi.amap.com/v3/geocode/geo"
BLOCK_PATTERNS = [
    "验证码",
    "verifycode",
    "antibot",
    "访问异常",
    "登录",
    "ke-passport",
]


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


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip())


def looks_blocked(html: str) -> bool:
    lower_html = (html or "").lower()
    return any(pattern.lower() in lower_html for pattern in BLOCK_PATTERNS)


def safe_request(session: requests.Session, url: str, *, timeout: int = 20) -> str:
    response = session.get(url, headers=COMMON_HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def sanitize_url(url: str) -> str:
    parsed = urlparse(url)
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key.lower() != "key"]
    return urlunparse(parsed._replace(query=urlencode(query)))


def clean_amap_name(name: str, category: str) -> str:
    cleaned = normalize_name(name)
    cleaned = re.sub(r"[（(]地铁站[）)]", "", cleaned)
    cleaned = cleaned.replace("地铁站", "")
    if category == "street":
        cleaned = cleaned.replace("街道办事处", "街道")
        cleaned = cleaned.replace("办事处", "")
    return cleaned


class BaseSource:
    source_name = "base"

    def fetch(self, session: requests.Session, max_pages: int) -> tuple[list[PlaceRecord], list[str]]:
        raise NotImplementedError


class LianjiaCommunitySource(BaseSource):
    source_name = "lianjia"
    base_url = "https://bj.lianjia.com/xiaoqu/"

    def fetch(self, session: requests.Session, max_pages: int) -> tuple[list[PlaceRecord], list[str]]:
        records: list[PlaceRecord] = []
        warnings: list[str] = []
        for page in range(1, max_pages + 1):
            url = self.base_url if page == 1 else f"{self.base_url}pg{page}/"
            html = safe_request(session, url)
            if looks_blocked(html):
                warnings.append(f"{self.source_name}: page {page} returned login/anti-bot content")
                break

            soup = BeautifulSoup(html, "lxml")
            cards = soup.select(".xiaoquListItem, .content__list--item, li[class*='clear']")
            if not cards:
                warnings.append(f"{self.source_name}: no supported listing cards found on page {page}")
                break

            for card in cards:
                name = ""
                district = ""
                for selector in [".title a", ".xiaoquListItemTitle a", "a[title]"]:
                    node = card.select_one(selector)
                    if node and node.get_text(strip=True):
                        name = node.get_text(strip=True)
                        break
                for selector in [".district", ".positionInfo", ".xiaoquListItemDistrict", ".position a"]:
                    node = card.select_one(selector)
                    if node and node.get_text(" ", strip=True):
                        district = extract_district(node.get_text(" ", strip=True))
                        if district:
                            break
                if name and district:
                    records.append(
                        PlaceRecord(
                            name=normalize_name(name),
                            district=district,
                            category="community",
                            source=self.source_name,
                            source_url=url,
                            address="",
                            location="",
                            adcode="",
                        )
                    )
            time.sleep(random.uniform(1.2, 2.0))
        return deduplicate(records), warnings


class AnjukeCommunitySource(BaseSource):
    source_name = "anjuke"
    base_url = "https://beijing.anjuke.com/community/"

    def fetch(self, session: requests.Session, max_pages: int) -> tuple[list[PlaceRecord], list[str]]:
        records: list[PlaceRecord] = []
        warnings: list[str] = []
        for page in range(1, max_pages + 1):
            url = self.base_url if page == 1 else f"{self.base_url}p{page}/"
            html = safe_request(session, url)
            if looks_blocked(html):
                warnings.append(f"{self.source_name}: page {page} returned anti-bot content")
                break

            soup = BeautifulSoup(html, "lxml")
            cards = soup.select(".li-itemmod, .list-item, li[class*='list']")
            if not cards:
                warnings.append(f"{self.source_name}: no supported listing cards found on page {page}")
                break

            for card in cards:
                name = ""
                district = ""
                for selector in [".li-info h3 a", ".item-title a", "a[title]"]:
                    node = card.select_one(selector)
                    if node and node.get_text(strip=True):
                        name = node.get_text(strip=True)
                        break
                full_text = card.get_text(" ", strip=True)
                district = extract_district(full_text)
                if name and district:
                    records.append(
                        PlaceRecord(
                            name=normalize_name(name),
                            district=district,
                            category="community",
                            source=self.source_name,
                            source_url=url,
                            address="",
                            location="",
                            adcode="",
                        )
                    )
            time.sleep(random.uniform(1.4, 2.3))
        return deduplicate(records), warnings


class BeijingSubwaySource(BaseSource):
    source_name = "beijing_subway"
    base_url = "https://map.bjsubway.com/"

    def fetch(self, session: requests.Session, max_pages: int) -> tuple[list[PlaceRecord], list[str]]:
        del max_pages
        html = safe_request(session, self.base_url)
        soup = BeautifulSoup(html, "lxml")
        scripts = "\n".join(script.get_text(" ", strip=True) for script in soup.select("script"))
        candidates = set()
        for match in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,12}(?:站)?", scripts):
            text = normalize_name(match)
            if is_plausible_station_name(text):
                candidates.add(text.removesuffix("站"))

        warnings: list[str] = []
        if not candidates:
            warnings.append(f"{self.source_name}: no station names extracted from page scripts")
        records = [
            PlaceRecord(
                name=name,
                district="",
                category="subway_station",
                source=self.source_name,
                source_url=self.base_url,
                address="",
                location="",
                adcode="",
            )
            for name in sorted(candidates)
        ]
        return records, warnings


class AmapTextSource(BaseSource):
    source_name = "amap"
    endpoint = "https://restapi.amap.com/v3/place/text"

    def __init__(self, api_key: str, keyword: str, category: str) -> None:
        self.api_key = api_key
        self.keyword = keyword
        self.category = category

    def fetch(self, session: requests.Session, max_pages: int) -> tuple[list[PlaceRecord], list[str]]:
        records: list[PlaceRecord] = []
        warnings: list[str] = []
        for district in DISTRICT_NAMES:
            for page in range(1, max_pages + 1):
                params = {
                    "key": self.api_key,
                    "keywords": f"{district}{self.keyword}",
                    "city": "北京",
                    "citylimit": "true",
                    "offset": 20,
                    "page": page,
                    "extensions": "base",
                    "types": "",
                }
                response = session.get(self.endpoint, params=params, timeout=20, headers=COMMON_HEADERS)
                response.raise_for_status()
                payload = response.json()
                pois = payload.get("pois") or []
                if not pois:
                    break
                for poi in pois:
                    poi_district = normalize_district(poi.get("adname") or "")
                    if poi_district != district:
                        continue
                    name = clean_amap_name(poi.get("name") or "", self.category)
                    if not name:
                        continue
                    if self.category == "street" and "街道" not in name:
                        continue
                    if self.category == "street" and "驻北京" in name:
                        continue
                    records.append(
                        PlaceRecord(
                            name=name,
                            district=poi_district,
                            category=self.category,
                            source=f"{self.source_name}_{self.category}",
                            source_url=sanitize_url(response.url),
                            address=normalize_name(poi.get("address") or poi.get("pname") or ""),
                            location=normalize_name(poi.get("location") or ""),
                            adcode=normalize_name(poi.get("adcode") or ""),
                        )
                    )
                time.sleep(random.uniform(0.8, 1.2))
        if not records:
            warnings.append(f"{self.source_name}_{self.category}: no records returned")
        return deduplicate(records), warnings


def normalize_district(text: str) -> str:
    cleaned = normalize_name(text)
    return cleaned if cleaned in DISTRICT_NAMES else ""


def extract_district(text: str) -> str:
    cleaned = normalize_name(text)
    for district in DISTRICT_NAMES:
        if district in cleaned or district.replace("区", "") in cleaned:
            return district
    return ""


def is_plausible_station_name(text: str) -> bool:
    if not text:
        return False
    if len(text) < 2 or len(text) > 8:
        return False
    blacklist = {
        "首页", "详情", "分钟", "票价", "复制", "路线", "拥挤", "方向",
        "信息公开", "联系我们", "输入站点名称", "关闭详情", "乘车方案",
        "出发时间", "返回", "取消", "关闭", "展开详情", "详情展开",
    }
    if text in blacklist:
        return False
    if any(word in text for word in ["分钟", "方案", "时间", "拥挤", "路段", "详情", "运营", "出发", "换乘", "完全程"]):
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def resolve_missing_districts(
    session: requests.Session,
    records: list[PlaceRecord],
    amap_key: str,
) -> list[PlaceRecord]:
    if not amap_key:
        return records

    resolved: list[PlaceRecord] = []
    for record in records:
        if record.district:
            resolved.append(record)
            continue
        params = {"key": amap_key, "address": f"北京市{record.name}"}
        try:
            response = session.get(GEOCODE_ENDPOINT, params=params, headers=COMMON_HEADERS, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            resolved.append(record)
            continue

        geocodes = payload.get("geocodes") or []
        district = ""
        if geocodes:
            district = normalize_district(geocodes[0].get("district") or "")
        if district:
            resolved.append(
                PlaceRecord(
                    name=record.name,
                    district=district,
                    category=record.category,
                    source=record.source,
                    source_url=record.source_url,
                    address=record.address,
                    location=record.location,
                    adcode=record.adcode,
                )
            )
        else:
            resolved.append(record)
        time.sleep(random.uniform(0.2, 0.5))
    return resolved


def deduplicate(records: Iterable[PlaceRecord]) -> list[PlaceRecord]:
    unique: dict[tuple[str, str, str], PlaceRecord] = {}
    for record in records:
        key = (record.name, record.district, record.category)
        unique[key] = record
    return list(unique.values())


def merge_to_dictionary(records: list[PlaceRecord]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {district: set() for district in DISTRICT_NAMES}
    for record in records:
        if record.district in grouped and record.name:
            grouped[record.district].add(record.name)
    return {district: sorted(values) for district, values in grouped.items()}


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_outputs(records: list[PlaceRecord], warnings: list[str], output_json: str, output_jsonl: str) -> None:
    dictionary = merge_to_dictionary(records)
    ensure_parent(output_json)
    ensure_parent(output_jsonl)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    report_path = os.path.join(os.path.dirname(output_json), "crawl_report.json")
    report = {
        "record_count": len(records),
        "warning_count": len(warnings),
        "warnings": warnings,
        "district_counts": {district: len(names) for district, names in dictionary.items()},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def build_sources(source_names: list[str], amap_key: str) -> list[BaseSource]:
    mapping: dict[str, BaseSource] = {
        "lianjia": LianjiaCommunitySource(),
        "anjuke": AnjukeCommunitySource(),
        "subway": BeijingSubwaySource(),
    }
    if amap_key:
        mapping["amap_subway"] = AmapTextSource(amap_key, "地铁站", "subway_station")
        mapping["amap_streets"] = AmapTextSource(amap_key, "街道办事处", "street")
        mapping["amap_communities"] = AmapTextSource(amap_key, "小区", "community")
    return [mapping[name] for name in source_names if name in mapping]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an additive Beijing place dictionary.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["lianjia", "anjuke", "subway"],
        help="Sources to crawl. Optional amap_subway/amap_streets require AMAP_API_KEY.",
    )
    parser.add_argument("--max-pages", type=int, default=2, help="Max pages per HTML source.")
    parser.add_argument(
        "--output-json",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "beijing_districts_extra.json"),
    )
    parser.add_argument(
        "--output-jsonl",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "beijing_place_records.jsonl"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    amap_key = os.getenv("AMAP_API_KEY", "").strip()
    session = requests.Session()

    all_records: list[PlaceRecord] = []
    warnings: list[str] = []
    for source in build_sources(args.sources, amap_key):
        source_records, source_warnings = source.fetch(session, args.max_pages)
        all_records.extend(source_records)
        warnings.extend(source_warnings)

    all_records = resolve_missing_districts(session, all_records, amap_key)
    all_records = deduplicate(all_records)
    if args.dry_run:
        print(json.dumps(
            {
                "record_count": len(all_records),
                "warnings": warnings,
                "sample_records": [asdict(record) for record in all_records[:10]],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    write_outputs(all_records, warnings, args.output_json, args.output_jsonl)
    print(
        json.dumps(
            {
                "record_count": len(all_records),
                "warning_count": len(warnings),
                "output_json": args.output_json,
                "output_jsonl": args.output_jsonl,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
