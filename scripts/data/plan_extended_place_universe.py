from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.core.paths import get_place_taxonomy_dir, get_universe_v2_dir


BEIJING_DISTRICTS = [
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

PRIORITY_DISTRICTS = [
    "朝阳区",
    "海淀区",
    "丰台区",
    "通州区",
    "昌平区",
    "大兴区",
]

BUS_STOP_KEYWORDS = [
    "公交站",
    "公交场站",
    "公交枢纽",
    "首末站",
    "客运站",
    "公交站点",
]

COMMUNITY_HIGH_PRECISION_KEYWORDS = [
    "家园",
    "花园",
    "名苑",
    "雅苑",
    "嘉园",
    "佳园",
    "佳苑",
    "公寓",
    "新村",
    "家属院",
    "家属楼",
    "生活区",
    "宿舍",
    "别墅",
    "豪庭",
    "华庭",
    "公馆",
    "国际",
    "庄园",
    "名城",
]

COMMUNITY_MEDIUM_PRECISION_KEYWORDS = [
    "东区",
    "西区",
    "南区",
    "北区",
    "一区",
    "二区",
    "三区",
    "四区",
    "五区",
    "苑",
    "园",
    "城",
    "湾",
    "都",
    "轩",
    "阁",
    "庭",
]

GRID_BBOX = {
    "lng_start": 115.90,
    "lng_end": 116.85,
    "lat_start": 39.60,
    "lat_end": 40.25,
    "step": 0.01,
    "radius_meters": 1000,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def count_taxonomy_entries(path: Path) -> int:
    data = load_json(path)
    return sum(len(values) for values in data.values())


def generate_grid_points() -> list[dict]:
    points = []
    lng = GRID_BBOX["lng_start"]
    while lng <= GRID_BBOX["lng_end"] + 1e-9:
        lat = GRID_BBOX["lat_start"]
        while lat <= GRID_BBOX["lat_end"] + 1e-9:
            points.append(
                {
                    "lng": round(lng, 4),
                    "lat": round(lat, 4),
                    "radius_meters": GRID_BBOX["radius_meters"],
                }
            )
            lat += GRID_BBOX["step"]
        lng += GRID_BBOX["step"]
    return points


def filter_beijing_roads(raw_path: Path) -> tuple[list[dict], dict]:
    raw_rows = load_jsonl(raw_path)
    filtered = []
    seen = set()
    source_districts = set()
    for row in raw_rows:
        district = (row.get("district") or "").strip()
        road = (row.get("road") or "").strip()
        township = (row.get("township") or "").strip()
        if district not in BEIJING_DISTRICTS or not road:
            continue
        source_districts.add(district)
        key = (road, district, township)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(
            {
                "road": road,
                "district": district,
                "township": township,
                "high_precision_zone": (row.get("high_precision_zone") or "").strip(),
                "source": "amap_regeo_road",
            }
        )
    summary = {
        "raw_records": len(raw_rows),
        "beijing_records": len(filtered),
        "unique_beijing_roads": len({row["road"] for row in filtered}),
        "districts_covered": sorted(source_districts),
    }
    return filtered, summary


def build_text_query_universe(districts: list[str], keywords: list[str], category: str, tier: str) -> list[dict]:
    rows = []
    for district in districts:
        for keyword in keywords:
            rows.append(
                {
                    "district": district,
                    "keyword": keyword,
                    "category": category,
                    "tier": tier,
                    "query_text": f"{district}{keyword}",
                }
            )
    return rows


def build_summary_markdown(report: dict) -> str:
    current = report["current_coverage"]
    road_existing = report["road_existing_raw_source"]
    universe = report["proposed_universe"]
    return "\n".join(
        [
            "# 扩展地点抓取 Universe 规划",
            "",
            "## 当前覆盖",
            f"- 小区词条：`{current['community']}`",
            f"- 道路词条：`{current['road']}`",
            f"- 公交站词条：`{current['bus_stop']}`",
            f"- 地铁站词条：`{current['subway_station']}`",
            f"- 杂项 POI：`{current['poi_misc']}`",
            "",
            "## 已有但未充分利用的道路原始源",
            f"- 原始道路记录：`{road_existing['raw_records']}`",
            f"- 过滤后北京道路记录：`{road_existing['beijing_records']}`",
            f"- 过滤后唯一北京道路：`{road_existing['unique_beijing_roads']}`",
            "",
            "## 新 Universe",
            f"- 道路逆地理网格点：`{universe['road_regeo_grid']['grid_points']}`",
            f"- 小区逆地理网格点：`{universe['community_regeo_grid']['grid_points']}`",
            f"- 公交站文本查询：`{universe['bus_stop_text']['query_count']}`",
            f"- 小区长尾高精查询：`{universe['community_text_high_precision']['query_count']}`",
            f"- 小区长尾中精查询：`{universe['community_text_medium_precision']['query_count']}`",
            "",
            "## 推荐执行顺序",
            "1. 先接入并过滤现有 `road_township_dictionary.jsonl`，这是最低成本的道路增量。",
            "2. 再跑公交站文本查询，成本低、命中率高、对留言归区很有帮助。",
            "3. 然后跑小区逆地理网格，补标准 POI 长尾。",
            "4. 最后再跑小区长尾文本查询，用于追补网格未覆盖的名称变体。",
        ]
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    universe_dir = get_universe_v2_dir()
    place_taxonomy_dir = get_place_taxonomy_dir()

    current_coverage = {
        "community": count_taxonomy_entries(place_taxonomy_dir / "community.json"),
        "road": count_taxonomy_entries(place_taxonomy_dir / "road.json"),
        "bus_stop": count_taxonomy_entries(place_taxonomy_dir / "bus_stop.json"),
        "subway_station": count_taxonomy_entries(place_taxonomy_dir / "subway_station.json"),
        "poi_misc": count_taxonomy_entries(place_taxonomy_dir / "poi_misc.json"),
    }

    road_raw_path = project_root / "tools" / "road_township_dictionary.jsonl"
    beijing_road_rows, road_existing_summary = filter_beijing_roads(road_raw_path)
    save_jsonl(universe_dir / "road_township_beijing.jsonl", beijing_road_rows)

    grid_points = generate_grid_points()
    save_json(universe_dir / "road_regeo_grid_points.json", grid_points)
    save_json(universe_dir / "community_regeo_grid_points.json", grid_points)

    bus_queries = build_text_query_universe(BEIJING_DISTRICTS, BUS_STOP_KEYWORDS, "bus_stop", "high")
    community_high_queries = build_text_query_universe(
        PRIORITY_DISTRICTS, COMMUNITY_HIGH_PRECISION_KEYWORDS, "community", "high"
    )
    community_medium_queries = build_text_query_universe(
        PRIORITY_DISTRICTS, COMMUNITY_MEDIUM_PRECISION_KEYWORDS, "community", "medium"
    )

    save_json(universe_dir / "bus_stop_queries.json", bus_queries)
    save_json(universe_dir / "community_longtail_high_queries.json", community_high_queries)
    save_json(universe_dir / "community_longtail_medium_queries.json", community_medium_queries)

    report = {
        "current_coverage": current_coverage,
        "road_existing_raw_source": road_existing_summary,
        "proposed_universe": {
            "road_regeo_grid": {
                "grid_points": len(grid_points),
                "source": "amap_regeo",
                "purpose": "补道路名和街乡镇高精绑定",
                "output": str(universe_dir / "road_regeo_grid_points.json"),
            },
            "community_regeo_grid": {
                "grid_points": len(grid_points),
                "source": "amap_regeo",
                "purpose": "补标准住宅 POI 和小区长尾",
                "output": str(universe_dir / "community_regeo_grid_points.json"),
            },
            "bus_stop_text": {
                "query_count": len(bus_queries),
                "districts": len(BEIJING_DISTRICTS),
                "keywords": BUS_STOP_KEYWORDS,
                "output": str(universe_dir / "bus_stop_queries.json"),
            },
            "community_text_high_precision": {
                "query_count": len(community_high_queries),
                "districts": PRIORITY_DISTRICTS,
                "keywords": COMMUNITY_HIGH_PRECISION_KEYWORDS,
                "output": str(universe_dir / "community_longtail_high_queries.json"),
            },
            "community_text_medium_precision": {
                "query_count": len(community_medium_queries),
                "districts": PRIORITY_DISTRICTS,
                "keywords": COMMUNITY_MEDIUM_PRECISION_KEYWORDS,
                "output": str(universe_dir / "community_longtail_medium_queries.json"),
            },
        },
        "execution_order": [
            "integrate_existing_beijing_road_raw_source",
            "crawl_bus_stop_text_queries",
            "crawl_community_regeo_grid",
            "crawl_community_text_longtail_queries",
        ],
        "notes": [
            "旧 bulk 小区 universe 已完成，不建议继续追旧 remaining pages。",
            "现有 road_township_dictionary.jsonl 含京外记录，先过滤再并入更划算。",
            "公交站优先用文本查询，小区优先用逆地理网格补标准 POI。",
            "社区长尾文本查询只建议先跑重点区，避免把大量歧义后缀一次性放大。",
        ],
    }

    save_json(universe_dir / "extended_place_universe_report.json", report)
    save_markdown(universe_dir / "EXTENDED_PLACE_UNIVERSE_PLAN.md", build_summary_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
