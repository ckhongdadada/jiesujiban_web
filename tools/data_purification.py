"""
接诉即办 - 地理数据全局提纯与分类引擎
=====================================
功能：
  1. 跨源去重：合并 AMap + Baidu 数据，基于 (名称, 经纬度3位) 全局排重
  2. 智能分拣：将"非住宅噪音"按类别归入对应数据集（学校、医院、交通...）
  3. 标准化输出：生成最终干净的小区词库 + 各类别专属词库

输入：
  - data/amap_community_bulk_records.jsonl   (AMap 主数据)
  - tools/baidu_community_records.jsonl      (百度补全数据)
  - tools/road_township_dictionary.jsonl     (道路数据)

输出（全部写入 data/purified/ 目录）：
  - community_clean.jsonl       ← 纯净小区数据（核心资产）
  - school.jsonl                ← 学校
  - hospital.jsonl              ← 医院/卫生
  - transit.jsonl               ← 公交/地铁/交通
  - government.jsonl            ← 政府/办事处
  - commercial.jsonl            ← 商业（酒店/超市/银行...）
  - other_poi.jsonl             ← 无法归类的杂项
  - purification_report.json    ← 统计报告
"""

import json
import os
import sys
from collections import Counter, defaultdict

# Windows 终端 GBK 编码兼容
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


# ================= 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INPUT_FILES = {
    "amap_main": os.path.join(DATA_DIR, "amap_community_bulk_records.jsonl"),
    "baidu": os.path.join(SCRIPT_DIR, "baidu_community_records.jsonl"),
}
ROAD_FILE = os.path.join(SCRIPT_DIR, "road_township_dictionary.jsonl")

OUTPUT_DIR = os.path.join(DATA_DIR, "purified")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 分类规则引擎 =================
# 每个类别对应一组关键词，命中即归入该类别
CATEGORY_RULES = {
    "school": [
        "幼儿园", "小学", "中学", "高中", "大学", "学院", "学校",
        "附小", "附中", "实验学校", "教育", "培训", "托儿所",
    ],
    "hospital": [
        "医院", "卫生", "诊所", "门诊", "药房", "药店", "保健",
        "急救", "疾控", "防疫", "康复", "护理",
    ],
    "transit": [
        "公交", "地铁", "车站", "火车站", "客运", "枢纽",
        "停车场", "停车楼", "加油站", "充电站", "充电桩",
    ],
    "government": [
        "派出所", "公安", "法院", "检察", "办事处", "街道办",
        "居委会", "村委会", "政府", "行政", "税务", "工商",
        "城管", "消防", "民政", "信访", "司法",
    ],
    "commercial": [
        "酒店", "宾馆", "旅馆", "招待所", "民宿",
        "超市", "便利店", "商场", "商城", "百货",
        "银行", "信用社", "证券",
        "餐厅", "饭店", "饭馆", "食堂", "快餐",
        "美容", "理发", "洗浴", "足疗",
        "洗车", "修理", "汽修", "4S店",
        "快递", "物流", "邮局",
        "写字楼", "办公楼", "科技园", "产业园",
        "公司", "集团", "工厂", "车间",
        "菜市场", "批发市场", "农贸",
    ],
}


def classify_name(name: str) -> str:
    """根据名称中的关键词，判断其所属类别。返回类别名或 'community'。"""
    for category, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            if kw in name:
                return category
    return "community"


def make_dedup_key(name: str, location: str) -> str:
    """生成去重主键：名称 + 经纬度保留3位小数"""
    try:
        parts = location.split(",")
        lng = round(float(parts[0]), 3)
        lat = round(float(parts[1]), 3)
        return f"{name}|{lng}|{lat}"
    except (ValueError, IndexError):
        return f"{name}|{location}"


def load_jsonl(filepath: str) -> list[dict]:
    """加载 JSONL 文件，返回字典列表"""
    records = []
    if not os.path.exists(filepath):
        print(f"  ⚠️ 文件不存在，跳过: {filepath}")
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def normalize_record(record: dict, source: str) -> dict:
    """统一不同来源的数据格式"""
    return {
        "name": (record.get("name") or "").strip(),
        "district": (record.get("district") or "").strip(),
        "location": (record.get("location") or "").strip(),
        "address": (record.get("address") or "").strip(),
        "source": source,
    }


# ================= 主流程 =================
def main():
    print("=" * 60)
    print("🧪 接诉即办：地理数据全局提纯引擎")
    print("=" * 60)

    # ── 阶段 1：读取 & 跨源去重 ──
    print("\n📥 阶段 1/3：读取全部数据源并跨源去重...")

    all_records = []

    for source_name, filepath in INPUT_FILES.items():
        raw = load_jsonl(filepath)
        print(f"  [{source_name}] 读取 {len(raw)} 条原始记录")
        for r in raw:
            all_records.append(normalize_record(r, source_name))

    print(f"  合计原始记录：{len(all_records)} 条")

    # 去重：优先保留 AMap 数据（坐标系一致）
    seen_keys = set()
    unique_records = []
    dup_count = 0

    # AMap 优先排序
    all_records.sort(key=lambda x: 0 if x["source"] == "amap_main" else 1)

    for rec in all_records:
        if not rec["name"]:
            continue
        key = make_dedup_key(rec["name"], rec["location"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_records.append(rec)
        else:
            dup_count += 1

    print(f"  去重完毕：剔除 {dup_count} 条重复 → 剩余 {len(unique_records)} 条唯一记录")

    # ── 阶段 2：智能分拣 ──
    print("\n🔬 阶段 2/3：按关键词进行智能分类...")

    categorized = defaultdict(list)
    category_counter = Counter()

    for rec in unique_records:
        cat = classify_name(rec["name"])
        categorized[cat].append(rec)
        category_counter[cat] += 1

    for cat, count in category_counter.most_common():
        emoji = {"community": "🏠", "school": "🏫", "hospital": "🏥",
                 "transit": "🚇", "government": "🏛️", "commercial": "🏪"}.get(cat, "📍")
        print(f"  {emoji} {cat}: {count} 条")

    # ── 阶段 3：标准化输出 ──
    print(f"\n💾 阶段 3/3：写入提纯结果到 {OUTPUT_DIR}/ ...")

    output_files = {}
    for cat, records in categorized.items():
        filename = f"{cat}_clean.jsonl" if cat == "community" else f"{cat}.jsonl"
        filepath = os.path.join(OUTPUT_DIR, filename)
        output_files[cat] = filepath

        # 按区排序，便于人工抽样检查
        records.sort(key=lambda x: (x.get("district", ""), x.get("name", "")))

        with open(filepath, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  ✅ {filename}: {len(records)} 条")

    # ── 同步更新 place_taxonomy ──
    print("\n📦 同步更新 place_taxonomy 分类词库...")

    taxonomy_dir = os.path.join(DATA_DIR, "place_taxonomy")
    taxonomy_mapping = {
        "community": "community.json",
        "school": "school.json",
        "hospital": "hospital.json",
        "transit": "bus_stop.json",  # 交通类统一归入
    }

    for cat, taxonomy_file in taxonomy_mapping.items():
        if cat not in categorized:
            continue

        # 按区归类
        by_district = defaultdict(list)
        for rec in categorized[cat]:
            district = rec.get("district", "未知")
            if district:
                by_district[district].append(rec["name"])

        # 去重并排序
        for district in by_district:
            by_district[district] = sorted(set(by_district[district]))

        taxonomy_path = os.path.join(taxonomy_dir, taxonomy_file)
        with open(taxonomy_path, "w", encoding="utf-8") as f:
            json.dump(dict(by_district), f, ensure_ascii=False, indent=2)

        total_entries = sum(len(v) for v in by_district.values())
        print(f"  ✅ {taxonomy_file}: {len(by_district)} 个区, {total_entries} 条")

    # ── 生成报告 ──
    report = {
        "input_sources": {name: len(load_jsonl(path)) for name, path in INPUT_FILES.items()},
        "total_raw_records": len(all_records),
        "duplicates_removed": dup_count,
        "unique_records": len(unique_records),
        "category_breakdown": dict(category_counter.most_common()),
        "output_files": {cat: os.path.basename(path) for cat, path in output_files.items()},
        "taxonomy_updated": list(taxonomy_mapping.keys()),
    }

    report_path = os.path.join(OUTPUT_DIR, "purification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📊 提纯报告已保存: {report_path}")
    print("\n" + "=" * 60)
    print(f"🎉 提纯完成！核心小区净资产：{category_counter['community']} 条")
    print(f"   附加资产（学校/医院/交通/商业等）：{len(unique_records) - category_counter['community']} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
