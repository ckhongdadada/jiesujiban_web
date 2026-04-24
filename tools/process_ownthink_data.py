"""
OwnThink知识图谱数据处理脚本
============================
从OwnThink开源知识图谱中筛选北京/政务相关实体

数据来源: https://github.com/ownthink/KnowledgeGraphData
下载方式: 百度网盘 (提取码: 3hpp)

使用方法:
  1. 从百度网盘下载 ownthink_v2.csv
  2. 放置到 data/raw/ownthink/ 目录
  3. 运行: python tools/process_ownthink_data.py

输出:
  - data/processed/ownthink_beijing.jsonl  (北京相关)
  - data/processed/ownthink_government.jsonl (政务相关)
  - data/processed/ownthink_places.jsonl (地名相关)
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Iterator, Optional
from pathlib import Path


@dataclass
class KnowledgeTriple:
    entity: str
    attribute: str
    value: str
    category: str = "general"


BEIJING_KEYWORDS = [
    "北京", "东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区",
    "门头沟区", "房山区", "通州区", "顺义区", "昌平区", "大兴区",
    "怀柔区", "平谷区", "密云区", "延庆区",
    "天安门", "故宫", "颐和园", "长城", "鸟巢", "水立方",
    "中关村", "国贸", "三里屯", "王府井", "西单", "望京",
    "地铁", "公交", "首都", "京"
]

GOVERNMENT_KEYWORDS = [
    "政府", "部门", "局", "委", "办", "厅", "署", "院",
    "街道", "社区", "居委会", "村委会",
    "政策", "法规", "条例", "规定", "办法", "通知", "公告",
    "审批", "许可", "登记", "备案", "申请", "办理",
    "公共服务", "政务服务", "便民", "民生",
    "投诉", "举报", "信访", "热线"
]

PLACE_KEYWORDS = [
    "小区", "家园", "花园", "公寓", "社区", "新村", "嘉园", "华庭",
    "大厦", "广场", "中心", "商城", "市场",
    "学校", "医院", "公园", "体育场", "图书馆", "博物馆",
    "街道", "路", "巷", "胡同", "门", "桥",
    "镇", "乡", "村", "县", "区", "市"
]

FILTER_CATEGORIES = {
    "beijing": BEIJING_KEYWORDS,
    "government": GOVERNMENT_KEYWORDS,
    "places": PLACE_KEYWORDS
}


def matches_keywords(text: str, keywords: list[str]) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            return True
    return False


def categorize_triple(entity: str, attribute: str, value: str) -> list[str]:
    categories = []
    full_text = f"{entity} {attribute} {value}"
    
    for category, keywords in FILTER_CATEGORIES.items():
        if matches_keywords(full_text, keywords):
            categories.append(category)
    
    return categories if categories else ["general"]


def read_ownthink_csv(filepath: str) -> Iterator[KnowledgeTriple]:
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        
        for row in reader:
            if len(row) >= 3:
                entity, attribute, value = row[0], row[1], row[2]
                categories = categorize_triple(entity, attribute, value)
                
                for cat in categories:
                    yield KnowledgeTriple(
                        entity=entity,
                        attribute=attribute,
                        value=value,
                        category=cat
                    )


def process_ownthink_data(
    input_path: str,
    output_dir: str,
    max_per_category: int = 100000
) -> dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    
    counters = defaultdict(int)
    files = {}
    
    for category in list(FILTER_CATEGORIES.keys()) + ["general"]:
        output_path = os.path.join(output_dir, f"ownthink_{category}.jsonl")
        files[category] = open(output_path, 'w', encoding='utf-8')
    
    try:
        total_processed = 0
        total_matched = 0
        
        for triple in read_ownthink_csv(input_path):
            total_processed += 1
            
            if total_processed % 1000000 == 0:
                print(f"已处理: {total_processed:,} 条, 匹配: {total_matched:,} 条")
            
            category = triple.category
            if category != "general" and counters[category] < max_per_category:
                files[category].write(json.dumps(asdict(triple), ensure_ascii=False) + '\n')
                counters[category] += 1
                total_matched += 1
        
        print(f"\n处理完成!")
        print(f"总处理: {total_processed:,} 条")
        print(f"总匹配: {total_matched:,} 条")
        print(f"\n各类别统计:")
        for category, count in counters.items():
            if count > 0:
                print(f"  {category}: {count:,} 条")
        
        return dict(counters)
    
    finally:
        for f in files.values():
            f.close()


def create_consolidated_graph(
    input_dir: str,
    output_path: str,
    categories: list[str] = None
) -> int:
    if categories is None:
        categories = ["beijing", "government", "places"]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_count = 0
    entities = defaultdict(dict)
    
    for category in categories:
        input_path = os.path.join(input_dir, f"ownthink_{category}.jsonl")
        if not os.path.exists(input_path):
            print(f"跳过: {input_path} (不存在)")
            continue
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    triple = json.loads(line)
                    entity = triple['entity']
                    attr = triple['attribute']
                    value = triple['value']
                    
                    if entity not in entities:
                        entities[entity] = {'name': entity, 'category': category}
                    
                    entities[entity][attr] = value
                    total_count += 1
                except json.JSONDecodeError:
                    continue
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entity_data in entities.values():
            f.write(json.dumps(entity_data, ensure_ascii=False) + '\n')
    
    print(f"合并完成: {len(entities):,} 个实体, {total_count:,} 条关系")
    return len(entities)


def main():
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "raw" / "ownthink" / "ownthink_v2.csv"
    output_dir = project_root / "data" / "processed" / "ownthink"
    consolidated_path = project_root / "data" / "processed" / "knowledge_graph_entities.jsonl"
    
    if not input_path.exists():
        print("=" * 60)
        print("OwnThink 数据文件不存在!")
        print("=" * 60)
        print(f"\n请按以下步骤操作:")
        print(f"1. 访问 https://github.com/ownthink/KnowledgeGraphData")
        print(f"2. 从百度网盘下载 ownthink_v2.csv (提取码: 3hpp)")
        print(f"3. 解压后将文件放到: {input_path}")
        print(f"4. 重新运行此脚本")
        print()
        return
    
    print("=" * 60)
    print("开始处理 OwnThink 知识图谱数据")
    print("=" * 60)
    print(f"输入: {input_path}")
    print(f"输出: {output_dir}")
    print()
    
    counters = process_ownthink_data(
        str(input_path),
        str(output_dir),
        max_per_category=100000
    )
    
    print("\n" + "=" * 60)
    print("合并知识图谱")
    print("=" * 60)
    
    entity_count = create_consolidated_graph(
        str(output_dir),
        str(consolidated_path),
        categories=["beijing", "government", "places"]
    )
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"知识图谱实体文件: {consolidated_path}")
    print(f"实体数量: {entity_count:,}")


if __name__ == "__main__":
    main()
