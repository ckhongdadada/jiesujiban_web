"""
导入完整OwnThink数据到知识图谱
================================
将完整的OwnThink知识图谱数据导入到项目知识图谱中

使用方法:
  1. 先下载 ownthink_v2.csv 到 data/raw/ownthink/
  2. 运行: python tools/import_full_ownthink.py [--max-entities N]

注意:
  - 完整数据有1.4亿条，建议使用 --max-entities 限制导入数量
  - 导入过程可能需要较长时间
  - 建议启用Neo4j以获得更好的查询性能
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class GraphEntity:
    id: str
    name: str
    type: str
    properties: dict[str, Any]
    relations: list[dict[str, Any]]


class OwnThinkImporter:
    def __init__(self, max_entities: int = None):
        self.max_entities = max_entities
        self.entities: dict[str, GraphEntity] = {}
        self.entity_name_index: dict[str, str] = {}
        self.entity_counter = 0
        self.relation_counter = 0
        self.stats = {
            "total_triples": 0,
            "unique_entities": 0,
            "total_relations": 0,
            "skipped_triples": 0,
            "attribute_types": defaultdict(int),
        }
    
    def _get_entity_type(self, attribute: str) -> str:
        type_mapping = {
            "中文名": "Entity",
            "别名": "Entity",
            "描述": "Entity",
            "标签": "Category",
            "所属": "Location",
            "位于": "Location",
            "地址": "Location",
            "首都": "Location",
            "省会": "Location",
            "面积": "Location",
            "人口": "Location",
            "气候": "Location",
            "景点": "Location",
            "特产": "Location",
            "上级": "Organization",
            "下级": "Organization",
            "职能": "Organization",
            "成立时间": "Organization",
            "创始人": "Person",
            "作者": "Person",
            "导演": "Person",
            "演员": "Person",
            "歌手": "Person",
            "出生日期": "Person",
            "逝世日期": "Person",
            "国籍": "Person",
            "职业": "Person",
            "上映时间": "Work",
            "发行时间": "Work",
            "出版社": "Work",
            "ISBN": "Work",
        }
        return type_mapping.get(attribute, "Entity")
    
    def _get_relation_type(self, attribute: str) -> str:
        relation_mapping = {
            "描述": "HAS_DESCRIPTION",
            "中文名": "HAS_NAME",
            "别名": "HAS_ALIAS",
            "简称": "HAS_ABBREVIATION",
            "所属": "BELONGS_TO",
            "位于": "LOCATED_AT",
            "地址": "HAS_ADDRESS",
            "电话": "HAS_PHONE",
            "邮编": "HAS_POSTCODE",
            "面积": "HAS_AREA",
            "人口": "HAS_POPULATION",
            "气候": "HAS_CLIMATE",
            "特产": "HAS_SPECIALTY",
            "景点": "HAS_ATTRACTION",
            "上级": "REPORTS_TO",
            "下级": "HAS_SUBORDINATE",
            "职能": "HAS_FUNCTION",
            "成立时间": "ESTABLISHED_AT",
            "类型": "HAS_TYPE",
            "标签": "HAS_CATEGORY",
        }
        return relation_mapping.get(attribute, f"HAS_{attribute.upper()[:20]}")
    
    def add_entity(self, name: str, entity_type: str = "Entity") -> str:
        if name in self.entity_name_index:
            return self.entity_name_index[name]
        
        entity_id = f"Entity_{self.entity_counter}"
        self.entity_counter += 1
        
        entity = GraphEntity(
            id=entity_id,
            name=name,
            type=entity_type,
            properties={"name": name},
            relations=[]
        )
        
        self.entities[entity_id] = entity
        self.entity_name_index[name] = entity_id
        self.stats["unique_entities"] += 1
        
        return entity_id
    
    def add_relation(self, entity_id: str, relation_type: str, value: str):
        if entity_id not in self.entities:
            return
        
        if len(value) > 500:
            value = value[:500] + "..."
        
        relation = {
            "type": relation_type,
            "value": value
        }
        
        self.entities[entity_id].relations.append(relation)
        self.relation_counter += 1
        self.stats["total_relations"] += 1
    
    def import_from_csv(self, filepath: str, progress_interval: int = 1000000) -> int:
        print(f"开始导入: {filepath}")
        print(f"最大实体数: {self.max_entities or '无限制'}")
        
        start_time = time.time()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            
            for row in reader:
                if self.max_entities and self.stats["unique_entities"] >= self.max_entities:
                    print(f"\n已达到最大实体数限制: {self.max_entities}")
                    break
                
                if len(row) < 3:
                    self.stats["skipped_triples"] += 1
                    continue
                
                entity_name = row[0].strip()
                attribute = row[1].strip()
                value = row[2].strip()
                
                if not entity_name or not attribute or not value:
                    self.stats["skipped_triples"] += 1
                    continue
                
                self.stats["total_triples"] += 1
                self.stats["attribute_types"][attribute] += 1
                
                entity_type = self._get_entity_type(attribute)
                entity_id = self.add_entity(entity_name, entity_type)
                
                relation_type = self._get_relation_type(attribute)
                self.add_relation(entity_id, relation_type, value)
                
                if self.stats["total_triples"] % progress_interval == 0:
                    elapsed = time.time() - start_time
                    rate = self.stats["total_triples"] / elapsed
                    print(f"  已处理: {self.stats['total_triples']:,} 三元组, "
                          f"{self.stats['unique_entities']:,} 实体, "
                          f"速率: {rate:.0f} 条/秒")
        
        elapsed = time.time() - start_time
        print(f"\n导入完成!")
        print(f"  总耗时: {elapsed:.1f} 秒")
        print(f"  总三元组: {self.stats['total_triples']:,}")
        print(f"  唯一实体: {self.stats['unique_entities']:,}")
        print(f"  总关系数: {self.stats['total_relations']:,}")
        print(f"  跳过三元组: {self.stats['skipped_triples']:,}")
        
        return self.stats["unique_entities"]
    
    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n保存知识图谱到: {output_path}")
        
        top_attributes = sorted(
            self.stats["attribute_types"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        graph_data = {
            "metadata": {
                "version": "2.0",
                "source": "ownthink_full",
                "stats": {
                    "total_triples": self.stats["total_triples"],
                    "unique_entities": self.stats["unique_entities"],
                    "total_relations": self.stats["total_relations"],
                    "skipped_triples": self.stats["skipped_triples"],
                    "top_attributes": dict(top_attributes),
                }
            },
            "entities": {eid: asdict(e) for eid, e in self.entities.items()}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  文件大小: {file_size:.1f} MB")
        print(f"  实体数量: {len(self.entities):,}")


def main():
    parser = argparse.ArgumentParser(description="导入完整OwnThink知识图谱")
    parser.add_argument("--input", type=str, help="OwnThink CSV文件路径")
    parser.add_argument("--output", type=str, help="输出知识图谱路径")
    parser.add_argument("--max-entities", type=int, default=None, 
                        help="最大导入实体数（默认无限制）")
    parser.add_argument("--preview", type=int, default=0,
                        help="仅预览前N条数据")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    input_path = args.input or str(
        project_root / "data" / "raw" / "ownthink" / "ownthink_v2.csv"
    )
    output_path = args.output or str(
        project_root / "data" / "runtime" / "knowledge_graph_full.json"
    )
    
    print("=" * 60)
    print("OwnThink 完整知识图谱导入工具")
    print("=" * 60)
    
    if not os.path.exists(input_path):
        print(f"\n错误: 数据文件不存在: {input_path}")
        print("\n请先下载数据:")
        print("  百度网盘: https://pan.baidu.com/s/1LZjs9Dsta0yD9NH-1y0sAw")
        print("  提取码: 3hpp")
        print("  解压密码: https://www.ownthink.com/")
        return
    
    if args.preview > 0:
        print(f"\n预览模式: 仅处理前 {args.preview} 条数据")
        args.max_entities = args.preview
    
    importer = OwnThinkImporter(max_entities=args.max_entities)
    
    importer.import_from_csv(input_path)
    
    importer.save(output_path)
    
    print("\n" + "=" * 60)
    print("导入完成!")
    print("=" * 60)
    print(f"\n下一步:")
    print(f"1. 更新 config.json:")
    print(f'   "knowledge_graph_path": "data/runtime/knowledge_graph_full.json"')
    print(f"2. 如需使用Neo4j:")
    print(f'   "feature_enable_neo4j": true')
    print(f'   "neo4j_uri": "bolt://localhost:7687"')
    print(f"3. 重启应用")


if __name__ == "__main__":
    main()
