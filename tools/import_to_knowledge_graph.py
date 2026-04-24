"""
知识图谱导入脚本
================
将处理后的OwnThink数据导入到项目知识图谱中

使用方法:
  python tools/import_to_knowledge_graph.py [--input INPUT_PATH] [--output OUTPUT_PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


class KnowledgeGraphBuilder:
    def __init__(self):
        self.entities: dict[str, GraphEntity] = {}
        self.entity_name_index: dict[str, list[str]] = defaultdict(list)
        self.stats = {
            "total_entities": 0,
            "total_relations": 0,
            "by_type": defaultdict(int)
        }
    
    def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: dict[str, Any] = None
    ) -> str:
        entity_id = f"{entity_type}_{len(self.entities)}"
        
        if name in self.entity_name_index:
            for existing_id in self.entity_name_index[name]:
                if self.entities[existing_id].type == entity_type:
                    return existing_id
        
        entity = GraphEntity(
            id=entity_id,
            name=name,
            type=entity_type,
            properties=properties or {},
            relations=[]
        )
        
        self.entities[entity_id] = entity
        self.entity_name_index[name].append(entity_id)
        self.stats["total_entities"] += 1
        self.stats["by_type"][entity_type] += 1
        
        return entity_id
    
    def add_relation(
        self,
        from_entity: str,
        relation_type: str,
        to_entity: str = None,
        value: str = None,
        properties: dict[str, Any] = None
    ):
        relation = {
            "type": relation_type,
            "properties": properties or {}
        }
        
        if to_entity:
            relation["target"] = to_entity
        elif value:
            relation["value"] = value
        
        if from_entity in self.entities:
            self.entities[from_entity].relations.append(relation)
            self.stats["total_relations"] += 1
    
    def import_from_ownthink(self, input_path: str, max_entities: int = 50000) -> int:
        if not os.path.exists(input_path):
            print(f"错误: 文件不存在 {input_path}")
            return 0
        
        print(f"导入知识图谱数据: {input_path}")
        
        count = 0
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= max_entities:
                    break
                
                try:
                    data = json.loads(line)
                    name = data.get('name', '')
                    category = data.get('category', 'general')
                    
                    if not name:
                        continue
                    
                    entity_type = self._map_category_to_type(category)
                    entity_id = self.add_entity(name, entity_type, data)
                    
                    for key, value in data.items():
                        if key in ['name', 'category']:
                            continue
                        
                        if isinstance(value, str) and len(value) < 500:
                            self.add_relation(
                                entity_id,
                                self._map_attribute_to_relation(key),
                                value=value
                            )
                    
                    count += 1
                    if count % 10000 == 0:
                        print(f"  已导入: {count:,} 个实体")
                
                except json.JSONDecodeError:
                    continue
        
        print(f"导入完成: {count:,} 个实体")
        return count
    
    def _map_category_to_type(self, category: str) -> str:
        mapping = {
            "beijing": "Location",
            "government": "Organization",
            "places": "Location",
            "general": "Entity"
        }
        return mapping.get(category, "Entity")
    
    def _map_attribute_to_relation(self, attribute: str) -> str:
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
            "职能": "HAS_FUNCTION",
            "成立时间": "ESTABLISHED_AT",
            "类型": "HAS_TYPE",
        }
        return relation_mapping.get(attribute, f"HAS_{attribute.upper()}")
    
    def build_domain_graph(self) -> int:
        print("构建领域知识图谱...")
        
        domain_entities = [
            {
                "name": "北京市",
                "type": "Location",
                "properties": {
                    "level": "直辖市",
                    "code": "110000"
                }
            },
            {
                "name": "东城区",
                "type": "Location",
                "properties": {"level": "区", "code": "110101"}
            },
            {
                "name": "西城区",
                "type": "Location",
                "properties": {"level": "区", "code": "110102"}
            },
            {
                "name": "朝阳区",
                "type": "Location",
                "properties": {"level": "区", "code": "110105"}
            },
            {
                "name": "丰台区",
                "type": "Location",
                "properties": {"level": "区", "code": "110106"}
            },
            {
                "name": "石景山区",
                "type": "Location",
                "properties": {"level": "区", "code": "110107"}
            },
            {
                "name": "海淀区",
                "type": "Location",
                "properties": {"level": "区", "code": "110108"}
            },
            {
                "name": "北京市政务服务管理局",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "政务服务管理"}
            },
            {
                "name": "北京市城市管理综合行政执法局",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "城市管理执法"}
            },
            {
                "name": "北京市住房和城乡建设委员会",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "住房建设管理"}
            },
            {
                "name": "北京市交通委员会",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "交通管理"}
            },
            {
                "name": "北京市生态环境局",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "环境保护"}
            },
            {
                "name": "北京市市场监督管理局",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "市场监督管理"}
            },
            {
                "name": "北京市卫生健康委员会",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "卫生健康管理"}
            },
            {
                "name": "北京市教育局",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "教育管理"}
            },
            {
                "name": "北京市公安局",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "公共安全"}
            },
            {
                "name": "北京市人力资源和社会保障局",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "人力资源和社会保障"}
            },
            {
                "name": "北京市民政局",
                "type": "Organization",
                "properties": {"level": "市级", "职能": "民政事务"}
            },
            {
                "name": "12345热线",
                "type": "Service",
                "properties": {"类型": "政务服务热线", "电话": "12345"}
            },
            {
                "name": "接诉即办",
                "type": "Policy",
                "properties": {"类型": "工作机制", "描述": "市民诉求快速响应机制"}
            }
        ]
        
        beijing_id = None
        district_ids = {}
        org_ids = {}
        
        for entity_data in domain_entities:
            entity_id = self.add_entity(
                entity_data["name"],
                entity_data["type"],
                entity_data.get("properties", {})
            )
            
            if entity_data["name"] == "北京市":
                beijing_id = entity_id
            elif entity_data["type"] == "Location" and entity_data["name"] != "北京市":
                district_ids[entity_data["name"]] = entity_id
            elif entity_data["type"] == "Organization":
                org_ids[entity_data["name"]] = entity_id
        
        if beijing_id:
            for district_name, district_id in district_ids.items():
                self.add_relation(district_id, "BELONGS_TO", to_entity=beijing_id)
        
        for org_name, org_id in org_ids.items():
            if beijing_id:
                self.add_relation(org_id, "LOCATED_AT", to_entity=beijing_id)
        
        service_id = self.add_entity("12345热线", "Service", {"电话": "12345"})
        policy_id = self.add_entity("接诉即办", "Policy", {"描述": "市民诉求快速响应机制"})
        
        for org_id in org_ids.values():
            self.add_relation(org_id, "USES_SERVICE", to_entity=service_id)
            self.add_relation(org_id, "FOLLOWS_POLICY", to_entity=policy_id)
        
        print(f"领域知识图谱构建完成: {len(domain_entities)} 个核心实体")
        return len(domain_entities)
    
    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        graph_data = {
            "metadata": {
                "version": "1.0",
                "source": "ownthink + domain",
                "stats": {
                    "total_entities": self.stats["total_entities"],
                    "total_relations": self.stats["total_relations"],
                    "by_type": dict(self.stats["by_type"])
                }
            },
            "entities": {eid: asdict(e) for eid, e in self.entities.items()}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n知识图谱已保存: {output_path}")
        print(f"  实体总数: {self.stats['total_entities']:,}")
        print(f"  关系总数: {self.stats['total_relations']:,}")
        print(f"  按类型统计:")
        for entity_type, count in sorted(self.stats["by_type"].items()):
            print(f"    {entity_type}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="导入知识图谱数据")
    parser.add_argument("--input", type=str, help="OwnThink处理后的数据路径")
    parser.add_argument("--output", type=str, help="输出知识图谱路径")
    parser.add_argument("--max-entities", type=int, default=50000, help="最大导入实体数")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    input_path = args.input or str(
        project_root / "data" / "processed" / "knowledge_graph_entities.jsonl"
    )
    output_path = args.output or str(
        project_root / "data" / "runtime" / "knowledge_graph.json"
    )
    
    print("=" * 60)
    print("知识图谱导入工具")
    print("=" * 60)
    
    builder = KnowledgeGraphBuilder()
    
    builder.build_domain_graph()
    
    if os.path.exists(input_path):
        builder.import_from_ownthink(input_path, max_entities=args.max_entities)
    else:
        print(f"\n注意: OwnThink数据文件不存在: {input_path}")
        print("仅使用领域知识图谱")
    
    builder.save(output_path)
    
    print("\n" + "=" * 60)
    print("导入完成!")
    print("=" * 60)
    print(f"\n下一步:")
    print(f"1. 在 config.json 中设置:")
    print(f'   "feature_enable_knowledge_graph": true')
    print(f'   "knowledge_graph_path": "data/runtime/knowledge_graph.json"')
    print(f"2. 重启应用")


if __name__ == "__main__":
    main()
