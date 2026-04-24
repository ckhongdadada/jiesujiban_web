"""
信息融合引擎
融合多源信息，检测冲突，计算置信度
"""

from __future__ import annotations

import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class FusedFact:
    """融合后的事实"""
    entity_name: str
    fact_type: str
    value: Any
    sources: List[str]
    confidence: float
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    fusion_time: str = ""
    
    def __post_init__(self):
        if not self.fusion_time:
            self.fusion_time = datetime.now().isoformat()


@dataclass
class ConflictInfo:
    """冲突信息"""
    entity_name: str
    fact_type: str
    conflicting_values: List[Tuple[str, Any, str]]
    resolution: str = "pending"
    resolved_value: Any = None


class InformationFusion:
    """信息融合引擎"""
    
    SOURCE_WEIGHTS = {
        "government_website": 1.0,
        "official_announcement": 0.95,
        "user_verified": 0.90,
        "news_api": 0.70,
        "social_media": 0.50,
        "unknown": 0.30
    }
    
    STATUS_ORDER = [
        "未开始", "规划中", "前期手续", "主体施工",
        "进行中", "收尾阶段", "完工验收", "已完成", "运营中"
    ]
    
    def __init__(self):
        self.fused_facts: Dict[str, Dict[str, FusedFact]] = defaultdict(dict)
        self.conflicts: List[ConflictInfo] = []
    
    def fuse_multi_source_info(
        self,
        sources: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        融合多源信息
        
        Args:
            sources: 数据源名称到数据列表的映射
            
        Returns:
            融合结果
        """
        all_facts: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        
        for source_name, data_list in sources.items():
            for data in data_list:
                self._extract_facts_from_data(data, source_name, all_facts)
        
        fused_results = {}
        
        for entity_name, facts_by_type in all_facts.items():
            for fact_type, fact_list in facts_by_type.items():
                fused_fact = self._fuse_facts(entity_name, fact_type, fact_list)
                self.fused_facts[entity_name][fact_type] = fused_fact
                
                if fused_fact.conflicts:
                    conflict = ConflictInfo(
                        entity_name=entity_name,
                        fact_type=fact_type,
                        conflicting_values=[
                            (f["source"], f["value"], f.get("time", ""))
                            for f in fact_list
                        ]
                    )
                    self.conflicts.append(conflict)
                
                fused_results[f"{entity_name}_{fact_type}"] = {
                    "entity_name": entity_name,
                    "fact_type": fact_type,
                    "value": fused_fact.value,
                    "confidence": fused_fact.confidence,
                    "sources": fused_fact.sources,
                    "has_conflict": len(fused_fact.conflicts) > 0
                }
        
        return {
            "fused_facts": fused_results,
            "conflicts": [self._conflict_to_dict(c) for c in self.conflicts],
            "statistics": {
                "total_entities": len(all_facts),
                "total_facts": sum(len(facts) for facts in all_facts.values()),
                "conflict_count": len(self.conflicts)
            }
        }
    
    def _extract_facts_from_data(
        self,
        data: Dict[str, Any],
        source_name: str,
        all_facts: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ) -> None:
        """从数据中提取事实"""
        entity_name = data.get("project") or data.get("name") or data.get("title")
        
        if not entity_name:
            return
        
        source_weight = self.SOURCE_WEIGHTS.get(source_name, 0.5)
        
        if data.get("status"):
            all_facts[entity_name]["status"].append({
                "value": data["status"],
                "source": source_name,
                "weight": source_weight,
                "time": data.get("publish_date", data.get("update_time", ""))
            })
        
        if data.get("demolition_status"):
            all_facts[entity_name]["demolition_status"].append({
                "value": data["demolition_status"],
                "source": source_name,
                "weight": source_weight,
                "time": data.get("publish_date", data.get("update_time", ""))
            })
        
        if data.get("responsible_unit") or data.get("unit"):
            all_facts[entity_name]["responsible_unit"].append({
                "value": data.get("responsible_unit") or data.get("unit"),
                "source": source_name,
                "weight": source_weight,
                "time": data.get("publish_date", data.get("update_time", ""))
            })
    
    def _fuse_facts(
        self,
        entity_name: str,
        fact_type: str,
        fact_list: List[Dict[str, Any]]
    ) -> FusedFact:
        """融合单个事实"""
        if not fact_list:
            return FusedFact(
                entity_name=entity_name,
                fact_type=fact_type,
                value=None,
                sources=[],
                confidence=0.0
            )
        
        value_counts: Dict[Any, float] = defaultdict(float)
        sources = []
        conflicts = []
        
        for fact in fact_list:
            value = fact["value"]
            weight = fact["weight"]
            source = fact["source"]
            
            value_counts[value] += weight
            sources.append(source)
        
        unique_values = set(f["value"] for f in fact_list)
        
        if len(unique_values) > 1:
            if fact_type in ["status", "demolition_status"]:
                resolved_value = self._resolve_status_conflict(fact_list)
            else:
                resolved_value = self._resolve_by_weight(fact_list)
            
            for fact in fact_list:
                if fact["value"] != resolved_value:
                    conflicts.append({
                        "source": fact["source"],
                        "value": fact["value"],
                        "weight": fact["weight"]
                    })
        else:
            resolved_value = fact_list[0]["value"]
        
        total_weight = sum(f["weight"] for f in fact_list)
        confidence = min(total_weight / len(fact_list), 1.0)
        
        return FusedFact(
            entity_name=entity_name,
            fact_type=fact_type,
            value=resolved_value,
            sources=list(set(sources)),
            confidence=confidence,
            conflicts=conflicts
        )
    
    def _resolve_status_conflict(self, fact_list: List[Dict[str, Any]]) -> str:
        """解决状态冲突"""
        status_scores: Dict[str, float] = defaultdict(float)
        
        for fact in fact_list:
            status = fact["value"]
            weight = fact["weight"]
            
            try:
                status_idx = self.STATUS_ORDER.index(status)
                recency_bonus = 0.1 if fact.get("time") else 0
                status_scores[status] += weight + recency_bonus
            except ValueError:
                status_scores[status] += weight * 0.5
        
        return max(status_scores.items(), key=lambda x: x[1])[0]
    
    def _resolve_by_weight(self, fact_list: List[Dict[str, Any]]) -> Any:
        """根据权重解决冲突"""
        value_weights: Dict[Any, float] = defaultdict(float)
        
        for fact in fact_list:
            value_weights[fact["value"]] += fact["weight"]
        
        return max(value_weights.items(), key=lambda x: x[1])[0]
    
    def _conflict_to_dict(self, conflict: ConflictInfo) -> Dict[str, Any]:
        """转换冲突信息为字典"""
        return {
            "entity_name": conflict.entity_name,
            "fact_type": conflict.fact_type,
            "conflicting_values": conflict.conflicting_values,
            "resolution": conflict.resolution,
            "resolved_value": conflict.resolved_value
        }
    
    def get_fused_fact(self, entity_name: str, fact_type: str) -> Optional[FusedFact]:
        """获取融合后的事实"""
        return self.fused_facts.get(entity_name, {}).get(fact_type)
    
    def get_all_conflicts(self) -> List[ConflictInfo]:
        """获取所有冲突"""
        return self.conflicts
    
    def resolve_conflict(
        self,
        entity_name: str,
        fact_type: str,
        resolved_value: Any
    ) -> bool:
        """
        解决冲突
        
        Args:
            entity_name: 实体名称
            fact_type: 事实类型
            resolved_value: 解决后的值
            
        Returns:
            是否解决成功
        """
        for conflict in self.conflicts:
            if conflict.entity_name == entity_name and conflict.fact_type == fact_type:
                conflict.resolution = "manual"
                conflict.resolved_value = resolved_value
                
                if entity_name in self.fused_facts and fact_type in self.fused_facts[entity_name]:
                    self.fused_facts[entity_name][fact_type].value = resolved_value
                
                return True
        
        return False
    
    def export_to_kb_format(self) -> List[Dict[str, Any]]:
        """导出为知识库格式"""
        kb_facts = []
        
        for entity_name, facts_by_type in self.fused_facts.items():
            for fact_type, fused_fact in facts_by_type.items():
                if fused_fact.value is None:
                    continue
                
                kb_fact = {
                    "entity": entity_name,
                    "type": fact_type,
                    "value": fused_fact.value,
                    "confidence": fused_fact.confidence,
                    "sources": fused_fact.sources,
                    "fusion_time": fused_fact.fusion_time
                }
                
                if fact_type == "status":
                    kb_fact["project"] = entity_name
                    kb_fact["status"] = fused_fact.value
                elif fact_type == "demolition_status":
                    kb_fact["project"] = entity_name
                    kb_fact["demolition_status"] = fused_fact.value
                elif fact_type == "responsible_unit":
                    kb_fact["project"] = entity_name
                    kb_fact["unit"] = fused_fact.value
                
                kb_facts.append(kb_fact)
        
        return kb_facts
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_entities": len(self.fused_facts),
            "total_facts": sum(len(facts) for facts in self.fused_facts.values()),
            "conflict_count": len(self.conflicts),
            "unresolved_conflicts": sum(1 for c in self.conflicts if c.resolution == "pending")
        }


def fuse_multi_source_data(
    sources: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    融合多源数据的便捷函数
    
    Args:
        sources: 数据源映射
        
    Returns:
        融合结果
    """
    fusion_engine = InformationFusion()
    return fusion_engine.fuse_multi_source_info(sources)


if __name__ == "__main__":
    sources = {
        "government_website": [
            {
                "project": "鲁疃西路南延",
                "status": "完工验收",
                "demolition_status": "已完成",
                "responsible_unit": "北京市公联公路联络线有限责任公司",
                "publish_date": "2024-03-15"
            }
        ],
        "news_api": [
            {
                "project": "鲁疃西路南延",
                "status": "已完成",
                "publish_date": "2024-03-20"
            }
        ],
        "user_verified": [
            {
                "project": "鲁疃西路南延",
                "status": "进行中",
                "update_time": "2024-03-10"
            }
        ]
    }
    
    print("融合多源信息...")
    fusion_engine = InformationFusion()
    result = fusion_engine.fuse_multi_source_info(sources)
    
    print("\n融合结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n冲突信息:")
    for conflict in result["conflicts"]:
        print(f"  - {conflict['entity_name']} ({conflict['fact_type']})")
        for source, value, time in conflict["conflicting_values"]:
            print(f"    {source}: {value}")
    
    print("\n统计信息:")
    stats = fusion_engine.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
