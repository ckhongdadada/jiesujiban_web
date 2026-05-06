"""
结构化知识库
管理项目状态、公共资源、责任主体映射等结构化信息
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
from threading import Lock


@dataclass
class ProjectInfo:
    """项目信息"""
    name: str
    status: str = ""
    demolition_status: str = ""
    current_phase: str = ""
    expected_completion: str = ""
    responsible_unit: str = ""
    blocking_issue: str = ""
    district: str = ""
    street: str = ""
    update_time: str = ""
    source: str = ""
    confidence: float = 0.0


@dataclass
class PublicResource:
    """公共资源"""
    name: str
    resource_type: str  # library, hospital, school, park, etc.
    status: str = ""  # 已运营, 建设中, 规划中
    count: int = 0
    facilities: List[Dict[str, str]] = field(default_factory=list)
    district: str = ""
    street: str = ""
    update_time: str = ""
    source: str = ""


@dataclass
class UnitMapping:
    """单位映射"""
    project_name: str
    district: str
    street: str
    responsible_unit: str
    contact_info: str = ""
    update_time: str = ""


class StructuredKnowledgeBase:
    """结构化知识库管理器"""
    
    def __init__(self, kb_path: str | None = None):
        self.kb_path = Path(kb_path) if kb_path else self._get_default_path()
        self._lock = Lock()
        
        self.project_status: Dict[str, ProjectInfo] = {}
        self.public_resources: Dict[str, PublicResource] = {}
        self.unit_mapping: Dict[str, UnitMapping] = {}
        
        self._load()
    
    def _get_default_path(self) -> Path:
        from src.jsjb.core.paths import get_runtime_dir
        return get_runtime_dir() / "structured_kb.json"
    
    def _load(self) -> None:
        if not self.kb_path.exists():
            self._initialize_empty_kb()
            return
        
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for name, info in data.get("project_status", {}).items():
                self.project_status[name] = ProjectInfo(**info)
            
            for name, info in data.get("public_resources", {}).items():
                self.public_resources[name] = PublicResource(**info)
            
            for name, info in data.get("unit_mapping", {}).items():
                self.unit_mapping[name] = UnitMapping(**info)
                
        except Exception as e:
            print(f"加载知识库失败: {e}")
            self._initialize_empty_kb()
    
    def _initialize_empty_kb(self) -> None:
        self.project_status = {}
        self.public_resources = {}
        self.unit_mapping = {}
        
        self._add_sample_data()
    
    def _add_sample_data(self) -> None:
        """添加示例数据"""
        self.project_status["鲁疃西路南延"] = ProjectInfo(
            name="鲁疃西路南延",
            status="完工验收",
            demolition_status="已完成",
            current_phase="推进绿化、信号接入附属工程",
            expected_completion="五一前通车",
            responsible_unit="北京市公联公路联络线有限责任公司",
            district="昌平区",
            update_time=datetime.now().isoformat(),
            source="官方公告",
            confidence=0.95
        )
        
        self.public_resources["旧宫镇图书室"] = PublicResource(
            name="旧宫镇图书室",
            resource_type="library",
            status="已运营",
            count=7,
            facilities=[
                {"name": "一卡通联网分馆", "address": "旧宫镇文化中心", "phone": "87975965"},
                {"name": "清欣园社区图书室", "address": "清欣园社区", "distance": "3公里"}
            ],
            district="大兴区",
            street="旧宫镇",
            update_time=datetime.now().isoformat(),
            source="官方数据"
        )
        
        self.unit_mapping["南大街腾退项目"] = UnitMapping(
            project_name="南大街腾退项目",
            district="通州区",
            street="中仓街道",
            responsible_unit="通州区中仓街道",
            update_time=datetime.now().isoformat()
        )
    
    def _save_internal(self) -> None:
        """内部保存（不获取锁，调用方需已持有锁）"""
        self.kb_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "project_status": {name: asdict(info) for name, info in self.project_status.items()},
            "public_resources": {name: asdict(info) for name, info in self.public_resources.items()},
            "unit_mapping": {name: asdict(info) for name, info in self.unit_mapping.items()},
            "update_time": datetime.now().isoformat()
        }

        with open(self.kb_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save(self) -> None:
        """保存知识库"""
        with self._lock:
            self._save_internal()
    
    def get_project_status(self, project_name: str) -> Optional[ProjectInfo]:
        """获取项目状态"""
        return self.project_status.get(project_name)
    
    def update_project_status(self, project_info: ProjectInfo) -> None:
        """更新项目状态"""
        with self._lock:
            project_info.update_time = datetime.now().isoformat()
            self.project_status[project_info.name] = project_info
            self._save_internal()
    
    def get_public_resource(self, resource_name: str) -> Optional[PublicResource]:
        """获取公共资源"""
        return self.public_resources.get(resource_name)
    
    def update_public_resource(self, resource: PublicResource) -> None:
        """更新公共资源"""
        with self._lock:
            resource.update_time = datetime.now().isoformat()
            self.public_resources[resource.name] = resource
            self._save_internal()
    
    def get_unit_mapping(self, project_name: str) -> Optional[UnitMapping]:
        """获取单位映射"""
        return self.unit_mapping.get(project_name)
    
    def update_unit_mapping(self, mapping: UnitMapping) -> None:
        """更新单位映射"""
        with self._lock:
            mapping.update_time = datetime.now().isoformat()
            self.unit_mapping[mapping.project_name] = mapping
            self._save_internal()
    
    def search_projects(self, query: str, district: str = "") -> List[ProjectInfo]:
        """搜索项目"""
        results = []
        query_lower = query.lower()
        
        for project in self.project_status.values():
            if query_lower in project.name.lower():
                if not district or project.district == district:
                    results.append(project)
        
        return results
    
    def search_resources(self, resource_type: str, district: str = "") -> List[PublicResource]:
        """搜索公共资源"""
        results = []
        
        for resource in self.public_resources.values():
            if resource.resource_type == resource_type or not resource_type:
                if not district or resource.district == district:
                    results.append(resource)
        
        return results
    
    def add_fact(self, fact: Dict[str, Any]) -> bool:
        """添加事实"""
        fact_type = fact.get("type")
        
        try:
            if fact_type == "project_status":
                project_name = fact.get("project", "")
                existing = self.get_project_status(project_name) if project_name else None
                project_info = ProjectInfo(
                    name=project_name,
                    status=fact.get("status", "") or (existing.status if existing else ""),
                    demolition_status=fact.get("demolition_status", "") or (existing.demolition_status if existing else ""),
                    current_phase=fact.get("current_phase", "") or (existing.current_phase if existing else ""),
                    expected_completion=fact.get("expected_completion", "") or (existing.expected_completion if existing else ""),
                    responsible_unit=fact.get("responsible_unit", "") or (existing.responsible_unit if existing else ""),
                    district=fact.get("district", "") or (existing.district if existing else ""),
                    street=fact.get("street", "") or (existing.street if existing else ""),
                    source=fact.get("source", "user_verified"),
                    confidence=fact.get("confidence", 0.8)
                )
                self.update_project_status(project_info)
                return True

            elif fact_type == "demolition_status":
                project_name = fact.get("project", "")
                existing = self.get_project_status(project_name) if project_name else None
                project_info = ProjectInfo(
                    name=project_name,
                    status=existing.status if existing else "",
                    demolition_status=fact.get("demolition_status", "") or (existing.demolition_status if existing else ""),
                    current_phase=existing.current_phase if existing else "",
                    expected_completion=existing.expected_completion if existing else "",
                    responsible_unit=existing.responsible_unit if existing else "",
                    district=fact.get("district", "") or (existing.district if existing else ""),
                    street=fact.get("street", "") or (existing.street if existing else ""),
                    source=fact.get("source", "user_verified"),
                    confidence=fact.get("confidence", 0.8)
                )
                self.update_project_status(project_info)
                return True
            
            elif fact_type == "public_resource":
                resource = PublicResource(
                    name=fact.get("name", ""),
                    resource_type=fact.get("resource_type", ""),
                    status=fact.get("status", ""),
                    count=fact.get("count", 0),
                    facilities=fact.get("facilities", []),
                    district=fact.get("district", ""),
                    street=fact.get("street", ""),
                    source=fact.get("source", "user_verified")
                )
                self.update_public_resource(resource)
                return True
            
            elif fact_type in {"unit_mapping", "responsible_unit"}:
                mapping = UnitMapping(
                    project_name=fact.get("project", ""),
                    district=fact.get("district", ""),
                    street=fact.get("street", ""),
                    responsible_unit=fact.get("unit", "")
                )
                self.update_unit_mapping(mapping)
                if mapping.project_name:
                    existing = self.get_project_status(mapping.project_name)
                    if existing:
                        existing.responsible_unit = mapping.responsible_unit or existing.responsible_unit
                        existing.district = mapping.district or existing.district
                        existing.street = mapping.street or existing.street
                        self.update_project_status(existing)
                return True
            
        except Exception as e:
            print(f"添加事实失败: {e}")
            return False
        
        return False
    
    def mark_for_update(self, fact: Dict[str, Any]) -> None:
        """标记需要更新的事实"""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "project_count": len(self.project_status),
            "resource_count": len(self.public_resources),
            "unit_mapping_count": len(self.unit_mapping),
            "last_update": max(
                [p.update_time for p in self.project_status.values()] or ["无"]
            )
        }
    
    def export_to_rag_format(self) -> List[Dict[str, Any]]:
        """导出为RAG格式"""
        documents = []
        
        for name, project in self.project_status.items():
            content = f"项目名称: {name}\n"
            content += f"项目状态: {project.status}\n"
            content += f"拆迁状态: {project.demolition_status}\n"
            content += f"当前阶段: {project.current_phase}\n"
            content += f"预计完工: {project.expected_completion}\n"
            content += f"责任单位: {project.responsible_unit}\n"
            content += f"所属区域: {project.district}"
            
            documents.append({
                "doc_type": "project_status",
                "title": name,
                "content": content,
                "metadata": {
                    "project_status": project.status,
                    "demolition_status": project.demolition_status,
                    "responsible_unit": project.responsible_unit,
                    "district": project.district
                }
            })
        
        for name, resource in self.public_resources.items():
            content = f"资源名称: {name}\n"
            content += f"资源类型: {resource.resource_type}\n"
            content += f"运营状态: {resource.status}\n"
            content += f"数量: {resource.count}\n"
            content += f"所属区域: {resource.district} {resource.street}\n"
            
            if resource.facilities:
                content += "设施列表:\n"
                for facility in resource.facilities:
                    content += f"  - {facility.get('name', '')}"
                    if facility.get('address'):
                        content += f", 地址: {facility['address']}"
                    if facility.get('phone'):
                        content += f", 电话: {facility['phone']}"
                    content += "\n"
            
            documents.append({
                "doc_type": "public_resource",
                "title": name,
                "content": content,
                "metadata": {
                    "resource_status": resource.status,
                    "resource_type": resource.resource_type,
                    "district": resource.district
                }
            })
        
        return documents


_kb_instance: Optional[StructuredKnowledgeBase] = None


def get_knowledge_base(kb_path: str | None = None) -> StructuredKnowledgeBase:
    """获取知识库单例"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = StructuredKnowledgeBase(kb_path)
    return _kb_instance


if __name__ == "__main__":
    kb = get_knowledge_base()
    
    print("知识库统计:")
    print(json.dumps(kb.get_statistics(), ensure_ascii=False, indent=2))
    
    print("\n项目状态示例:")
    project = kb.get_project_status("鲁疃西路南延")
    if project:
        print(json.dumps(asdict(project), ensure_ascii=False, indent=2))
    
    print("\n导出RAG格式示例:")
    docs = kb.export_to_rag_format()
    if docs:
        print(json.dumps(docs[0], ensure_ascii=False, indent=2))
