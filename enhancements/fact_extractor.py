"""
事实提取器
从用户反馈和正确回复中提取结构化事实
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ExtractedFact:
    """提取的事实"""
    fact_type: str
    content: Dict[str, Any]
    confidence: float
    source: str
    extracted_at: str


class FactExtractor:
    """事实提取器"""
    
    PROJECT_STATUS_PATTERNS = [
        (r"(.+?)(?:道路|工程|项目).{0,5}(?:已(?:完成|完工|竣工|交付))", "已完成"),
        (r"(.+?)(?:道路|工程|项目).{0,5}(?:正在(?:施工|建设|推进|办理))", "进行中"),
        (r"(.+?)(?:道路|工程|项目).{0,5}(?:暂未(?:开始|启动|建设))", "未开始"),
        (r"(.+?)(?:道路|工程|项目).{0,5}(?:规划中)", "规划中"),
    ]
    
    DEMOLITION_PATTERNS = [
        (r"(.+?)(?:拆迁|征收).{0,5}(?:已(?:完成|结束))", "已完成"),
        (r"(.+?)(?:拆迁|征收).{0,5}(?:正在(?:进行|推进))", "进行中"),
        (r"(.+?)(?:拆迁|征收).{0,5}(?:暂未(?:开始|启动))", "未开始"),
    ]
    
    RESPONSIBLE_UNIT_PATTERNS = [
        r"(.+?(?:道路|工程|项目)).{0,10}(?:由|责任单位[是为])(.+?(?:公司|单位|集团|局|委|街道))",
        r"(.+?(?:道路|工程|项目)).{0,10}(?:承办单位[是为])(.+?(?:公司|单位|集团|局|委|街道))",
        r"(.+?(?:道路|工程|项目)).{0,10}(?:负责单位[是为])(.+?(?:公司|单位|集团|局|委|街道))",
    ]
    
    PUBLIC_RESOURCE_PATTERNS = [
        (r"(.+?(?:图书室|图书馆|阅览室)).{0,5}(?:已(?:运营|开放|建成))", "已运营"),
        (r"(.+?(?:医院|卫生院)).{0,5}(?:已(?:运营|开放|建成))", "已运营"),
        (r"(.+?(?:学校|幼儿园)).{0,5}(?:已(?:运营|开放|建成))", "已运营"),
        (r"(.+?(?:公园|广场)).{0,5}(?:已(?:运营|开放|建成))", "已运营"),
    ]
    
    CONTACT_INFO_PATTERNS = [
        r"电话[：:]\s*(\d{3,4}[-\s]?\d{7,8})",
        r"联系电话[：:]\s*(\d{3,4}[-\s]?\d{7,8})",
        r"(\d{3,4}[-\s]?\d{7,8})",
    ]
    
    ADDRESS_PATTERNS = [
        r"地址[：:]\s*(.+?)(?:，|。|电话|$)",
        r"位于\s*(.+?)(?:，|。|电话|$)",
    ]
    
    DISTRICT_PATTERNS = [
        r"(.+?区)",
        r"(.+?县)",
    ]
    
    def __init__(self):
        pass
    
    def extract_facts_from_reply(
        self,
        reply: str,
        reference_reply: str = "",
        feedback_data: Optional[Dict[str, Any]] = None
    ) -> List[ExtractedFact]:
        """
        从回复中提取事实
        
        Args:
            reply: 生成的回复
            reference_reply: 参考的正确回复
            feedback_data: 反馈数据
            
        Returns:
            提取的事实列表
        """
        facts = []
        
        text = reference_reply if reference_reply else reply
        
        project_facts = self._extract_project_status(text)
        facts.extend(project_facts)
        
        demolition_facts = self._extract_demolition_status(text)
        facts.extend(demolition_facts)
        
        unit_facts = self._extract_responsible_unit(text)
        facts.extend(unit_facts)
        
        resource_facts = self._extract_public_resources(text)
        facts.extend(resource_facts)
        
        if feedback_data:
            district = feedback_data.get("district", "")
            for fact in facts:
                if "district" not in fact.content:
                    fact.content["district"] = district
        
        return facts
    
    def _extract_project_status(self, text: str) -> List[ExtractedFact]:
        """提取项目状态"""
        facts = []
        
        for pattern, status in self.PROJECT_STATUS_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                project_name = match.group(1).strip()
                
                if self._is_valid_project_name(project_name):
                    facts.append(ExtractedFact(
                        fact_type="project_status",
                        content={
                            "project": project_name,
                            "status": status,
                            "source_text": match.group(0)
                        },
                        confidence=0.85,
                        source="user_verified",
                        extracted_at=datetime.now().isoformat()
                    ))
        
        return facts
    
    def _extract_demolition_status(self, text: str) -> List[ExtractedFact]:
        """提取拆迁状态"""
        facts = []
        
        for pattern, status in self.DEMOLITION_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                project_name = match.group(1).strip()
                
                if self._is_valid_project_name(project_name):
                    facts.append(ExtractedFact(
                        fact_type="demolition_status",
                        content={
                            "project": project_name,
                            "demolition_status": status,
                            "source_text": match.group(0)
                        },
                        confidence=0.85,
                        source="user_verified",
                        extracted_at=datetime.now().isoformat()
                    ))
        
        return facts
    
    def _extract_responsible_unit(self, text: str) -> List[ExtractedFact]:
        """提取责任单位"""
        facts = []
        
        for pattern in self.RESPONSIBLE_UNIT_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                project_name = match.group(1).strip()
                unit_name = match.group(2).strip()
                
                if self._is_valid_project_name(project_name) and self._is_valid_unit_name(unit_name):
                    facts.append(ExtractedFact(
                        fact_type="responsible_unit",
                        content={
                            "project": project_name,
                            "unit": unit_name,
                            "source_text": match.group(0)
                        },
                        confidence=0.90,
                        source="user_verified",
                        extracted_at=datetime.now().isoformat()
                    ))
        
        return facts
    
    def _extract_public_resources(self, text: str) -> List[ExtractedFact]:
        """提取公共资源"""
        facts = []
        
        for pattern, status in self.PUBLIC_RESOURCE_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                resource_name = match.group(1).strip()
                
                resource_type = self._determine_resource_type(resource_name)
                
                contact_info = self._extract_contact_info(text, match.start(), match.end())
                address = self._extract_address(text, match.start(), match.end())
                
                content = {
                    "name": resource_name,
                    "resource_type": resource_type,
                    "status": status,
                    "source_text": match.group(0)
                }
                
                if contact_info:
                    content["phone"] = contact_info
                if address:
                    content["address"] = address
                
                facts.append(ExtractedFact(
                    fact_type="public_resource",
                    content=content,
                    confidence=0.80,
                    source="user_verified",
                    extracted_at=datetime.now().isoformat()
                ))
        
        return facts
    
    def _extract_contact_info(self, text: str, start: int, end: int) -> Optional[str]:
        """提取联系信息"""
        search_range = 200
        search_start = max(0, start - search_range)
        search_end = min(len(text), end + search_range)
        search_text = text[search_start:search_end]
        
        for pattern in self.CONTACT_INFO_PATTERNS:
            match = re.search(pattern, search_text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_address(self, text: str, start: int, end: int) -> Optional[str]:
        """提取地址"""
        search_range = 200
        search_start = max(0, start - search_range)
        search_end = min(len(text), end + search_range)
        search_text = text[search_start:search_end]
        
        for pattern in self.ADDRESS_PATTERNS:
            match = re.search(pattern, search_text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _is_valid_project_name(self, name: str) -> bool:
        """判断是否为有效的项目名称"""
        if not name or len(name) < 2:
            return False
        
        invalid_patterns = [
            r"^[一二三四五六七八九十]+$",
            r"^(?:的|了|在|有|是)$",
            r"^(?:经|由|已|正)$",
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, name):
                return False
        
        return True
    
    def _is_valid_unit_name(self, name: str) -> bool:
        """判断是否为有效的单位名称"""
        if not name or len(name) < 3:
            return False
        
        valid_suffixes = ["公司", "单位", "集团", "局", "委", "街道", "镇", "乡", "处", "办"]
        
        return any(name.endswith(suffix) for suffix in valid_suffixes)
    
    def _determine_resource_type(self, name: str) -> str:
        """确定资源类型"""
        if "图书" in name or "阅览" in name:
            return "library"
        elif "医院" in name or "卫生" in name:
            return "hospital"
        elif "学校" in name or "幼儿园" in name:
            return "school"
        elif "公园" in name:
            return "park"
        elif "广场" in name:
            return "plaza"
        else:
            return "other"
    
    def extract_district(self, text: str) -> Optional[str]:
        """提取行政区"""
        for pattern in self.DISTRICT_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def facts_to_kb_format(self, facts: List[ExtractedFact]) -> List[Dict[str, Any]]:
        """将事实转换为知识库格式"""
        kb_facts = []
        
        for fact in facts:
            kb_fact = {
                "type": fact.fact_type,
                "confidence": fact.confidence,
                "source": fact.source,
                "extracted_at": fact.extracted_at
            }
            
            if fact.fact_type == "project_status":
                kb_fact["project"] = fact.content.get("project", "")
                kb_fact["status"] = fact.content.get("status", "")
            
            elif fact.fact_type == "demolition_status":
                kb_fact["project"] = fact.content.get("project", "")
                kb_fact["demolition_status"] = fact.content.get("demolition_status", "")
            
            elif fact.fact_type == "responsible_unit":
                kb_fact["project"] = fact.content.get("project", "")
                kb_fact["unit"] = fact.content.get("unit", "")
            
            elif fact.fact_type == "public_resource":
                kb_fact["name"] = fact.content.get("name", "")
                kb_fact["resource_type"] = fact.content.get("resource_type", "")
                kb_fact["status"] = fact.content.get("status", "")
                if fact.content.get("phone"):
                    kb_fact["phone"] = fact.content.get("phone")
                if fact.content.get("address"):
                    kb_fact["address"] = fact.content.get("address")
            
            if "district" in fact.content:
                kb_fact["district"] = fact.content["district"]
            
            kb_facts.append(kb_fact)
        
        return kb_facts


def extract_facts_from_feedback(
    feedback_data: Dict[str, Any],
    reference_reply: str = ""
) -> List[Dict[str, Any]]:
    """
    从反馈中提取事实的便捷函数
    
    Args:
        feedback_data: 反馈数据
        reference_reply: 参考的正确回复
        
    Returns:
        知识库格式的事实列表
    """
    extractor = FactExtractor()
    
    reply = feedback_data.get("reply", "")
    
    facts = extractor.extract_facts_from_reply(
        reply=reply,
        reference_reply=reference_reply,
        feedback_data=feedback_data
    )
    
    return extractor.facts_to_kb_format(facts)


if __name__ == "__main__":
    test_reply = """
    经核实，鲁疃西路南延工程已完工验收，拆迁已完成并交付施工。
    该项目由北京市公联公路联络线有限责任公司负责建设。
    目前正在推进绿化、信号接入等附属工程，计划五一前通车。
    
    另外，旧宫镇图书室已运营，共有7个分馆。
    一卡通联网分馆地址：旧宫镇文化中心，电话：87975965。
    """
    
    extractor = FactExtractor()
    facts = extractor.extract_facts_from_reply(test_reply)
    
    print("提取的事实:")
    for fact in facts:
        print(f"\n类型: {fact.fact_type}")
        print(f"内容: {json.dumps(fact.content, ensure_ascii=False, indent=2)}")
        print(f"置信度: {fact.confidence}")
    
    print("\n知识库格式:")
    kb_facts = extractor.facts_to_kb_format(facts)
    print(json.dumps(kb_facts, ensure_ascii=False, indent=2))
