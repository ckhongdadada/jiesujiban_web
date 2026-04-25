"""
事实验证模块
实现否定词检测、状态一致性检查、责任主体对齐、高风险标记
"""

from __future__ import annotations

import re
import json
from typing import Any, List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from src.jsjb.core.paths import get_fact_kb_path


@dataclass
class VerificationWarning:
    """验证警告"""
    warning_type: str
    message: str
    context: str
    severity: str  # low, medium, high
    suggestion: str


class FactVerifier:
    """事实验证引擎"""
    
    NEGATION_PATTERNS = [
        (r"不涉及", "不涉及"),
        (r"未涉及", "未涉及"),
        (r"暂未建设", "暂未建设"),
        (r"尚未建设", "尚未建设"),
        (r"不存在", "不存在"),
        (r"无此", "无此"),
        (r"未发现", "未发现"),
        (r"未有", "未有"),
        (r"不属", "不属于"),
        (r"非集团校", "非集团校"),
        (r"无.*?征收", "无征收"),
    ]
    
    AFFIRMATION_PATTERNS = [
        (r"已(?:完成|建成|交付)", "已完成"),
        (r"正在(?:推进|施工|建设)", "进行中"),
        (r"涉及征收", "涉及征收"),
        (r"属于集团校", "属于集团校"),
        (r"已建设", "已建设"),
        (r"已建成", "已建成"),
        (r"已发现", "已发现"),
        (r"已有", "已有"),
        (r"属于", "属于"),
    ]
    
    HIGH_RISK_KEYWORDS = [
        "征地", "拆迁", "集团校", "招生资格", "产权",
        "不动产", "土地征收", "腾退", "产权归属",
        "学区", "入学资格", "房产证", "土地证"
    ]
    
    PROJECT_STATUS_PATTERNS = [
        (r"已(?:完成|完工|竣工|交付)", "已完成"),
        (r"正在(?:施工|建设|推进|办理)", "进行中"),
        (r"暂未(?:开始|启动|建设)", "未开始"),
        (r"规划中", "规划中"),
        (r"前期手续", "前期手续"),
        (r"主体施工", "主体施工"),
        (r"完工验收", "完工验收"),
        (r"运营中", "运营中"),
    ]
    
    DEMOLITION_PATTERNS = [
        (r"拆迁已(?:完成|结束)", "已完成"),
        (r"拆迁正在(?:进行|推进)", "进行中"),
        (r"因拆迁(?:影响|阻碍)", "受阻"),
        (r"受拆迁影响", "受阻"),
    ]
    
    def __init__(self, knowledge_base_path: str | None = None):
        self.kb_path = Path(knowledge_base_path) if knowledge_base_path else None
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        if not self.kb_path or not self.kb_path.exists():
            return {
                "project_status": {},
                "public_resources": {},
                "unit_mapping": {}
            }
        
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {
                "project_status": {},
                "public_resources": {},
                "unit_mapping": {}
            }
    
    def verify_reply(
        self,
        generated_reply: str,
        retrieval_hits: List[Dict[str, Any]],
        location_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        验证生成回复的事实准确性
        
        Args:
            generated_reply: 生成的回复文本
            retrieval_hits: RAG检索结果
            location_result: 地名识别结果
            
        Returns:
            验证结果字典
        """
        warnings = []
        
        negation_warnings = self._check_negation_consistency(generated_reply, retrieval_hits)
        warnings.extend(negation_warnings)
        
        status_warnings = self._check_project_status(generated_reply, retrieval_hits)
        warnings.extend(status_warnings)
        
        unit_warnings = self._check_responsible_unit(generated_reply, retrieval_hits, location_result)
        warnings.extend(unit_warnings)
        
        resource_warnings = self._check_public_resources(generated_reply, retrieval_hits)
        warnings.extend(resource_warnings)
        
        high_risk = self._check_high_risk_content(generated_reply)
        
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for warning in warnings:
            severity_counts[warning.severity] += 1
        
        return {
            "is_valid": len(warnings) == 0 and not high_risk,
            "warnings": [self._warning_to_dict(w) for w in warnings],
            "high_risk": high_risk,
            "needs_review": len(warnings) > 0 or high_risk,
            "severity_counts": severity_counts,
            "summary": self._generate_summary(warnings, high_risk)
        }
    
    def _check_negation_consistency(
        self,
        generated_reply: str,
        retrieval_hits: List[Dict[str, Any]]
    ) -> List[VerificationWarning]:
        """检查否定词一致性"""
        warnings = []
        
        gen_negations = self._extract_patterns(generated_reply, self.NEGATION_PATTERNS)
        
        if not gen_negations:
            return warnings
        
        reference_texts = []
        for hit in retrieval_hits:
            snippet = hit.get("snippet", "") or hit.get("content", "")
            if snippet:
                reference_texts.append(snippet)
        
        for neg_pattern, neg_label in gen_negations:
            for ref_text in reference_texts:
                ref_affirmations = self._extract_patterns(ref_text, self.AFFIRMATION_PATTERNS)
                
                for aff_pattern, aff_label in ref_affirmations:
                    if self._is_contradiction(neg_label, aff_label):
                        warnings.append(VerificationWarning(
                            warning_type="negation_contradiction",
                            message=f"生成回复包含否定表述「{neg_label}」，但参考材料显示「{aff_label}」",
                            context=f"生成: {neg_pattern}\n参考: {aff_pattern}",
                            severity="high",
                            suggestion="请核实该事项的实际状态，确保结论正确"
                        ))
        
        return warnings
    
    def _check_project_status(
        self,
        generated_reply: str,
        retrieval_hits: List[Dict[str, Any]]
    ) -> List[VerificationWarning]:
        """检查项目状态一致性"""
        warnings = []
        
        gen_status = self._extract_patterns(generated_reply, self.PROJECT_STATUS_PATTERNS)
        gen_demolition = self._extract_patterns(generated_reply, self.DEMOLITION_PATTERNS)
        
        for hit in retrieval_hits:
            project_status = hit.get("project_status", "")
            demolition_status = hit.get("demolition_status", "")
            
            if project_status:
                for gen_pattern, gen_label in gen_status:
                    if self._is_status_conflict(gen_label, project_status):
                        warnings.append(VerificationWarning(
                            warning_type="status_conflict",
                            message=f"项目状态不一致：生成回复为「{gen_label}」，知识库为「{project_status}」",
                            context=gen_pattern,
                            severity="high",
                            suggestion=f"请使用知识库中的状态「{project_status}」"
                        ))
            
            if demolition_status:
                for gen_pattern, gen_label in gen_demolition:
                    if demolition_status == "已完成" and gen_label == "受阻":
                        warnings.append(VerificationWarning(
                            warning_type="demolition_status_error",
                            message="拆迁已完成，但生成回复称「因拆迁影响」",
                            context=gen_pattern,
                            severity="high",
                            suggestion="拆迁已完成，不应作为当前阻碍原因"
                        ))
        
        return warnings
    
    def _check_responsible_unit(
        self,
        generated_reply: str,
        retrieval_hits: List[Dict[str, Any]],
        location_result: Dict[str, Any]
    ) -> List[VerificationWarning]:
        """检查责任主体一致性"""
        warnings = []
        
        units_in_reply = self._extract_units(generated_reply)
        
        expected_unit = None
        expected_street = None
        
        for hit in retrieval_hits:
            if hit.get("unit"):
                expected_unit = hit["unit"]
            if hit.get("responsible_unit"):
                expected_unit = hit["responsible_unit"]
        
        if location_result:
            expected_street = location_result.get("street", "")
        
        for unit in units_in_reply:
            if expected_unit and unit != expected_unit:
                if not self._is_unit_compatible(unit, expected_unit):
                    warnings.append(VerificationWarning(
                        warning_type="unit_mismatch",
                        message=f"责任单位不一致：生成回复为「{unit}」，应为「{expected_unit}」",
                        context=unit,
                        severity="medium",
                        suggestion=f"请使用正确的责任单位「{expected_unit}」"
                    ))
            
            if expected_street and "街道" in unit:
                if expected_street not in unit and unit not in expected_street:
                    warnings.append(VerificationWarning(
                        warning_type="street_mismatch",
                        message=f"承办街道不一致：生成回复为「{unit}」，应为「{expected_street}」",
                        context=unit,
                        severity="high",
                        suggestion=f"请使用正确的承办街道「{expected_street}」"
                    ))
        
        return warnings
    
    def _check_public_resources(
        self,
        generated_reply: str,
        retrieval_hits: List[Dict[str, Any]]
    ) -> List[VerificationWarning]:
        """检查公共资源状态"""
        warnings = []
        
        denial_patterns = [
            r"暂未建设",
            r"尚未建设",
            r"不存在",
            r"无此设施",
            r"没有.*?图书",
            r"没有.*?阅览室",
        ]
        
        for pattern in denial_patterns:
            if re.search(pattern, generated_reply):
                for hit in retrieval_hits:
                    resource_status = hit.get("resource_status", "")
                    if resource_status == "已运营":
                        warnings.append(VerificationWarning(
                            warning_type="resource_denial",
                            message="生成回复否认资源存在，但知识库显示资源已运营",
                            context=pattern,
                            severity="high",
                            suggestion="请引用检索结果中的具体资源信息"
                        ))
                        break
        
        return warnings
    
    def _check_high_risk_content(self, generated_reply: str) -> bool:
        """检查高风险内容"""
        for keyword in self.HIGH_RISK_KEYWORDS:
            if keyword in generated_reply:
                return True
        return False
    
    def _extract_patterns(self, text: str, patterns: List[tuple]) -> List[tuple]:
        """提取匹配的模式"""
        results = []
        for pattern, label in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                results.append((match.group(), label))
        return results
    
    def _is_contradiction(self, neg_label: str, aff_label: str) -> bool:
        """判断是否矛盾"""
        contradiction_pairs = [
            ("不涉及", "涉及征收"),
            ("未涉及", "涉及"),
            ("暂未建设", "已建设"),
            ("暂未建设", "已建成"),
            ("不存在", "已有"),
            ("非集团校", "属于集团校"),
        ]
        
        for neg, aff in contradiction_pairs:
            if neg in neg_label and aff in aff_label:
                return True
        return False
    
    def _is_status_conflict(self, gen_status: str, kb_status: str) -> bool:
        """判断状态是否冲突"""
        status_order = ["未开始", "规划中", "前期手续", "主体施工", "进行中", "完工验收", "已完成", "运营中"]
        
        try:
            gen_idx = next(i for i, s in enumerate(status_order) if s in gen_status)
            kb_idx = next(i for i, s in enumerate(status_order) if s in kb_status)
            
            return abs(gen_idx - kb_idx) > 1
        except StopIteration:
            return False
    
    def _extract_units(self, text: str) -> List[str]:
        """提取单位名称"""
        units = []
        
        patterns = [
            r"经(.+?(?:街道|镇|乡|局|委|处|办))",
            r"(.+?(?:街道|镇|乡|局|委|处|办)).*?回复",
            r"由(.+?(?:公司|单位|集团))负责",
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                unit = match.group(1).strip()
                if unit and unit not in units:
                    units.append(unit)
        
        return units
    
    def _is_unit_compatible(self, unit1: str, unit2: str) -> bool:
        """判断单位是否兼容"""
        return unit1 in unit2 or unit2 in unit1
    
    def _warning_to_dict(self, warning: VerificationWarning) -> Dict[str, str]:
        """转换警告为字典"""
        return {
            "warning_type": warning.warning_type,
            "message": warning.message,
            "context": warning.context,
            "severity": warning.severity,
            "suggestion": warning.suggestion
        }
    
    def _generate_summary(self, warnings: List[VerificationWarning], high_risk: bool) -> str:
        """生成验证摘要"""
        if not warnings and not high_risk:
            return "验证通过，未发现事实性问题"
        
        parts = []
        
        if warnings:
            high_count = sum(1 for w in warnings if w.severity == "high")
            medium_count = sum(1 for w in warnings if w.severity == "medium")
            
            if high_count > 0:
                parts.append(f"发现{high_count}个高风险问题")
            if medium_count > 0:
                parts.append(f"发现{medium_count}个中风险问题")
        
        if high_risk:
            parts.append("包含高风险关键词，建议人工复审")
        
        return "，".join(parts) + "。"


def verify_generated_reply(
    generated_reply: str,
    retrieval_hits: List[Dict[str, Any]],
    location_result: Dict[str, Any],
    knowledge_base_path: str | None = None
) -> Dict[str, Any]:
    """
    验证生成回复的便捷函数
    
    Args:
        generated_reply: 生成的回复
        retrieval_hits: RAG检索结果
        location_result: 地名识别结果
        knowledge_base_path: 知识库路径
        
    Returns:
        验证结果
    """
    kb_path = knowledge_base_path or str(get_fact_kb_path())
    verifier = FactVerifier(kb_path)
    return verifier.verify_reply(generated_reply, retrieval_hits, location_result)


if __name__ == "__main__":
    test_reply = "经核实，该道路目前处于施工建设阶段，因受拆迁影响，暂不具备通车条件。"
    test_hits = [
        {
            "snippet": "拆迁已完成并交付施工，主体工程已完成，计划五一前通车。",
            "project_status": "完工验收",
            "demolition_status": "已完成"
        }
    ]
    test_location = {"district": "昌平区"}
    
    result = verify_generated_reply(test_reply, test_hits, test_location)
    print("验证结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
