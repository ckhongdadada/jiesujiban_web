"""Generated-reply quality attribution and error extraction."""

from __future__ import annotations

import re
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from src.jsjb.reply_generation.atomic_facts import split_reply_to_atomic_facts
from src.jsjb.reply_generation.fact_extraction import FactExtractor
from src.jsjb.reply_generation.verification import verify_generated_reply


BEIJING_DISTRICTS = [
    "东城区",
    "西城区",
    "朝阳区",
    "海淀区",
    "丰台区",
    "石景山区",
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

DISTRICT_PATTERN = re.compile("|".join(re.escape(item) for item in BEIJING_DISTRICTS))
STREET_PATTERN = re.compile(r"([\u4e00-\u9fa5]{2,12}(?:街道|镇|乡|办事处))")


class ReplyErrorExtractor:
    """Compare a generated reply with references/RAG evidence and attribute quality issues."""

    GREETING_PATTERNS = [
        r"^尊敬的.*?[，,！!\n]",
        r"^您好\s*[！!，,：:\n]?",
        r"^你好\s*[！!，,：:\n]?",
        r"^市民朋友\s*[，,！!\n]?",
    ]

    CLOSING_PATTERNS = [
        r"感谢您.*$",
        r"欢迎您.*$",
        r"祝您.*$",
        r"特此回复.*$",
        r"如有疑问.*$",
        r"\n\s*[\u4e00-\u9fa5]{2,20}(?:委员会|办公室|管理局|街道办|镇政府|政府|局)\s*\n?\s*\d{4}年\d{1,2}月\d{1,2}日\s*$",
        r"\d{4}年\d{1,2}月\d{1,2}日\s*$",
    ]

    ACTION_KEYWORDS = [
        "已安排",
        "已责成",
        "已督促",
        "已协调",
        "已处理",
        "已整改",
        "现场核查",
        "现场查看",
        "维修",
        "清理",
        "整治",
        "协调",
        "办理",
        "解决",
        "反馈",
        "下一步",
        "将继续",
    ]

    FORMAT_VIOLATION_PATTERNS = [
        (r"您好|你好|尊敬的", "包含问候语，不符合当前“直接输出办理结果”的生成约束。"),
        (r"感谢您|欢迎您|祝您|特此回复", "包含结束套话，不符合当前精简回复格式。"),
        (r"联系电话[:：]?\s*\d", "包含联系方式，可能偏离演示系统的简洁回复要求。"),
    ]

    GENERIC_PROGRESS_PATTERNS = [
        r"正在.*?(研究|推进|协调)",
        r"请.*?耐心等待",
        r"已转.*?部门",
        r"请.*?关注.*?进展",
    ]

    def __init__(self):
        self.fact_extractor = FactExtractor()

    def analyze(
        self,
        generated_reply: str,
        reference_reply: str,
        retrieval_hits: Optional[List[Dict[str, Any]]] = None,
        location_result: Optional[Dict[str, Any]] = None,
        expected_unit: str = "",
        feedback_type: str = "",
        comments: str = "",
    ) -> Dict[str, Any]:
        generated_reply = generated_reply or ""
        reference_reply = reference_reply or ""
        retrieval_hits = retrieval_hits or []
        location_result = location_result or {}

        normalized_generated = self.normalize_reply(generated_reply)
        normalized_reference = self.normalize_reply(reference_reply)

        generated_facts = [asdict(item) for item in self.fact_extractor.extract_facts_from_reply(generated_reply)]
        reference_facts = [asdict(item) for item in self.fact_extractor.extract_facts_from_reply(reference_reply)]
        generated_atomic_facts = split_reply_to_atomic_facts(generated_reply)
        reference_atomic_facts = split_reply_to_atomic_facts(reference_reply)

        errors: List[Dict[str, Any]] = []
        errors.extend(self._detect_format_violations(generated_reply))
        errors.extend(self._compare_facts(generated_facts, reference_facts))
        errors.extend(self._compare_locations(normalized_generated, normalized_reference, location_result))
        errors.extend(self._compare_actions(normalized_generated, normalized_reference))
        errors.extend(self._detect_actionability(normalized_generated))
        errors.extend(self._detect_expected_unit_issue(normalized_generated, expected_unit))
        errors.extend(self._detect_retrieval_grounding_issues(normalized_generated, retrieval_hits))
        errors.extend(self._detect_user_reported_issue(feedback_type, comments))

        verification = None
        if retrieval_hits:
            verification = verify_generated_reply(
                generated_reply=generated_reply,
                retrieval_hits=retrieval_hits,
                location_result=location_result,
            )
            for warning in verification.get("warnings", []):
                errors.append(
                    {
                        "error_type": warning.get("warning_type", "verification_warning"),
                        "severity": warning.get("severity", "medium"),
                        "message": warning.get("message", ""),
                        "generated_value": warning.get("context", ""),
                        "reference_value": "",
                    }
                )
            if verification.get("high_risk"):
                errors.append(
                    {
                        "error_type": "unsafe_high_risk",
                        "severity": "high",
                        "message": "回复包含高风险内容，建议人工复核。",
                        "generated_value": generated_reply[:200],
                        "reference_value": "",
                    }
                )

        deduped_errors = self._dedupe_errors(errors)
        severity = self._overall_severity(deduped_errors)
        error_types = sorted({item["error_type"] for item in deduped_errors})
        quality_dimensions = self._build_quality_dimensions(
            errors=deduped_errors,
            retrieval_hits=retrieval_hits,
            normalized_generated=normalized_generated,
            normalized_reference=normalized_reference,
        )
        summary = self._build_summary(deduped_errors, normalized_generated, normalized_reference)

        return {
            "summary": summary,
            "severity": severity,
            "error_count": len(deduped_errors),
            "error_types": error_types,
            "errors": deduped_errors,
            "normalized_generated_reply": normalized_generated,
            "normalized_reference_reply": normalized_reference,
            "generated_facts": generated_facts,
            "reference_facts": reference_facts,
            "generated_atomic_facts": generated_atomic_facts,
            "reference_atomic_facts": reference_atomic_facts,
            "similarity": self._build_similarity(normalized_generated, normalized_reference),
            "verification": verification,
            "quality_dimensions": quality_dimensions,
            "routing_recommendations": self._build_routing_recommendations(error_types, severity),
            "needs_review": severity in {"medium", "high"} or bool(deduped_errors),
        }

    def normalize_reply(self, text: str) -> str:
        text = (text or "").strip()
        for pattern in self.GREETING_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        for pattern in self.CLOSING_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)
        text = re.sub(r"^\s*[：:,，。！!\n]+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _detect_format_violations(self, generated_reply: str) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        for pattern, message in self.FORMAT_VIOLATION_PATTERNS:
            if re.search(pattern, generated_reply):
                violations.append(
                    {
                        "error_type": "format_violation",
                        "severity": "low",
                        "message": message,
                        "generated_value": generated_reply[:200],
                        "reference_value": "",
                    }
                )
        if len(generated_reply.strip()) > 260:
            violations.append(
                {
                    "error_type": "format_violation",
                    "severity": "low",
                    "message": "回复偏长，可能不符合简洁政务回复格式。",
                    "generated_value": str(len(generated_reply.strip())),
                    "reference_value": "",
                }
            )
        return violations

    def _compare_facts(
        self,
        generated_facts: List[Dict[str, Any]],
        reference_facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not reference_facts:
            return []

        errors: List[Dict[str, Any]] = []
        generated_index = self._index_facts(generated_facts)
        reference_index = self._index_facts(reference_facts)

        for key, ref_fact in reference_index.items():
            gen_fact = generated_index.get(key)
            if gen_fact is None:
                errors.append(
                    {
                        "error_type": "missing_fact",
                        "severity": "medium",
                        "message": f"标准回复包含 {ref_fact['fact_type']}，生成回复缺失。",
                        "generated_value": "",
                        "reference_value": str(ref_fact.get("content", {})),
                    }
                )
                continue

            comparison = self._compare_fact_values(gen_fact, ref_fact)
            if comparison is not None:
                errors.append(comparison)

        for key, gen_fact in generated_index.items():
            if key not in reference_index:
                errors.append(
                    {
                        "error_type": "hallucinated_fact",
                        "severity": "medium",
                        "message": f"生成回复出现标准回复中没有的 {gen_fact['fact_type']}。",
                        "generated_value": str(gen_fact.get("content", {})),
                        "reference_value": "",
                    }
                )

        return errors

    def _index_facts(self, facts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for fact in facts:
            content = fact.get("content", {})
            key_parts = [fact.get("fact_type", "")]
            for field in ("project", "name", "unit"):
                if field in content:
                    key_parts.append(str(content.get(field, "")))
            indexed["|".join(key_parts)] = fact
        return indexed

    def _compare_fact_values(
        self,
        generated_fact: Dict[str, Any],
        reference_fact: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        gen_content = generated_fact.get("content", {})
        ref_content = reference_fact.get("content", {})
        fact_type = reference_fact.get("fact_type", "")

        field_by_type = {
            "project_status": "status",
            "demolition_status": "demolition_status",
            "responsible_unit": "unit",
            "public_resource": "status",
        }
        field = field_by_type.get(fact_type)
        if not field:
            return None
        if gen_content.get(field) == ref_content.get(field):
            return None
        error_type = "unit_mismatch" if fact_type == "responsible_unit" else "status_conflict"
        return {
            "error_type": error_type,
            "severity": "high" if fact_type in {"project_status", "demolition_status", "responsible_unit"} else "medium",
            "message": f"{fact_type} 与标准回复不一致。",
            "generated_value": gen_content.get(field, ""),
            "reference_value": ref_content.get(field, ""),
        }

    def _compare_locations(
        self,
        normalized_generated: str,
        normalized_reference: str,
        location_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        expected_district = location_result.get("district", "")
        expected_street = location_result.get("street") or location_result.get("subdistrict") or ""

        generated_districts = set(DISTRICT_PATTERN.findall(normalized_generated))
        reference_districts = set(DISTRICT_PATTERN.findall(normalized_reference))
        if expected_district:
            reference_districts.add(expected_district)

        if reference_districts and generated_districts and not generated_districts.issubset(reference_districts):
            errors.append(
                {
                    "error_type": "district_or_street_mismatch",
                    "severity": "high",
                    "message": "生成回复出现与标准答案或识别地点不一致的行政区。",
                    "generated_value": "、".join(sorted(generated_districts)),
                    "reference_value": "、".join(sorted(reference_districts)),
                }
            )

        generated_streets = set(STREET_PATTERN.findall(normalized_generated))
        if expected_street and generated_streets and expected_street not in generated_streets:
            errors.append(
                {
                    "error_type": "district_or_street_mismatch",
                    "severity": "medium",
                    "message": "生成回复中的街道信息与识别结果不一致。",
                    "generated_value": "、".join(sorted(generated_streets)),
                    "reference_value": expected_street,
                }
            )
        return errors

    def _compare_actions(self, normalized_generated: str, normalized_reference: str) -> List[Dict[str, Any]]:
        if not normalized_reference:
            return []
        reference_actions = [item for item in self.ACTION_KEYWORDS if item in normalized_reference]
        generated_actions = [item for item in self.ACTION_KEYWORDS if item in normalized_generated]
        if reference_actions and not generated_actions:
            return [
                {
                    "error_type": "action_missing",
                    "severity": "medium",
                    "message": "标准回复包含明确办理动作，但生成回复未体现。",
                    "generated_value": "",
                    "reference_value": "、".join(reference_actions),
                }
            ]
        return []

    def _detect_actionability(self, normalized_generated: str) -> List[Dict[str, Any]]:
        if not normalized_generated:
            return [
                {
                    "error_type": "empty_reply",
                    "severity": "high",
                    "message": "生成回复为空。",
                    "generated_value": "",
                    "reference_value": "",
                }
            ]
        has_action = any(keyword in normalized_generated for keyword in self.ACTION_KEYWORDS)
        has_generic_progress = any(re.search(pattern, normalized_generated) for pattern in self.GENERIC_PROGRESS_PATTERNS)
        if has_generic_progress and len(normalized_generated) < 120:
            return [
                {
                    "error_type": "low_actionability",
                    "severity": "medium",
                    "message": "回复偏向进度性套话，缺少具体核实情况或办理结果。",
                    "generated_value": normalized_generated[:200],
                    "reference_value": "",
                }
            ]
        if not has_action and len(normalized_generated) < 180:
            return [
                {
                    "error_type": "low_actionability",
                    "severity": "medium",
                    "message": "回复缺少明确办理动作，建议补充核实、处理、整改或协调结果。",
                    "generated_value": normalized_generated[:200],
                    "reference_value": "",
                }
            ]
        return []

    def _detect_expected_unit_issue(self, normalized_generated: str, expected_unit: str) -> List[Dict[str, Any]]:
        expected_unit = (expected_unit or "").strip()
        if not expected_unit:
            return []
        if expected_unit in normalized_generated:
            return []
        unit_suffix_hit = re.search(r"[\u4e00-\u9fa5]{2,20}(?:委|局|办事处|政府|街道|中心)", normalized_generated)
        if unit_suffix_hit:
            return [
                {
                    "error_type": "unit_mismatch",
                    "severity": "medium",
                    "message": "生成回复中出现其他疑似承办单位，且未体现当前预测单位。",
                    "generated_value": unit_suffix_hit.group(0),
                    "reference_value": expected_unit,
                }
            ]
        return []

    def _detect_retrieval_grounding_issues(
        self,
        normalized_generated: str,
        retrieval_hits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not retrieval_hits:
            return [
                {
                    "error_type": "weak_evidence",
                    "severity": "medium",
                    "message": "本次反馈没有携带 RAG 参考文档，生成回复缺少可追溯依据。",
                    "generated_value": normalized_generated[:200],
                    "reference_value": "",
                }
            ]

        scores = []
        for hit in retrieval_hits:
            for key in ("score", "similarity", "final_score"):
                if key in hit:
                    try:
                        scores.append(float(hit[key]))
                    except (TypeError, ValueError):
                        pass
                    break
        if scores and max(scores) < 0.2:
            return [
                {
                    "error_type": "weak_evidence",
                    "severity": "medium",
                    "message": "RAG 检索最高相关度较低，生成回复可能缺少充分依据。",
                    "generated_value": str(round(max(scores), 4)),
                    "reference_value": "",
                }
            ]
        return []

    def _detect_user_reported_issue(self, feedback_type: str, comments: str) -> List[Dict[str, Any]]:
        text = f"{feedback_type} {comments}".strip()
        if not text:
            return []
        mapping = [
            ("单位", "unit_mismatch", "用户反馈指出责任单位可能错误。"),
            ("地", "district_or_street_mismatch", "用户反馈指出地点或辖区可能错误。"),
            ("事实", "fact_conflict", "用户反馈指出回复事实可能错误。"),
            ("无用", "low_actionability", "用户反馈指出回复帮助不足。"),
        ]
        errors = []
        for marker, error_type, message in mapping:
            if marker in text:
                errors.append(
                    {
                        "error_type": error_type,
                        "severity": "medium",
                        "message": message,
                        "generated_value": comments[:200],
                        "reference_value": "",
                    }
                )
        return errors

    def _build_similarity(self, normalized_generated: str, normalized_reference: str) -> Dict[str, float]:
        if not normalized_reference:
            return {
                "char_overlap": 0.0,
                "sequence_ratio": 0.0,
                "generated_length": float(len(normalized_generated)),
                "reference_length": 0.0,
            }
        generated_chars = set(normalized_generated)
        reference_chars = set(normalized_reference)
        overlap = len(generated_chars & reference_chars) / max(len(reference_chars), 1)
        ratio = SequenceMatcher(None, normalized_generated, normalized_reference).ratio()
        return {
            "char_overlap": round(overlap, 4),
            "sequence_ratio": round(ratio, 4),
            "generated_length": float(len(normalized_generated)),
            "reference_length": float(len(normalized_reference)),
        }

    def _dedupe_errors(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for item in errors:
            key = (
                item.get("error_type", ""),
                item.get("message", ""),
                item.get("generated_value", ""),
                item.get("reference_value", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _overall_severity(self, errors: List[Dict[str, Any]]) -> str:
        severities = {item.get("severity", "low") for item in errors}
        if "high" in severities:
            return "high"
        if "medium" in severities:
            return "medium"
        return "low"

    def _build_quality_dimensions(
        self,
        *,
        errors: List[Dict[str, Any]],
        retrieval_hits: List[Dict[str, Any]],
        normalized_generated: str,
        normalized_reference: str,
    ) -> Dict[str, str]:
        error_types = {item.get("error_type", "") for item in errors}
        return {
            "format_compliance": "fail" if "format_violation" in error_types else "pass",
            "grounding": "weak" if "weak_evidence" in error_types else ("strong" if retrieval_hits else "none"),
            "fact_consistency": "fail" if {"status_conflict", "fact_conflict", "hallucinated_fact"} & error_types else "pass",
            "actionability": "warn" if {"low_actionability", "action_missing"} & error_types else "pass",
            "location_consistency": "fail" if "district_or_street_mismatch" in error_types else "pass",
            "unit_consistency": "fail" if "unit_mismatch" in error_types else "pass",
            "reference_similarity": "available" if normalized_reference else "not_available",
            "reply_length": "empty" if not normalized_generated else ("long" if len(normalized_generated) > 260 else "normal"),
        }

    def _build_routing_recommendations(self, error_types: List[str], severity: str) -> List[str]:
        recommendations = []
        if severity in {"medium", "high"}:
            recommendations.append("manual_review")
        if "weak_evidence" in error_types:
            recommendations.append("review_rag_corpus_or_query_rewrite")
        if "unit_mismatch" in error_types:
            recommendations.append("add_to_unit_classifier_active_learning")
        if {"status_conflict", "fact_conflict", "hallucinated_fact"} & set(error_types):
            recommendations.append("queue_fact_verification")
        if "low_actionability" in error_types:
            recommendations.append("adjust_generation_prompt_or_finetune_data")
        return recommendations

    def _build_summary(
        self,
        errors: List[Dict[str, Any]],
        normalized_generated: str,
        normalized_reference: str,
    ) -> str:
        if not errors:
            return "未检测到明显结构化问题。"
        similarity = self._build_similarity(normalized_generated, normalized_reference)
        if normalized_reference:
            return (
                f"共识别 {len(errors)} 类问题，序列相似度 {similarity['sequence_ratio']:.2f}，"
                "建议重点复核事实、责任单位、地点和办理动作。"
            )
        return (
            f"共识别 {len(errors)} 类问题；本次没有标准回复，"
            "归因主要依据格式约束、RAG 依据、地点/单位一致性和回复可操作性。"
        )
