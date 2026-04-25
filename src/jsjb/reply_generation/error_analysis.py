from __future__ import annotations

import re
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from src.jsjb.reply_generation.fact_extraction import FactExtractor
from src.jsjb.reply_generation.verification import verify_generated_reply


DISTRICT_PATTERN = re.compile(r"(东城区|西城区|朝阳区|海淀区|丰台区|石景山区|门头沟区|房山区|通州区|顺义区|昌平区|大兴区|怀柔区|平谷区|密云区|延庆区)")
STREET_PATTERN = re.compile(r"([\u4e00-\u9fa5]{2,12}(?:街道|镇|乡))")


class ReplyErrorExtractor:
    """Compare a generated reply against a reference reply and emit structured errors."""

    GREETING_PATTERNS = [
        r"^尊敬的.*?[，,：:\n]",
        r"^您好[！!，,：:\n]?",
        r"^你好[！!，,：:\n]?",
        r"^市民朋友[，,：:\n]?",
    ]

    CLOSING_PATTERNS = [
        r"感谢您.*$",
        r"欢迎您.*$",
        r"祝您.*$",
        r"特此回复.*$",
        r"如有疑问.*$",
        r"[\u4e00-\u9fa5]{2,20}(?:委员会|办公室|管理局|街道办|镇政府|政府|局)\s*\n?\s*\d{4}年\d{1,2}月\d{1,2}日\s*$",
        r"\d{4}年\d{1,2}月\d{1,2}日\s*$",
    ]

    ACTION_KEYWORDS = [
        "已安排",
        "已责成",
        "已督促",
        "将继续",
        "立即整改",
        "现场核查",
        "维修",
        "清理",
        "整治",
        "协调",
        "反馈",
        "办理",
        "解决",
    ]

    FORMAT_VIOLATION_PATTERNS = [
        (r"您好|你好|尊敬的", "包含问候语"),
        (r"感谢您|欢迎您|祝您|特此回复", "包含结束套话"),
        (r"联系电话[:：]?\s*\d", "包含联系方式，可能偏离简洁回复格式"),
    ]

    def __init__(self):
        self.fact_extractor = FactExtractor()

    def analyze(
        self,
        generated_reply: str,
        reference_reply: str,
        retrieval_hits: Optional[List[Dict[str, Any]]] = None,
        location_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        generated_reply = generated_reply or ""
        reference_reply = reference_reply or ""
        retrieval_hits = retrieval_hits or []
        location_result = location_result or {}

        normalized_generated = self.normalize_reply(generated_reply)
        normalized_reference = self.normalize_reply(reference_reply)

        generated_facts = [asdict(item) for item in self.fact_extractor.extract_facts_from_reply(generated_reply)]
        reference_facts = [asdict(item) for item in self.fact_extractor.extract_facts_from_reply(reference_reply)]

        errors: List[Dict[str, Any]] = []
        errors.extend(self._detect_format_violations(generated_reply))
        errors.extend(self._compare_facts(generated_facts, reference_facts))
        errors.extend(
            self._compare_locations(
                normalized_generated,
                normalized_reference,
                location_result,
            )
        )
        errors.extend(self._compare_actions(normalized_generated, normalized_reference))

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
            "similarity": self._build_similarity(normalized_generated, normalized_reference),
            "verification": verification,
            "needs_review": severity in {"medium", "high"} or bool(deduped_errors),
        }

    def normalize_reply(self, text: str) -> str:
        text = (text or "").strip()
        for pattern in self.GREETING_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        for pattern in self.CLOSING_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)
        text = re.sub(r"^\s*[：:,，。；;、\n]+", "", text)
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
            if key not in reference_index and reference_index:
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
            if "project" in content:
                key_parts.append(str(content.get("project", "")))
            if "name" in content:
                key_parts.append(str(content.get("name", "")))
            if "unit" in content:
                key_parts.append(str(content.get("unit", "")))
            key = "|".join(key_parts)
            indexed[key] = fact
        return indexed

    def _compare_fact_values(
        self,
        generated_fact: Dict[str, Any],
        reference_fact: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        gen_content = generated_fact.get("content", {})
        ref_content = reference_fact.get("content", {})
        fact_type = reference_fact.get("fact_type", "")

        if fact_type == "project_status":
            if gen_content.get("status") != ref_content.get("status"):
                return {
                    "error_type": "status_conflict",
                    "severity": "high",
                    "message": "项目状态与标准回复不一致。",
                    "generated_value": gen_content.get("status", ""),
                    "reference_value": ref_content.get("status", ""),
                }
        elif fact_type == "demolition_status":
            if gen_content.get("demolition_status") != ref_content.get("demolition_status"):
                return {
                    "error_type": "status_conflict",
                    "severity": "high",
                    "message": "征拆状态与标准回复不一致。",
                    "generated_value": gen_content.get("demolition_status", ""),
                    "reference_value": ref_content.get("demolition_status", ""),
                }
        elif fact_type == "responsible_unit":
            if gen_content.get("unit") != ref_content.get("unit"):
                return {
                    "error_type": "unit_mismatch",
                    "severity": "high",
                    "message": "责任单位与标准回复不一致。",
                    "generated_value": gen_content.get("unit", ""),
                    "reference_value": ref_content.get("unit", ""),
                }
        elif fact_type == "public_resource":
            if gen_content.get("status") != ref_content.get("status"):
                return {
                    "error_type": "status_conflict",
                    "severity": "medium",
                    "message": "公共资源状态与标准回复不一致。",
                    "generated_value": gen_content.get("status", ""),
                    "reference_value": ref_content.get("status", ""),
                }
        return None

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
                    "message": "生成回复出现与标准答案/识别地点不一致的行政区。",
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

    def _build_similarity(self, normalized_generated: str, normalized_reference: str) -> Dict[str, float]:
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

    def _build_summary(
        self,
        errors: List[Dict[str, Any]],
        normalized_generated: str,
        normalized_reference: str,
    ) -> str:
        if not errors:
            return "未检测到明显结构化错误。"
        similarity = self._build_similarity(normalized_generated, normalized_reference)
        return (
            f"共识别 {len(errors)} 类问题，"
            f"序列相似度 {similarity['sequence_ratio']:.2f}，"
            f"重点建议人工复核高风险事实、责任单位和办理动作。"
        )
