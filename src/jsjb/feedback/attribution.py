"""Feedback attribution for the continuous-learning loop."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.jsjb.unit_classifier.catalog import canonicalize_unit, load_unit_catalog


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


UNIT_ERROR_MARKERS = ["单位", "部门", "应由", "应该由", "归", "负责", "不是"]
LOCATION_ERROR_MARKERS = ["区", "街道", "地点", "辖区", "位置", "不是"]
RAG_ERROR_MARKERS = ["材料", "参考", "检索", "案例", "政策", "不相关", "无用"]
FACT_ERROR_MARKERS = ["事实", "编造", "不准确", "不对", "矛盾", "不是事实"]
FORMAT_ERROR_MARKERS = ["格式", "套话", "问候", "落款", "太长"]
GENERIC_ERROR_MARKERS = ["空泛", "笼统", "没有措施", "没说怎么处理", "无帮助"]


@dataclass
class FeedbackAttributionResult:
    """Structured attribution from one user feedback record."""

    primary_error: str = "unknown"
    error_tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    corrected_unit: str = ""
    corrected_district: str = ""
    recommended_actions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackAttributor:
    """Infer where a bad result should be routed for improvement."""

    def __init__(self) -> None:
        self.unit_catalog = load_unit_catalog()
        unit_meta = self.unit_catalog.get("unit_meta", {}) or {}
        aliases = self.unit_catalog.get("alias_to_unit", {}) or {}
        candidates = set(unit_meta.keys()) | set(aliases.keys()) | set(aliases.values())
        self.known_units = sorted(
            {canonicalize_unit(item, self.unit_catalog) for item in candidates if item},
            key=len,
            reverse=True,
        )

    def analyze(
        self,
        *,
        feedback_record: dict[str, Any],
        request_data: dict[str, Any],
        quality_attribution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        quality_attribution = quality_attribution or {}
        comments = str(feedback_record.get("comments") or request_data.get("comments") or "").strip()
        feedback_type = str(feedback_record.get("feedback_type") or request_data.get("feedback_type") or "").strip()
        text = f"{feedback_type} {comments}".strip()
        units = request_data.get("units") or []
        retrieval_feedback = request_data.get("retrieval_feedback") or request_data.get("doc_feedback") or []
        retrieval_hits = request_data.get("retrieval") or request_data.get("retrieval_hits") or request_data.get("rag_docs") or []

        tags: set[str] = set(quality_attribution.get("error_types") or [])
        evidence: list[str] = []

        corrected_unit = self._extract_corrected_unit(text)
        corrected_district = self._extract_corrected_district(text)

        predicted_unit = str(feedback_record.get("unit") or "").strip()
        top_units = [
            str(item.get("unit", "")).strip()
            for item in units
            if isinstance(item, dict) and item.get("unit")
        ]

        if corrected_unit:
            tags.add("unit_wrong")
            evidence.append(f"用户反馈中识别到纠正单位: {corrected_unit}")
            if top_units and corrected_unit not in top_units[:3]:
                tags.add("unit_top3_missing")
                evidence.append("纠正单位不在当前 Top3 候选中")
            elif top_units:
                evidence.append("纠正单位在当前候选列表中，可用于排序/校准")
            if predicted_unit and corrected_unit != canonicalize_unit(predicted_unit):
                evidence.append(f"当前预测单位为: {predicted_unit}")
        elif self._contains_any(text, UNIT_ERROR_MARKERS) or "unit_mismatch" in tags:
            tags.add("unit_wrong")
            evidence.append("反馈或质量归因指向责任单位错误")

        if corrected_district:
            tags.add("district_wrong")
            evidence.append(f"用户反馈中识别到纠正区县: {corrected_district}")
            current_district = str(feedback_record.get("district") or "").strip()
            if current_district and corrected_district != current_district:
                evidence.append(f"当前识别区县为: {current_district}")
        elif self._contains_any(text, LOCATION_ERROR_MARKERS) or "district_or_street_mismatch" in tags:
            tags.add("location_missing")

        if self._has_negative_doc_feedback(retrieval_feedback) or self._contains_any(text, RAG_ERROR_MARKERS):
            tags.add("rag_irrelevant")
            evidence.append("反馈指向参考材料/检索结果不相关或无用")
            if not retrieval_hits:
                tags.add("rag_missing")

        if self._contains_any(text, FACT_ERROR_MARKERS) or {"status_conflict", "fact_conflict", "hallucinated_fact"} & tags:
            tags.add("fact_conflict")
            evidence.append("反馈或质量归因指向事实冲突/幻觉")

        if self._contains_any(text, FORMAT_ERROR_MARKERS) or "format_violation" in tags:
            tags.add("format_violation")

        if self._contains_any(text, GENERIC_ERROR_MARKERS) or {"low_actionability", "action_missing"} & tags:
            tags.add("too_generic")

        if not tags and feedback_record.get("is_helpful") in (False, 0, "false", "False"):
            tags.add("negative_feedback")

        actions = self._recommend_actions(tags, corrected_unit=corrected_unit)
        primary = self._pick_primary(tags)
        confidence = self._score_confidence(
            tags=tags,
            corrected_unit=corrected_unit,
            corrected_district=corrected_district,
            evidence=evidence,
        )
        result = FeedbackAttributionResult(
            primary_error=primary,
            error_tags=sorted(tags),
            confidence=confidence,
            corrected_unit=corrected_unit,
            corrected_district=corrected_district,
            recommended_actions=actions,
            evidence=evidence,
        )
        return result.to_dict()

    def _extract_corrected_unit(self, text: str) -> str:
        if not text:
            return ""

        patterns = [
            r"(?:正确(?:单位|部门)?(?:是|为)?|应为|应该是|应该由|应由|归|属于|负责单位(?:是|为)?)[：:\s]*([\u4e00-\u9fa5]{2,30}(?:委|局|办|办事处|政府|中心|公司|集团|镇|街道))",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = canonicalize_unit(match.group(1), self.unit_catalog)
                if candidate:
                    return candidate

        for unit in self.known_units:
            if len(unit) >= 3 and unit in text:
                return unit
        return ""

    def _extract_corrected_district(self, text: str) -> str:
        for district in BEIJING_DISTRICTS:
            if district in text:
                return district
        return ""

    @staticmethod
    def _contains_any(text: str, markers: list[str]) -> bool:
        return any(marker in text for marker in markers)

    @staticmethod
    def _has_negative_doc_feedback(retrieval_feedback: Any) -> bool:
        if isinstance(retrieval_feedback, dict):
            retrieval_feedback = [retrieval_feedback]
        if not isinstance(retrieval_feedback, list):
            return False
        for item in retrieval_feedback:
            if isinstance(item, dict) and item.get("is_helpful") in (False, 0, "false", "False"):
                return True
        return False

    @staticmethod
    def _pick_primary(tags: set[str]) -> str:
        priority = [
            "unit_top3_missing",
            "unit_wrong",
            "district_wrong",
            "rag_irrelevant",
            "fact_conflict",
            "format_violation",
            "too_generic",
            "negative_feedback",
        ]
        for tag in priority:
            if tag in tags:
                return tag
        return sorted(tags)[0] if tags else "unknown"

    @staticmethod
    def _recommend_actions(tags: set[str], corrected_unit: str = "") -> list[str]:
        actions: list[str] = []
        if {"unit_wrong", "unit_top3_missing"} & tags:
            actions.append("add_classifier_training_candidate")
        if "rag_irrelevant" in tags or "rag_missing" in tags:
            actions.append("adjust_rag_feedback_weight")
        if {"fact_conflict", "format_violation", "too_generic"} & tags:
            actions.append("save_reply_error_case")
        if corrected_unit or "fact_conflict" in tags:
            actions.append("queue_knowledge_review_candidate")
        if not actions:
            actions.append("manual_review")
        return actions

    @staticmethod
    def _score_confidence(
        *,
        tags: set[str],
        corrected_unit: str,
        corrected_district: str,
        evidence: list[str],
    ) -> float:
        score = 0.35
        if corrected_unit:
            score += 0.30
        if corrected_district:
            score += 0.15
        if tags:
            score += min(len(tags) * 0.06, 0.24)
        if evidence:
            score += 0.08
        return round(min(score, 0.98), 4)
