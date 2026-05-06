"""Automatic feedback routing for RAG, active learning, and reply QA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


NEGATIVE_FEEDBACK_TYPES = {
    "bad",
    "negative",
    "not_helpful",
    "wrong_unit",
    "wrong_reply",
    "rag_not_useful",
    "事实错误",
    "单位错误",
    "回复无用",
    "检索无用",
}


@dataclass
class FeedbackAutomationResult:
    """Summary of automatic actions triggered by one feedback record."""

    doc_feedback_recorded: int = 0
    active_learning_collected: bool = False
    reply_error_analysis_id: Optional[int] = None
    queued_fact_count: int = 0
    routing_reasons: List[str] = field(default_factory=list)
    quality_attribution: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeedbackAutomationRouter:
    """Route feedback into downstream learning and quality-control queues."""

    def __init__(
        self,
        *,
        feedback_db,
        reply_error_extractor,
        sample_collector=None,
    ) -> None:
        self.feedback_db = feedback_db
        self.reply_error_extractor = reply_error_extractor
        self.sample_collector = sample_collector

    def run(
        self,
        *,
        feedback_id: int,
        feedback_record: Dict[str, Any],
        request_data: Dict[str, Any],
    ) -> FeedbackAutomationResult:
        result = FeedbackAutomationResult()
        is_helpful = bool(feedback_record.get("is_helpful"))
        is_negative = self._is_negative_feedback(feedback_record)
        retrieval_hits = self._extract_retrieval_hits(request_data)
        reference_reply = (request_data.get("reference_reply") or "").strip()
        location_result = self._extract_location(feedback_record, request_data)
        query = self._build_query(feedback_record, request_data)

        result.doc_feedback_recorded = self._record_rag_doc_feedback(
            retrieval_hits=retrieval_hits,
            request_data=request_data,
            query=query,
            default_helpful=is_helpful,
        )
        if result.doc_feedback_recorded:
            result.routing_reasons.append("rag_doc_feedback_recorded")

        should_analyze = self._should_analyze_reply(
            is_negative=is_negative,
            reference_reply=reference_reply,
            retrieval_hits=retrieval_hits,
            request_data=request_data,
        )
        if should_analyze:
            quality_attribution = self.reply_error_extractor.analyze(
                generated_reply=feedback_record.get("reply", ""),
                reference_reply=reference_reply,
                retrieval_hits=retrieval_hits,
                location_result=location_result,
                expected_unit=feedback_record.get("unit", ""),
                feedback_type=feedback_record.get("feedback_type", ""),
                comments=feedback_record.get("comments", ""),
            )
            result.quality_attribution = quality_attribution

            should_store = (
                is_negative
                or bool(reference_reply)
                or bool(quality_attribution.get("needs_review"))
            )
            if should_store:
                result.reply_error_analysis_id = self.feedback_db.save_reply_error_analysis(
                    feedback_id=feedback_id,
                    analysis=quality_attribution,
                    reference_reply=reference_reply,
                )
                result.routing_reasons.append("reply_quality_attribution_saved")

            if reference_reply and result.reply_error_analysis_id is not None:
                result.queued_fact_count = self.feedback_db.queue_knowledge_graph_facts(
                    feedback_id=feedback_id,
                    analysis_id=result.reply_error_analysis_id,
                    analysis=quality_attribution,
                    source_reply_type="reference",
                )
                if result.queued_fact_count:
                    result.routing_reasons.append("reference_facts_queued_for_review")

        result.active_learning_collected = self._collect_active_learning_sample(
            feedback_id=feedback_id,
            feedback_record=feedback_record,
            request_data=request_data,
            is_negative=is_negative,
            quality_attribution=result.quality_attribution,
            doc_feedback_recorded=result.doc_feedback_recorded,
        )
        if result.active_learning_collected:
            result.routing_reasons.append("active_learning_sample_collected")

        return result

    def _is_negative_feedback(self, feedback_record: Dict[str, Any]) -> bool:
        if not bool(feedback_record.get("is_helpful")):
            return True
        feedback_type = str(feedback_record.get("feedback_type", "")).strip().lower()
        comments = str(feedback_record.get("comments", "")).strip()
        if feedback_type in NEGATIVE_FEEDBACK_TYPES:
            return True
        return any(marker in comments for marker in ["错误", "不对", "无用", "不准确", "不相关"])

    def _extract_retrieval_hits(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        hits = (
            request_data.get("retrieval")
            or request_data.get("retrieval_hits")
            or request_data.get("rag_docs")
            or []
        )
        return [item for item in hits if isinstance(item, dict)]

    def _extract_location(
        self,
        feedback_record: Dict[str, Any],
        request_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        location = request_data.get("location") or {}
        if not isinstance(location, dict):
            location = {}
        return {
            **location,
            "district": location.get("district") or feedback_record.get("district", ""),
            "street": location.get("street") or request_data.get("street", ""),
            "subdistrict": location.get("subdistrict") or request_data.get("subdistrict", ""),
        }

    def _build_query(self, feedback_record: Dict[str, Any], request_data: Dict[str, Any]) -> str:
        return (
            request_data.get("query")
            or request_data.get("message")
            or f"{feedback_record.get('title', '')} {feedback_record.get('body', '')}".strip()
        )

    def _record_rag_doc_feedback(
        self,
        *,
        retrieval_hits: List[Dict[str, Any]],
        request_data: Dict[str, Any],
        query: str,
        default_helpful: bool,
    ) -> int:
        per_doc_feedback = request_data.get("retrieval_feedback") or request_data.get("doc_feedback") or []
        if isinstance(per_doc_feedback, dict):
            per_doc_feedback = [per_doc_feedback]

        recorded = 0
        if per_doc_feedback:
            for item in per_doc_feedback:
                if not isinstance(item, dict):
                    continue
                doc_id = self._doc_id_from_hit(item)
                if not doc_id:
                    continue
                self.feedback_db.record_doc_feedback(
                    doc_id=doc_id,
                    is_helpful=bool(item.get("is_helpful", default_helpful)),
                    query=item.get("query") or query,
                )
                recorded += 1
            return recorded

        for hit in retrieval_hits:
            doc_id = self._doc_id_from_hit(hit)
            if not doc_id:
                continue
            self.feedback_db.record_doc_feedback(
                doc_id=doc_id,
                is_helpful=default_helpful,
                query=query,
            )
            recorded += 1
        return recorded

    def _doc_id_from_hit(self, hit: Dict[str, Any]) -> str:
        for key in ("doc_id", "id", "chunk_id", "source_id"):
            value = str(hit.get(key, "")).strip()
            if value:
                return value
        title = str(hit.get("title", "")).strip()
        source = str(hit.get("source", "")).strip()
        if title or source:
            return f"{source}|{title}".strip("|")
        return ""

    def _should_analyze_reply(
        self,
        *,
        is_negative: bool,
        reference_reply: str,
        retrieval_hits: List[Dict[str, Any]],
        request_data: Dict[str, Any],
    ) -> bool:
        verification = request_data.get("verification") or {}
        return (
            is_negative
            or bool(reference_reply)
            or bool(retrieval_hits)
            or bool(verification.get("needs_review"))
        )

    def _collect_active_learning_sample(
        self,
        *,
        feedback_id: int,
        feedback_record: Dict[str, Any],
        request_data: Dict[str, Any],
        is_negative: bool,
        quality_attribution: Optional[Dict[str, Any]],
        doc_feedback_recorded: int,
    ) -> bool:
        if self.sample_collector is None:
            return False

        needs_review = bool((quality_attribution or {}).get("needs_review"))
        if not (is_negative or needs_review):
            return False

        units = request_data.get("units", []) or []
        prediction_probs = {
            unit.get("unit", ""): unit.get("confidence", 0)
            for unit in units[:5]
            if isinstance(unit, dict) and unit.get("unit")
        }
        top_confidence = units[0].get("confidence", 0.0) if units and isinstance(units[0], dict) else 0.0
        error_types = (quality_attribution or {}).get("error_types", [])
        trigger_reason = (
            feedback_record.get("feedback_type")
            or feedback_record.get("comments", "")[:120]
            or ", ".join(error_types)
            or "feedback automation"
        )
        sample_source = "generated_reply_quality_issue" if needs_review else "user_negative_feedback"

        return bool(
            self.sample_collector.add_sample(
                sample_id=f"feedback_{feedback_id}",
                tag=feedback_record.get("tag", ""),
                title=feedback_record.get("title", ""),
                body=feedback_record.get("body", ""),
                district=feedback_record.get("district", ""),
                predicted_unit=feedback_record.get("unit", ""),
                confidence=top_confidence,
                prediction_probs=prediction_probs,
                user_feedback=feedback_record.get("comments") or feedback_record.get("feedback_type"),
                sample_source=sample_source,
                trigger_reason=trigger_reason,
                risk_flags={
                    "feedback_type": feedback_record.get("feedback_type", ""),
                    "reply_error_types": error_types,
                    "reply_error_severity": (quality_attribution or {}).get("severity", ""),
                    "doc_feedback_recorded": doc_feedback_recorded,
                    "quality_dimensions": (quality_attribution or {}).get("quality_dimensions", {}),
                },
            )
        )
