from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _doc_id(hit: dict[str, Any]) -> str:
    return str(hit.get("doc_id") or hit.get("id") or hit.get("title") or "")


@dataclass
class RAGActiveLearningConfig:
    enabled: bool = True
    score_threshold: float = 0.42
    min_uncertainty: float = 0.18
    max_queue_docs: int = 5


class RAGActiveLearningSampler:
    """BADGE-lite sampler for RAG review candidates.

    This is intentionally lightweight: it does not require gradients or model
    retraining. It uses uncertainty, diversity, risk, and feedback conflict to
    decide whether a retrieval case deserves manual review.
    """

    def __init__(self, config: RAGActiveLearningConfig | None = None):
        self.config = config or RAGActiveLearningConfig()

    def build_candidate(
        self,
        *,
        trace_id: str = "",
        query: str,
        title: str = "",
        body: str = "",
        district: str = "",
        tag: str = "",
        unit: str = "",
        retrieval_hits: list[dict[str, Any]] | None = None,
        evidence_strength: str = "",
        verification: dict[str, Any] | None = None,
        feedback_type: str = "",
        feedback_comment: str = "",
    ) -> dict[str, Any]:
        hits = list(retrieval_hits or [])
        scores = [_safe_float(hit.get("score")) for hit in hits]
        top1_score = scores[0] if scores else 0.0
        top2_score = scores[1] if len(scores) > 1 else 0.0
        score_gap = max(0.0, top1_score - top2_score) if len(scores) > 1 else top1_score

        uncertainty_score = self._uncertainty_score(hits, top1_score, score_gap)
        diversity_score = self._diversity_score(hits)
        risk_score, risk_reasons = self._risk_score(
            hits=hits,
            evidence_strength=evidence_strength,
            verification=verification,
            district=district,
        )
        feedback_score, feedback_reasons = self._feedback_signal(hits, feedback_type, feedback_comment)

        badge_lite_score = (
            0.40 * uncertainty_score
            + 0.22 * diversity_score
            + 0.28 * risk_score
            + 0.10 * feedback_score
        )
        reasons = self._selection_reasons(
            uncertainty_score=uncertainty_score,
            diversity_score=diversity_score,
            risk_score=risk_score,
            feedback_score=feedback_score,
            risk_reasons=risk_reasons,
            feedback_reasons=feedback_reasons,
            has_hits=bool(hits),
        )

        return {
            "trace_id": trace_id,
            "query": query,
            "title": title,
            "body": body,
            "district": district,
            "tag": tag,
            "unit": unit,
            "retrieved_docs": self._compact_hits(hits),
            "top1_doc_id": _doc_id(hits[0]) if hits else "",
            "top1_score": round(top1_score, 6),
            "top2_score": round(top2_score, 6),
            "score_gap": round(score_gap, 6),
            "uncertainty_score": round(uncertainty_score, 6),
            "diversity_score": round(diversity_score, 6),
            "risk_score": round(risk_score, 6),
            "feedback_score": round(feedback_score, 6),
            "badge_lite_score": round(badge_lite_score, 6),
            "feedback_type": feedback_type,
            "feedback_comment": feedback_comment,
            "selection_reason": reasons,
            "status": "pending_review",
        }

    def should_enqueue(self, candidate: dict[str, Any]) -> bool:
        if not self.config.enabled:
            return False
        score = _safe_float(candidate.get("badge_lite_score"))
        uncertainty = _safe_float(candidate.get("uncertainty_score"))
        if not candidate.get("retrieved_docs"):
            return True
        return score >= self.config.score_threshold or uncertainty >= self.config.min_uncertainty

    def _uncertainty_score(self, hits: list[dict[str, Any]], top1_score: float, score_gap: float) -> float:
        if not hits:
            return 1.0
        # Small Top1-Top2 margin means the retriever is unsure.
        margin_uncertainty = max(0.0, min(1.0, 1.0 - score_gap))
        low_confidence = max(0.0, min(1.0, 0.65 - top1_score)) / 0.65
        return max(margin_uncertainty, low_confidence)

    def _diversity_score(self, hits: list[dict[str, Any]]) -> float:
        if len(hits) <= 1:
            return 0.0
        doc_types = {str(hit.get("doc_type", "")) for hit in hits if hit.get("doc_type")}
        sources = {str(hit.get("source", "")) for hit in hits if hit.get("source")}
        districts = {str(hit.get("district", "")) for hit in hits if hit.get("district")}
        raw = 0.0
        raw += min(len(doc_types), 4) / 4 * 0.4
        raw += min(len(sources), 4) / 4 * 0.3
        raw += min(len(districts), 4) / 4 * 0.3
        return min(1.0, raw)

    def _risk_score(
        self,
        *,
        hits: list[dict[str, Any]],
        evidence_strength: str,
        verification: dict[str, Any] | None,
        district: str,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0
        if not hits:
            reasons.append("no_retrieval_hit")
            score += 0.45
        if evidence_strength in {"none", "weak"}:
            reasons.append(f"weak_evidence:{evidence_strength or 'unknown'}")
            score += 0.25
        if verification and verification.get("needs_review"):
            reasons.append("generation_fact_verification_warning")
            score += 0.25
        if district and hits:
            same_district = any(str(hit.get("district", "")) == district for hit in hits[:3])
            if not same_district:
                reasons.append("top3_no_same_district_doc")
                score += 0.15
        return min(1.0, score), reasons

    def _feedback_signal(
        self,
        hits: list[dict[str, Any]],
        feedback_type: str,
        feedback_comment: str,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0
        if feedback_type and feedback_type not in {"helpful", "useful", "positive"}:
            reasons.append(f"negative_or_specific_feedback:{feedback_type}")
            score += 0.55
        if feedback_comment.strip():
            reasons.append("has_user_comment")
            score += 0.25
        if hits and any(_safe_float(hit.get("feedback_boost")) < 0 for hit in hits[:3]):
            reasons.append("retrieved_doc_has_negative_feedback_history")
            score += 0.20
        return min(1.0, score), reasons

    def _selection_reasons(
        self,
        *,
        uncertainty_score: float,
        diversity_score: float,
        risk_score: float,
        feedback_score: float,
        risk_reasons: list[str],
        feedback_reasons: list[str],
        has_hits: bool,
    ) -> list[str]:
        reasons: list[str] = []
        if uncertainty_score >= self.config.min_uncertainty:
            reasons.append("high_retrieval_uncertainty")
        if diversity_score >= 0.45:
            reasons.append("diverse_candidate_docs")
        if risk_score >= 0.25:
            reasons.append("high_generation_or_grounding_risk")
        if feedback_score > 0:
            reasons.append("feedback_signal_available")
        if not has_hits:
            reasons.append("needs_corpus_or_query_rewrite_review")
        reasons.extend(risk_reasons)
        reasons.extend(feedback_reasons)
        return list(dict.fromkeys(reasons))

    def _compact_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact = []
        for hit in hits[: self.config.max_queue_docs]:
            compact.append(
                {
                    "doc_id": _doc_id(hit),
                    "title": hit.get("title", ""),
                    "doc_type": hit.get("doc_type", ""),
                    "district": hit.get("district", ""),
                    "source": hit.get("source", ""),
                    "score": _safe_float(hit.get("score")),
                    "dense_score": _safe_float(hit.get("dense_score")),
                    "sparse_score": _safe_float(hit.get("sparse_score")),
                    "feedback_boost": _safe_float(hit.get("feedback_boost")),
                    "matched_terms": hit.get("matched_terms", []),
                }
            )
        return compact


def encode_candidate_docs(candidate: dict[str, Any]) -> str:
    return json.dumps(candidate.get("retrieved_docs", []), ensure_ascii=False)
