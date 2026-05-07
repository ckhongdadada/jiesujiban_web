from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _safe_probs(probs: Any) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        return np.asarray([1.0], dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return np.ones_like(arr) / arr.size
    return arr / total


def entropy(probs: Any) -> float:
    p = _safe_probs(probs)
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum() / math.log(max(len(p), 2)))


def margin_uncertainty(probs: Any) -> float:
    p = np.sort(_safe_probs(probs))[::-1]
    if len(p) == 1:
        return 0.0
    return float(max(0.0, 1.0 - (p[0] - p[1])))


@dataclass
class BadgeCandidate:
    sample_id: str
    tag: str = ""
    title: str = ""
    body: str = ""
    district: str = ""
    predicted_unit: str = ""
    probabilities: dict[str, float] | None = None
    feature_vector: list[float] | None = None
    user_feedback: str = ""


class ClassifierBadgeSampler:
    """BADGE-style sampler for unit-classifier active learning.

    It builds a gradient embedding from predicted probabilities and optional
    feature vectors, then greedily selects uncertain and diverse samples.
    """

    def gradient_embedding(self, candidate: BadgeCandidate) -> np.ndarray:
        probs_dict = candidate.probabilities or {}
        probs = _safe_probs(list(probs_dict.values()) or [1.0])
        pred = int(np.argmax(probs))
        residual = probs.copy()
        residual[pred] -= 1.0
        feature = np.asarray(candidate.feature_vector or [1.0], dtype=np.float64)
        if feature.ndim != 1 or feature.size == 0:
            feature = np.asarray([1.0], dtype=np.float64)
        embedding = np.outer(residual, feature).reshape(-1)
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm
        return embedding

    def score_candidate(self, candidate: BadgeCandidate) -> dict[str, float]:
        probs = list((candidate.probabilities or {}).values())
        ent = entropy(probs)
        margin = margin_uncertainty(probs)
        feedback_boost = 0.15 if candidate.user_feedback.strip() else 0.0
        score = min(1.0, 0.55 * ent + 0.35 * margin + feedback_boost)
        return {
            "badge_uncertainty": round(score, 6),
            "entropy": round(ent, 6),
            "margin_uncertainty": round(margin, 6),
        }

    def select_batch(self, candidates: list[BadgeCandidate], batch_size: int = 20) -> list[dict[str, Any]]:
        if not candidates or batch_size <= 0:
            return []
        embeddings = [self.gradient_embedding(candidate) for candidate in candidates]
        scores = [self.score_candidate(candidate) for candidate in candidates]
        selected: list[int] = []
        remaining = set(range(len(candidates)))

        first = max(remaining, key=lambda idx: scores[idx]["badge_uncertainty"])
        selected.append(first)
        remaining.remove(first)

        while remaining and len(selected) < batch_size:
            def value(idx: int) -> float:
                diversity = min(float(np.linalg.norm(embeddings[idx] - embeddings[j])) for j in selected)
                return 0.65 * scores[idx]["badge_uncertainty"] + 0.35 * diversity

            nxt = max(remaining, key=value)
            selected.append(nxt)
            remaining.remove(nxt)

        result = []
        for rank, idx in enumerate(selected, start=1):
            candidate = candidates[idx]
            payload = {
                "rank": rank,
                "sample_id": candidate.sample_id,
                "tag": candidate.tag,
                "title": candidate.title,
                "body": candidate.body,
                "district": candidate.district,
                "predicted_unit": candidate.predicted_unit,
                "probabilities": candidate.probabilities or {},
                "selection_strategy": "badge",
            }
            payload.update(scores[idx])
            result.append(payload)
        return result

    def enqueue_batch(self, collector: Any, candidates: list[BadgeCandidate], batch_size: int = 20) -> dict[str, Any]:
        selected = self.select_batch(candidates, batch_size=batch_size)
        inserted = 0
        for item in selected:
            probabilities = item.get("probabilities") or {}
            confidence = max(probabilities.values()) if probabilities else 0.0
            ok = collector.add_sample(
                sample_id=item["sample_id"],
                tag=item.get("tag", ""),
                title=item.get("title", ""),
                body=item.get("body", ""),
                district=item.get("district", ""),
                predicted_unit=item.get("predicted_unit", ""),
                confidence=float(confidence),
                prediction_probs=probabilities,
                user_feedback="",
                sample_source="classifier_badge",
                trigger_reason="badge_uncertainty_diversity",
                risk_flags={
                    "badge_uncertainty": item.get("badge_uncertainty", 0.0),
                    "entropy": item.get("entropy", 0.0),
                    "margin_uncertainty": item.get("margin_uncertainty", 0.0),
                    "selection_rank": item.get("rank"),
                },
            )
            inserted += int(bool(ok))
        return {
            "status": "ok",
            "selected_count": len(selected),
            "inserted_count": inserted,
            "strategy": "badge",
            "selected": selected,
        }
