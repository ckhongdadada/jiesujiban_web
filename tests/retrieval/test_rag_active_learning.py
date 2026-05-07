from __future__ import annotations

from src.jsjb.retrieval.active_learning import RAGActiveLearningConfig, RAGActiveLearningSampler


def test_badge_lite_sampler_scores_uncertain_retrieval():
    sampler = RAGActiveLearningSampler(RAGActiveLearningConfig(score_threshold=0.3))
    candidate = sampler.build_candidate(
        query="小区垃圾清运不及时",
        title="垃圾清运",
        body="小区垃圾清运不及时",
        district="朝阳区",
        tag="环境卫生",
        unit="城管委",
        retrieval_hits=[
            {"doc_id": "d1", "title": "案例A", "score": 0.61, "doc_type": "案例", "district": "朝阳区"},
            {"doc_id": "d2", "title": "政策B", "score": 0.59, "doc_type": "政策", "district": "海淀区"},
        ],
        evidence_strength="medium",
    )

    assert candidate["uncertainty_score"] > 0.9
    assert "high_retrieval_uncertainty" in candidate["selection_reason"]
    assert sampler.should_enqueue(candidate) is True


def test_badge_lite_sampler_queues_no_hit_cases():
    sampler = RAGActiveLearningSampler()
    candidate = sampler.build_candidate(query="完全没有命中的问题", retrieval_hits=[], evidence_strength="none")

    assert candidate["risk_score"] > 0
    assert candidate["retrieved_docs"] == []
    assert sampler.should_enqueue(candidate) is True
