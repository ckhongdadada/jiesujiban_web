from __future__ import annotations

from src.jsjb.active_learning.sample_collector import SampleCollector
from src.jsjb.unit_classifier.active_learning import BadgeCandidate, ClassifierBadgeSampler
from src.jsjb.unit_classifier.hierarchy import UnitHierarchy
from src.jsjb.unit_classifier.training.class_balance import build_class_weights, effective_num_class_weights


def test_effective_num_weights_upweight_tail_classes():
    labels = [0] * 100 + [1] * 10 + [2]

    weights = effective_num_class_weights(labels, num_classes=3, beta=0.99, min_weight=0.1, max_weight=20.0)

    assert weights[2] > weights[1] > weights[0]
    assert weights.shape[0] == 3


def test_build_class_weights_can_disable_weighting():
    weights = build_class_weights([0, 0, 1], num_classes=2, strategy="none")

    assert weights.tolist() == [1.0, 1.0]


def test_unit_hierarchy_reports_coarse_lift_for_related_units():
    id2label = {"0": "朝阳区政府", "1": "海淀区政府", "2": "市城管委"}
    metrics = UnitHierarchy().metrics([0, 2], [1, 1], id2label)

    assert metrics["fine_acc"] == 0.0
    assert metrics["coarse_acc"] == 0.5
    assert metrics["coarse_lift"] == 0.5


def test_badge_sampler_selects_uncertain_and_diverse_batch():
    sampler = ClassifierBadgeSampler()
    candidates = [
        BadgeCandidate(sample_id="certain", probabilities={"A": 0.95, "B": 0.05}, feature_vector=[1.0, 0.0]),
        BadgeCandidate(sample_id="uncertain-a", probabilities={"A": 0.52, "B": 0.48}, feature_vector=[1.0, 0.0]),
        BadgeCandidate(sample_id="uncertain-b", probabilities={"A": 0.51, "B": 0.49}, feature_vector=[0.0, 1.0]),
    ]

    selected = sampler.select_batch(candidates, batch_size=2)
    selected_ids = {item["sample_id"] for item in selected}

    assert "certain" not in selected_ids
    assert selected_ids == {"uncertain-a", "uncertain-b"}
    assert all(item["selection_strategy"] == "badge" for item in selected)


def test_badge_sampler_can_enqueue_to_active_learning_collector(tmp_path):
    collector = SampleCollector(str(tmp_path / "active_learning.db"))
    sampler = ClassifierBadgeSampler()
    candidates = [
        BadgeCandidate(
            sample_id="badge-1",
            tag="环境卫生",
            title="垃圾清运",
            body="小区垃圾无人清运",
            district="朝阳区",
            predicted_unit="城管委",
            probabilities={"城管委": 0.51, "环卫中心": 0.49},
            feature_vector=[1.0, 0.0],
        )
    ]

    result = sampler.enqueue_batch(collector, candidates, batch_size=1)
    pending = collector.get_pending_samples(limit=5)

    assert result["inserted_count"] == 1
    assert pending[0]["sample_source"] == "classifier_badge"
    assert pending[0]["trigger_reason"] == "badge_uncertainty_diversity"
    assert pending[0]["risk_flags"]["selection_rank"] == 1
