from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from src.jsjb.unit_classifier.catalog import normalize_unit_text


@dataclass(frozen=True)
class UnitHierarchyLabel:
    fine: str
    coarse: str
    level: str = "fine"


class UnitHierarchy:
    """Rule-based hierarchy for official reply units.

    The current training labels are fine-grained units. This hierarchy adds a
    stable coarse layer for evaluation and future two-stage classifiers without
    changing the existing label_map.json.
    """

    def coarse_label(self, unit: str) -> str:
        unit = normalize_unit_text(unit)
        if not unit:
            return "unknown"
        if "街道" in unit or "办事处" in unit:
            return "street_town"
        if "镇" in unit or "乡" in unit:
            return "street_town"
        if "政府" in unit:
            return "district_government"
        if "委" in unit:
            return "commission"
        if "局" in unit:
            return "bureau"
        if "中心" in unit:
            return "public_service_center"
        if "公司" in unit or "集团" in unit:
            return "enterprise"
        if "学校" in unit or "医院" in unit:
            return "public_institution"
        return "other_unit"

    def build_maps(self, id2label: dict[str, str]) -> dict[str, Any]:
        fine_to_coarse = {}
        coarse_to_fine: dict[str, list[str]] = defaultdict(list)
        for _, label in sorted(id2label.items(), key=lambda item: int(item[0])):
            fine = normalize_unit_text(label)
            coarse = self.coarse_label(fine)
            fine_to_coarse[fine] = coarse
            coarse_to_fine[coarse].append(fine)
        return {
            "fine_to_coarse": fine_to_coarse,
            "coarse_to_fine": dict(coarse_to_fine),
            "coarse_distribution": dict(Counter(fine_to_coarse.values())),
        }

    def metrics(self, y_true: list[int], y_pred: list[int], id2label: dict[str, str]) -> dict[str, float]:
        if not y_true:
            return {"fine_acc": 0.0, "coarse_acc": 0.0, "coarse_lift": 0.0}
        fine_correct = 0
        coarse_correct = 0
        for true_id, pred_id in zip(y_true, y_pred):
            true_label = normalize_unit_text(id2label[str(int(true_id))])
            pred_label = normalize_unit_text(id2label[str(int(pred_id))])
            if true_label == pred_label:
                fine_correct += 1
            if self.coarse_label(true_label) == self.coarse_label(pred_label):
                coarse_correct += 1
        fine_acc = fine_correct / len(y_true)
        coarse_acc = coarse_correct / len(y_true)
        return {
            "fine_acc": round(fine_acc, 6),
            "coarse_acc": round(coarse_acc, 6),
            "coarse_lift": round(coarse_acc - fine_acc, 6),
        }
