"""Automation helpers for active-learning review and training packages."""

from __future__ import annotations

import json
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.jsjb.core.paths import get_runtime_dir
from src.jsjb.active_learning.sample_collector import SampleCollector


@dataclass
class TrainingPackageResult:
    """Files produced for one active-learning training package."""

    status: str
    sample_count: int
    data_path: str
    manifest_path: str
    message: str
    classifier_data_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActiveLearningAutomationService:
    """Build annotation plans and export reviewed samples into training packages."""

    def __init__(self, sample_collector: SampleCollector):
        self.collector = sample_collector

    def build_review_plan(self, limit: int = 30) -> Dict[str, Any]:
        """Return a diversified, reasoned list of samples for the next annotation batch."""
        priority_samples = self.collector.get_pending_samples(limit=limit, order_by="priority")
        uncertain_samples = self.collector.get_pending_samples(limit=limit, max_confidence=0.7, order_by="confidence")
        recent_samples = self.collector.get_pending_samples(limit=limit, order_by="created_at")
        selected = self._dedupe_samples([*priority_samples, *uncertain_samples, *recent_samples])[:limit]

        clusters = self._cluster_samples(selected)
        source_counts = Counter(sample.get("sample_source") or "unknown" for sample in selected)
        risk_counts = Counter()
        for sample in selected:
            for flag in self._extract_risk_labels(sample):
                risk_counts[flag] += 1

        return {
            "status": "ok",
            "generated_at": datetime.now().isoformat(),
            "limit": limit,
            "sample_count": len(selected),
            "summary": {
                "source_distribution": dict(source_counts),
                "risk_distribution": dict(risk_counts),
                "cluster_count": len(clusters),
            },
            "clusters": clusters,
            "samples": [self._format_sample(sample) for sample in selected],
            "next_action": self.suggest_next_action(),
        }

    def suggest_next_action(self, min_training_samples: int = 20) -> Dict[str, Any]:
        stats = self.collector.get_statistics()
        ready = int(stats.get("unused_samples", 0) or 0)
        pending = int(stats.get("pending_samples", 0) or 0)
        negative = int(stats.get("negative_feedback_samples", 0) or 0)

        if ready >= min_training_samples:
            return {
                "action": "prepare_training_package",
                "reason": f"已有 {ready} 条已标注未训练样本，可先导出增训包再训练。",
                "ready_samples": ready,
                "min_training_samples": min_training_samples,
            }
        if negative > 0:
            return {
                "action": "annotate_negative_feedback",
                "reason": f"有 {negative} 条负反馈样本，建议优先人工确认正确单位。",
                "pending_samples": pending,
            }
        if pending > 0:
            return {
                "action": "annotate_pending",
                "reason": f"有 {pending} 条待标注样本，建议按优先级/低置信度批量标注。",
                "pending_samples": pending,
            }
        return {
            "action": "wait",
            "reason": "当前主动学习样本较少，等待更多低置信度或负反馈样本。",
            "pending_samples": pending,
            "ready_samples": ready,
        }

    def prepare_training_package(
        self,
        output_dir: str | None = None,
        min_samples: int = 1,
        unused_only: bool = True,
    ) -> TrainingPackageResult:
        """Export annotated samples and a manifest for safe incremental training."""
        samples = self.collector.get_annotated_samples(unused_only=unused_only)
        if len(samples) < min_samples:
            return TrainingPackageResult(
                status="not_enough_samples",
                sample_count=len(samples),
                data_path="",
                manifest_path="",
                classifier_data_path="",
                message=f"可用已标注样本 {len(samples)} 条，少于阈值 {min_samples} 条。",
            )

        if output_dir is None:
            output_dir = str(get_runtime_dir() / "active_learning_exports")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = output_path / f"active_learning_training_{timestamp}.jsonl"
        classifier_data_path = output_path / f"active_learning_classifier_rows_{timestamp}.csv"
        manifest_path = output_path / f"active_learning_training_{timestamp}.manifest.json"

        with data_path.open("w", encoding="utf-8") as fp:
            for sample in samples:
                fp.write(json.dumps(self._training_record(sample), ensure_ascii=False) + "\n")
        with classifier_data_path.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["留言标签", "留言标题", "留言正文", "官方回复单位"])
            writer.writeheader()
            for sample in samples:
                writer.writerow(
                    {
                        "留言标签": sample.get("tag", ""),
                        "留言标题": sample.get("title", ""),
                        "留言正文": sample.get("body", ""),
                        "官方回复单位": sample.get("correct_unit", ""),
                    }
                )

        manifest = {
            "created_at": datetime.now().isoformat(),
            "sample_count": len(samples),
            "unused_only": unused_only,
            "data_path": str(data_path),
            "classifier_data_path": str(classifier_data_path),
            "source_distribution": self._count(samples, "sample_source"),
            "unit_distribution": self._count(samples, "correct_unit"),
            "district_distribution": self._count(samples, "district"),
            "recommended_command_note": "将 data_path 作为增量数据源，正式训练前建议先 dry-run 验证标签体系。",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        return TrainingPackageResult(
            status="ok",
            sample_count=len(samples),
            data_path=str(data_path),
            manifest_path=str(manifest_path),
            classifier_data_path=str(classifier_data_path),
            message="主动学习增训包已生成。",
        )

    def _dedupe_samples(self, samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        result = []
        for sample in samples:
            sample_id = sample.get("sample_id")
            if not sample_id or sample_id in seen:
                continue
            seen.add(sample_id)
            result.append(sample)
        return result

    def _cluster_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            risk_labels = self._extract_risk_labels(sample)
            key = "|".join(
                [
                    sample.get("sample_source") or "unknown",
                    sample.get("predicted_unit") or "unknown_unit",
                    risk_labels[0] if risk_labels else "no_risk_flag",
                ]
            )
            grouped[key].append(sample)

        clusters = []
        for key, items in grouped.items():
            source, predicted_unit, main_risk = key.split("|", 2)
            clusters.append(
                {
                    "cluster_key": key,
                    "sample_count": len(items),
                    "sample_ids": [item.get("sample_id") for item in items[:10]],
                    "source": source,
                    "predicted_unit": predicted_unit,
                    "main_risk": main_risk,
                    "avg_confidence": round(
                        sum(float(item.get("confidence") or 0.0) for item in items) / max(len(items), 1),
                        4,
                    ),
                }
            )
        return sorted(clusters, key=lambda item: item["sample_count"], reverse=True)

    def _format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sample_id": sample.get("sample_id"),
            "tag": sample.get("tag", ""),
            "title": sample.get("title", ""),
            "body": sample.get("body", ""),
            "district": sample.get("district", ""),
            "predicted_unit": sample.get("predicted_unit", ""),
            "confidence": sample.get("confidence", 0.0),
            "priority": sample.get("priority", 0),
            "sample_source": sample.get("sample_source", ""),
            "trigger_reason": sample.get("trigger_reason", ""),
            "risk_flags": sample.get("risk_flags", {}),
            "created_at": sample.get("created_at", ""),
        }

    def _extract_risk_labels(self, sample: Dict[str, Any]) -> List[str]:
        flags = sample.get("risk_flags") or {}
        labels = []
        for value in flags.get("reply_error_types", []) or []:
            labels.append(str(value))
        severity = flags.get("reply_error_severity")
        if severity:
            labels.append(f"severity:{severity}")
        feedback_type = flags.get("feedback_type")
        if feedback_type:
            labels.append(f"feedback:{feedback_type}")
        if not labels and sample.get("sample_source"):
            labels.append(str(sample.get("sample_source")))
        return labels

    def _training_record(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "tag": sample.get("tag", ""),
            "title": sample.get("title", ""),
            "body": sample.get("body", ""),
            "district": sample.get("district", ""),
            "unit": sample.get("correct_unit", ""),
            "source": "active_learning",
            "sample_id": sample.get("sample_id", ""),
            "annotated_by": sample.get("annotated_by", ""),
            "annotated_at": sample.get("annotated_at", ""),
            "notes": sample.get("notes", ""),
            "confidence_before": sample.get("confidence_before", 0.0),
        }

    def _count(self, samples: List[Dict[str, Any]], key: str) -> Dict[str, int]:
        return dict(Counter(str(sample.get(key) or "unknown") for sample in samples))
