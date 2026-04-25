"""
标注管理器
提供标注界面和标注流程管理
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from src.jsjb.active_learning.sample_collector import SampleCollector


class AnnotationManager:
    """标注管理器"""
    
    def __init__(self, sample_collector: SampleCollector):
        self.collector = sample_collector
    
    def get_next_batch(self, batch_size: int = 10, strategy: str = "priority") -> List[Dict]:
        """
        获取下一批待标注样本
        
        Args:
            batch_size: 批次大小
            strategy: 采样策略
                - priority: 按优先级（默认）
                - uncertainty: 按不确定性（低置信度优先）
                - diversity: 多样性采样
                - time: 按时间（最新优先）
            
        Returns:
            样本列表
        """
        if strategy == "priority":
            return self.collector.get_pending_samples(limit=batch_size, order_by="priority")
        
        elif strategy == "uncertainty":
            # 低置信度优先
            return self.collector.get_pending_samples(
                limit=batch_size,
                max_confidence=0.7,
                order_by="confidence"
            )
        
        elif strategy == "diversity":
            # 多样性采样：从不同置信度区间采样
            samples = []
            intervals = [
                (0.0, 0.3, batch_size // 3),
                (0.3, 0.5, batch_size // 3),
                (0.5, 0.7, batch_size - 2 * (batch_size // 3))
            ]
            for min_conf, max_conf, count in intervals:
                batch = self.collector.get_pending_samples(
                    limit=count,
                    min_confidence=min_conf,
                    max_confidence=max_conf
                )
                samples.extend(batch)
            return samples
        
        elif strategy == "time":
            return self.collector.get_pending_samples(limit=batch_size, order_by="created_at")
        
        else:
            raise ValueError(f"未知的采样策略: {strategy}")
    
    def format_for_annotation(self, sample: Dict) -> Dict:
        """
        格式化样本用于标注界面显示
        
        Returns:
            {
                "id": "...",
                "content": "...",
                "predicted": "...",
                "confidence": 0.xx,
                "top_predictions": [...],
                "feedback": "...",
                "priority": "高/中/低"
            }
        """
        # 获取Top-K预测
        probs = sample.get("prediction_probs", {})
        top_predictions = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 判断优先级级别
        priority_score = sample.get("priority", 0)
        if priority_score >= 90:
            priority_label = "高"
        elif priority_score >= 60:
            priority_label = "中"
        else:
            priority_label = "低"
        
        return {
            "id": sample["sample_id"],
            "tag": sample["tag"],
            "title": sample["title"],
            "body": sample["body"],
            "district": sample["district"],
            "content": f"【{sample['tag']}】{sample['title']}\n{sample['body']}",
            "predicted": sample["predicted_unit"],
            "confidence": round(sample["confidence"], 3),
            "top_predictions": [
                {"unit": unit, "prob": round(prob, 3)}
                for unit, prob in top_predictions
            ],
            "feedback": sample.get("user_feedback", ""),
            "priority": priority_label,
            "created_at": sample["created_at"]
        }
    
    def submit_annotation(
        self,
        sample_id: str,
        correct_unit: str,
        annotator: str,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        提交标注
        
        Returns:
            {"success": bool, "message": str}
        """
        success = self.collector.annotate_sample(
            sample_id=sample_id,
            correct_unit=correct_unit,
            annotated_by=annotator,
            notes=notes
        )
        
        if success:
            return {
                "success": True,
                "message": "标注成功"
            }
        else:
            return {
                "success": False,
                "message": "标注失败，请重试"
            }
    
    def submit_batch_annotations(
        self,
        annotations: List[Dict],
        annotator: str
    ) -> Dict[str, Any]:
        """
        批量提交标注
        
        Args:
            annotations: [{"sample_id": "...", "correct_unit": "...", "notes": "..."}, ...]
            annotator: 标注人
            
        Returns:
            {"success_count": int, "total": int, "failed": [...]}
        """
        total = len(annotations)
        failed = []
        
        for ann in annotations:
            ann["annotated_by"] = annotator
            success = self.collector.annotate_sample(
                sample_id=ann["sample_id"],
                correct_unit=ann["correct_unit"],
                annotated_by=annotator,
                notes=ann.get("notes", "")
            )
            if not success:
                failed.append(ann["sample_id"])
        
        success_count = total - len(failed)
        
        return {
            "success_count": success_count,
            "total": total,
            "failed": failed,
            "message": f"成功标注 {success_count}/{total} 条样本"
        }
    
    def get_annotation_progress(self) -> Dict[str, Any]:
        """
        获取标注进度
        
        Returns:
            {
                "pending": int,
                "annotated": int,
                "used_in_training": int,
                "ready_for_training": int,
                "progress_percent": float
            }
        """
        stats = self.collector.get_statistics()
        
        pending = stats["pending_samples"]
        annotated = stats["annotated_samples"]
        ready = stats["unused_samples"]
        
        total = pending + annotated
        progress = (annotated / total * 100) if total > 0 else 0
        
        return {
            "pending": pending,
            "annotated": annotated,
            "used_in_training": annotated - ready,
            "ready_for_training": ready,
            "progress_percent": round(progress, 2),
            "avg_confidence": stats["avg_confidence"],
            "negative_feedback_count": stats["negative_feedback_samples"]
        }
    
    def get_annotation_quality_report(self) -> Dict[str, Any]:
        """
        获取标注质量报告
        
        Returns:
            {
                "agreement_rate": float,  # 标注与预测一致率
                "low_confidence_accuracy": float,  # 低置信度样本的标注准确率
                "annotator_stats": {...}  # 各标注人统计
            }
        """
        samples = self.collector.get_annotated_samples(unused_only=False)
        
        if not samples:
            return {
                "agreement_rate": 0,
                "low_confidence_accuracy": 0,
                "annotator_stats": {}
            }
        
        # 计算一致率
        agreement_count = 0
        low_conf_samples = []
        annotator_stats = {}
        
        for sample in samples:
            # 获取原始预测
            conn = self.collector._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT predicted_unit FROM pending_samples WHERE sample_id = ?",
                (sample["sample_id"],)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                predicted = row[0]
                if predicted == sample["correct_unit"]:
                    agreement_count += 1
                
                # 低置信度样本
                if sample["confidence_before"] < 0.5:
                    low_conf_samples.append(sample)
            
            # 标注人统计
            annotator = sample["annotated_by"]
            if annotator not in annotator_stats:
                annotator_stats[annotator] = {"count": 0, "avg_confidence": []}
            annotator_stats[annotator]["count"] += 1
            annotator_stats[annotator]["avg_confidence"].append(sample["confidence_before"])
        
        # 计算指标
        agreement_rate = agreement_count / len(samples) if samples else 0
        
        low_conf_agreement = 0
        if low_conf_samples:
            for sample in low_conf_samples:
                conn = self.collector._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT predicted_unit FROM pending_samples WHERE sample_id = ?",
                    (sample["sample_id"],)
                )
                row = cursor.fetchone()
                conn.close()
                if row and row[0] == sample["correct_unit"]:
                    low_conf_agreement += 1
            low_conf_accuracy = low_conf_agreement / len(low_conf_samples)
        else:
            low_conf_accuracy = 0
        
        # 整理标注人统计
        for annotator, stats in annotator_stats.items():
            avg_conf = sum(stats["avg_confidence"]) / len(stats["avg_confidence"])
            stats["avg_confidence"] = round(avg_conf, 3)
        
        return {
            "total_samples": len(samples),
            "agreement_rate": round(agreement_rate, 3),
            "low_confidence_samples": len(low_conf_samples),
            "low_confidence_accuracy": round(low_conf_accuracy, 3),
            "annotator_stats": annotator_stats
        }
    
    def suggest_next_action(self) -> Dict[str, Any]:
        """
        建议下一步操作
        
        Returns:
            {
                "action": "annotate/train/wait",
                "reason": "...",
                "details": {...}
            }
        """
        stats = self.collector.get_statistics()
        
        pending = stats["pending_samples"]
        ready = stats["unused_samples"]
        negative = stats["negative_feedback_samples"]
        
        # 如果有负面反馈样本，优先标注
        if negative > 0:
            return {
                "action": "annotate",
                "reason": f"有 {negative} 条用户反馈不满意的样本需要优先标注",
                "details": {
                    "priority_samples": negative,
                    "suggested_batch_size": min(negative, 20)
                }
            }
        
        # 如果有足够的已标注样本，建议训练
        if ready >= 50:
            return {
                "action": "train",
                "reason": f"已有 {ready} 条标注样本可用于训练",
                "details": {
                    "ready_samples": ready,
                    "recommended": True
                }
            }
        
        # 如果待标注样本较多，建议继续标注
        if pending > 10:
            return {
                "action": "annotate",
                "reason": f"还有 {pending} 条待标注样本",
                "details": {
                    "pending_samples": pending,
                    "suggested_batch_size": min(pending, 20)
                }
            }
        
        # 否则等待更多样本
        return {
            "action": "wait",
            "reason": "当前样本较少，建议等待更多低置信度样本积累",
            "details": {
                "pending_samples": pending,
                "ready_samples": ready,
                "threshold": 50
            }
        }
