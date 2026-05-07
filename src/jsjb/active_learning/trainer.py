"""
增量训练器
基于新标注数据进行增量训练
"""

from __future__ import annotations

import os
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

import pandas as pd

from src.jsjb.active_learning.sample_collector import SampleCollector


class IncrementalTrainer:
    """增量训练器，用于主动学习样本的增量训练"""
    
    def __init__(
        self,
        sample_collector: SampleCollector,
        base_data_path: str,
        model_dir: str,
        training_script: str = "training/train_unit_classifier_hybrid.py"
    ):
        self.collector = sample_collector
        self.base_data_path = base_data_path
        self.model_dir = model_dir
        self.training_script = training_script
        self._evaluation_cache = {}
    
    def prepare_training_data(self, output_path: str = None) -> str:
        """
        准备训练数据
        
        合并基础数据和新标注数据，并输出为当前分类训练脚本可直接读取的 CSV。

        这里不能把主动学习 jsonl 直接追加到 xlsx/csv 后面，否则会破坏
        原始文件格式；统一落成 CSV 后，训练脚本会走标准
        “留言标签/留言标题/留言正文/官方回复单位”字段。
        
        Returns:
            训练数据文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                os.path.dirname(self.base_data_path) if self.base_data_path else "data/runtime",
                f"active_learning_classifier_training_{timestamp}.csv"
            )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        base_df = self._load_base_training_frame()
        new_samples = self.collector.get_annotated_samples(unused_only=True)
        active_df = self._annotated_samples_to_frame(new_samples)
        merged_df = pd.concat([base_df, active_df], ignore_index=True)
        merged_df = merged_df.dropna(subset=["留言标题", "留言正文", "官方回复单位"])
        merged_df = merged_df[
            merged_df["留言标题"].astype(str).str.strip().ne("")
            & merged_df["留言正文"].astype(str).str.strip().ne("")
            & merged_df["官方回复单位"].astype(str).str.strip().ne("")
        ].copy()

        merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        print(f"[增量训练] 基础数据 {len(base_df)} 条，新标注 {len(active_df)} 条，合并后 {len(merged_df)} 条")
        print(f"[增量训练] 训练数据已保存到: {output_path}")
        
        return output_path

    def _load_base_training_frame(self) -> pd.DataFrame:
        """Load the configured base data and normalize it to classifier columns."""
        required = ["留言标签", "留言标题", "留言正文", "官方回复单位"]
        if not self.base_data_path or not os.path.exists(self.base_data_path):
            print("[主动学习] 基础数据不存在，将仅使用已标注样本生成增训集")
            return pd.DataFrame(columns=required)

        suffix = Path(self.base_data_path).suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(self.base_data_path, engine="openpyxl")
        elif suffix == ".csv":
            df = pd.read_csv(self.base_data_path, low_memory=False)
        elif suffix in {".jsonl", ".json"}:
            records = []
            with open(self.base_data_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        records.append(json.loads(line))
            df = pd.DataFrame(records)
        else:
            raise ValueError(f"不支持的基础训练数据格式: {self.base_data_path}")

        if all(column in df.columns for column in required):
            return df[required].fillna("").copy()

        master_mapping = {
            "message_tag_level1": "留言标签",
            "message_title": "留言标题",
            "message_body": "留言正文",
            "reply_unit_norm": "官方回复单位",
        }
        active_mapping = {
            "tag": "留言标签",
            "title": "留言标题",
            "body": "留言正文",
            "unit": "官方回复单位",
        }
        for mapping in (master_mapping, active_mapping):
            if all(column in df.columns for column in mapping):
                return df.rename(columns=mapping)[required].fillna("").copy()

        missing = [column for column in required if column not in df.columns]
        raise ValueError(f"基础训练数据缺少分类训练字段: {missing}")

    def _annotated_samples_to_frame(self, samples: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert human-confirmed active-learning samples to classifier rows."""
        records = []
        for sample in samples:
            records.append(
                {
                    "留言标签": sample.get("tag", ""),
                    "留言标题": sample.get("title", ""),
                    "留言正文": sample.get("body", ""),
                    "官方回复单位": sample.get("correct_unit", ""),
                    "active_learning_sample_id": sample.get("sample_id", ""),
                    "active_learning_source": "badge_or_low_confidence_review",
                    "active_learning_annotated_at": sample.get("annotated_at", ""),
                    "active_learning_annotated_by": sample.get("annotated_by", ""),
                    "active_learning_notes": sample.get("notes", ""),
                    "active_learning_confidence_before": sample.get("confidence_before", 0.0),
                }
            )
        columns = [
            "留言标签",
            "留言标题",
            "留言正文",
            "官方回复单位",
            "active_learning_sample_id",
            "active_learning_source",
            "active_learning_annotated_at",
            "active_learning_annotated_by",
            "active_learning_notes",
            "active_learning_confidence_before",
        ]
        return pd.DataFrame(records, columns=columns)
    
    def train(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        执行增量训练
        
        Args:
            epochs: 训练轮数（增量训练建议较少轮数）
            batch_size: 批次大小
            learning_rate: 学习率
            dry_run: 是否只准备数据不实际训练
            
        Returns:
            训练结果
        """
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        before_metrics = self.evaluate_model()
        
        training_data_path = self.prepare_training_data()
        
        new_sample_count = len(self.collector.get_annotated_samples(unused_only=True))
        
        if dry_run:
            return {
                "status": "dry_run",
                "batch_id": batch_id,
                "training_data": training_data_path,
                "new_samples": new_sample_count,
                "metrics_before": before_metrics,
                "message": "数据准备完成，未执行训练"
            }
        
        backup_dir = f"{self.model_dir}_backup_{batch_id}"
        if os.path.exists(self.model_dir):
            shutil.copytree(self.model_dir, backup_dir)
            print(f"[增量训练] 已备份当前模型到: {backup_dir}")
        
        current_weights = os.path.join(self.model_dir, "pytorch_model.bin")
        cmd = [
            sys.executable,
            self.training_script,
            "--data-path", training_data_path,
            "--save-dir", self.model_dir,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--train-from-scratch",
            "--no-resume",
        ]
        if os.path.exists(current_weights):
            cmd.extend(["--init-from", current_weights])
        
        print(f"[增量训练] 开始训练...")
        print(f"[增量训练] 命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                print(f"[增量训练] 训练完成")
                
                sample_ids = [s["sample_id"] for s in self.collector.get_annotated_samples(unused_only=True)]
                self.collector.mark_samples_used(sample_ids, batch_id)
                
                after_metrics = self.evaluate_model()
                self._record_training_history(
                    batch_id=batch_id,
                    sample_count=new_sample_count,
                    status="completed",
                    model_path=self.model_dir,
                    accuracy_before=before_metrics.get("accuracy") if "error" not in before_metrics else None,
                    accuracy_after=after_metrics.get("accuracy") if "error" not in after_metrics else None,
                    f1_before=before_metrics.get("f1") if "error" not in before_metrics else None,
                    f1_after=after_metrics.get("f1") if "error" not in after_metrics else None,
                )
                
                return {
                    "status": "success",
                    "batch_id": batch_id,
                    "new_samples": new_sample_count,
                    "model_dir": self.model_dir,
                    "backup_dir": backup_dir,
                    "metrics_before": before_metrics,
                    "metrics_after": after_metrics,
                    "message": "训练成功完成"
                }
            else:
                print(f"[增量训练] 训练失败")
                print(f"错误输出: {result.stderr}")
                
                if os.path.exists(backup_dir):
                    if os.path.exists(self.model_dir):
                        shutil.rmtree(self.model_dir)
                    shutil.move(backup_dir, self.model_dir)
                    print("[主动学习] 已恢复模型备份")
                
                self._record_training_history(
                    batch_id=batch_id,
                    sample_count=new_sample_count,
                    status="failed",
                    notes=result.stderr[:500]
                )
                
                return {
                    "status": "failed",
                    "batch_id": batch_id,
                    "error": result.stderr,
                    "message": "训练失败，已恢复备份模型"
                }
        
        except subprocess.TimeoutExpired:
            print(f"[增量训练] 训练超时")
            
            if os.path.exists(backup_dir):
                if os.path.exists(self.model_dir):
                    shutil.rmtree(self.model_dir)
                shutil.move(backup_dir, self.model_dir)
            
            self._record_training_history(
                batch_id=batch_id,
                sample_count=new_sample_count,
                status="timeout"
            )
            
            return {
                "status": "timeout",
                "batch_id": batch_id,
                "message": "训练超时，已恢复备份模型"
            }
        
        except Exception as e:
            print(f"[增量训练] 训练异常: {e}")
            
            if os.path.exists(backup_dir):
                if os.path.exists(self.model_dir):
                    shutil.rmtree(self.model_dir)
                shutil.move(backup_dir, self.model_dir)
            
            self._record_training_history(
                batch_id=batch_id,
                sample_count=new_sample_count,
                status="error",
                notes=str(e)
            )
            
            return {
                "status": "error",
                "batch_id": batch_id,
                "error": str(e),
                "message": "训练异常，已恢复备份模型"
            }
    
    def _record_training_history(
        self,
        batch_id: str,
        sample_count: int,
        status: str,
        model_path: str = None,
        accuracy_before: float = None,
        accuracy_after: float = None,
        f1_before: float = None,
        f1_after: float = None,
        notes: str = None
    ):
        """记录训练历史"""
        import sqlite3
        
        conn = sqlite3.connect(self.collector.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO training_history
        (batch_id, sample_count, accuracy_before, accuracy_after, 
         f1_before, f1_after, model_path, started_at, completed_at, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id, sample_count, accuracy_before, accuracy_after,
            f1_before, f1_after, model_path,
            datetime.now().isoformat(),
            datetime.now().isoformat() if status in ["completed", "failed"] else None,
            status, notes
        ))
        
        conn.commit()
        conn.close()
    
    def get_training_history(self, limit: int = 10) -> List[Dict]:
        """获取训练历史"""
        import sqlite3
        
        conn = sqlite3.connect(self.collector.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM training_history
        ORDER BY started_at DESC
        LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def evaluate_model(self, test_data_path: str = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_data_path: 测试数据路径，如果为None则使用已标注样本
            
        Returns:
            {"accuracy": float, "f1": float, "precision": float, "recall": float, "sample_count": int}
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            print("[主动学习] 缺少依赖: transformers, torch")
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "sample_count": 0,
                "error": "缺少依赖库"
            }
        
        if not os.path.exists(self.model_dir):
            print(f"[主动学习] 模型目录不存在: {self.model_dir}")
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "sample_count": 0,
                "error": "模型目录不存在"
            }
        
        label_map_path = os.path.join(self.model_dir, "label_map.json")
        if not os.path.exists(label_map_path):
            print(f"[主动学习] 标签映射文件不存在: {label_map_path}")
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "sample_count": 0,
                "error": "标签映射文件不存在"
            }
        
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        id_to_label = {int(k): v for k, v in label_map.items() if str(k).isdigit()}
        valid_labels = set(id_to_label.values())
        
        if test_data_path is None:
            samples = self.collector.get_annotated_samples(unused_only=False, limit=100)
            if not samples:
                return {
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "sample_count": 0,
                    "error": "无测试样本"
                }
        else:
            samples = []
            with open(test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            model.eval()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            correct = 0
            total = 0
            predictions = []
            labels = []
            
            for sample in samples:
                text = f"{sample.get('tag', '')} {sample.get('title', '')} {sample.get('body', '')}"
                true_label = sample.get("correct_unit") or sample.get("unit")
                
                if not true_label or true_label not in valid_labels:
                    continue
                
                inputs = tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    pred_id = torch.argmax(outputs.logits, dim=-1).item()
                    pred_label = id_to_label.get(pred_id, "未知")
                
                predictions.append(pred_label)
                labels.append(true_label)
                
                if pred_label == true_label:
                    correct += 1
                total += 1
            
            if total == 0:
                return {
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "sample_count": 0
                }
            
            accuracy = correct / total
            
            from collections import Counter
            label_counts = Counter(labels)
            pred_counts = Counter(predictions)
            
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for label in set(labels + predictions):
                tp = sum(1 for p, l in zip(predictions, labels) if p == label and l == label)
                fp = sum(1 for p, l in zip(predictions, labels) if p == label and l != label)
                fn = sum(1 for p, l in zip(predictions, labels) if p != label and l == label)
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                
                precision_scores.append(prec)
                recall_scores.append(rec)
                f1_scores.append(f1)
            
            return {
                "accuracy": round(accuracy, 4),
                "f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
                "precision": round(sum(precision_scores) / len(precision_scores), 4) if precision_scores else 0.0,
                "recall": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0,
                "sample_count": total
            }
            
        except Exception as e:
            print(f"[主动学习] 模型评估失败: {e}")
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "sample_count": 0,
                "error": str(e)
            }
    
    def should_trigger_training(self, threshold: int = 50) -> bool:
        """
        判断是否应该触发训练
        
        Args:
            threshold: 最小样本数阈值
            
        Returns:
            是否应该训练
        """
        unused_count = len(self.collector.get_annotated_samples(unused_only=True))
        return unused_count >= threshold
