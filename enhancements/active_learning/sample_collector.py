from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


NEGATIVE_FEEDBACK_MARKERS = ["不满意", "无用", "错误", "不对", "不准确", "negative", "bad"]


class SampleCollector:
    """Collect, rank, annotate, and export active-learning samples."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = "data/runtime/active_learning.db"
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT UNIQUE,
                tag TEXT,
                title TEXT,
                body TEXT,
                district TEXT,
                predicted_unit TEXT,
                confidence REAL,
                prediction_probs TEXT,
                user_feedback TEXT,
                is_negative_feedback INTEGER DEFAULT 0,
                priority INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                annotated_at TEXT,
                annotated_by TEXT,
                correct_unit TEXT,
                notes TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS annotated_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT,
                tag TEXT,
                title TEXT,
                body TEXT,
                district TEXT,
                correct_unit TEXT,
                annotated_by TEXT,
                annotated_at TEXT,
                confidence_before REAL,
                used_in_training INTEGER DEFAULT 0,
                training_batch TEXT,
                notes TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT UNIQUE,
                sample_count INTEGER,
                accuracy_before REAL,
                accuracy_after REAL,
                f1_before REAL,
                f1_after REAL,
                model_path TEXT,
                started_at TEXT,
                completed_at TEXT,
                status TEXT,
                notes TEXT
            )
            """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON pending_samples(confidence)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON pending_samples(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON pending_samples(priority DESC)")

        self._ensure_column(cursor, "pending_samples", "sample_source", "TEXT DEFAULT 'classifier_low_confidence'")
        self._ensure_column(cursor, "pending_samples", "trigger_reason", "TEXT DEFAULT ''")
        self._ensure_column(cursor, "pending_samples", "risk_flags", "TEXT DEFAULT ''")

        conn.commit()
        conn.close()

    def _ensure_column(self, cursor, table_name: str, column_name: str, column_def: str) -> None:
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_sample(
        self,
        sample_id: str,
        tag: str,
        title: str,
        body: str,
        district: str,
        predicted_unit: str,
        confidence: float,
        prediction_probs: Optional[Dict[str, float]] = None,
        user_feedback: Optional[str] = None,
        sample_source: str = "classifier_low_confidence",
        trigger_reason: str = "",
        risk_flags: Optional[Dict[str, Any]] = None,
    ) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            priority = self._calculate_priority(confidence, user_feedback)
            feedback_text = (user_feedback or "").lower()
            is_negative = 1 if any(marker in feedback_text for marker in NEGATIVE_FEEDBACK_MARKERS) else 0

            cursor.execute(
                """
                INSERT OR REPLACE INTO pending_samples (
                    sample_id, tag, title, body, district, predicted_unit, confidence,
                    prediction_probs, user_feedback, is_negative_feedback, priority, created_at,
                    sample_source, trigger_reason, risk_flags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    tag,
                    title,
                    body,
                    district,
                    predicted_unit,
                    confidence,
                    json.dumps(prediction_probs or {}, ensure_ascii=False),
                    user_feedback,
                    is_negative,
                    priority,
                    datetime.now().isoformat(),
                    sample_source,
                    trigger_reason,
                    json.dumps(risk_flags or {}, ensure_ascii=False),
                ),
            )
            conn.commit()
            return True
        except Exception as exc:
            print(f"[sample_collector] failed to add sample: {exc}")
            return False
        finally:
            conn.close()

    def _calculate_priority(self, confidence: float, user_feedback: Optional[str] = None) -> int:
        if user_feedback and any(marker in user_feedback.lower() for marker in NEGATIVE_FEEDBACK_MARKERS):
            return 100
        if confidence < 0.3:
            return 90
        if confidence < 0.5:
            return 70
        if confidence < 0.7:
            return 50
        return 30

    def get_pending_samples(
        self,
        limit: int = 100,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        order_by: str = "priority",
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        where_clauses = ["status = 'pending'"]
        params: List[Any] = []

        if min_confidence is not None:
            where_clauses.append("confidence >= ?")
            params.append(min_confidence)
        if max_confidence is not None:
            where_clauses.append("confidence <= ?")
            params.append(max_confidence)

        if order_by == "priority":
            order_clause = "ORDER BY priority DESC, confidence ASC"
        elif order_by == "confidence":
            order_clause = "ORDER BY confidence ASC"
        else:
            order_clause = "ORDER BY created_at DESC"

        query = f"""
        SELECT * FROM pending_samples
        WHERE {' AND '.join(where_clauses)}
        {order_clause}
        LIMIT ?
        """
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        samples = []
        for row in rows:
            sample = dict(row)
            sample["prediction_probs"] = json.loads(sample.get("prediction_probs") or "{}")
            sample["risk_flags"] = json.loads(sample.get("risk_flags") or "{}")
            samples.append(sample)
        return samples

    def annotate_sample(
        self,
        sample_id: str,
        correct_unit: str,
        annotated_by: str,
        notes: str = "",
    ) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                UPDATE pending_samples
                SET status = 'annotated',
                    correct_unit = ?,
                    annotated_by = ?,
                    annotated_at = ?,
                    notes = ?
                WHERE sample_id = ?
                """,
                (correct_unit, annotated_by, datetime.now().isoformat(), notes, sample_id),
            )

            cursor.execute("SELECT * FROM pending_samples WHERE sample_id = ?", (sample_id,))
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    """
                    INSERT INTO annotated_samples (
                        sample_id, tag, title, body, district, correct_unit,
                        annotated_by, annotated_at, confidence_before, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id,
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        correct_unit,
                        annotated_by,
                        datetime.now().isoformat(),
                        row[7],
                        notes,
                    ),
                )

            conn.commit()
            return True
        except Exception as exc:
            print(f"[sample_collector] failed to annotate sample: {exc}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def batch_annotate(self, annotations: List[Dict[str, Any]]) -> int:
        success_count = 0
        for annotation in annotations:
            if self.annotate_sample(
                annotation["sample_id"],
                annotation["correct_unit"],
                annotation.get("annotated_by", "unknown"),
                annotation.get("notes", ""),
            ):
                success_count += 1
        return success_count

    def get_annotated_samples(
        self,
        limit: Optional[int] = None,
        unused_only: bool = True,
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        where_clause = "WHERE used_in_training = 0" if unused_only else ""
        limit_clause = f"LIMIT {limit}" if limit else ""
        cursor.execute(
            f"""
            SELECT * FROM annotated_samples
            {where_clause}
            ORDER BY annotated_at DESC
            {limit_clause}
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def mark_samples_used(self, sample_ids: List[str], batch_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        placeholders = ",".join(["?" for _ in sample_ids])
        cursor.execute(
            f"""
            UPDATE annotated_samples
            SET used_in_training = 1, training_batch = ?
            WHERE sample_id IN ({placeholders})
            """,
            [batch_id] + sample_ids,
        )
        conn.commit()
        conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM pending_samples WHERE status = 'pending'")
        pending_count = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence) FROM pending_samples WHERE status = 'pending'")
        avg_confidence = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM pending_samples WHERE is_negative_feedback = 1 AND status = 'pending'")
        negative_feedback_count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT sample_source, COUNT(*)
            FROM pending_samples
            WHERE status = 'pending'
            GROUP BY sample_source
            ORDER BY COUNT(*) DESC
            """
        )
        source_distribution = [
            {"sample_source": row[0] or "unknown", "count": row[1]}
            for row in cursor.fetchall()
        ]

        cursor.execute("SELECT COUNT(*) FROM annotated_samples")
        annotated_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM annotated_samples WHERE used_in_training = 0")
        unused_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM training_history")
        training_count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT accuracy_after, f1_after
            FROM training_history
            WHERE status = 'completed'
            ORDER BY completed_at DESC
            LIMIT 1
            """
        )
        latest_metrics = cursor.fetchone()
        conn.close()

        return {
            "pending_samples": pending_count,
            "avg_confidence": round(avg_confidence, 3),
            "negative_feedback_samples": negative_feedback_count,
            "annotated_samples": annotated_count,
            "unused_samples": unused_count,
            "training_rounds": training_count,
            "latest_accuracy": latest_metrics[0] if latest_metrics else None,
            "latest_f1": latest_metrics[1] if latest_metrics else None,
            "source_distribution": source_distribution,
        }

    def export_training_data(self, output_path: str, unused_only: bool = True) -> None:
        samples = self.get_annotated_samples(unused_only=unused_only)
        with open(output_path, "w", encoding="utf-8") as fp:
            for sample in samples:
                record = {
                    "tag": sample["tag"],
                    "title": sample["title"],
                    "body": sample["body"],
                    "district": sample["district"],
                    "unit": sample["correct_unit"],
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[sample_collector] exported {len(samples)} samples to {output_path}")
