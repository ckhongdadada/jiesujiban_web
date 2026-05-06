"""
用户反馈SQLite存储模块
实现用户反馈数据的持久化存储和查询
"""

import sqlite3
import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from src.jsjb.core.paths import get_feedback_db_path


class FeedbackDatabase:
    """用户反馈数据库管理器"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, db_path: str = None):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: str = None):
        if db_path and not hasattr(self, 'db_path'):
            self.db_path = db_path
            self._conn = None
            self._conn_lock = threading.Lock()
            self._init_database()
    
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._conn_lock:
                if self._conn is None:
                    self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA busy_timeout=5000")
                    self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        with self._conn_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    @classmethod
    def reset_singleton(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

    def _init_database(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 创建反馈表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                tag TEXT,
                title TEXT,
                body TEXT,
                reply TEXT,
                unit TEXT,
                district TEXT,
                is_helpful INTEGER,
                feedback_type TEXT,
                comments TEXT,
                client_ip TEXT,
                user_agent TEXT,
                processing_time REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建错误分析表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id INTEGER NOT NULL,
                error_type TEXT,
                error_description TEXT,
                error_entities TEXT,
                correct_facts TEXT,
                needs_correction INTEGER DEFAULT 0,
                analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (feedback_id) REFERENCES user_feedback(id)
            )
        ''')
        
        # 创建事实提取表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id INTEGER NOT NULL,
                fact_type TEXT,
                fact_content TEXT,
                confidence REAL,
                source TEXT,
                verified INTEGER DEFAULT 0,
                extracted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (feedback_id) REFERENCES user_feedback(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generated_reply_error_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id INTEGER NOT NULL,
                reference_reply TEXT,
                normalized_generated_reply TEXT,
                normalized_reference_reply TEXT,
                summary TEXT,
                severity TEXT,
                error_types TEXT,
                error_items TEXT,
                generated_facts TEXT,
                reference_facts TEXT,
                verification_warnings TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (feedback_id) REFERENCES user_feedback(id)
            )
        ''')
        self._ensure_column(cursor, "generated_reply_error_analysis", "quality_dimensions", "TEXT DEFAULT ''")
        self._ensure_column(cursor, "generated_reply_error_analysis", "routing_recommendations", "TEXT DEFAULT ''")

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_graph_fact_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id INTEGER NOT NULL,
                analysis_id INTEGER,
                fact_type TEXT NOT NULL,
                fact_content TEXT NOT NULL,
                source_reply_type TEXT DEFAULT 'reference',
                review_status TEXT DEFAULT 'pending',
                reviewer TEXT DEFAULT '',
                review_notes TEXT DEFAULT '',
                reviewed_at TEXT,
                imported_to_graph INTEGER DEFAULT 0,
                import_result TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (feedback_id) REFERENCES user_feedback(id),
                FOREIGN KEY (analysis_id) REFERENCES generated_reply_error_analysis(id)
            )
        ''')
        
        # 创建索引提升查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON user_feedback(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_helpful ON user_feedback(is_helpful)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tag ON user_feedback(tag)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_district ON user_feedback(district)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_type ON error_analysis(error_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fact_type ON extracted_facts(fact_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_reply_error_feedback_id ON generated_reply_error_analysis(feedback_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_reply_error_severity ON generated_reply_error_analysis(severity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_kg_fact_queue_status ON knowledge_graph_fact_queue(review_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_kg_fact_queue_feedback_id ON knowledge_graph_fact_queue(feedback_id)')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doc_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                is_helpful INTEGER NOT NULL,
                query TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_feedback_doc_id ON doc_feedback(doc_id)')

        # 创建统计视图
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS feedback_stats AS
            SELECT 
                COUNT(*) as total_count,
                SUM(CASE WHEN is_helpful = 1 THEN 1 ELSE 0 END) as helpful_count,
                SUM(CASE WHEN is_helpful = 0 THEN 1 ELSE 0 END) as unhelpful_count,
                AVG(CASE WHEN is_helpful IS NOT NULL THEN 1.0 ELSE 0 END) as helpful_rate,
                COUNT(DISTINCT tag) as tag_count,
                COUNT(DISTINCT unit) as unit_count,
                COUNT(DISTINCT district) as district_count
            FROM user_feedback
        ''')
        
        conn.commit()

    def _ensure_column(self, cursor, table_name: str, column_name: str, column_def: str) -> None:
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")

    def save_reply_error_analysis(
        self,
        feedback_id: int,
        analysis: Dict[str, Any],
        reference_reply: str = "",
    ) -> int:
        """Save structured generated-vs-reference reply error analysis."""
        conn = self._get_conn()
        cursor = conn.cursor()

        verification = analysis.get("verification") or {}
        cursor.execute(
            '''
            INSERT INTO generated_reply_error_analysis (
                feedback_id,
                reference_reply,
                normalized_generated_reply,
                normalized_reference_reply,
                summary,
                severity,
                error_types,
                error_items,
                generated_facts,
                reference_facts,
                verification_warnings,
                quality_dimensions,
                routing_recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                feedback_id,
                reference_reply,
                analysis.get("normalized_generated_reply", ""),
                analysis.get("normalized_reference_reply", ""),
                analysis.get("summary", ""),
                analysis.get("severity", "low"),
                json.dumps(analysis.get("error_types", []), ensure_ascii=False),
                json.dumps(analysis.get("errors", []), ensure_ascii=False),
                json.dumps(analysis.get("generated_facts", []), ensure_ascii=False),
                json.dumps(analysis.get("reference_facts", []), ensure_ascii=False),
                json.dumps(verification.get("warnings", []), ensure_ascii=False),
                json.dumps(analysis.get("quality_dimensions", {}), ensure_ascii=False),
                json.dumps(analysis.get("routing_recommendations", []), ensure_ascii=False),
            ),
        )

        analysis_id = cursor.lastrowid
        conn.commit()
        return analysis_id

    def get_reply_error_analysis(self, feedback_id: int) -> Optional[Dict[str, Any]]:
        """Return the latest structured reply error analysis for a feedback record."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            '''
            SELECT * FROM generated_reply_error_analysis
            WHERE feedback_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            ''',
            (feedback_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        record = dict(row)
        for key in [
            "error_types",
            "error_items",
            "generated_facts",
            "reference_facts",
            "verification_warnings",
            "quality_dimensions",
            "routing_recommendations",
        ]:
            try:
                fallback = "{}" if key == "quality_dimensions" else "[]"
                record[key] = json.loads(record.get(key) or fallback)
            except json.JSONDecodeError:
                record[key] = {} if key == "quality_dimensions" else []
        return record

    def record_doc_feedback(self, doc_id: str, is_helpful: bool, query: str = "") -> None:
        """Record whether a retrieved document was helpful for RAG feedback loop."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO doc_feedback (doc_id, is_helpful, query) VALUES (?, ?, ?)",
            (doc_id, 1 if is_helpful else 0, query),
        )
        conn.commit()

    def get_doc_feedback_scores(self) -> Dict[str, float]:
        """Get aggregated helpful/unhelpful scores per doc for RAG boosting."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT doc_id, SUM(CASE WHEN is_helpful = 1 THEN 1 ELSE -1 END) "
            "FROM doc_feedback GROUP BY doc_id"
        )
        rows = cursor.fetchall()
        return {row[0]: max(min(float(row[1]) * 0.05, 0.30), -0.30) for row in rows}

    def get_doc_feedback_summary(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return document-level helpful/unhelpful feedback ranking for dashboard use."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT
                doc_id,
                COUNT(*) AS total_count,
                SUM(CASE WHEN is_helpful = 1 THEN 1 ELSE 0 END) AS helpful_count,
                SUM(CASE WHEN is_helpful = 0 THEN 1 ELSE 0 END) AS unhelpful_count,
                SUM(CASE WHEN is_helpful = 1 THEN 1 ELSE -1 END) AS net_votes,
                MAX(created_at) AS latest_feedback_at
            FROM doc_feedback
            GROUP BY doc_id
            ORDER BY ABS(net_votes) DESC, total_count DESC, latest_feedback_at DESC
            LIMIT ?
            ''',
            (limit,),
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            total = row["total_count"] or 0
            helpful = row["helpful_count"] or 0
            results.append(
                {
                    "doc_id": row["doc_id"],
                    "total_count": total,
                    "helpful_count": helpful,
                    "unhelpful_count": row["unhelpful_count"] or 0,
                    "net_votes": row["net_votes"] or 0,
                    "helpful_rate": round(helpful / total * 100, 2) if total else 0.0,
                    "feedback_boost": round(max(min((row["net_votes"] or 0) * 0.05, 0.30), -0.30), 4),
                    "latest_feedback_at": row["latest_feedback_at"],
                }
            )
        return results

    def get_quality_error_distribution(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Aggregate generated-reply attribution error types."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT error_types
            FROM generated_reply_error_analysis
            WHERE error_types IS NOT NULL AND TRIM(error_types) != ''
            '''
        )
        counts: Dict[str, int] = {}
        for row in cursor.fetchall():
            try:
                error_types = json.loads(row["error_types"] or "[]")
            except json.JSONDecodeError:
                error_types = []
            for error_type in error_types:
                counts[str(error_type)] = counts.get(str(error_type), 0) + 1
        return [
            {"error_type": key, "count": value}
            for key, value in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
        ]

    def get_feedback_dashboard(self, limit: int = 20) -> Dict[str, Any]:
        """Build a dashboard payload for the feedback learning loop."""
        stats = self.get_statistics()
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM generated_reply_error_analysis")
        reply_error_count = cursor.fetchone()[0]
        cursor.execute("SELECT severity, COUNT(*) FROM generated_reply_error_analysis GROUP BY severity ORDER BY COUNT(*) DESC")
        severity_distribution = [
            {"severity": row[0] or "unknown", "count": row[1]}
            for row in cursor.fetchall()
        ]
        cursor.execute("SELECT COUNT(*) FROM knowledge_graph_fact_queue WHERE review_status = 'pending'")
        pending_kg_facts = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM doc_feedback")
        doc_feedback_count = cursor.fetchone()[0]

        latest_negative = self.search_feedback(is_helpful=False, limit=limit)
        return {
            "status": "ok",
            "stats": stats,
            "feedback_statistics": stats,
            "reply_error_count": reply_error_count,
            "reply_error_analysis_count": reply_error_count,
            "severity_distribution": severity_distribution,
            "quality_error_distribution": self.get_quality_error_distribution(limit=limit),
            "doc_feedback_count": doc_feedback_count,
            "doc_feedback_ranking": self.get_doc_feedback_summary(limit=limit),
            "pending_kg_facts": pending_kg_facts,
            "pending_kg_fact_count": pending_kg_facts,
            "latest_negative_feedback": latest_negative,
        }

    def queue_knowledge_graph_facts(
        self,
        feedback_id: int,
        analysis_id: int,
        analysis: Dict[str, Any],
        source_reply_type: str = "reference",
    ) -> int:
        """Queue extracted facts for manual review before importing into the knowledge graph."""
        facts = analysis.get("reference_facts", []) if source_reply_type == "reference" else analysis.get("generated_facts", [])
        if not facts:
            return 0

        conn = self._get_conn()
        cursor = conn.cursor()
        inserted = 0

        for fact in facts:
            fact_type = fact.get("fact_type", "")
            fact_content = json.dumps(fact.get("content", {}), ensure_ascii=False)
            cursor.execute(
                '''
                INSERT INTO knowledge_graph_fact_queue (
                    feedback_id,
                    analysis_id,
                    fact_type,
                    fact_content,
                    source_reply_type
                ) VALUES (?, ?, ?, ?, ?)
                ''',
                (feedback_id, analysis_id, fact_type, fact_content, source_reply_type),
            )
            inserted += 1

        conn.commit()
        return inserted

    def list_knowledge_graph_fact_queue(
        self,
        status: str = "pending",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM knowledge_graph_fact_queue
            WHERE review_status = ?
            ORDER BY created_at ASC, id ASC
            LIMIT ?
            ''',
            (status, limit),
        )
        rows = cursor.fetchall()

        results = []
        for row in rows:
            item = dict(row)
            try:
                item["fact_content"] = json.loads(item.get("fact_content") or "{}")
            except json.JSONDecodeError:
                item["fact_content"] = {}
            results.append(item)
        return results

    def get_knowledge_graph_fact_candidate(self, candidate_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM knowledge_graph_fact_queue WHERE id = ?',
            (candidate_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["fact_content"] = json.loads(item.get("fact_content") or "{}")
        except json.JSONDecodeError:
            item["fact_content"] = {}
        return item

    def review_knowledge_graph_fact_candidate(
        self,
        candidate_id: int,
        action: str,
        reviewer: str = "",
        review_notes: str = "",
        imported_to_graph: bool = False,
        import_result: str = "",
    ) -> bool:
        review_status = "approved" if action == "approve" else "rejected"
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            '''
            UPDATE knowledge_graph_fact_queue
            SET review_status = ?,
                reviewer = ?,
                review_notes = ?,
                reviewed_at = ?,
                imported_to_graph = ?,
                import_result = ?
            WHERE id = ?
            ''',
            (
                review_status,
                reviewer,
                review_notes,
                datetime.now().isoformat(),
                1 if imported_to_graph else 0,
                import_result,
                candidate_id,
            ),
        )
        success = cursor.rowcount > 0
        conn.commit()
        return success
    
    def add_feedback(self, feedback_data: Dict[str, Any]) -> int:
        """
        添加用户反馈记录
        
        Args:
            feedback_data: 反馈数据字典
            
        Returns:
            记录ID
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback (
                timestamp, tag, title, body, reply, unit, district,
                is_helpful, feedback_type, comments, client_ip, user_agent, processing_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_data.get('timestamp', datetime.now().isoformat()),
            feedback_data.get('tag', ''),
            feedback_data.get('title', ''),
            feedback_data.get('body', ''),
            feedback_data.get('reply', ''),
            feedback_data.get('unit', ''),
            feedback_data.get('district', ''),
            1 if feedback_data.get('is_helpful') else 0,
            feedback_data.get('feedback_type', ''),
            feedback_data.get('comments', ''),
            feedback_data.get('client_ip', ''),
            feedback_data.get('user_agent', ''),
            feedback_data.get('processing_time', 0)
        ))
        
        feedback_id = cursor.lastrowid
        conn.commit()
        
        return feedback_id
    
    def get_feedback(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取反馈记录列表
        
        Args:
            limit: 返回记录数量
            offset: 偏移量
            
        Returns:
            反馈记录列表
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM user_feedback 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_feedback_by_id(self, feedback_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取单条反馈"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_feedback WHERE id = ?', (feedback_id,))
        row = cursor.fetchone()
        
        return dict(row) if row else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取反馈统计数据
        
        Returns:
            统计数据字典
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 基础统计
        cursor.execute('SELECT COUNT(*) FROM user_feedback')
        total_count = cursor.fetchone()[0]
        
        # 有用/无用统计
        cursor.execute('SELECT COUNT(*) FROM user_feedback WHERE is_helpful = 1')
        helpful_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM user_feedback WHERE is_helpful = 0')
        unhelpful_count = cursor.fetchone()[0]
        
        # 标签分布
        cursor.execute('''
            SELECT tag, COUNT(*) as count 
            FROM user_feedback 
            GROUP BY tag 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        tag_distribution = [{'tag': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # 单位分布
        cursor.execute('''
            SELECT unit, COUNT(*) as count 
            FROM user_feedback 
            GROUP BY unit 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        unit_distribution = [{'unit': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # 地区分布
        cursor.execute('''
            SELECT district, COUNT(*) as count 
            FROM user_feedback 
            WHERE district != '' 
            GROUP BY district 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        district_distribution = [{'district': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # 近期趋势（最近7天）
        cursor.execute('''
            SELECT DATE(created_at) as date, COUNT(*) as count,
                   SUM(CASE WHEN is_helpful = 1 THEN 1 ELSE 0 END) as helpful
            FROM user_feedback
            WHERE created_at >= DATE('now', '-7 days')
            GROUP BY DATE(created_at)
            ORDER BY date
        ''')
        recent_trend = [
            {'date': row[0], 'count': row[1], 'helpful': row[2]} 
            for row in cursor.fetchall()
        ]

        # 反馈类型分布与类型内有用率
        cursor.execute('''
            SELECT
                CASE
                    WHEN feedback_type IS NULL OR TRIM(feedback_type) = '' THEN 'unknown'
                    ELSE feedback_type
                END AS feedback_type,
                COUNT(*) AS count,
                SUM(CASE WHEN is_helpful = 1 THEN 1 ELSE 0 END) AS helpful_count
            FROM user_feedback
            GROUP BY
                CASE
                    WHEN feedback_type IS NULL OR TRIM(feedback_type) = '' THEN 'unknown'
                    ELSE feedback_type
                END
            ORDER BY count DESC
        ''')
        feedback_type_distribution = []
        for row in cursor.fetchall():
            ftype = row[0]
            count = row[1] or 0
            helpful = row[2] or 0
            rate = (helpful / count * 100) if count > 0 else 0
            feedback_type_distribution.append({
                'feedback_type': ftype,
                'count': count,
                'helpful_count': helpful,
                'helpful_rate': round(rate, 2),
            })
        
        
        helpful_rate = (helpful_count / total_count * 100) if total_count > 0 else 0
        
        return {
            'total_count': total_count,
            'helpful_count': helpful_count,
            'unhelpful_count': unhelpful_count,
            'helpful_rate': round(helpful_rate, 2),
            'tag_distribution': tag_distribution,
            'unit_distribution': unit_distribution,
            'district_distribution': district_distribution,
            'recent_trend': recent_trend,
            'feedback_type_distribution': feedback_type_distribution,
        }
    
    def search_feedback(
        self,
        keyword: str = None,
        tag: str = None,
        district: str = None,
        feedback_type: str = None,
        is_helpful: bool = None,
        start_date: str = None,
        end_date: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        搜索反馈记录
        
        Args:
            keyword: 关键词搜索
            tag: 标签筛选
            district: 地区筛选
            is_helpful: 有用/无用筛选
            start_date: 开始日期
            end_date: 结束日期
            limit: 返回数量
            offset: 偏移量
            
        Returns:
            符合条件的反馈记录列表
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if keyword:
            conditions.append('(title LIKE ? OR body LIKE ? OR reply LIKE ?)')
            keyword_pattern = f'%{keyword}%'
            params.extend([keyword_pattern, keyword_pattern, keyword_pattern])
        
        if tag:
            conditions.append('tag = ?')
            params.append(tag)
        
        if district:
            conditions.append('district = ?')
            params.append(district)

        if feedback_type:
            conditions.append('feedback_type = ?')
            params.append(feedback_type)
        
        if is_helpful is not None:
            conditions.append('is_helpful = ?')
            params.append(1 if is_helpful else 0)
        
        if start_date:
            conditions.append('created_at >= ?')
            params.append(start_date)
        
        if end_date:
            conditions.append('created_at <= ?')
            params.append(end_date)
        
        where_clause = ' AND '.join(conditions) if conditions else '1=1'
        
        cursor.execute(f'''
            SELECT * FROM user_feedback 
            WHERE {where_clause}
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        ''', (*params, limit, offset))
        
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def delete_feedback(self, feedback_id: int) -> bool:
        """删除反馈记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM user_feedback WHERE id = ?', (feedback_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        
        return deleted
    
    def export_to_json(self, filepath: str) -> int:
        """导出所有数据到JSON文件"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_feedback ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        data = [dict(row) for row in rows]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        
        return len(data)
    
    def analyze_feedback(self, feedback_id: int) -> Dict[str, Any]:
        """
        自动分析反馈，识别错误类型
        
        Args:
            feedback_id: 反馈ID
            
        Returns:
            分析结果字典
        """
        feedback = self.get_feedback_by_id(feedback_id)
        if not feedback:
            return {"error": "反馈不存在"}
        
        if feedback.get('is_helpful') == 1:
            return {"status": "正面反馈，无需分析"}
        
        error_type = self._classify_error(feedback)
        error_entities = self._extract_error_entities(feedback)
        correct_facts = self._infer_correct_facts(feedback)
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO error_analysis (
                feedback_id, error_type, error_description, 
                error_entities, correct_facts, needs_correction
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id,
            error_type,
            f"检测到{error_type}类型错误",
            json.dumps(error_entities, ensure_ascii=False),
            json.dumps(correct_facts, ensure_ascii=False),
            1 if error_type != "未知错误" else 0
        ))
        
        conn.commit()
        
        return {
            "feedback_id": feedback_id,
            "error_type": error_type,
            "error_entities": error_entities,
            "correct_facts": correct_facts
        }
    
    def _classify_error(self, feedback: Dict[str, Any]) -> str:
        """分类错误类型"""
        reply = feedback.get('reply', '')
        title = feedback.get('title', '')
        body = feedback.get('body', '')
        comments = feedback.get('comments', '')
        
        text = f"{title} {body} {reply} {comments}"
        
        error_patterns = {
            "A: 建设主体张冠李戴": [
                "单位错误", "责任单位不对", "不是这个单位负责",
                "主体错误", "找错单位了"
            ],
            "B: 是/否结论颠倒": [
                "结论错误", "事实相反", "说反了", "结论颠倒",
                "不是这样的", "实际情况是"
            ],
            "C: 项目阶段倒退": [
                "状态错误", "已经完工", "已经完成", "不是建设中",
                "项目状态不对"
            ],
            "D: 阻碍原因归因错误": [
                "原因错误", "不是拆迁", "拆迁已完成", "阻碍原因不对"
            ],
            "E: 现有资源错报": [
                "资源存在", "有这个设施", "图书馆存在", "设施已运营",
                "不是没有"
            ],
            "F: 承办街道错配": [
                "街道错误", "不是这个街道", "街道不对", "承办街道错误"
            ]
        }
        
        for error_type, patterns in error_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return error_type
        
        return "未知错误"
    
    def _extract_error_entities(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """提取错误相关的实体"""
        import re
        
        reply = feedback.get('reply', '')
        entities = {
            "units": [],
            "projects": [],
            "locations": [],
            "statuses": []
        }
        
        unit_patterns = [
            r"经(.+?(?:街道|镇|乡|局|委|处|办))",
            r"由(.+?(?:公司|单位|集团))负责"
        ]
        
        for pattern in unit_patterns:
            matches = re.finditer(pattern, reply)
            for match in matches:
                unit = match.group(1).strip()
                if unit not in entities["units"]:
                    entities["units"].append(unit)
        
        status_patterns = [
            r"(已(?:完成|完工|竣工|交付))",
            r"(正在(?:施工|建设|推进|办理))",
            r"(暂未(?:开始|启动|建设))"
        ]
        
        for pattern in status_patterns:
            matches = re.finditer(pattern, reply)
            for match in matches:
                status = match.group(1)
                if status not in entities["statuses"]:
                    entities["statuses"].append(status)
        
        return entities
    
    def _infer_correct_facts(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """推断正确的事实"""
        comments = feedback.get('comments', '')
        correct_facts = {}
        
        import re
        
        status_match = re.search(r"实际(?:状态|情况)[是为：:\s]*(.+?)(?:，|。|$)", comments)
        if status_match:
            correct_facts["status"] = status_match.group(1).strip()
        
        unit_match = re.search(r"正确(?:单位|部门)[是为：:\s]*(.+?)(?:，|。|$)", comments)
        if unit_match:
            correct_facts["unit"] = unit_match.group(1).strip()
        
        return correct_facts
    
    def update_feedback(self, feedback_id: int, updates: Dict[str, Any]) -> bool:
        """
        更新反馈记录
        
        Args:
            feedback_id: 反馈ID
            updates: 更新字段字典
            
        Returns:
            是否更新成功
        """
        if not updates:
            return False
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['is_helpful', 'feedback_type', 'comments']:
                set_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not set_clauses:
            return False
        
        params.append(feedback_id)
        
        cursor.execute(
            f"UPDATE user_feedback SET {', '.join(set_clauses)} WHERE id = ?",
            params
        )
        
        success = cursor.rowcount > 0
        conn.commit()
        
        return success
    
    def get_recent_feedback(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        获取最近几天的反馈
        
        Args:
            days: 天数
            
        Returns:
            反馈列表
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM user_feedback 
            WHERE created_at >= DATE('now', ?)
            ORDER BY created_at DESC
        ''', (f'-{days} days',))
        
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误分析统计"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT error_type, COUNT(*) as count
            FROM error_analysis
            GROUP BY error_type
            ORDER BY count DESC
        ''')
        
        error_distribution = [
            {"error_type": row[0], "count": row[1]}
            for row in cursor.fetchall()
        ]
        
        cursor.execute('SELECT COUNT(*) FROM error_analysis WHERE needs_correction = 1')
        needs_correction_count = cursor.fetchone()[0]
        
        
        return {
            "error_distribution": error_distribution,
            "needs_correction_count": needs_correction_count
        }
    
    def add_extracted_fact(
        self,
        feedback_id: int,
        fact_type: str,
        fact_content: str,
        confidence: float = 0.8,
        source: str = "user_verified"
    ) -> int:
        """
        添加提取的事实
        
        Args:
            feedback_id: 反馈ID
            fact_type: 事实类型
            fact_content: 事实内容
            confidence: 置信度
            source: 来源
            
        Returns:
            事实ID
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO extracted_facts (
                feedback_id, fact_type, fact_content, confidence, source
            ) VALUES (?, ?, ?, ?, ?)
        ''', (feedback_id, fact_type, fact_content, confidence, source))
        
        fact_id = cursor.lastrowid
        conn.commit()
        
        return fact_id
    
    def get_unverified_facts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取未验证的事实"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ef.*, uf.reply, uf.title, uf.body
            FROM extracted_facts ef
            JOIN user_feedback uf ON ef.feedback_id = uf.id
            WHERE ef.verified = 0
            ORDER BY ef.extracted_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def verify_fact(self, fact_id: int, verified: bool = True) -> bool:
        """验证事实"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE extracted_facts SET verified = ? WHERE id = ?',
            (1 if verified else 0, fact_id)
        )
        
        success = cursor.rowcount > 0
        conn.commit()
        
        return success


def get_feedback_database(db_path: str = None) -> FeedbackDatabase:
    """
    获取反馈数据库实例
    
    Args:
        db_path: 数据库文件路径
        
    Returns:
        FeedbackDatabase实例
    """
    if db_path is None:
        db_path = str(get_feedback_db_path())
    
    return FeedbackDatabase(db_path)


# 兼容旧版本的函数别名
def init_feedback_db(db_path: str = None) -> FeedbackDatabase:
    """初始化反馈数据库"""
    return get_feedback_database(db_path)


if __name__ == "__main__":
    # 测试数据库功能
    db = get_feedback_database("test_feedback.db")
    
    # 添加测试数据
    test_data = {
        'timestamp': datetime.now().isoformat(),
        'tag': '投诉',
        'title': '小区垃圾没人清理',
        'body': '我们小区垃圾堆积严重',
        'reply': '已安排人员处理',
        'unit': '城管委',
        'district': '朝阳区',
        'is_helpful': True,
        'client_ip': '127.0.0.1',
        'processing_time': 1.5
    }
    
    feedback_id = db.add_feedback(test_data)
    print(f"添加反馈记录，ID: {feedback_id}")
    
    # 获取统计
    stats = db.get_statistics()
    print(f"统计信息: {stats}")
    
    # 搜索测试
    results = db.search_feedback(keyword='垃圾')
    print(f"搜索结果: {len(results)} 条")
    
    print("数据库测试完成")
