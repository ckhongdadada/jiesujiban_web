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
            self._init_database()
    
    def _init_database(self) -> None:
        """初始化数据库和表结构"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
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
        
        # 创建索引提升查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON user_feedback(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_helpful ON user_feedback(is_helpful)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tag ON user_feedback(tag)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_district ON user_feedback(district)')
        
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
        conn.close()
    
    def add_feedback(self, feedback_data: Dict[str, Any]) -> int:
        """
        添加用户反馈记录
        
        Args:
            feedback_data: 反馈数据字典
            
        Returns:
            记录ID
        """
        conn = sqlite3.connect(self.db_path)
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
        conn.close()
        
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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM user_feedback 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_feedback_by_id(self, feedback_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取单条反馈"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_feedback WHERE id = ?', (feedback_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取反馈统计数据
        
        Returns:
            统计数据字典
        """
        conn = sqlite3.connect(self.db_path)
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
        
        conn.close()
        
        helpful_rate = (helpful_count / total_count * 100) if total_count > 0 else 0
        
        return {
            'total_count': total_count,
            'helpful_count': helpful_count,
            'unhelpful_count': unhelpful_count,
            'helpful_rate': round(helpful_rate, 2),
            'tag_distribution': tag_distribution,
            'unit_distribution': unit_distribution,
            'district_distribution': district_distribution,
            'recent_trend': recent_trend
        }
    
    def search_feedback(
        self,
        keyword: str = None,
        tag: str = None,
        district: str = None,
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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
        conn.close()
        
        return [dict(row) for row in rows]
    
    def delete_feedback(self, feedback_id: int) -> bool:
        """删除反馈记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM user_feedback WHERE id = ?', (feedback_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def export_to_json(self, filepath: str) -> int:
        """导出所有数据到JSON文件"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_feedback ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        data = [dict(row) for row in rows]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        conn.close()
        
        return len(data)


def get_feedback_database(db_path: str = None) -> FeedbackDatabase:
    """
    获取反馈数据库实例
    
    Args:
        db_path: 数据库文件路径
        
    Returns:
        FeedbackDatabase实例
    """
    if db_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        db_path = os.path.join(data_dir, "feedback.db")
    
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