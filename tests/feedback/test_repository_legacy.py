"""
反馈数据库单元测试
"""

import unittest
from tests import BaseTestCase
from enhancements.feedback_db import FeedbackDatabase


class TestFeedbackDatabase(BaseTestCase):
    """反馈数据库测试"""
    
    def setUp(self):
        """每个测试前的初始化"""
        self.db_path = str(self.test_data_dir / "test_feedback.db")
        self.db = FeedbackDatabase(self.db_path)
    
    def test_add_feedback(self):
        """测试添加反馈"""
        feedback_data = {
            'tag': '投诉',
            'title': '测试标题',
            'body': '测试内容',
            'reply': '测试回复',
            'unit': '测试单位',
            'district': '朝阳区',
            'is_helpful': True,
            'client_ip': '127.0.0.1'
        }
        
        feedback_id = self.db.add_feedback(feedback_data)
        
        self.assertIsNotNone(feedback_id)
        self.assertTrue(feedback_id > 0)
    
    def test_get_feedback_by_id(self):
        """测试根据ID获取反馈"""
        feedback_data = {
            'tag': '咨询',
            'title': '测试查询',
            'is_helpful': False
        }
        feedback_id = self.db.add_feedback(feedback_data)
        
        retrieved = self.db.get_feedback_by_id(feedback_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['tag'], '咨询')
        self.assertEqual(retrieved['is_helpful'], 0)
    
    def test_get_feedback_list(self):
        """测试获取反馈列表"""
        for i in range(5):
            self.db.add_feedback({
                'tag': f'标签{i}',
                'title': f'标题{i}'
            })
        
        feedbacks = self.db.get_feedback(limit=3)
        
        self.assertEqual(len(feedbacks), 3)
    
    def test_search_feedback(self):
        """测试搜索反馈"""
        self.db.add_feedback({
            'tag': '投诉',
            'title': '垃圾问题',
            'body': '垃圾堆积严重'
        })
        self.db.add_feedback({
            'tag': '咨询',
            'title': '政策咨询'
        })
        
        results = self.db.search_feedback(keyword='垃圾')
        
        self.assertTrue(len(results) > 0)
        self.assertTrue(all('垃圾' in r['title'] or '垃圾' in r['body'] for r in results))
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        self.db.add_feedback({'is_helpful': True})
        self.db.add_feedback({'is_helpful': True})
        self.db.add_feedback({'is_helpful': False})
        
        stats = self.db.get_statistics()
        
        self.assertEqual(stats['total_count'], 3)
        self.assertEqual(stats['helpful_count'], 2)
        self.assertEqual(stats['unhelpful_count'], 1)
    
    def test_analyze_feedback(self):
        """测试分析反馈"""
        feedback_data = {
            'title': '测试标题',
            'body': '单位错误，不是这个单位负责',
            'reply': '经某某单位处理',
            'is_helpful': False,
            'comments': '正确的单位是另一个'
        }
        feedback_id = self.db.add_feedback(feedback_data)
        
        result = self.db.analyze_feedback(feedback_id)
        
        self.assertIn('error_type', result)
    
    def test_update_feedback(self):
        """测试更新反馈"""
        feedback_data = {'title': '原标题', 'is_helpful': None}
        feedback_id = self.db.add_feedback(feedback_data)
        
        success = self.db.update_feedback(feedback_id, {'is_helpful': True})
        
        self.assertTrue(success)
        
        updated = self.db.get_feedback_by_id(feedback_id)
        self.assertEqual(updated['is_helpful'], 1)
    
    def test_delete_feedback(self):
        """测试删除反馈"""
        feedback_data = {'title': '待删除'}
        feedback_id = self.db.add_feedback(feedback_data)
        
        success = self.db.delete_feedback(feedback_id)
        
        self.assertTrue(success)
        
        deleted = self.db.get_feedback_by_id(feedback_id)
        self.assertIsNone(deleted)
    
    def test_get_recent_feedback(self):
        """测试获取最近反馈"""
        self.db.add_feedback({'title': '最近反馈1'})
        self.db.add_feedback({'title': '最近反馈2'})
        
        feedbacks = self.db.get_recent_feedback(days=1)
        
        self.assertTrue(len(feedbacks) >= 2)
    
    def test_add_extracted_fact(self):
        """测试添加提取的事实"""
        feedback_data = {'title': '测试'}
        feedback_id = self.db.add_feedback(feedback_data)
        
        fact_id = self.db.add_extracted_fact(
            feedback_id=feedback_id,
            fact_type="project_status",
            fact_content="项目已完成",
            confidence=0.9
        )
        
        self.assertIsNotNone(fact_id)


if __name__ == "__main__":
    unittest.main()
