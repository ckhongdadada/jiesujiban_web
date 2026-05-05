"""
集成测试
测试各模块之间的协作
"""

import unittest
import json
from tests import BaseTestCase
from src.jsjb.reply_generation.verification import FactVerifier
from src.jsjb.reply_generation.fact_extraction import FactExtractor
from src.jsjb.knowledge.structured_kb import StructuredKnowledgeBase, ProjectInfo
from src.jsjb.feedback.repository import FeedbackDatabase
from src.jsjb.knowledge.updater import KnowledgeUpdater


class TestIntegration(BaseTestCase):
    """集成测试"""
    
    def setUp(self):
        """每个测试前的初始化"""
        FeedbackDatabase.reset_singleton()
        if hasattr(FeedbackDatabase, 'db_path'):
            del FeedbackDatabase.db_path
        self.kb_path = str(self.test_data_dir / "test_kb.json")
        self.db_path = str(self.test_data_dir / "test_feedback.db")
        
        self.kb = StructuredKnowledgeBase(self.kb_path)
        self.db = FeedbackDatabase(self.db_path)
        self.verifier = FactVerifier()
        self.extractor = FactExtractor()
    
    def test_fact_verification_and_extraction_flow(self):
        """测试事实验证和提取流程"""
        reply = "经核实，鲁疃西路南延工程已完工验收，拆迁已完成。"
        hits = [{
            "project_status": "进行中",
            "demolition_status": "未开始",
            "snippet": "项目正在建设中"
        }]
        location = {"district": "昌平区"}
        
        verification = self.verifier.verify_reply(reply, hits, location)
        
        self.assertFalse(verification["is_valid"])
        
        facts = self.extractor.extract_facts_from_reply(reply)
        
        self.assertTrue(len(facts) > 0)
        
        kb_facts = self.extractor.facts_to_kb_format(facts)
        
        for fact in kb_facts:
            self.kb.add_fact(fact)
        
        project = self.kb.get_project_status("鲁疃西路南延")
        self.assertIsNotNone(project)
    
    def test_feedback_to_knowledge_update_flow(self):
        """测试反馈到知识更新的流程"""
        feedback_data = {
            'title': '鲁疃西路南延工程进展',
            'body': '咨询道路建设进度',
            'reply': '经核实，鲁疃西路南延工程已完工验收，计划五一前通车。',
            'is_helpful': True,
            'district': '昌平区'
        }
        
        feedback_id = self.db.add_feedback(feedback_data)
        
        feedback = self.db.get_feedback_by_id(feedback_id)
        
        facts = self.extractor.extract_facts_from_reply(
            reply=feedback['reply'],
            feedback_data=feedback
        )
        
        self.assertTrue(len(facts) > 0)
        
        kb_facts = self.extractor.facts_to_kb_format(facts)
        
        for fact in kb_facts:
            self.kb.add_fact(fact)
        
        project = self.kb.get_project_status("鲁疃西路南延")
        self.assertIsNotNone(project)
    
    def test_error_feedback_analysis_flow(self):
        """测试错误反馈分析流程"""
        feedback_data = {
            'title': '项目状态错误反馈',
            'body': '回复说正在施工，实际已经完工',
            'reply': '该道路正在施工建设中',
            'is_helpful': False,
            'comments': '实际状态：已完工'
        }
        
        feedback_id = self.db.add_feedback(feedback_data)
        
        analysis = self.db.analyze_feedback(feedback_id)
        
        self.assertIn('error_type', analysis)
    
    def test_full_pipeline(self):
        """测试完整流程"""
        project_name = "集成测试项目"
        
        project = ProjectInfo(
            name=project_name,
            status="进行中",
            district="朝阳区"
        )
        self.kb.update_project_status(project)
        
        reply = f"经核实，{project_name}已完工验收。"
        hits = [{
            "project_status": "进行中",
            "snippet": f"{project_name}正在建设中"
        }]
        location = {"district": "朝阳区"}
        
        verification = self.verifier.verify_reply(reply, hits, location)
        
        self.assertFalse(verification["is_valid"])
        
        facts = self.extractor.extract_facts_from_reply(reply)
        kb_facts = self.extractor.facts_to_kb_format(facts)
        
        for fact in kb_facts:
            self.kb.add_fact(fact)
        
        updated_project = self.kb.get_project_status(project_name)
        self.assertIsNotNone(updated_project)


if __name__ == "__main__":
    unittest.main()
