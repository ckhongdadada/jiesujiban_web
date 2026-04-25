"""
结构化知识库单元测试
"""

import unittest
import json
from tests import BaseTestCase
from src.jsjb.knowledge.structured_kb import (
    StructuredKnowledgeBase, ProjectInfo, PublicResource, UnitMapping
)


class TestStructuredKnowledgeBase(BaseTestCase):
    """结构化知识库测试"""
    
    def setUp(self):
        """每个测试前的初始化"""
        self.kb_path = self.test_data_dir / "test_kb.json"
        self.kb = StructuredKnowledgeBase(str(self.kb_path))
    
    def test_add_project_status(self):
        """测试添加项目状态"""
        project = ProjectInfo(
            name="测试道路工程",
            status="进行中",
            demolition_status="已完成",
            responsible_unit="测试单位",
            district="朝阳区"
        )
        
        self.kb.update_project_status(project)
        
        retrieved = self.kb.get_project_status("测试道路工程")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.status, "进行中")
        self.assertEqual(retrieved.demolition_status, "已完成")
    
    def test_add_public_resource(self):
        """测试添加公共资源"""
        resource = PublicResource(
            name="测试图书馆",
            resource_type="library",
            status="已运营",
            count=1,
            district="海淀区"
        )
        
        self.kb.update_public_resource(resource)
        
        retrieved = self.kb.get_public_resource("测试图书馆")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.status, "已运营")
        self.assertEqual(retrieved.resource_type, "library")
    
    def test_add_unit_mapping(self):
        """测试添加单位映射"""
        mapping = UnitMapping(
            project_name="测试项目",
            district="通州区",
            street="测试街道",
            responsible_unit="测试街道办"
        )
        
        self.kb.update_unit_mapping(mapping)
        
        retrieved = self.kb.get_unit_mapping("测试项目")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.responsible_unit, "测试街道办")
    
    def test_search_projects(self):
        """测试项目搜索"""
        project = ProjectInfo(
            name="鲁疃西路南延",
            status="已完工",
            district="昌平区"
        )
        self.kb.update_project_status(project)
        
        results = self.kb.search_projects("鲁疃", district="昌平区")
        
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].name, "鲁疃西路南延")
    
    def test_search_resources(self):
        """测试资源搜索"""
        resource = PublicResource(
            name="测试图书室",
            resource_type="library",
            status="已运营",
            district="大兴区"
        )
        self.kb.update_public_resource(resource)
        
        results = self.kb.search_resources("library", district="大兴区")
        
        self.assertTrue(len(results) > 0)
    
    def test_export_to_rag_format(self):
        """测试导出RAG格式"""
        project = ProjectInfo(
            name="测试项目",
            status="已完工",
            district="朝阳区"
        )
        self.kb.update_project_status(project)
        
        documents = self.kb.export_to_rag_format()
        
        self.assertTrue(len(documents) > 0)
        
        project_doc = next((d for d in documents if "测试项目" in d.get("title", "")), None)
        self.assertIsNotNone(project_doc)
        self.assertEqual(project_doc["doc_type"], "project_status")
    
    def test_add_fact(self):
        """测试添加事实"""
        fact = {
            "type": "project_status",
            "project": "事实测试项目",
            "status": "规划中",
            "district": "海淀区"
        }
        
        success = self.kb.add_fact(fact)
        
        self.assertTrue(success)
        
        retrieved = self.kb.get_project_status("事实测试项目")
        self.assertIsNotNone(retrieved)
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        stats = self.kb.get_statistics()
        
        self.assertIn("project_count", stats)
        self.assertIn("resource_count", stats)
        self.assertIn("unit_mapping_count", stats)
    
    def test_persistence(self):
        """测试持久化"""
        project = ProjectInfo(
            name="持久化测试项目",
            status="进行中",
            district="丰台区"
        )
        self.kb.update_project_status(project)
        
        new_kb = StructuredKnowledgeBase(str(self.kb_path))
        retrieved = new_kb.get_project_status("持久化测试项目")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.status, "进行中")


if __name__ == "__main__":
    unittest.main()
