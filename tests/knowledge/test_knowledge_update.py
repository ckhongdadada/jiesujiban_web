"""
知识入库自动化测试
覆盖：StructuredKnowledgeBase add_fact、更新操作、死锁修复验证、持久化
"""

import json
import os
import unittest
from pathlib import Path

from tests import BaseTestCase
from src.jsjb.knowledge.structured_kb import (
    StructuredKnowledgeBase, ProjectInfo, PublicResource, UnitMapping
)


class TestStructuredKBAddFact(BaseTestCase):
    """add_fact 接口测试"""

    def setUp(self):
        self.kb_path = self.test_data_dir / "test_kb_update.json"
        self.kb = StructuredKnowledgeBase(str(self.kb_path))

    def test_add_fact_project_status(self):
        result = self.kb.add_fact({
            "type": "project_status",
            "project": "道路改造",
            "status": "进行中",
            "responsible_unit": "住建委",
            "district": "朝阳区",
            "confidence": 0.9,
        })
        self.assertTrue(result)
        proj = self.kb.get_project_status("道路改造")
        self.assertIsNotNone(proj)
        self.assertEqual(proj.status, "进行中")
        self.assertEqual(proj.responsible_unit, "住建委")

    def test_add_fact_public_resource(self):
        result = self.kb.add_fact({
            "type": "public_resource",
            "name": "社区图书馆",
            "resource_type": "library",
            "status": "已运营",
            "count": 3,
            "district": "海淀区",
        })
        self.assertTrue(result)
        resource = self.kb.get_public_resource("社区图书馆")
        self.assertIsNotNone(resource)
        self.assertEqual(resource.resource_type, "library")
        self.assertEqual(resource.count, 3)

    def test_add_fact_unit_mapping(self):
        result = self.kb.add_fact({
            "type": "unit_mapping",
            "project": "排水工程",
            "district": "通州区",
            "street": "梨园街道",
            "unit": "通州区水务局",
        })
        self.assertTrue(result)
        mapping = self.kb.get_unit_mapping("排水工程")
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.responsible_unit, "通州区水务局")

    def test_add_fact_unknown_type_returns_false(self):
        result = self.kb.add_fact({"type": "unknown_type"})
        self.assertFalse(result)

    def test_add_fact_project_with_all_fields(self):
        result = self.kb.add_fact({
            "type": "project_status",
            "project": "老旧小区综合整治",
            "status": "主体施工",
            "demolition_status": "已拆除",
            "current_phase": "外墙保温",
            "expected_completion": "2027-06",
            "responsible_unit": "朝阳区住建委",
            "district": "朝阳区",
            "street": "望京街道",
            "source": "government_website",
            "confidence": 0.95,
        })
        self.assertTrue(result)
        proj = self.kb.get_project_status("老旧小区综合整治")
        self.assertEqual(proj.demolition_status, "已拆除")
        self.assertEqual(proj.current_phase, "外墙保温")
        self.assertEqual(proj.expected_completion, "2027-06")


class TestStructuredKBNoDeadlock(BaseTestCase):
    """验证 Lock 不再死锁（修复 _save_internal）"""

    def setUp(self):
        self.kb_path = self.test_data_dir / "test_kb_deadlock.json"
        if self.kb_path.exists():
            os.unlink(self.kb_path)
        self.kb = StructuredKnowledgeBase(str(self.kb_path))

    def test_rapid_sequential_updates_no_deadlock(self):
        for i in range(20):
            self.kb.add_fact({
                "type": "project_status",
                "project": f"批量项目{i}",
                "status": "进行中",
                "district": "测试区",
            })
        self.assertTrue(len(self.kb.project_status) >= 20)

    def test_mixed_operations_no_deadlock(self):
        self.kb.update_project_status(ProjectInfo(name="A", status="S", district="D"))
        self.kb.update_public_resource(PublicResource(name="R", resource_type="T", status="S", count=1, district="D"))
        self.kb.update_unit_mapping(UnitMapping(project_name="A", district="D", street="", responsible_unit="U"))
        self.kb.save()

        self.assertIsNotNone(self.kb.get_project_status("A"))
        self.assertIsNotNone(self.kb.get_public_resource("R"))
        self.assertIsNotNone(self.kb.get_unit_mapping("A"))

    def test_save_then_update_no_deadlock(self):
        self.kb.save()
        self.kb.add_fact({
            "type": "project_status",
            "project": "save_then_add",
            "status": "规划中",
        })
        self.kb.save()
        self.assertIsNotNone(self.kb.get_project_status("save_then_add"))


class TestStructuredKBPersistence(BaseTestCase):
    """持久化与重载测试"""

    def test_update_survives_reload(self):
        kb_path = self.test_data_dir / "test_kb_persist.json"
        kb = StructuredKnowledgeBase(str(kb_path))
        kb.add_fact({
            "type": "project_status",
            "project": "持久化项目",
            "status": "已完成",
            "district": "昌平区",
        })
        kb.save()

        kb2 = StructuredKnowledgeBase(str(kb_path))
        proj = kb2.get_project_status("持久化项目")
        self.assertIsNotNone(proj)
        self.assertEqual(proj.status, "已完成")

    def test_multiple_facts_persist(self):
        kb_path = self.test_data_dir / "test_kb_multi.json"
        kb = StructuredKnowledgeBase(str(kb_path))

        kb.add_fact({"type": "project_status", "project": "P1", "status": "S1"})
        kb.add_fact({"type": "project_status", "project": "P2", "status": "S2"})
        kb.add_fact({"type": "public_resource", "name": "R1", "resource_type": "lib", "status": "open", "count": 1})
        kb.add_fact({"type": "unit_mapping", "project": "P1", "unit": "U1"})

        kb2 = StructuredKnowledgeBase(str(kb_path))
        self.assertIsNotNone(kb2.get_project_status("P1"))
        self.assertIsNotNone(kb2.get_project_status("P2"))
        self.assertIsNotNone(kb2.get_public_resource("R1"))
        self.assertIsNotNone(kb2.get_unit_mapping("P1"))


class TestStructuredKBSearch(BaseTestCase):
    """搜索功能测试"""

    def setUp(self):
        self.kb_path = self.test_data_dir / "test_kb_search.json"
        if self.kb_path.exists():
            os.unlink(self.kb_path)
        self.kb = StructuredKnowledgeBase(str(self.kb_path))
        self.kb.add_fact({"type": "project_status", "project": "朝阳区排水改造", "status": "进行中", "district": "朝阳区"})
        self.kb.add_fact({"type": "project_status", "project": "海淀区道路修缮", "status": "已完成", "district": "海淀区"})
        self.kb.add_fact({"type": "project_status", "project": "朝阳区老旧小区改造", "status": "规划中", "district": "朝阳区"})

    def test_search_by_name(self):
        results = self.kb.search_projects("排水")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "朝阳区排水改造")

    def test_search_by_district(self):
        results = self.kb.search_projects("", district="朝阳区")
        self.assertEqual(len(results), 2)

    def test_search_by_name_and_district(self):
        results = self.kb.search_projects("老旧小区", district="朝阳区")
        self.assertEqual(len(results), 1)

    def test_search_no_match(self):
        results = self.kb.search_projects("不存在")
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
