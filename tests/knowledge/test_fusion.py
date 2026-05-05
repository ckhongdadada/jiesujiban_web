"""
信息融合引擎自动化测试
覆盖：多源融合、冲突检测、置信度计算、状态冲突解决
"""

import unittest

from src.jsjb.knowledge.fusion import InformationFusion, FusedFact, ConflictInfo


class TestInformationFusion(unittest.TestCase):
    """信息融合引擎测试"""

    def setUp(self):
        self.fusion = InformationFusion()

    def test_single_source_no_conflict(self):
        sources = {
            "government_website": [
                {"project": "道路改造", "status": "进行中", "responsible_unit": "住建委"}
            ]
        }
        result = self.fusion.fuse_multi_source_info(sources)

        self.assertIn("fused_facts", result)
        self.assertIn("statistics", result)
        self.assertEqual(result["statistics"]["conflict_count"], 0)

        fact = result["fused_facts"]["道路改造_status"]
        self.assertEqual(fact["value"], "进行中")
        self.assertFalse(fact["has_conflict"])
        self.assertGreater(fact["confidence"], 0)

    def test_multi_source_consistent(self):
        sources = {
            "government_website": [
                {"project": "排水工程", "status": "进行中", "responsible_unit": "水务局"}
            ],
            "user_verified": [
                {"project": "排水工程", "status": "进行中", "responsible_unit": "水务局"}
            ]
        }
        result = self.fusion.fuse_multi_source_info(sources)

        fact = result["fused_facts"]["排水工程_status"]
        self.assertFalse(fact["has_conflict"])
        self.assertEqual(fact["value"], "进行中")
        self.assertTrue(len(fact["sources"]) >= 2)

    def test_multi_source_status_conflict(self):
        sources = {
            "government_website": [
                {"project": "老旧小区改造", "status": "进行中"}
            ],
            "social_media": [
                {"project": "老旧小区改造", "status": "已完成"}
            ]
        }
        result = self.fusion.fuse_multi_source_info(sources)

        fact = result["fused_facts"]["老旧小区改造_status"]
        self.assertTrue(fact["has_conflict"])
        self.assertIn(fact["value"], ["进行中", "已完成"])
        self.assertGreater(fact["confidence"], 0)

        self.assertTrue(len(result["conflicts"]) > 0)

    def test_status_resolution_by_order(self):
        sources = {
            "social_media": [
                {"project": "工程A", "status": "规划中"}
            ],
            "government_website": [
                {"project": "工程A", "status": "主体施工"}
            ]
        }
        result = self.fusion.fuse_multi_source_info(sources)

        fact = result["fused_facts"]["工程A_status"]
        self.assertTrue(fact["has_conflict"])
        self.assertEqual(fact["value"], "主体施工")

    def test_responsible_unit_conflict(self):
        sources = {
            "government_website": [
                {"project": "项目B", "responsible_unit": "住建委"}
            ],
            "news_api": [
                {"project": "项目B", "responsible_unit": "城管委"}
            ]
        }
        result = self.fusion.fuse_multi_source_info(sources)

        fact = result["fused_facts"]["项目B_responsible_unit"]
        self.assertTrue(fact["has_conflict"])
        self.assertEqual(fact["value"], "住建委")

    def test_source_weights(self):
        self.assertEqual(self.fusion.SOURCE_WEIGHTS["government_website"], 1.0)
        self.assertEqual(self.fusion.SOURCE_WEIGHTS["official_announcement"], 0.95)
        self.assertEqual(self.fusion.SOURCE_WEIGHTS["user_verified"], 0.90)
        self.assertEqual(self.fusion.SOURCE_WEIGHTS["news_api"], 0.70)
        self.assertEqual(self.fusion.SOURCE_WEIGHTS["social_media"], 0.50)
        self.assertEqual(self.fusion.SOURCE_WEIGHTS["unknown"], 0.30)

    def test_unknown_source_weight(self):
        sources = {
            "unknown_source": [
                {"project": "项目C", "status": "进行中"}
            ]
        }
        result = self.fusion.fuse_multi_source_info(sources)

        fact = result["fused_facts"]["项目C_status"]
        self.assertLessEqual(fact["confidence"], 0.5)

    def test_empty_sources(self):
        result = self.fusion.fuse_multi_source_info({})
        self.assertEqual(result["statistics"]["total_entities"], 0)
        self.assertEqual(result["statistics"]["conflict_count"], 0)

    def test_multiple_entities(self):
        sources = {
            "government_website": [
                {"project": "项目X", "status": "进行中"},
                {"project": "项目Y", "status": "已完成"},
                {"project": "项目Z", "status": "规划中"}
            ]
        }
        result = self.fusion.fuse_multi_source_info(sources)

        self.assertEqual(result["statistics"]["total_entities"], 3)
        self.assertIn("项目X_status", result["fused_facts"])
        self.assertIn("项目Y_status", result["fused_facts"])
        self.assertIn("项目Z_status", result["fused_facts"])

    def test_conflict_info_structure(self):
        sources = {
            "government_website": [
                {"project": "工程D", "status": "进行中"}
            ],
            "social_media": [
                {"project": "工程D", "status": "已完成"}
            ]
        }
        result = self.fusion.fuse_multi_source_info(sources)

        self.assertTrue(len(result["conflicts"]) > 0)
        conflict = result["conflicts"][0]
        self.assertIn("entity_name", conflict)
        self.assertIn("fact_type", conflict)
        self.assertIn("conflicting_values", conflict)
        self.assertIn("resolution", conflict)

    def test_statistics_structure(self):
        sources = {
            "government_website": [
                {"project": "A", "status": "进行中", "responsible_unit": "X"}
            ]
        }
        result = self.fusion.fuse_multi_source_info(sources)

        stats = result["statistics"]
        self.assertIn("total_entities", stats)
        self.assertIn("total_facts", stats)
        self.assertIn("conflict_count", stats)

    def test_demolition_status_fusion(self):
        sources = {
            "government_website": [
                {"project": "腾退项目", "demolition_status": "已拆除"}
            ],
            "user_verified": [
                {"project": "腾退项目", "demolition_status": "已拆除"}
            ]
        }
        result = self.fusion.fuse_multi_source_info(sources)

        fact = result["fused_facts"]["腾退项目_demolition_status"]
        self.assertEqual(fact["value"], "已拆除")
        self.assertFalse(fact["has_conflict"])


class TestFusedFactDataClass(unittest.TestCase):
    """FusedFact 数据类测试"""

    def test_create_fused_fact(self):
        fact = FusedFact(
            entity_name="项目A",
            fact_type="status",
            value="进行中",
            sources=["gov"],
            confidence=0.9
        )
        self.assertEqual(fact.entity_name, "项目A")
        self.assertEqual(fact.value, "进行中")
        self.assertEqual(len(fact.conflicts), 0)
        self.assertTrue(fact.fusion_time)

    def test_fused_fact_with_conflicts(self):
        fact = FusedFact(
            entity_name="项目B",
            fact_type="status",
            value="进行中",
            sources=["gov", "media"],
            confidence=0.7,
            conflicts=[{"source": "media", "value": "已完成"}]
        )
        self.assertEqual(len(fact.conflicts), 1)


if __name__ == "__main__":
    unittest.main()
