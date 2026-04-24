"""
事实验证模块单元测试
"""

import unittest
from enhancements.fact_verifier import FactVerifier, verify_generated_reply


class TestFactVerifier(unittest.TestCase):
    """事实验证器测试"""
    
    def setUp(self):
        """测试初始化"""
        self.verifier = FactVerifier(knowledge_base_path=None)
    
    def test_negation_contradiction(self):
        """测试否定词矛盾检测"""
        reply = "经核实，该区域暂未建设图书馆。"
        hits = [
            {
                "snippet": "该图书馆已建成并投入使用，现已开放借阅服务。",
                "title": "社区图书馆建设情况",
            }
        ]
        location = {"district": "朝阳区"}
        
        result = self.verifier.verify_reply(reply, hits, location)
        
        self.assertFalse(result["is_valid"])
        self.assertTrue(result["needs_review"])
        self.assertGreater(len(result["warnings"]), 0)
        self.assertEqual(result["warnings"][0]["warning_type"], "negation_contradiction")
    
    def test_project_status_conflict(self):
        """测试项目状态冲突检测"""
        reply = "该道路目前正在施工建设中。"
        hits = [
            {
                "snippet": "道路已完工并通车。",
                "project_status": "已完成",
                "title": "道路建设进展",
            }
        ]
        location = {"district": "海淀区"}
        
        result = self.verifier.verify_reply(reply, hits, location)
        
        self.assertFalse(result["is_valid"])
        self.assertTrue(result["needs_review"])
        warnings = [w for w in result["warnings"] if w["warning_type"] == "status_conflict"]
        self.assertGreater(len(warnings), 0)
    
    def test_demolition_status_error(self):
        """测试拆迁状态错误检测"""
        reply = "因拆迁影响，该项目暂时无法推进。"
        hits = [
            {
                "snippet": "拆迁工作已全部完成。",
                "demolition_status": "已完成",
                "title": "拆迁进展通报",
            }
        ]
        location = {"district": "昌平区"}
        
        result = self.verifier.verify_reply(reply, hits, location)
        
        self.assertFalse(result["is_valid"])
        warnings = [w for w in result["warnings"] if w["warning_type"] == "demolition_status_error"]
        self.assertGreater(len(warnings), 0)
    
    def test_responsible_unit_mismatch(self):
        """测试责任单位不一致检测"""
        reply = "经朝阳区城管委核实处理。"
        hits = [
            {
                "snippet": "该问题由海淀区城管委负责。",
                "responsible_unit": "海淀区城管委",
                "title": "责任单位说明",
            }
        ]
        location = {"district": "海淀区"}
        
        result = self.verifier.verify_reply(reply, hits, location)
        
        self.assertFalse(result["is_valid"])
        warnings = [w for w in result["warnings"] if w["warning_type"] == "unit_mismatch"]
        self.assertGreater(len(warnings), 0)
    
    def test_resource_denial_conflict(self):
        """测试资源否认冲突检测"""
        reply = "该社区暂未建设阅览室。"
        hits = [
            {
                "snippet": "社区阅览室已运营两年。",
                "resource_status": "已运营",
                "title": "社区资源情况",
            }
        ]
        location = {"district": "西城区"}
        
        result = self.verifier.verify_reply(reply, hits, location)
        
        self.assertFalse(result["is_valid"])
        warnings = [w for w in result["warnings"] if w["warning_type"] == "resource_denial"]
        self.assertGreater(len(warnings), 0)
    
    def test_high_risk_content(self):
        """测试高风险内容检测"""
        reply = "该区域涉及征地拆迁，属于集团校招生范围。"
        hits = []
        location = {"district": "东城区"}
        
        result = self.verifier.verify_reply(reply, hits, location)
        
        self.assertTrue(result["high_risk"])
        self.assertTrue(result["needs_review"])
    
    def test_valid_reply(self):
        """测试有效回复（无问题）"""
        reply = "经核实，该道路已完工通车，感谢您的关注。"
        hits = [
            {
                "snippet": "道路已完工并通车。",
                "project_status": "已完成",
                "title": "道路建设进展",
            }
        ]
        location = {"district": "丰台区"}
        
        result = self.verifier.verify_reply(reply, hits, location)
        
        self.assertTrue(result["is_valid"])
        self.assertFalse(result["needs_review"])
        self.assertEqual(len(result["warnings"]), 0)
    
    def test_empty_retrieval_hits(self):
        """测试空检索结果"""
        reply = "经核实，将进一步调查处理。"
        hits = []
        location = {"district": "石景山区"}
        
        result = self.verifier.verify_reply(reply, hits, location)
        
        # 空检索结果不应产生警告
        self.assertTrue(result["is_valid"])
        self.assertFalse(result["needs_review"])
    
    def test_severity_counts(self):
        """测试严重程度统计"""
        reply = "该区域暂未建设，因拆迁影响无法推进。"
        hits = [
            {
                "snippet": "已建成运营，拆迁已完成。",
                "project_status": "已完成",
                "demolition_status": "已完成",
                "resource_status": "已运营",
            }
        ]
        location = {"district": "通州区"}
        
        result = self.verifier.verify_reply(reply, hits, location)
        
        self.assertIn("severity_counts", result)
        self.assertGreater(result["severity_counts"]["high"], 0)
    
    def test_verify_generated_reply_function(self):
        """测试便捷函数"""
        reply = "经核实，该项目正在推进中。"
        hits = [{"snippet": "项目已完工。", "project_status": "已完成"}]
        location = {"district": "大兴区"}
        
        result = verify_generated_reply(reply, hits, location)
        
        self.assertIn("is_valid", result)
        self.assertIn("warnings", result)
        self.assertIn("needs_review", result)


class TestPatternExtraction(unittest.TestCase):
    """模式提取测试"""
    
    def setUp(self):
        self.verifier = FactVerifier(knowledge_base_path=None)
    
    def test_extract_negation_patterns(self):
        """测试否定词提取"""
        text = "该区域不涉及征收，暂未建设相关设施。"
        patterns = self.verifier._extract_patterns(text, self.verifier.NEGATION_PATTERNS)
        
        self.assertGreater(len(patterns), 0)
        self.assertTrue(any("不涉及" in p[0] for p in patterns))
    
    def test_extract_affirmation_patterns(self):
        """测试肯定词提取"""
        text = "该项目已完成建设，正在推进验收工作。"
        patterns = self.verifier._extract_patterns(text, self.verifier.AFFIRMATION_PATTERNS)
        
        self.assertGreater(len(patterns), 0)
        self.assertTrue(any("已完成" in p[1] for p in patterns))
    
    def test_extract_units(self):
        """测试单位提取"""
        text = "经朝阳区城管委核实，由望京街道办事处负责处理。"
        units = self.verifier._extract_units(text)
        
        self.assertGreater(len(units), 0)
        self.assertTrue(any("城管委" in u or "街道" in u for u in units))


class TestContradictionDetection(unittest.TestCase):
    """矛盾检测测试"""
    
    def setUp(self):
        self.verifier = FactVerifier(knowledge_base_path=None)
    
    def test_is_contradiction(self):
        """测试矛盾判断"""
        self.assertTrue(self.verifier._is_contradiction("不涉及", "涉及征收"))
        self.assertTrue(self.verifier._is_contradiction("暂未建设", "已建设"))
        self.assertTrue(self.verifier._is_contradiction("非集团校", "属于集团校"))
        self.assertFalse(self.verifier._is_contradiction("正在建设", "已建设"))
    
    def test_is_status_conflict(self):
        """测试状态冲突判断"""
        self.assertTrue(self.verifier._is_status_conflict("未开始", "已完成"))
        self.assertTrue(self.verifier._is_status_conflict("进行中", "运营中"))
        self.assertFalse(self.verifier._is_status_conflict("进行中", "主体施工"))
    
    def test_is_unit_compatible(self):
        """测试单位兼容性判断"""
        self.assertTrue(self.verifier._is_unit_compatible("朝阳区城管委", "城管委"))
        self.assertTrue(self.verifier._is_unit_compatible("城管委", "朝阳区城管委"))
        self.assertFalse(self.verifier._is_unit_compatible("朝阳区城管委", "海淀区城管委"))


if __name__ == "__main__":
    unittest.main()
