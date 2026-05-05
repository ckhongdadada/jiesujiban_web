"""
实体提取器自动化测试
覆盖：项目、地点、单位、政策、日期、金额、状态提取
"""

import unittest

from src.jsjb.knowledge.entity_extractor import EntityExtractor


class TestEntityExtractorExtraction(unittest.TestCase):
    """实体提取测试"""

    @classmethod
    def setUpClass(cls):
        cls.extractor = EntityExtractor(use_ner=False)

    def _extract(self, text: str, district: str = "") -> dict:
        doc = {"title": text[:50], "content": text, "district": district}
        return self.extractor.extract_from_document(doc)

    def test_extract_project_name(self):
        text = "朝阳区望京街道花家地西里小区外墙脱落整治工程已于2026年3月开工"
        result = self._extract(text, "朝阳区")
        names = [p["name"] for p in result["projects"]]
        self.assertTrue(any("花家地西里" in n for n in names))

    def test_extract_location_district(self):
        text = "海淀区中关村南大街积水严重，影响居民出行"
        result = self._extract(text, "海淀区")
        loc_names = [l["name"] for l in result["locations"]]
        self.assertIn("海淀区", loc_names)

    def test_extract_location_street(self):
        text = "丰台区方庄街道芳古园小区广场舞噪声扰民"
        result = self._extract(text, "丰台区")
        loc_names = [l["name"] for l in result["locations"]]
        self.assertTrue(any("方庄街道" in n for n in loc_names))

    def test_extract_organization(self):
        text = "朝阳区住建委已将该项目列入2026年改造计划，由朝阳区房管局负责实施"
        result = self._extract(text, "朝阳区")
        org_names = [o["name"] for o in result["organizations"]]
        self.assertTrue(any("住建委" in n for n in org_names))

    def test_extract_amount(self):
        text = "预算约3500万元，工期18个月"
        result = self._extract(text)
        self.assertTrue(len(result["amounts"]) > 0)
        self.assertTrue(any("3500" in a for a in result["amounts"]))

    def test_extract_date(self):
        text = "该项目预计2026年6月开工，2027年12月完工"
        result = self._extract(text)
        self.assertTrue(len(result["dates"]) >= 2)

    def test_extract_phone(self):
        text = "联系电话：010-65299999，如有问题请拨打电话：010-85712345"
        result = self._extract(text)
        self.assertTrue(len(result["contacts"]) >= 1)

    def test_extract_phone_no_prefix(self):
        text = "请拨打65299999咨询"
        result = self._extract(text)
        self.assertEqual(len(result["contacts"]), 0)

    def test_extract_policy(self):
        text = "根据《北京市排水条例》相关规定，市水务局负责排水管理"
        result = self._extract(text)
        titles = [p["title"] for p in result["policies"]]
        self.assertTrue(any("排水条例" in t for t in titles))

    def test_extract_relationships(self):
        text = "朝阳区住建委负责花家地西里小区改造工程，该项目位于朝阳区"
        result = self._extract(text, "朝阳区")
        self.assertTrue(len(result["relationships"]) > 0)

    def test_empty_text(self):
        result = self._extract("")
        self.assertEqual(len(result["projects"]), 0)
        self.assertEqual(len(result["locations"]), 0)
        self.assertEqual(len(result["organizations"]), 0)

    def test_complex_real_case(self):
        text = (
            "海淀区上地街道安宁庄路路面破损严重，多处坑洼影响行车安全。"
            "海淀区城市管理委已安排养护单位进行修复，预计2026年5月中旬完工。"
            "根据《北京市城市道路管理办法》，市交通委负责全市道路养护监督。"
            "预算约120万元。"
        )
        result = self._extract(text, "海淀区")
        self.assertTrue(len(result["locations"]) > 0)
        self.assertTrue(len(result["organizations"]) > 0)
        self.assertTrue(len(result["amounts"]) > 0)
        self.assertTrue(len(result["policies"]) > 0)
        self.assertTrue(len(result["dates"]) > 0)


class TestEntityExtractorHelpers(unittest.TestCase):
    """提取器辅助功能测试"""

    @classmethod
    def setUpClass(cls):
        cls.extractor = EntityExtractor(use_ner=False)

    def test_infer_project_type(self):
        self.assertEqual(self.extractor._infer_project_type("道路改造工程"), "道路建设")
        self.assertEqual(self.extractor._infer_project_type("供水设施"), "供水设施")

    def test_infer_org_type(self):
        self.assertEqual(self.extractor._infer_org_type("海淀区住建委"), "住房建设")
        self.assertEqual(self.extractor._infer_org_type("朝阳区公安分局"), "公安管理")

    def test_infer_org_level(self):
        self.assertEqual(self.extractor._infer_org_level("市交通委"), "市级")
        self.assertEqual(self.extractor._infer_org_level("海淀区住建委"), "区级")
        self.assertEqual(self.extractor._infer_org_level("望京街道办"), "街道级")

    def test_deduplicate_relationships(self):
        rels = [
            {"from": "A", "to": "B", "rel_type": "LOCATED_IN"},
            {"from": "A", "to": "B", "rel_type": "LOCATED_IN"},
            {"from": "A", "to": "C", "rel_type": "LOCATED_IN"},
        ]
        unique = self.extractor._deduplicate_relationships(rels)
        self.assertEqual(len(unique), 2)

    def test_beijing_districts_coverage(self):
        self.assertIn("朝阳区", self.extractor.BEIJING_DISTRICTS)
        self.assertIn("海淀区", self.extractor.BEIJING_DISTRICTS)
        self.assertIn("通州区", self.extractor.BEIJING_DISTRICTS)
        self.assertEqual(len(self.extractor.BEIJING_DISTRICTS), 18)


if __name__ == "__main__":
    unittest.main()
