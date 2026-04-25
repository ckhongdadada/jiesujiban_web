from __future__ import annotations

import unittest

from src.jsjb.location.rule_based import (
    BeijingDistrictResolver,
    LocationNER,
    DISTRICT_NAMES,
)


class TestNERBasic(unittest.TestCase):
    """常见行政区识别与小区/街道/地标识别"""

    @classmethod
    def setUpClass(cls):
        cls.resolver = BeijingDistrictResolver(enable_lac=False)
        cls.ner = LocationNER(enable_lac=False)

    def test_direct_district_name(self):
        result = self.resolver.resolve("朝阳区望京街道垃圾清运问题")
        self.assertEqual(result["district"], "朝阳区")
        self.assertGreater(result["confidence"], 0.5)

    def test_all_sixteen_districts(self):
        for district in DISTRICT_NAMES:
            result = self.resolver.resolve(f"{district}某小区物业管理问题")
            self.assertEqual(result["district"], district, f"未能识别行政区: {district}")

    def test_district_short_name(self):
        result = self.resolver.resolve("海淀中关村附近噪声扰民")
        self.assertEqual(result["district"], "海淀区")

    def test_community_name(self):
        result = self.resolver.resolve("朝阳区望京西园四区停车问题")
        self.assertEqual(result["district"], "朝阳区")

    def test_landmark_name(self):
        result = self.resolver.resolve("海淀区五道口周边环境问题")
        self.assertEqual(result["district"], "海淀区")

    def test_street_name(self):
        result = self.resolver.resolve("昌平区回龙观街道物业服务问题")
        self.assertEqual(result["district"], "昌平区")

    def test_no_district_mention(self):
        result = self.resolver.resolve("小区门口垃圾堆积如山")
        self.assertIsNone(result["district"])

    def test_multiple_districts_picks_first(self):
        result = self.resolver.resolve("朝阳区与海淀区交界处噪声问题")
        self.assertIn(result["district"], {"朝阳区", "海淀区"})

    def test_ner_extract_district(self):
        result = self.ner.extract_district("丰台区方庄路路灯损坏")
        self.assertEqual(result["district"], "丰台区")

    def test_confidence_direct_district_high(self):
        result = self.resolver.resolve("东城区物业管理问题")
        self.assertGreaterEqual(result["confidence"], 0.9)

    def test_result_has_expected_keys(self):
        result = self.resolver.resolve("西城区金融街停车问题")
        for key in ("district", "confidence", "method", "places", "districts", "categories"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
