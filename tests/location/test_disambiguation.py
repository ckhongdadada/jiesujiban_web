from __future__ import annotations

import unittest

from src.jsjb.location.disambiguation import (
    ContextFeatureExtractor,
    DisambiguationResult,
    GeoCodingValidator,
    LocationCandidate,
    LocationDisambiguator,
    MultiSourceFusion,
    disambiguate_location,
)


class TestDisambiguationSingleCandidate(unittest.TestCase):
    def test_single_candidate_returns_directly(self):
        disambiguator = LocationDisambiguator(enable_context_disambiguation=False, enable_geo_validation=False)
        candidates = [
            LocationCandidate(matched_text="朝阳区", district="朝阳区", source="ner", confidence=0.95)
        ]
        result = disambiguator.disambiguate("朝阳区望京停车问题", "朝阳区", candidates)
        self.assertIsNotNone(result)
        self.assertEqual(result.district, "朝阳区")
        self.assertEqual(result.disambiguation_method, "single_candidate")


class TestDisambiguationSingleDistrict(unittest.TestCase):
    def test_same_district_multiple_sources(self):
        disambiguator = LocationDisambiguator(enable_context_disambiguation=False, enable_geo_validation=False)
        candidates = [
            LocationCandidate(matched_text="朝阳区", district="朝阳区", source="ner", confidence=0.95),
            LocationCandidate(matched_text="望京", district="朝阳区", source="alias", confidence=0.85),
        ]
        result = disambiguator.disambiguate("朝阳区望京停车问题", "朝阳区", candidates)
        self.assertIsNotNone(result)
        self.assertEqual(result.district, "朝阳区")
        self.assertEqual(result.disambiguation_method, "single_district")


class TestDisambiguationMultiSourceFusion(unittest.TestCase):
    def test_multi_district_fusion(self):
        disambiguator = LocationDisambiguator(enable_context_disambiguation=False, enable_geo_validation=False)
        candidates = [
            LocationCandidate(matched_text="朝阳区", district="朝阳区", source="ner", confidence=0.95),
            LocationCandidate(matched_text="海淀区", district="海淀区", source="alias", confidence=0.80),
        ]
        result = disambiguator.disambiguate("朝阳区与海淀区交界处噪声问题", "朝阳区", candidates)
        self.assertIsNotNone(result)
        self.assertIn(result.district, {"朝阳区", "海淀区"})
        self.assertEqual(result.disambiguation_method, "multi_source_fusion")


class TestContextFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = ContextFeatureExtractor()

    def test_district_keyword_extraction(self):
        features = self.extractor.extract_features("望京CBD停车问题", "望京")
        self.assertIn("district_keyword_朝阳区", features)
        self.assertGreater(features["district_keyword_朝阳区"], 0)

    def test_context_score_calculation(self):
        score = self.extractor.calculate_context_score("望京CBD停车问题", "望京", "朝阳区")
        self.assertGreater(score, 0)

    def test_no_keyword_low_score(self):
        score = self.extractor.calculate_context_score("随机文本无关键词", "某地", "密云区")
        self.assertLess(score, 0.5)


class TestGeoCodingValidator(unittest.TestCase):
    def setUp(self):
        self.validator = GeoCodingValidator(amap_api_key=None)

    def test_district_centers_defined(self):
        for district in [
            "东城区", "西城区", "朝阳区", "海淀区", "丰台区",
            "石景山区", "通州区", "顺义区", "昌平区", "大兴区",
        ]:
            self.assertIn(district, self.validator.district_centers)

    def test_is_within_district(self):
        self.assertTrue(self.validator.is_within_district((116.418, 39.928), "东城区"))
        self.assertFalse(self.validator.is_within_district((116.656, 39.909), "东城区"))

    def test_calculate_geo_score_no_api(self):
        score = self.validator.calculate_geo_score("望京", "朝阳区")
        self.assertEqual(score, 0.5)


class TestMultiSourceFusion(unittest.TestCase):
    def setUp(self):
        self.fusion = MultiSourceFusion()

    def test_empty_candidates_returns_none(self):
        result = self.fusion.fuse_results([])
        self.assertIsNone(result)

    def test_fusion_picks_highest_score(self):
        candidates = [
            LocationCandidate(matched_text="朝阳区", district="朝阳区", source="direct_match", confidence=0.99),
            LocationCandidate(matched_text="海淀区", district="海淀区", source="alias_match", confidence=0.60),
        ]
        result = self.fusion.fuse_results(candidates)
        self.assertIsNotNone(result)
        self.assertEqual(result.district, "朝阳区")


class TestDisambiguateLocationConvenience(unittest.TestCase):
    def test_convenience_function(self):
        candidates = [
            {"matched_text": "朝阳区", "district": "朝阳区", "source": "ner", "confidence": 0.95},
        ]
        result = disambiguate_location("朝阳区望京停车问题", "朝阳区", candidates)
        self.assertIsNotNone(result)
        self.assertEqual(result["district"], "朝阳区")

    def test_empty_candidates_returns_none(self):
        result = disambiguate_location("测试文本", "某地", [])
        self.assertIsNone(result)


class TestDisambiguatorStats(unittest.TestCase):
    def test_stats_initial(self):
        disambiguator = LocationDisambiguator(enable_context_disambiguation=False, enable_geo_validation=False)
        stats = disambiguator.get_stats()
        self.assertEqual(stats["total_disambiguations"], 0)

    def test_stats_after_disambiguation(self):
        disambiguator = LocationDisambiguator(enable_context_disambiguation=False, enable_geo_validation=False)
        candidates = [
            LocationCandidate(matched_text="朝阳区", district="朝阳区", source="ner", confidence=0.95),
        ]
        disambiguator.disambiguate("朝阳区停车问题", "朝阳区", candidates)
        stats = disambiguator.get_stats()
        self.assertEqual(stats["total_disambiguations"], 1)


if __name__ == "__main__":
    unittest.main()
