from __future__ import annotations

import unittest

from src.jsjb.unit_classifier.catalog import (
    canonicalize_unit,
    infer_unit_district,
    normalize_unit_text,
    score_unit_candidate,
    GENERIC_BAD_UNITS,
    DISTRICT_NAMES,
)


class TestTopKOutput(unittest.TestCase):
    """TopK 概率有序 & 单位规范化正确"""

    def test_normalize_basic(self):
        self.assertEqual(normalize_unit_text("  环卫中心  "), "环卫中心")

    def test_normalize_strip_at(self):
        self.assertEqual(normalize_unit_text("@北京12345"), "北京12345")

    def test_normalize_trailing_punctuation(self):
        self.assertEqual(normalize_unit_text("住建委，"), "住建委")

    def test_normalize_fullwidth_parens(self):
        self.assertEqual(normalize_unit_text("住建委（回复）"), "住建委")

    def test_normalize_housing_committee(self):
        self.assertEqual(normalize_unit_text("住房和城乡建设委员会"), "住建委")

    def test_normalize_health_committee(self):
        self.assertEqual(normalize_unit_text("卫生健康委员会"), "卫健委")

    def test_normalize_street_office(self):
        self.assertEqual(normalize_unit_text("街道办事处"), "街道办")

    def test_normalize_hotline(self):
        self.assertEqual(normalize_unit_text("北京12345热线"), "北京12345")

    def test_normalize_empty(self):
        self.assertEqual(normalize_unit_text(""), "")
        self.assertEqual(normalize_unit_text(None), "")

    def test_normalize_reply_unit_prefix(self):
        self.assertEqual(normalize_unit_text("回复单位：环卫中心"), "环卫中心")


class TestCanonicalizeUnit(unittest.TestCase):
    def test_canonicalize_empty(self):
        self.assertEqual(canonicalize_unit(""), "")
        self.assertEqual(canonicalize_unit(None), "")

    def test_canonicalize_bad_unit(self):
        result = canonicalize_unit("认领交办")
        self.assertEqual(result, "")

    def test_canonicalize_with_catalog(self):
        catalog = {
            "alias_to_unit": {"环卫": "环卫中心"},
            "unit_meta": {"环卫中心": {"districts": ["朝阳区"], "category": "department"}},
        }
        result = canonicalize_unit("环卫", catalog=catalog)
        self.assertEqual(result, "环卫中心")


class TestInferUnitDistrict(unittest.TestCase):
    def test_infer_from_catalog(self):
        catalog = {
            "alias_to_unit": {},
            "unit_meta": {"环卫中心": {"districts": ["朝阳区", "海淀区"]}},
        }
        result = infer_unit_district("环卫中心", catalog=catalog)
        self.assertEqual(result, "朝阳区")

    def test_infer_from_name(self):
        result = infer_unit_district("朝阳区环卫中心")
        self.assertEqual(result, "朝阳区")

    def test_infer_empty(self):
        result = infer_unit_district("")
        self.assertEqual(result, "")


class TestScoreUnitCandidate(unittest.TestCase):
    def test_bad_unit_negative_score(self):
        score = score_unit_candidate("认领交办")
        self.assertLess(score, 0)

    def test_empty_unit_negative_score(self):
        score = score_unit_candidate("")
        self.assertLess(score, 0)

    def test_district_match_positive(self):
        catalog = {
            "alias_to_unit": {},
            "unit_meta": {"环卫中心": {"districts": ["朝阳区"], "category": "department", "top_tags": []}},
        }
        score = score_unit_candidate("环卫中心", district="朝阳区", catalog=catalog)
        self.assertGreater(score, 0)

    def test_district_mismatch_penalty(self):
        catalog = {
            "alias_to_unit": {},
            "unit_meta": {"环卫中心": {"districts": ["海淀区"], "category": "department", "top_tags": []}},
        }
        score = score_unit_candidate("环卫中心", district="朝阳区", catalog=catalog)
        self.assertLess(score, 0)

    def test_tag_match_bonus(self):
        catalog = {
            "alias_to_unit": {},
            "unit_meta": {"环卫中心": {"districts": [], "category": "department", "top_tags": ["投诉"]}},
        }
        score_with_tag = score_unit_candidate("环卫中心", tag="投诉", catalog=catalog)
        score_without_tag = score_unit_candidate("环卫中心", catalog=catalog)
        self.assertGreater(score_with_tag, score_without_tag)


if __name__ == "__main__":
    unittest.main()
