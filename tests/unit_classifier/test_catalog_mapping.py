from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.jsjb.unit_classifier.catalog import (
    build_catalog_from_frames,
    canonicalize_unit,
    load_unit_catalog,
    normalize_unit_text,
)


class TestCatalogMapping(unittest.TestCase):
    """政府/街道办、历史单位别名的归一化映射"""

    def test_normalize_government_variants(self):
        self.assertEqual(normalize_unit_text("人民政府办公室"), "政府办")

    def test_normalize_street_variants(self):
        self.assertEqual(normalize_unit_text("街道办事处"), "街道办")

    def test_normalize_district_office(self):
        self.assertEqual(normalize_unit_text("地区办事处"), "地区办")

    def test_normalize_market_supervision(self):
        self.assertEqual(normalize_unit_text("市场监督管理局"), "市场监管局")

    def test_normalize_urban_management(self):
        self.assertEqual(normalize_unit_text("城市管理委员会"), "城管委")

    def test_normalize_urban_enforcement(self):
        self.assertEqual(normalize_unit_text("城市管理综合行政执法局"), "城管执法局")

    def test_normalize_hotline_variants(self):
        self.assertEqual(normalize_unit_text("政务服务便民热线"), "12345")

    def test_canonicalize_with_alias_mapping(self):
        catalog = {
            "alias_to_unit": {
                "朝阳区政府办": "政府办",
                "东城街道办": "街道办",
            },
            "unit_meta": {
                "政府办": {"districts": [], "category": "department", "top_tags": [], "aliases": ["朝阳区政府办"]},
                "街道办": {"districts": [], "category": "town_or_subdistrict", "top_tags": [], "aliases": ["东城街道办"]},
            },
        }
        self.assertEqual(canonicalize_unit("朝阳区政府办", catalog=catalog), "政府办")
        self.assertEqual(canonicalize_unit("东城街道办", catalog=catalog), "街道办")


class TestLoadUnitCatalog(unittest.TestCase):
    def test_load_missing_catalog(self):
        catalog = load_unit_catalog(path="/nonexistent/catalog.json")
        self.assertEqual(catalog, {"alias_to_unit": {}, "unit_meta": {}})

    def test_load_valid_catalog(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "unit_catalog.json")
        data = {
            "alias_to_unit": {"环卫": "环卫中心"},
            "unit_meta": {"环卫中心": {"districts": ["朝阳区"]}},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        catalog = load_unit_catalog(path=path)
        self.assertIn("环卫", catalog["alias_to_unit"])
        self.assertEqual(catalog["alias_to_unit"]["环卫"], "环卫中心")


class TestBuildCatalogFromFrames(unittest.TestCase):
    def test_build_from_empty_frames(self):
        import pandas as pd
        mapping_df = pd.DataFrame()
        master_df = pd.DataFrame()
        catalog = build_catalog_from_frames(mapping_df, master_df)
        self.assertIn("alias_to_unit", catalog)
        self.assertIn("unit_meta", catalog)

    def test_build_from_mapping_df(self):
        import pandas as pd
        mapping_df = pd.DataFrame({
            "normalized_unit": ["环卫中心", "住建委"],
            "unit_category": ["department", "department"],
            "sample_count": [100, 50],
            "source_type": ["training_data | manual", "training_data"],
            "district_examples": ["朝阳区 | 海淀区", "西城区"],
            "raw_unit_examples": ["环卫 | 环卫中心", "住建委"],
        })
        master_df = pd.DataFrame()
        catalog = build_catalog_from_frames(mapping_df, master_df)
        self.assertIn("环卫中心", catalog["unit_meta"])
        self.assertIn("住建委", catalog["unit_meta"])
        self.assertEqual(catalog["unit_meta"]["环卫中心"]["category"], "department")

    def test_build_from_master_df(self):
        import pandas as pd
        mapping_df = pd.DataFrame()
        master_df = pd.DataFrame({
            "reply_unit_norm": ["环卫中心", "住建委", "环卫中心"],
            "reply_unit_raw": ["朝阳区环卫中心", "住建委", "环卫"],
            "district_from_file": ["朝阳区", "海淀区", "丰台区"],
            "message_tag_level1": ["投诉", "建议", "投诉"],
        })
        catalog = build_catalog_from_frames(mapping_df, master_df)
        self.assertIn("环卫中心", catalog["unit_meta"])
        meta = catalog["unit_meta"]["环卫中心"]
        self.assertEqual(meta["sample_count"], 2)
        self.assertIn("朝阳区", meta["districts"])
        self.assertIn("丰台区", meta["districts"])


if __name__ == "__main__":
    unittest.main()
