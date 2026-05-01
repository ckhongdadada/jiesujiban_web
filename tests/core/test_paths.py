from __future__ import annotations

import os
import tempfile
import unittest

from src.jsjb.core.paths import (
    get_project_root,
    get_runtime_dir,
    get_raw_dir,
    get_processed_dir,
    get_reports_dir,
    get_runtime_catalog_path,
    get_runtime_district_file,
    get_policy_corpus_path,
    get_training_reports_dir,
)


class TestPathResolution(unittest.TestCase):
    """测试路径解析函数"""

    def test_get_project_root(self):
        """测试获取项目根目录"""
        root = get_project_root()
        self.assertTrue(os.path.isdir(root))
        self.assertIn("接诉即办项目", str(root))

    def test_get_runtime_dir(self):
        """测试获取运行时目录"""
        runtime_dir = get_runtime_dir()
        self.assertTrue(os.path.isdir(runtime_dir) or runtime_dir.parent.exists())

    def test_get_raw_dir(self):
        """测试获取原始数据目录"""
        raw_dir = get_raw_dir()
        self.assertTrue(os.path.isdir(raw_dir) or raw_dir.parent.exists())

    def test_get_processed_dir(self):
        """测试获取处理后数据目录"""
        processed_dir = get_processed_dir()
        self.assertTrue(os.path.isdir(processed_dir) or processed_dir.parent.exists())

    def test_get_reports_dir(self):
        """测试获取报告目录"""
        reports_dir = get_reports_dir()
        self.assertTrue(os.path.isdir(reports_dir) or reports_dir.parent.exists())

    def test_runtime_district_file(self):
        """测试获取区县文件路径"""
        path = get_runtime_district_file("beijing_districts_merged.json")
        self.assertTrue(path.exists() or path.parent.exists())

    def test_policy_corpus_path(self):
        """测试获取政策语料路径"""
        path = get_policy_corpus_path()
        self.assertTrue(path.parent.exists())

    def test_training_reports_dir(self):
        """测试获取训练报告目录"""
        path = get_training_reports_dir()
        self.assertTrue(os.path.isdir(path) or path.parent.exists())


if __name__ == "__main__":
    unittest.main()