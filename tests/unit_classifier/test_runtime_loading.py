from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.jsjb.unit_classifier.runtime import (
    _build_classifier_text,
    _detect_architecture,
    _load_json,
    classifier_status,
)
from src.jsjb.core.artifacts import inspect_classifier_artifacts


class TestRuntimeLoading(unittest.TestCase):
    """模型目录、label_map.json、model_meta.json 是否匹配"""

    def test_detect_architecture_hybrid(self):
        meta = {"architecture": "bert_cnn_attention_tfidf", "use_tfidf": True}
        self.assertEqual(_detect_architecture(meta), "bert_cnn_attention_tfidf")

    def test_build_classifier_text_uses_training_format(self):
        text = _build_classifier_text("环境卫生", "垃圾清运", "小区垃圾桶满溢")
        self.assertEqual(text, "【环境卫生】垃圾清运。小区垃圾桶满溢")

    def test_detect_architecture_legacy(self):
        meta = {"architecture": "", "use_cnn_attention": False}
        self.assertEqual(_detect_architecture(meta), "legacy_bert_sequence")

    def test_detect_architecture_from_flag(self):
        meta = {"architecture": "", "use_cnn_attention": True}
        self.assertEqual(_detect_architecture(meta), "bert_cnn_attention_tfidf")

    def test_load_json_missing_file(self):
        result = _load_json("/nonexistent/path/file.json")
        self.assertEqual(result, {})

    def test_load_json_valid_file(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"key": "value"}, f)
        result = _load_json(path)
        self.assertEqual(result, {"key": "value"})

    def test_load_json_invalid_json(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{invalid json")
        result = _load_json(path)
        self.assertEqual(result, {})

    def test_classifier_status_missing_dir(self):
        status = classifier_status("/nonexistent/model_dir")
        self.assertFalse(status["compatible_runtime_ready"])
        self.assertIn("model_meta", status)

    def test_classifier_status_with_meta(self):
        tmpdir = tempfile.mkdtemp()
        meta_path = os.path.join(tmpdir, "model_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "architecture": "bert_cnn_attention_tfidf",
                "use_tfidf": True,
                "tfidf_dim": 3000,
                "classifier_route": "hybrid_rankaware",
            }, f)
        status = classifier_status(tmpdir)
        self.assertTrue(status["model_meta"]["exists"])
        self.assertEqual(status["model_meta"]["architecture"], "bert_cnn_attention_tfidf")
        self.assertTrue(status["model_meta"]["use_tfidf"])

    def test_inspect_artifacts_missing_dir(self):
        result = inspect_classifier_artifacts("/nonexistent/dir")
        self.assertFalse(result["compatible_runtime_ready"])
        self.assertIn("label_map.json", result["missing"])

    def test_inspect_artifacts_partial_dir(self):
        tmpdir = tempfile.mkdtemp()
        label_map_path = os.path.join(tmpdir, "label_map.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({"0": "环卫中心", "1": "住建委"}, f)
        result = inspect_classifier_artifacts(tmpdir)
        self.assertFalse(result["compatible_runtime_ready"])
        self.assertTrue(any("pytorch_model.bin" in m or "model.safetensors" in m for m in result["missing"]))


if __name__ == "__main__":
    unittest.main()
