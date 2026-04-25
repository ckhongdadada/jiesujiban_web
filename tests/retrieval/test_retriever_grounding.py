from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.jsjb.retrieval.bge_retriever import PolicyRetriever
from src.jsjb.reply_generation.qwen_lora import _grounding_strength


def _write_sample_corpus(path: str, docs: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


SAMPLE_DOCS = [
    {
        "id": "g001",
        "title": "朝阳区垃圾清运案例",
        "doc_type": "案例",
        "district": "朝阳区",
        "source": "测试",
        "content": "朝阳区望京垃圾清运不及时",
        "tags": ["环境卫生"],
        "issue_type": "垃圾清运",
        "unit": "环卫中心",
        "applicable_tags": ["投诉"],
    },
    {
        "id": "g002",
        "title": "海淀区噪声扰民案例",
        "doc_type": "案例",
        "district": "海淀区",
        "source": "测试",
        "content": "海淀区中关村噪声扰民",
        "tags": ["噪声"],
        "issue_type": "噪声扰民",
        "unit": "住建委",
        "applicable_tags": ["投诉"],
    },
]


class TestRetrieverGrounding(unittest.TestCase):
    """命中弱证据时是否进入保守路径"""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.corpus_path = os.path.join(cls.tmpdir, "test_corpus.jsonl")
        _write_sample_corpus(cls.corpus_path, SAMPLE_DOCS)
        cls.retriever = PolicyRetriever(
            corpus_path=cls.corpus_path,
            backend="tfidf",
            enable_query_rewrite=False,
        )

    def test_empty_hits_returns_none(self):
        strength = _grounding_strength([], district="朝阳区")
        self.assertEqual(strength, "none")

    def test_cross_district_no_semantic_terms_is_weak(self):
        hits = [
            {
                "title": "海淀区噪声案例",
                "score": 0.5,
                "district": "海淀区",
                "matched_terms": ["海淀区"],
            }
        ]
        strength = _grounding_strength(hits, district="朝阳区")
        self.assertIn(strength, {"weak", "none"})

    def test_same_district_with_semantic_terms(self):
        hits = [
            {
                "title": "朝阳区垃圾清运案例",
                "score": 0.95,
                "district": "朝阳区",
                "matched_terms": ["朝阳区", "垃圾清运"],
            }
        ]
        strength = _grounding_strength(hits, district="朝阳区")
        self.assertIn(strength, {"strong", "medium"})

    def test_citywide_hit_with_semantic_terms(self):
        hits = [
            {
                "title": "全市停车秩序政策",
                "score": 0.80,
                "district": "北京市",
                "matched_terms": ["停车秩序", "北京市"],
            }
        ]
        strength = _grounding_strength(hits, district="朝阳区")
        self.assertIn(strength, {"medium", "weak"})

    def test_low_score_is_weak(self):
        hits = [
            {
                "title": "弱相关文档",
                "score": 0.3,
                "district": "朝阳区",
                "matched_terms": ["朝阳区"],
            }
        ]
        strength = _grounding_strength(hits, district="朝阳区")
        self.assertEqual(strength, "weak")


class TestRetrieverEmptyCorpus(unittest.TestCase):
    def test_empty_corpus_returns_empty(self):
        tmpdir = tempfile.mkdtemp()
        corpus_path = os.path.join(tmpdir, "empty.jsonl")
        with open(corpus_path, "w", encoding="utf-8") as f:
            pass
        retriever = PolicyRetriever(
            corpus_path=corpus_path,
            backend="tfidf",
            enable_query_rewrite=False,
        )
        results = retriever.search("垃圾清运", top_k=3)
        self.assertEqual(results, [])
        self.assertEqual(retriever.active_backend, "empty")


class TestRetrieverQueryRewrite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.corpus_path = os.path.join(cls.tmpdir, "test_corpus.jsonl")
        _write_sample_corpus(cls.corpus_path, SAMPLE_DOCS)

    def test_rewrite_disabled(self):
        retriever = PolicyRetriever(
            corpus_path=self.corpus_path,
            backend="tfidf",
            enable_query_rewrite=False,
        )
        results = retriever.search("垃圾清运", top_k=3)
        if results:
            self.assertTrue(len(results) > 0)

    def test_rewrite_enabled(self):
        retriever = PolicyRetriever(
            corpus_path=self.corpus_path,
            backend="tfidf",
            enable_query_rewrite=True,
            multi_query_count=3,
        )
        results = retriever.search("垃圾清运", top_k=3)
        if results:
            self.assertTrue(len(results) > 0)


if __name__ == "__main__":
    unittest.main()
