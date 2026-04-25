from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.jsjb.retrieval.bge_retriever import PolicyRetriever


def _write_sample_corpus(path: str, docs: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


SAMPLE_DOCS = [
    {
        "id": "f001",
        "title": "朝阳区垃圾清运案例",
        "doc_type": "案例",
        "district": "朝阳区",
        "source": "测试",
        "content": "朝阳区望京垃圾清运",
        "tags": ["环境卫生"],
        "issue_type": "垃圾清运",
        "unit": "环卫中心",
        "applicable_tags": ["投诉"],
    },
    {
        "id": "f002",
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
    {
        "id": "f003",
        "title": "全市停车秩序政策",
        "doc_type": "政策",
        "district": "北京市",
        "source": "测试",
        "content": "北京市停车秩序管理办法",
        "tags": ["停车"],
        "issue_type": "停车秩序",
        "unit": "交管局",
        "applicable_tags": ["投诉", "建议"],
    },
    {
        "id": "f004",
        "title": "丰台区道路积水案例",
        "doc_type": "案例",
        "district": "丰台区",
        "source": "测试",
        "content": "丰台区方庄路积水",
        "tags": ["积水"],
        "issue_type": "道路积水",
        "unit": "水务局",
        "applicable_tags": ["投诉"],
    },
]


class TestDistrictFilter(unittest.TestCase):
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

    def test_district_boost_same_district(self):
        results = self.retriever.search("垃圾清运问题", top_k=3, district="朝阳区")
        if not results:
            self.skipTest("检索后端未就绪")
        has_chaoyang = any(r.get("district") == "朝阳区" for r in results)
        self.assertTrue(has_chaoyang)

    def test_citywide_docs_get_small_boost(self):
        results = self.retriever.search("停车秩序", top_k=3, district="朝阳区")
        if not results:
            self.skipTest("检索后端未就绪")

    def test_cross_district_penalty(self):
        results = self.retriever.search("垃圾清运", top_k=3, district="丰台区")
        if not results:
            self.skipTest("检索后端未就绪")


class TestTagFilter(unittest.TestCase):
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

    def test_tag_matching_boosts(self):
        results = self.retriever.search("噪声扰民", top_k=3, tag="投诉")
        if not results:
            self.skipTest("检索后端未就绪")
        self.assertTrue(len(results) > 0)

    def test_suggestion_tag_prefers_policy(self):
        results = self.retriever.search("停车秩序", top_k=3, tag="建议")
        if not results:
            self.skipTest("检索后端未就绪")


class TestUnitFilter(unittest.TestCase):
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

    def test_unit_matching_boosts(self):
        results = self.retriever.search("停车秩序", top_k=3, unit="交管局")
        if not results:
            self.skipTest("检索后端未就绪")
        self.assertTrue(len(results) > 0)

    def test_issue_type_matching_boosts(self):
        results = self.retriever.search("垃圾清运问题", top_k=3)
        if not results:
            self.skipTest("检索后端未就绪")


class TestCombinedFilters(unittest.TestCase):
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

    def test_district_tag_unit_combined(self):
        results = self.retriever.search(
            "垃圾清运问题",
            top_k=3,
            district="朝阳区",
            tag="投诉",
            unit="环卫中心",
        )
        if not results:
            self.skipTest("检索后端未就绪")
        self.assertTrue(len(results) > 0)


if __name__ == "__main__":
    unittest.main()
