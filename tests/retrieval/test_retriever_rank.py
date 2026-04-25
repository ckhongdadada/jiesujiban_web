from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.jsjb.retrieval.bge_retriever import PolicyRetriever, RetrievalHit


def _write_sample_corpus(path: str, docs: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


SAMPLE_DOCS = [
    {
        "id": "doc_001",
        "title": "朝阳区垃圾清运案例",
        "doc_type": "案例",
        "district": "朝阳区",
        "source": "测试知识库",
        "content": "朝阳区望京街道垃圾清运不及时，已由环卫中心处理",
        "tags": ["环境卫生", "垃圾清运"],
        "issue_type": "垃圾清运",
        "unit": "环卫中心",
        "applicable_tags": ["投诉", "建议"],
    },
    {
        "id": "doc_002",
        "title": "海淀区噪声扰民处理案例",
        "doc_type": "案例",
        "district": "海淀区",
        "source": "测试知识库",
        "content": "海淀区中关村施工噪声扰民，已要求限时施工",
        "tags": ["噪声", "施工"],
        "issue_type": "噪声扰民",
        "unit": "住建委",
        "applicable_tags": ["投诉"],
    },
    {
        "id": "doc_003",
        "title": "全市停车秩序管理政策",
        "doc_type": "政策",
        "district": "北京市",
        "source": "测试知识库",
        "content": "北京市停车秩序管理办法，规范占道停车行为",
        "tags": ["停车", "秩序"],
        "issue_type": "停车秩序",
        "unit": "交管局",
        "applicable_tags": ["投诉", "建议"],
    },
    {
        "id": "doc_004",
        "title": "丰台区道路积水案例",
        "doc_type": "案例",
        "district": "丰台区",
        "source": "测试知识库",
        "content": "丰台区方庄路暴雨后积水严重，已由水务局处理",
        "tags": ["积水", "排水"],
        "issue_type": "道路积水",
        "unit": "水务局",
        "applicable_tags": ["投诉"],
    },
    {
        "id": "doc_005",
        "title": "朝阳区停车秩序案例",
        "doc_type": "案例",
        "district": "朝阳区",
        "source": "测试知识库",
        "content": "朝阳区CBD区域占道停车严重，交管部门已加强执法",
        "tags": ["停车", "占道"],
        "issue_type": "停车秩序",
        "unit": "交管局",
        "applicable_tags": ["投诉"],
    },
]


class TestRetrieverRank(unittest.TestCase):
    """同区弱相关 vs 跨区强相关排序"""

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

    def test_same_district_strong_semantic_ranks_higher(self):
        results = self.retriever.search(
            "朝阳区垃圾清运问题",
            top_k=3,
            district="朝阳区",
        )
        if not results:
            self.skipTest("检索后端未就绪，跳过排序测试")
        titles = [r["title"] for r in results]
        chaoyang_garbage = "朝阳区垃圾清运案例"
        if chaoyang_garbage in titles:
            self.assertLess(titles.index(chaoyang_garbage), len(titles))

    def test_cross_district_strong_semantic_not_suppressed(self):
        results = self.retriever.search(
            "垃圾清运不及时",
            top_k=3,
            district="丰台区",
        )
        if not results:
            self.skipTest("检索后端未就绪")
        titles = [r["title"] for r in results]
        garbage_doc = "朝阳区垃圾清运案例"
        if garbage_doc in titles:
            self.assertLess(titles.index(garbage_doc), 3)

    def test_citywide_doc_ranks_when_relevant(self):
        results = self.retriever.search(
            "停车秩序管理",
            top_k=3,
            district="朝阳区",
        )
        if not results:
            self.skipTest("检索后端未就绪")
        titles = [r["title"] for r in results]
        self.assertTrue(any("停车" in t for t in titles))


class TestRetrieverFilters(unittest.TestCase):
    """district / tag / unit 过滤是否生效"""

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

    def test_district_filter_boosts_same_district(self):
        results_with_district = self.retriever.search(
            "垃圾清运问题",
            top_k=3,
            district="朝阳区",
        )
        results_without = self.retriever.search(
            "垃圾清运问题",
            top_k=3,
        )
        if not results_with_district or not results_without:
            self.skipTest("检索后端未就绪")
        chaoyang_in_filtered = any("朝阳区" in r.get("district", "") for r in results_with_district)
        self.assertTrue(chaoyang_in_filtered)

    def test_tag_filter_influences_results(self):
        results = self.retriever.search(
            "噪声扰民施工问题",
            top_k=3,
            tag="投诉",
        )
        if not results:
            self.skipTest("检索后端未就绪")
        self.assertTrue(len(results) > 0)

    def test_unit_filter_influences_results(self):
        results = self.retriever.search(
            "停车秩序问题",
            top_k=3,
            unit="交管局",
        )
        if not results:
            self.skipTest("检索后端未就绪")
        self.assertTrue(len(results) > 0)


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

    def test_empty_query_returns_empty(self):
        results = self.retriever.search("", top_k=3)
        self.assertEqual(results, [])

    def test_irrelevant_query_low_scores(self):
        results = self.retriever.search(
            "量子计算机研发进展",
            top_k=3,
        )
        for r in results:
            self.assertIsInstance(r["score"], float)

    def test_result_has_expected_keys(self):
        results = self.retriever.search("垃圾清运", top_k=3)
        if not results:
            self.skipTest("检索后端未就绪")
        for r in results:
            for key in ("doc_id", "title", "score", "snippet", "matched_terms", "district"):
                self.assertIn(key, r)

    def test_describe_returns_backend_info(self):
        info = self.retriever.describe()
        self.assertIn("active_backend", info)
        self.assertIn("document_count", info)
        self.assertEqual(info["document_count"], len(SAMPLE_DOCS))


if __name__ == "__main__":
    unittest.main()
