from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from src.jsjb.core.paths import get_policy_corpus_path, get_policy_corpus_sample_path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


BEIJING_DISTRICTS = {
    "东城区",
    "西城区",
    "朝阳区",
    "丰台区",
    "石景山区",
    "海淀区",
    "门头沟区",
    "房山区",
    "通州区",
    "顺义区",
    "昌平区",
    "大兴区",
    "怀柔区",
    "平谷区",
    "密云区",
    "延庆区",
    "全市",
    "北京市",
}

ISSUE_RULES = [
    ("垃圾", "垃圾清运"),
    ("清运", "垃圾清运"),
    ("异味", "异味扰民"),
    ("噪声", "噪声扰民"),
    ("扰民", "扰民"),
    ("施工", "施工扰民"),
    ("停车", "停车秩序"),
    ("违停", "停车秩序"),
    ("积水", "道路积水"),
    ("物业", "物业服务"),
    ("消防", "消防通道"),
    ("占道", "占道经营"),
    ("路灯", "照明设施"),
    ("排水", "排水设施"),
    ("扬尘", "空气质量"),
    ("空气", "空气质量"),
    ("绿化", "园林绿化"),
]


@dataclass
class RetrievalHit:
    doc_id: str
    title: str
    doc_type: str
    district: str
    source: str
    score: float
    snippet: str
    matched_terms: list[str]
    retrieval_backend: str = ""
    rewrite_count: int = 1
    dense_score: float = 0.0
    sparse_score: float = 0.0


class PolicyRetriever:
    def __init__(
        self,
        corpus_path=None,
        backend=None,
        enable_query_rewrite=None,
        multi_query_count=None,
        dense_weight=None,
        sparse_weight=None,
    ):
        default_corpus_path = get_policy_corpus_path()
        if not default_corpus_path.exists():
            default_corpus_path = get_policy_corpus_sample_path()
        self.corpus_path = corpus_path or str(default_corpus_path)
        self.requested_backend = (backend or os.getenv("RAG_BACKEND", "hybrid")).lower()
        self.active_backend = "empty"
        self.docs = self._load_docs(self.corpus_path)
        self.vectorizer = None
        self.sparse_vectorizer = None
        self.embedding_model = None
        self.doc_vectors = None
        self.dense_doc_vectors = None
        self.sparse_doc_vectors = None
        self.enable_query_rewrite = self._resolve_bool(
            enable_query_rewrite,
            os.getenv("RAG_ENABLE_QUERY_REWRITE", "true"),
        )
        self.multi_query_count = self._resolve_int(
            multi_query_count,
            os.getenv("RAG_MULTI_QUERY_COUNT", "4"),
            minimum=1,
            maximum=8,
        )
        self.dense_weight = self._resolve_float(
            dense_weight,
            os.getenv("RAG_DENSE_WEIGHT", "0.68"),
            default=0.68,
        )
        self.sparse_weight = self._resolve_float(
            sparse_weight,
            os.getenv("RAG_SPARSE_WEIGHT", "0.32"),
            default=0.32,
        )
        self._build_index()

    @staticmethod
    def _resolve_bool(value, fallback):
        if value is None:
            value = fallback
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _resolve_int(value, fallback, minimum=None, maximum=None):
        try:
            resolved = int(value if value is not None else fallback)
        except (TypeError, ValueError):
            resolved = int(fallback)
        if minimum is not None:
            resolved = max(minimum, resolved)
        if maximum is not None:
            resolved = min(maximum, resolved)
        return resolved

    @staticmethod
    def _resolve_float(value, fallback, default=0.0):
        try:
            return float(value if value is not None else fallback)
        except (TypeError, ValueError):
            return default

    def _load_docs(self, path):
        if not os.path.exists(path):
            return []

        docs = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(self._normalize_doc(json.loads(line)))
        return docs

    def _normalize_doc(self, doc):
        normalized = dict(doc)
        normalized["id"] = normalized.get("id", "")
        normalized["title"] = normalized.get("title", "未命名材料")
        normalized["doc_type"] = normalized.get("doc_type", "参考材料")
        normalized["district"] = normalized.get("district", "全市")
        normalized["source"] = normalized.get("source", "本地知识库")
        normalized["content"] = normalized.get("content", "")
        normalized["tags"] = [str(tag) for tag in normalized.get("tags", [])]
        normalized["issue_type"] = str(normalized.get("issue_type", ""))
        normalized["unit"] = str(normalized.get("unit", ""))
        normalized["applicable_tags"] = [str(tag) for tag in normalized.get("applicable_tags", [])]
        return normalized

    def _build_index(self):
        if not self.docs:
            self.active_backend = "empty"
            return

        corpus = [self._compose_doc_text(doc) for doc in self.docs]
        dense_ready = False
        sparse_ready = False

        if self.requested_backend in {"bge", "dense", "hybrid"} and SENTENCE_TRANSFORMERS_AVAILABLE and NUMPY_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(
                    "BAAI/bge-small-zh-v1.5",
                    device="cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu",
                )
                self.dense_doc_vectors = self.embedding_model.encode(
                    corpus,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                dense_ready = True
            except Exception as exc:
                print("[RAG] BGE 初始化失败，回退到 TF-IDF:", exc)

        if SKLEARN_AVAILABLE:
            self.sparse_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            self.sparse_doc_vectors = self.sparse_vectorizer.fit_transform(corpus)
            self.vectorizer = self.sparse_vectorizer
            sparse_ready = True

        if dense_ready and sparse_ready and self.requested_backend == "hybrid":
            self.doc_vectors = self.dense_doc_vectors
            self.active_backend = "hybrid"
            print("[RAG] 使用 Hybrid 检索（BGE稠密 + TF-IDF稀疏），文档数:", len(self.docs))
        elif dense_ready:
            self.doc_vectors = self.dense_doc_vectors
            self.active_backend = "bge"
            print("[RAG] 使用 BGE 向量检索，文档数:", len(self.docs))
        elif sparse_ready:
            self.doc_vectors = self.sparse_doc_vectors
            self.active_backend = "tfidf"
            print("[RAG] 使用 TF-IDF 检索，文档数:", len(self.docs))
        else:
            self.active_backend = "empty"
            print("[RAG] 无可用检索后端")

    def _compose_doc_text(self, doc):
        parts = [
            doc.get("title", ""),
            doc.get("content", ""),
            " ".join(doc.get("tags", [])),
            doc.get("district", ""),
            doc.get("doc_type", ""),
            doc.get("issue_type", ""),
            doc.get("unit", ""),
            " ".join(doc.get("applicable_tags", [])),
        ]
        return "\n".join(part for part in parts if part)

    def extract_issue_keywords(self, text):
        keywords = []
        for token, label in ISSUE_RULES:
            if token in text and label not in keywords:
                keywords.append(label)
        return keywords

    def _collect_query_terms(self, query, district=None, tag=None, unit=None):
        terms = self.extract_issue_keywords(query)
        if district:
            terms.append(district)
            if district.endswith("区"):
                terms.append(district[:-1])
        if tag:
            terms.append(tag)
        if unit:
            terms.append(unit)
        terms.extend(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,12}", query))

        deduped = []
        for term in terms:
            if term and term not in deduped:
                deduped.append(term)
        return deduped

    def _matched_terms(self, doc, query_terms):
        haystack = self._compose_doc_text(doc)
        return [term for term in query_terms if term and term in haystack][:8]

    def _split_matched_terms(self, matched_terms, district=None, unit=None):
        locality_terms = set()
        semantic_terms = []

        if district:
            locality_terms.add(district)
            if district.endswith("区"):
                locality_terms.add(district[:-1])
        if unit:
            locality_terms.add(unit)

        for term in matched_terms:
            if term in locality_terms:
                continue
            semantic_terms.append(term)
        return locality_terms, semantic_terms

    def _preferred_doc_types(self, tag, preferred_doc_types):
        if preferred_doc_types:
            return set(preferred_doc_types)
        if not tag:
            return set()
        if "建言" in tag or "建议" in tag:
            return {"政策", "参考材料"}
        return {"案例", "政策"}

    def _build_query_for_vector_search(self, query, query_terms):
        extra = " ".join(term for term in query_terms if len(term) >= 2)
        return f"{query}\n{extra}".strip()

    def _rewrite_queries(self, query, query_terms, district=None, tag=None, unit=None):
        base_query = str(query or "").strip()
        if not self.enable_query_rewrite:
            return [self._build_query_for_vector_search(base_query, query_terms)]

        metadata_terms = [term for term in [district, tag, unit] if term]
        issue_terms = [term for term in query_terms if term and len(term) >= 2][:12]
        compact_query = " ".join(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}", base_query))

        candidates = [
            base_query,
            self._build_query_for_vector_search(base_query, query_terms),
            f"{base_query}\n{' '.join(metadata_terms + issue_terms)}".strip(),
            f"{compact_query}\n政策 案例 办理 责任 单位 {' '.join(metadata_terms)}".strip(),
            f"{base_query}\n同类诉求 处理依据 办理结果 {' '.join(issue_terms[:6])}".strip(),
        ]

        rewritten = []
        seen = set()
        for candidate in candidates:
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            rewritten.append(candidate)
            if len(rewritten) >= self.multi_query_count:
                break
        return rewritten or [base_query]

    def _normalize_scores(self, scores):
        if not NUMPY_AVAILABLE:
            return scores
        arr = np.asarray(scores, dtype=float)
        if arr.size == 0:
            return arr
        score_min = float(arr.min())
        score_max = float(arr.max())
        if score_max - score_min < 1e-9:
            return np.zeros_like(arr, dtype=float)
        return (arr - score_min) / (score_max - score_min)

    def _score_rewritten_queries(self, rewritten_queries):
        if not NUMPY_AVAILABLE:
            return None, None, None

        dense_scores = None
        sparse_scores = None

        if self.embedding_model is not None and self.dense_doc_vectors is not None:
            query_vectors = self.embedding_model.encode(
                rewritten_queries,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            dense_scores = np.max(np.dot(query_vectors, self.dense_doc_vectors.T), axis=0)

        if self.sparse_vectorizer is not None and self.sparse_doc_vectors is not None and SKLEARN_AVAILABLE:
            query_vectors = self.sparse_vectorizer.transform(rewritten_queries)
            sparse_scores = np.asarray(cosine_similarity(query_vectors, self.sparse_doc_vectors).max(axis=0)).ravel()

        if dense_scores is not None and sparse_scores is not None and self.active_backend == "hybrid":
            total_weight = max(self.dense_weight + self.sparse_weight, 1e-9)
            dense_weight = self.dense_weight / total_weight
            sparse_weight = self.sparse_weight / total_weight
            scores = dense_weight * self._normalize_scores(dense_scores) + sparse_weight * self._normalize_scores(sparse_scores)
        elif dense_scores is not None:
            scores = dense_scores
        elif sparse_scores is not None:
            scores = sparse_scores
        else:
            return None, None, None

        if dense_scores is None:
            dense_scores = np.zeros_like(scores, dtype=float)
        if sparse_scores is None:
            sparse_scores = np.zeros_like(scores, dtype=float)
        return scores, dense_scores, sparse_scores

    def _rerank_score(self, base_score, doc, query_terms, district=None, tag=None, unit=None, preferred_doc_types=None):
        boosted = float(base_score)
        matched_terms = self._matched_terms(doc, query_terms)
        doc_district = doc.get("district", "")
        _, semantic_terms = self._split_matched_terms(matched_terms, district=district, unit=unit)

        if district:
            if doc_district == district:
                boosted += 0.25
            elif doc_district in ("北京市", "全市"):
                boosted += 0.04
            elif doc_district and doc_district not in ("北京市", "全市"):
                boosted -= 0.16

        if tag and (tag in doc.get("applicable_tags", []) or tag in doc.get("tags", [])):
            boosted += 0.08

        if unit:
            doc_unit = doc.get("unit", "")
            if doc_unit and (doc_unit == unit or unit in doc_unit or doc_unit in unit):
                boosted += 0.08

        if preferred_doc_types and doc.get("doc_type") in preferred_doc_types:
            boosted += 0.06

        if doc.get("issue_type") and doc.get("issue_type") in query_terms:
            boosted += 0.10

        if matched_terms:
            boosted += min(len(matched_terms) * 0.03, 0.18)
        elif district and doc_district not in (district, "北京市", "全市"):
            boosted -= 0.08

        if semantic_terms:
            boosted += min(len(semantic_terms) * 0.09, 0.36)
        elif district and doc_district == district:
            # Same-district documents with no issue-level overlap are often less useful
            # than cross-district cases about the same problem.
            boosted -= 0.18

        if district and matched_terms:
            locality_terms = [term for term in matched_terms if term == district or term == district[:-1]]
            if locality_terms:
                boosted += 0.04

        return boosted, matched_terms

    def search(
        self,
        query,
        top_k=3,
        district=None,
        tag=None,
        unit=None,
        preferred_doc_types=None,
    ):
        if not query or self.active_backend == "empty":
            return []

        query_terms = self._collect_query_terms(query, district=district, tag=tag, unit=unit)
        rewritten_queries = self._rewrite_queries(query, query_terms, district=district, tag=tag, unit=unit)
        preferred_doc_types = self._preferred_doc_types(tag, preferred_doc_types)

        scores, dense_scores, sparse_scores = self._score_rewritten_queries(rewritten_queries)
        if scores is None:
            return []

        ranked_items = []
        for idx, (score, doc) in enumerate(zip(scores, self.docs)):
            boosted_score, matched_terms = self._rerank_score(
                score,
                doc,
                query_terms,
                district=district,
                tag=tag,
                unit=unit,
                preferred_doc_types=preferred_doc_types,
            )
            district_priority = 0
            if district:
                if doc.get("district") == district:
                    district_priority = 2
                elif doc.get("district") in ("北京市", "全市"):
                    district_priority = 1
            ranked_items.append((boosted_score, district_priority, matched_terms, doc, idx))

        ranked_items.sort(key=lambda item: (item[0], item[1]), reverse=True)

        seen_keys = set()
        results = []
        for score, _, matched_terms, doc, idx in ranked_items[: max(top_k * 6, 20)]:
            if score <= 0:
                continue
            dedup_key = (doc.get("title", ""), doc.get("source", ""))
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            hit = RetrievalHit(
                doc_id=doc.get("id", ""),
                title=doc.get("title", "未命名材料"),
                doc_type=doc.get("doc_type", "参考材料"),
                district=doc.get("district", ""),
                source=doc.get("source", "本地知识库"),
                score=round(score, 4),
                snippet=(doc.get("content", "")[:160] + "...") if len(doc.get("content", "")) > 160 else doc.get("content", ""),
                matched_terms=matched_terms,
                retrieval_backend=self.active_backend,
                rewrite_count=len(rewritten_queries),
                dense_score=round(float(dense_scores[idx]), 4),
                sparse_score=round(float(sparse_scores[idx]), 4),
            )
            results.append(asdict(hit))
            if len(results) >= top_k:
                break
        return results

    def describe(self):
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "document_count": len(self.docs),
            "corpus_path": self.corpus_path,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "dense_index_available": self.dense_doc_vectors is not None,
            "sparse_index_available": self.sparse_doc_vectors is not None,
            "query_rewrite_enabled": self.enable_query_rewrite,
            "multi_query_count": self.multi_query_count,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
        }


class RAGRetriever(PolicyRetriever):
    def __init__(self, data_dir=None, **kwargs):
        if data_dir:
            corpus_path = os.path.join(data_dir, "policy_case_corpus.jsonl")
            if not os.path.exists(corpus_path):
                corpus_path = os.path.join(data_dir, "policy_case_corpus.sample.jsonl")
            if os.path.exists(corpus_path):
                kwargs["corpus_path"] = corpus_path
        super().__init__(**kwargs)

    def retrieve(self, query, district=None, **kwargs):
        return self.search(query, district=district, **kwargs)
