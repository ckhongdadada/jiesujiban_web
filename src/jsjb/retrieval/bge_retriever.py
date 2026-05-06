from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from src.jsjb.core.paths import get_policy_corpus_path, get_policy_corpus_sample_path, get_feedback_db_path

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

try:
    from src.jsjb.knowledge.graph import InMemoryGraph

    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


from src.jsjb.retrieval.components import (
    DocumentChunk,
    FeedbackScoreCache,
    GraphAugmentor,
    Reranker,
    RetrievalHit,
    SemanticChunker,
)

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
    ("供暖", "供暖问题"),
    ("供热", "供暖问题"),
    ("充电", "消防安全"),
    ("电动自行车", "消防安全"),
    ("无障碍", "无障碍设施"),
    ("房屋", "房屋安全"),
    ("外墙", "房屋安全"),
]



class PolicyRetriever:
    def __init__(
        self,
        corpus_path=None,
        backend=None,
        enable_query_rewrite=None,
        multi_query_count=None,
        dense_weight=None,
        sparse_weight=None,
        enable_chunking=None,
        chunk_size=None,
        chunk_overlap=None,
        enable_reranker=None,
        enable_graph_augment=None,
        enable_feedback_boost=None,
        enable_post_processing=None,
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

        self.enable_chunking = self._resolve_bool(
            enable_chunking,
            os.getenv("RAG_ENABLE_CHUNKING", "true"),
        )
        self.chunk_size = self._resolve_int(
            chunk_size,
            os.getenv("RAG_CHUNK_SIZE", "400"),
            minimum=100,
            maximum=1000,
        )
        self.chunk_overlap = self._resolve_int(
            chunk_overlap,
            os.getenv("RAG_CHUNK_OVERLAP", "60"),
            minimum=0,
            maximum=200,
        )

        self.enable_reranker = self._resolve_bool(
            enable_reranker,
            os.getenv("RAG_ENABLE_RERANKER", "true"),
        )
        self.enable_graph_augment = self._resolve_bool(
            enable_graph_augment,
            os.getenv("RAG_ENABLE_GRAPH_AUGMENT", "true"),
        )
        self.enable_feedback_boost = self._resolve_bool(
            enable_feedback_boost,
            os.getenv("RAG_ENABLE_FEEDBACK_BOOST", "true"),
        )
        self.enable_post_processing = self._resolve_bool(
            enable_post_processing,
            os.getenv("RAG_ENABLE_POST_PROCESSING", "true"),
        )

        self.chunks: list[DocumentChunk] = []
        self.chunk_vectors = None
        self.chunk_sparse_vectors = None
        self.chunk_doc_map: dict[int, int] = {}
        self._chunker = SemanticChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self._reranker = Reranker() if self.enable_reranker else None
        self._graph_augmentor = GraphAugmentor() if self.enable_graph_augment else None
        self._feedback_cache = FeedbackScoreCache() if self.enable_feedback_boost else None

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

    def _build_chunk_index(self):
        self.chunks = []
        self.chunk_doc_map = {}
        for doc_idx, doc in enumerate(self.docs):
            doc_chunks = self._chunker.chunk_document(doc)
            for chunk in doc_chunks:
                chunk_idx = len(self.chunks)
                self.chunks.append(chunk)
                self.chunk_doc_map[chunk_idx] = doc_idx
        print(f"[RAG] 语义分块完成: {len(self.docs)} 文档 → {len(self.chunks)} 块")

    def _build_index(self):
        if not self.docs:
            self.active_backend = "empty"
            return

        if self.enable_chunking:
            self._build_chunk_index()

        use_chunks = self.enable_chunking and len(self.chunks) > 0
        corpus_items = self.chunks if use_chunks else self.docs
        corpus = [
            (self._compose_chunk_text(c) if isinstance(c, DocumentChunk) else self._compose_doc_text(c))
            for c in corpus_items
        ]

        dense_ready = False
        sparse_ready = False

        if self.requested_backend in {"bge", "dense", "hybrid"} and SENTENCE_TRANSFORMERS_AVAILABLE and NUMPY_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(
                    "BAAI/bge-small-zh-v1.5",
                    device="cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu",
                )
                vectors = self.embedding_model.encode(
                    corpus,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                if use_chunks:
                    self.chunk_vectors = vectors
                    self.dense_doc_vectors = self._aggregate_chunk_vectors_to_docs(vectors)
                else:
                    self.dense_doc_vectors = vectors
                dense_ready = True
            except Exception as exc:
                print("[RAG] BGE 初始化失败，回退到 TF-IDF:", exc)

        if SKLEARN_AVAILABLE:
            self.sparse_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            sparse_vectors = self.sparse_vectorizer.fit_transform(corpus)
            if use_chunks:
                self.chunk_sparse_vectors = sparse_vectors
                self.sparse_doc_vectors = self._aggregate_chunk_sparse_to_docs(sparse_vectors)
            else:
                self.sparse_doc_vectors = sparse_vectors
            self.vectorizer = self.sparse_vectorizer
            sparse_ready = True

        if dense_ready and sparse_ready and self.requested_backend == "hybrid":
            self.doc_vectors = self.dense_doc_vectors
            self.active_backend = "hybrid"
            mode = "分块" if use_chunks else "整文档"
            print(f"[RAG] 使用 Hybrid 检索（BGE稠密 + TF-IDF稀疏，{mode}），文档数:", len(self.docs))
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

    def _aggregate_chunk_vectors_to_docs(self, chunk_vectors) -> np.ndarray:
        if not NUMPY_AVAILABLE:
            return chunk_vectors
        doc_count = len(self.docs)
        doc_vectors = np.zeros((doc_count, chunk_vectors.shape[1]), dtype=np.float32)
        for chunk_idx, doc_idx in self.chunk_doc_map.items():
            doc_vectors[doc_idx] += chunk_vectors[chunk_idx]
        norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        doc_vectors /= norms
        return doc_vectors

    def _aggregate_chunk_sparse_to_docs(self, chunk_sparse_vectors):
        if not SKLEARN_AVAILABLE:
            return chunk_sparse_vectors
        doc_count = len(self.docs)
        n_features = chunk_sparse_vectors.shape[1]
        from scipy.sparse import lil_matrix
        doc_vectors = lil_matrix((doc_count, n_features))
        for chunk_idx, doc_idx in self.chunk_doc_map.items():
            doc_vectors[doc_idx] += chunk_sparse_vectors[chunk_idx]
        return doc_vectors.tocsr()

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

    @staticmethod
    def _compose_chunk_text(chunk: DocumentChunk) -> str:
        parts = [
            chunk.title,
            chunk.content,
            " ".join(chunk.tags),
            chunk.district,
            chunk.doc_type,
            chunk.issue_type,
            chunk.unit,
            " ".join(chunk.applicable_tags),
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

        if self._graph_augmentor:
            try:
                graph_terms = self._graph_augmentor.augment_query_terms(query, district=district)
                terms.extend(graph_terms)
            except Exception:
                pass

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
            boosted -= 0.18

        if district and matched_terms:
            locality_terms = [term for term in matched_terms if term == district or term == district[:-1]]
            if locality_terms:
                boosted += 0.04

        if self._feedback_cache:
            try:
                fb = self._feedback_cache.get_boost(doc.get("id", ""))
                boosted += fb
            except Exception:
                pass

        if self._graph_augmentor:
            try:
                graph_penalty = self._graph_augmentor.verify_fact_consistency(doc, " ".join(query_terms))
                boosted += graph_penalty
            except Exception:
                pass

        return boosted, matched_terms

    def _post_process_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        query_terms: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not results or not self.enable_post_processing:
            return results[:top_k]

        deduped = []
        seen_titles = set()
        for hit in results:
            title = hit.get("title", "")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            deduped.append(hit)

        type_count: dict[str, int] = defaultdict(int)
        diverse = []
        fallback = []
        for hit in deduped:
            dt = hit.get("doc_type", "参考材料")
            type_count[dt] += 1
            diverse.append(hit)
            if type_count[dt] > top_k:
                fallback.append(hit)

        if len(diverse) < top_k and fallback:
            diverse.extend(fallback[:top_k - len(diverse)])

        return diverse[:top_k]

    @staticmethod
    def _is_fact_contradictory(doc: dict[str, Any], query: str) -> bool:
        content = doc.get("content", "")
        negation_patterns = ["不存在", "暂未建设", "尚未建设", "未涉及", "不涉及", "无此", "未有"]
        has_negation = any(p in content for p in negation_patterns)
        if not has_negation:
            return False
        affirmation = ["已建成", "已建设", "正在施工", "正在建设", "已运营", "已开放", "已完成"]
        return any(a in query for a in affirmation)

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

        candidate_count = max(top_k * 6, 20)
        if self._reranker and self.enable_reranker:
            candidate_count = max(top_k * 4, 12)

        candidates = []
        seen_keys = set()
        for score, _, matched_terms, doc, idx in ranked_items[:candidate_count]:
            if score <= 0:
                continue
            if self.enable_post_processing and self._is_fact_contradictory(doc, query):
                continue
            dedup_key = (doc.get("title", ""), doc.get("source", ""))
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            candidates.append((score, doc, matched_terms, idx))

        if self._reranker and self.enable_reranker and len(candidates) > top_k:
            try:
                reranked = self._reranker.rerank(query, candidates, top_k=max(top_k * 2, 10))
                candidates = reranked
            except Exception:
                pass

        results = []
        for item in candidates[:max(top_k * 3, 15)]:
            if len(item) == 4:
                score, doc, matched_terms, idx = item
            else:
                score, doc, matched_terms, idx = item[0], item[1], item[2], item[3]

            fb = 0.0
            if self._feedback_cache:
                try:
                    fb = self._feedback_cache.get_boost(doc.get("id", ""))
                except Exception:
                    pass

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
                dense_score=round(float(dense_scores[idx]), 4) if idx < len(dense_scores) else 0.0,
                sparse_score=round(float(sparse_scores[idx]), 4) if idx < len(sparse_scores) else 0.0,
                full_content=doc.get("content", ""),
                rerank_score=round(score, 4) if self._reranker else 0.0,
                feedback_boost=round(fb, 4),
            )
            results.append(asdict(hit))

        if self.enable_graph_augment and self._graph_augmentor:
            try:
                graph_facts = self._graph_augmentor.query_related_facts(query, district=district, limit=max(1, top_k // 2))
                seen_ids = {r["doc_id"] for r in results}
                for fact in graph_facts:
                    if fact["id"] in seen_ids:
                        continue
                    graph_hit = RetrievalHit(
                        doc_id=fact["id"],
                        title=fact["title"],
                        doc_type=fact["doc_type"],
                        district=fact["district"],
                        source=fact["source"],
                        score=round(fact.get("_graph_relevance", 0.5), 4),
                        snippet=fact["content"][:160] + ("..." if len(fact["content"]) > 160 else ""),
                        matched_terms=[],
                        retrieval_backend="knowledge_graph",
                        rewrite_count=0,
                        dense_score=0.0,
                        sparse_score=0.0,
                        rerank_score=0.0,
                        feedback_boost=0.0,
                        full_content=fact["content"],
                    )
                    results.append(asdict(graph_hit))
            except Exception:
                pass

        results = self._post_process_results(results, query, query_terms, top_k)
        return results

    def describe(self):
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "document_count": len(self.docs),
            "chunk_count": len(self.chunks),
            "corpus_path": self.corpus_path,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "graph_available": GRAPH_AVAILABLE,
            "dense_index_available": self.dense_doc_vectors is not None,
            "sparse_index_available": self.sparse_doc_vectors is not None,
            "query_rewrite_enabled": self.enable_query_rewrite,
            "multi_query_count": self.multi_query_count,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "chunking_enabled": self.enable_chunking,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "reranker_enabled": self.enable_reranker,
            "graph_augment_enabled": self.enable_graph_augment,
            "feedback_boost_enabled": self.enable_feedback_boost,
            "post_processing_enabled": self.enable_post_processing,
        }

    def reload(self) -> dict[str, Any]:
        old_count = len(self.docs)
        old_chunk_count = len(self.chunks)
        self.docs = self._load_docs(self.corpus_path)
        self._build_index()
        return {
            "status": "ok",
            "old_document_count": old_count,
            "new_document_count": len(self.docs),
            "old_chunk_count": old_chunk_count,
            "new_chunk_count": len(self.chunks),
            "active_backend": self.active_backend,
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
