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
    BM25Index,
    DocumentChunk,
    FeedbackRelevanceScorer,
    FeedbackScoreCache,
    GraphAugmentor,
    Reranker,
    RetrievalHit,
    SemanticChunker,
    SemanticRetrievalCache,
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
        enable_parent_child=None,
        enable_adaptive_retrieval=None,
        adaptive_max_top_k=None,
        enable_semantic_chunking=None,
        semantic_chunk_threshold=None,
        enable_semantic_cache=None,
        semantic_cache_size=None,
        semantic_cache_threshold=None,
        semantic_cache_ttl=None,
        enable_bm25=None,
        bm25_weight=None,
        tfidf_weight=None,
        enable_hyde=None,
        hyde_trigger_threshold=None,
        hyde_max_queries=None,
        reranker_model_name=None,
        reranker_model_path=None,
        reranker_weight=None,
        enable_relevance_scorer=None,
        relevance_weight=None,
        graph_manager=None,
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
        self.bm25_index = None
        self.embedding_model = None
        self.doc_vectors = None
        self.dense_doc_vectors = None
        self.sparse_doc_vectors = None
        self.bm25_doc_scores = None
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
        self.enable_parent_child = self._resolve_bool(
            enable_parent_child,
            os.getenv("RAG_ENABLE_PARENT_CHILD", "true"),
        )
        self.enable_adaptive_retrieval = self._resolve_bool(
            enable_adaptive_retrieval,
            os.getenv("RAG_ENABLE_ADAPTIVE_RETRIEVAL", "true"),
        )
        self.adaptive_max_top_k = self._resolve_int(
            adaptive_max_top_k,
            os.getenv("RAG_ADAPTIVE_MAX_TOP_K", "8"),
            minimum=1,
            maximum=20,
        )
        self.enable_semantic_chunking = self._resolve_bool(
            enable_semantic_chunking,
            os.getenv("RAG_ENABLE_SEMANTIC_CHUNKING", "true"),
        )
        self.semantic_chunk_threshold = self._resolve_float(
            semantic_chunk_threshold,
            os.getenv("RAG_SEMANTIC_CHUNK_THRESHOLD", "0.72"),
            default=0.72,
        )
        self.enable_semantic_cache = self._resolve_bool(
            enable_semantic_cache,
            os.getenv("RAG_ENABLE_SEMANTIC_CACHE", "true"),
        )
        self.semantic_cache_size = self._resolve_int(
            semantic_cache_size,
            os.getenv("RAG_SEMANTIC_CACHE_SIZE", "256"),
            minimum=1,
            maximum=5000,
        )
        self.semantic_cache_threshold = self._resolve_float(
            semantic_cache_threshold,
            os.getenv("RAG_SEMANTIC_CACHE_THRESHOLD", "0.92"),
            default=0.92,
        )
        self.semantic_cache_ttl = self._resolve_int(
            semantic_cache_ttl,
            os.getenv("RAG_SEMANTIC_CACHE_TTL", "1800"),
            minimum=1,
            maximum=86400,
        )
        self.enable_bm25 = self._resolve_bool(
            enable_bm25,
            os.getenv("RAG_ENABLE_BM25", "true"),
        )
        self.bm25_weight = self._resolve_float(
            bm25_weight,
            os.getenv("RAG_BM25_WEIGHT", "0.75"),
            default=0.75,
        )
        self.tfidf_weight = self._resolve_float(
            tfidf_weight,
            os.getenv("RAG_TFIDF_WEIGHT", "0.25"),
            default=0.25,
        )
        self.enable_hyde = self._resolve_bool(
            enable_hyde,
            os.getenv("RAG_ENABLE_HYDE", "true"),
        )
        self.hyde_trigger_threshold = self._resolve_float(
            hyde_trigger_threshold,
            os.getenv("RAG_HYDE_TRIGGER_THRESHOLD", "0.24"),
            default=0.24,
        )
        self.hyde_max_queries = self._resolve_int(
            hyde_max_queries,
            os.getenv("RAG_HYDE_MAX_QUERIES", "2"),
            minimum=1,
            maximum=4,
        )
        self.reranker_model_name = str(
            reranker_model_name
            if reranker_model_name is not None
            else os.getenv("RAG_RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
        )
        self.reranker_model_path = str(
            reranker_model_path
            if reranker_model_path is not None
            else os.getenv("RAG_RERANKER_MODEL_PATH", "")
        )
        self.reranker_weight = self._resolve_float(
            reranker_weight,
            os.getenv("RAG_RERANKER_WEIGHT", "0.60"),
            default=0.60,
        )
        self.enable_relevance_scorer = self._resolve_bool(
            enable_relevance_scorer,
            os.getenv("RAG_ENABLE_RELEVANCE_SCORER", "true"),
        )
        self.relevance_weight = self._resolve_float(
            relevance_weight,
            os.getenv("RAG_RELEVANCE_WEIGHT", "0.12"),
            default=0.12,
        )

        self.chunks: list[DocumentChunk] = []
        self.chunk_vectors = None
        self.chunk_sparse_vectors = None
        self.chunk_bm25_scores = None
        self.chunk_doc_map: dict[int, int] = {}
        self._last_child_matches: dict[int, dict[str, Any]] = {}
        self._chunker = SemanticChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            enable_semantic_chunking=self.enable_semantic_chunking,
            semantic_threshold=self.semantic_chunk_threshold,
        )

        self._reranker = (
            Reranker(
                model_name=self.reranker_model_name,
                model_path=self.reranker_model_path,
                rerank_weight=self.reranker_weight,
            )
            if self.enable_reranker
            else None
        )
        self._graph_augmentor = GraphAugmentor(graph_manager=graph_manager) if self.enable_graph_augment else None
        self._feedback_cache = FeedbackScoreCache() if self.enable_feedback_boost else None
        self._relevance_scorer = FeedbackRelevanceScorer() if self.enable_relevance_scorer else None
        self._semantic_cache = (
            SemanticRetrievalCache(
                max_size=self.semantic_cache_size,
                similarity_threshold=self.semantic_cache_threshold,
                ttl_seconds=self.semantic_cache_ttl,
            )
            if self.enable_semantic_cache
            else None
        )

        self._build_index()

    def attach_graph_manager(self, graph_manager=None) -> bool:
        """Attach or replace the graph used by RAG graph augmentation."""
        if not self.enable_graph_augment:
            return False
        if self._graph_augmentor is None:
            self._graph_augmentor = GraphAugmentor(graph_manager=graph_manager)
            return bool(getattr(self._graph_augmentor, "_graph", None) is not None)
        return self._graph_augmentor.attach_graph_manager(graph_manager)

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

    def _build_chunk_index(self, embedding_model=None):
        self.chunks = []
        self.chunk_doc_map = {}
        for doc_idx, doc in enumerate(self.docs):
            doc_chunks = self._chunker.chunk_document(doc, embedding_model=embedding_model)
            for chunk in doc_chunks:
                chunk_idx = len(self.chunks)
                self.chunks.append(chunk)
                self.chunk_doc_map[chunk_idx] = doc_idx
        mode = "embedding语义边界" if self.enable_semantic_chunking and embedding_model is not None else "结构/长度兜底"
        print(f"[RAG] 分块完成({mode}): {len(self.docs)} 文档 → {len(self.chunks)} 块")

    def _build_index(self):
        if not self.docs:
            self.active_backend = "empty"
            return

        dense_requested = (
            self.requested_backend in {"bge", "dense", "hybrid"}
            and SENTENCE_TRANSFORMERS_AVAILABLE
            and NUMPY_AVAILABLE
        )
        if dense_requested:
            try:
                self.embedding_model = SentenceTransformer(
                    "BAAI/bge-small-zh-v1.5",
                    device="cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu",
                )
            except Exception as exc:
                self.embedding_model = None
                print("[RAG] BGE 初始化失败，语义分块/稠密检索将回退:", exc)

        if self.enable_chunking:
            semantic_model = self.embedding_model if self.enable_semantic_chunking else None
            self._build_chunk_index(embedding_model=semantic_model)

        use_chunks = self.enable_chunking and len(self.chunks) > 0
        corpus_items = self.chunks if use_chunks else self.docs
        corpus = [
            (self._compose_chunk_text(c) if isinstance(c, DocumentChunk) else self._compose_doc_text(c))
            for c in corpus_items
        ]

        dense_ready = False
        sparse_ready = False
        bm25_ready = False

        if dense_requested and self.embedding_model is not None:
            try:
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
                print("[RAG] BGE init failed, fallback to lexical retrieval:", exc)

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

        if self.enable_bm25:
            try:
                self.bm25_index = BM25Index(corpus)
                bm25_ready = True
            except Exception as exc:
                self.bm25_index = None
                print("[RAG] BM25 index build failed, skip BM25:", exc)

        lexical_ready = sparse_ready or bm25_ready

        if bm25_ready and self.requested_backend == "bm25":
            self.active_backend = "bm25"
            mode = "chunk" if use_chunks else "document"
            print(f"[RAG] BM25 retrieval enabled ({mode}), docs:", len(self.docs))
        elif dense_ready and lexical_ready and self.requested_backend == "hybrid":
            self.doc_vectors = self.dense_doc_vectors
            self.active_backend = "hybrid"
            mode = "chunk" if use_chunks else "document"
            lexical_name = "BM25+TF-IDF" if sparse_ready and bm25_ready else ("TF-IDF" if sparse_ready else "BM25")
            print(f"[RAG] Hybrid retrieval enabled (BGE dense + {lexical_name}, {mode}), docs:", len(self.docs))
        elif dense_ready:
            self.doc_vectors = self.dense_doc_vectors
            self.active_backend = "bge"
            print("[RAG] BGE dense retrieval enabled, docs:", len(self.docs))
        elif sparse_ready:
            self.doc_vectors = self.sparse_doc_vectors
            self.active_backend = "tfidf"
            print("[RAG] TF-IDF retrieval enabled, docs:", len(self.docs))
        elif bm25_ready:
            self.active_backend = "bm25"
            print("[RAG] BM25 retrieval enabled, docs:", len(self.docs))
        else:
            self.active_backend = "empty"
            print("[RAG] no retrieval backend available")
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

    def _child_context_for_doc(self, doc_idx: int, query_terms: list[str] | None = None) -> dict[str, Any]:
        child_match = self._last_child_matches.get(doc_idx)
        if not child_match:
            return {}
        chunk_idx = child_match.get("chunk_idx")
        if chunk_idx is None or chunk_idx >= len(self.chunks):
            return {}
        chunk = self.chunks[chunk_idx]
        chunk_doc = {
            "id": chunk.chunk_id,
            "title": chunk.title,
            "content": chunk.content,
            "tags": chunk.tags,
            "district": chunk.district,
            "doc_type": chunk.doc_type,
            "issue_type": chunk.issue_type,
            "unit": chunk.unit,
            "applicable_tags": chunk.applicable_tags,
        }
        matched_terms = self._matched_terms(chunk_doc, query_terms or [])
        return {
            "chunk": chunk,
            "matched_terms": matched_terms,
            "chunk_score": child_match.get("chunk_score", 0.0),
            "dense_score": child_match.get("dense_score", 0.0),
            "sparse_score": child_match.get("sparse_score", 0.0),
            "bm25_score": child_match.get("bm25_score", 0.0),
        }

    @staticmethod
    def _parent_context(content: str, child: DocumentChunk | None = None, limit: int = 800) -> str:
        content = content or ""
        if not content:
            return ""
        if child is None or child.start_char <= 0 and child.end_char <= 0:
            return content[:limit] + ("..." if len(content) > limit else "")
        center = max(0, (child.start_char + child.end_char) // 2)
        half = max(100, limit // 2)
        start = max(0, center - half)
        end = min(len(content), center + half)
        context = content[start:end].strip()
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""
        return f"{prefix}{context}{suffix}"

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

    def _should_use_hyde(self, scores) -> bool:
        if not self.enable_hyde or scores is None or not NUMPY_AVAILABLE:
            return False
        arr = np.asarray(scores, dtype=float)
        if arr.size == 0:
            return False
        return float(arr.max()) < float(self.hyde_trigger_threshold)

    def _build_hyde_queries(self, query, query_terms, district=None, tag=None, unit=None) -> list[str]:
        if not self.enable_hyde:
            return []
        base_query = str(query or "").strip()
        issue_terms = [term for term in query_terms if term and len(term) >= 2][:10]
        metadata = " ".join(term for term in [district, tag, unit] if term)
        pseudo_docs = [
            (
                f"\u653f\u52a1\u529e\u7406\u6848\u4f8b\uff1a\u5e02\u6c11\u53cd\u6620 {base_query}\u3002"
                f"\u533a\u57df\u548c\u8d23\u4efb\u5355\u4f4d\u7ebf\u7d22\uff1a{metadata}\u3002"
                f"\u76f8\u5173\u95ee\u9898\u5173\u952e\u8bcd\uff1a{' '.join(issue_terms)}\u3002"
                "\u529e\u7406\u7ed3\u679c\u901a\u5e38\u5305\u542b\u73b0\u573a\u6838\u5b9e\u3001\u8d23\u4efb\u5355\u4f4d\u3001\u653f\u7b56\u4f9d\u636e\u3001\u6574\u6539\u63aa\u65bd\u548c\u56de\u590d\u7ed3\u8bba\u3002"
            ),
            (
                f"\u540c\u7c7b\u8bc9\u6c42\u56de\u590d\uff1a{base_query}\u3002"
                f"\u6d89\u53ca {' '.join(issue_terms)} \u7c7b\u6295\u8bc9\uff0c\u9700\u7ed3\u5408\u73b0\u573a\u6838\u67e5\u3001\u90e8\u95e8\u534f\u540c\u3001\u5904\u7f6e\u8fdb\u5c55\u548c\u540e\u7eed\u76d1\u7763\u3002"
            ),
        ]
        rewritten = []
        seen = set()
        for pseudo_doc in pseudo_docs:
            pseudo_doc = re.sub(r"\s+", " ", pseudo_doc).strip()
            if pseudo_doc and pseudo_doc not in seen:
                seen.add(pseudo_doc)
                rewritten.append(pseudo_doc)
            if len(rewritten) >= self.hyde_max_queries:
                break
        return rewritten

    def _normalize_scores(self, scores):
        if not NUMPY_AVAILABLE:
            return scores
        arr = np.asarray(scores, dtype=float)
        if arr.size == 0:
            return arr
        score_min = float(arr.min())
        score_max = float(arr.max())
        if score_max - score_min < 1e-9:
            if score_max > 0:
                return np.ones_like(arr, dtype=float)
            return np.zeros_like(arr, dtype=float)
        return (arr - score_min) / (score_max - score_min)

    def _combine_lexical_scores(self, tfidf_scores, bm25_scores):
        if not NUMPY_AVAILABLE:
            return None
        parts = []
        weights = []
        if tfidf_scores is not None:
            parts.append(self._normalize_scores(tfidf_scores))
            weights.append(max(float(self.tfidf_weight), 0.0))
        if bm25_scores is not None:
            parts.append(self._normalize_scores(bm25_scores))
            weights.append(max(float(self.bm25_weight), 0.0))
        if not parts:
            return None
        total = sum(weights) or float(len(parts))
        lexical = np.zeros_like(parts[0], dtype=float)
        for part, weight in zip(parts, weights):
            lexical += part * (weight / total)
        return lexical

    def _bm25_scores_for_queries(self, rewritten_queries):
        if self.bm25_index is None or not NUMPY_AVAILABLE:
            return None
        per_query = [self.bm25_index.score(query) for query in rewritten_queries]
        if not per_query:
            return None
        return np.max(np.asarray(per_query, dtype=float), axis=0)

    def _score_rewritten_queries(self, rewritten_queries):
        if not NUMPY_AVAILABLE:
            return None, None, None, None

        scores = None
        dense_scores = None
        tfidf_scores = None
        bm25_scores = None
        lexical_scores = None

        dense_chunk_scores = None
        tfidf_chunk_scores = None
        bm25_chunk_scores = None

        use_chunk_scoring = (
            self.enable_parent_child
            and self.enable_chunking
            and len(self.chunks) > 0
            and (self.chunk_vectors is not None or self.chunk_sparse_vectors is not None or self.bm25_index is not None)
        )
        self._last_child_matches = {}

        if self.embedding_model is not None:
            query_vectors = self.embedding_model.encode(
                rewritten_queries,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if use_chunk_scoring:
                if self.chunk_vectors is not None:
                    dense_chunk_scores = np.max(np.dot(query_vectors, self.chunk_vectors.T), axis=0)
            elif self.dense_doc_vectors is not None:
                dense_scores = np.max(np.dot(query_vectors, self.dense_doc_vectors.T), axis=0)

        if self.sparse_vectorizer is not None and SKLEARN_AVAILABLE:
            query_tfidf = self.sparse_vectorizer.transform(rewritten_queries)
            if use_chunk_scoring and self.chunk_sparse_vectors is not None:
                tfidf_chunk_scores = np.asarray(cosine_similarity(query_tfidf, self.chunk_sparse_vectors).max(axis=0)).ravel()
            elif self.sparse_doc_vectors is not None:
                tfidf_scores = np.asarray(cosine_similarity(query_tfidf, self.sparse_doc_vectors).max(axis=0)).ravel()

        raw_bm25_scores = self._bm25_scores_for_queries(rewritten_queries)
        if raw_bm25_scores is not None:
            if use_chunk_scoring and len(raw_bm25_scores) == len(self.chunks):
                bm25_chunk_scores = raw_bm25_scores
            elif len(raw_bm25_scores) == len(self.docs):
                bm25_scores = raw_bm25_scores

        if use_chunk_scoring:
            lexical_chunk_scores = self._combine_lexical_scores(tfidf_chunk_scores, bm25_chunk_scores)
            chunk_scores = None

            if dense_chunk_scores is not None and lexical_chunk_scores is not None and self.active_backend == "hybrid":
                total_weight = max(self.dense_weight + self.sparse_weight, 1e-9)
                dense_weight = self.dense_weight / total_weight
                sparse_weight = self.sparse_weight / total_weight
                chunk_scores = (
                    dense_weight * self._normalize_scores(dense_chunk_scores)
                    + sparse_weight * self._normalize_scores(lexical_chunk_scores)
                )
            elif dense_chunk_scores is not None and self.active_backend in {"bge", "hybrid"}:
                chunk_scores = dense_chunk_scores
            elif lexical_chunk_scores is not None:
                chunk_scores = lexical_chunk_scores
            elif bm25_chunk_scores is not None:
                chunk_scores = bm25_chunk_scores

            if chunk_scores is not None:
                doc_count = len(self.docs)
                scores = np.zeros(doc_count, dtype=np.float32)
                dense_scores = np.zeros(doc_count, dtype=np.float32)
                lexical_scores = np.zeros(doc_count, dtype=np.float32)
                bm25_scores = np.zeros(doc_count, dtype=np.float32)
                for chunk_idx, doc_idx in self.chunk_doc_map.items():
                    chunk_score = float(chunk_scores[chunk_idx])
                    if chunk_score >= scores[doc_idx]:
                        scores[doc_idx] = chunk_score
                        dense_val = float(dense_chunk_scores[chunk_idx]) if dense_chunk_scores is not None else 0.0
                        lexical_val = float(lexical_chunk_scores[chunk_idx]) if lexical_chunk_scores is not None else 0.0
                        bm25_val = float(bm25_chunk_scores[chunk_idx]) if bm25_chunk_scores is not None else 0.0
                        dense_scores[doc_idx] = dense_val
                        lexical_scores[doc_idx] = lexical_val
                        bm25_scores[doc_idx] = bm25_val
                        self._last_child_matches[doc_idx] = {
                            "chunk_idx": chunk_idx,
                            "chunk_score": chunk_score,
                            "dense_score": dense_val,
                            "sparse_score": lexical_val,
                            "bm25_score": bm25_val,
                        }
                return scores, dense_scores, lexical_scores, bm25_scores

        lexical_scores = self._combine_lexical_scores(tfidf_scores, bm25_scores)

        if dense_scores is not None and lexical_scores is not None and self.active_backend == "hybrid":
            total_weight = max(self.dense_weight + self.sparse_weight, 1e-9)
            dense_weight = self.dense_weight / total_weight
            sparse_weight = self.sparse_weight / total_weight
            scores = dense_weight * self._normalize_scores(dense_scores) + sparse_weight * self._normalize_scores(lexical_scores)
        elif dense_scores is not None and self.active_backend in {"bge", "hybrid"}:
            scores = dense_scores
        elif lexical_scores is not None:
            scores = lexical_scores
        elif bm25_scores is not None:
            scores = bm25_scores
        else:
            return None, None, None, None

        if dense_scores is None:
            dense_scores = np.zeros_like(scores, dtype=float)
        if lexical_scores is None:
            lexical_scores = np.zeros_like(scores, dtype=float)
        if bm25_scores is None:
            bm25_scores = np.zeros_like(scores, dtype=float)
        return scores, dense_scores, lexical_scores, bm25_scores

    def _cache_query_vector(self, query: str, query_terms: list[str]) -> list[float] | None:
        if self.embedding_model is None:
            return None
        try:
            cache_query = self._build_query_for_vector_search(query, query_terms)
            vector = self.embedding_model.encode(
                [cache_query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            return [float(value) for value in vector]
        except Exception:
            return None

    def _corpus_version(self) -> str:
        try:
            mtime = os.path.getmtime(self.corpus_path)
        except OSError:
            mtime = 0
        phase2_flags = (
            f"bm25={int(self.enable_bm25)}:{self.bm25_weight}:{self.tfidf_weight}:"
            f"hyde={int(self.enable_hyde)}:{self.hyde_trigger_threshold}:"
            f"rel={int(self.enable_relevance_scorer)}:{self.relevance_weight}"
        )
        return f"{len(self.docs)}:{len(self.chunks)}:{int(mtime)}:{self.active_backend}:{phase2_flags}"

    @staticmethod
    def _mark_cache_hit(results: list[dict[str, Any]], cache_info: dict[str, Any]) -> list[dict[str, Any]]:
        marked = []
        for hit in results:
            item = dict(hit)
            item["cache_hit"] = bool(cache_info.get("hit"))
            item["cache_similarity"] = float(cache_info.get("similarity", 0.0) or 0.0)
            marked.append(item)
        return marked

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

        feedback_boost_val = 0.0
        if self._feedback_cache:
            try:
                fb = self._feedback_cache.get_boost_for_doc(doc)
                feedback_boost_val = fb
                boosted += fb
            except Exception:
                pass

        graph_score_val = 0.0
        graph_entities: list[str] = []
        if self._graph_augmentor:
            try:
                graph_penalty = self._graph_augmentor.verify_fact_consistency(doc, " ".join(query_terms))
                boosted += graph_penalty
                graph_boost, graph_entities = self._graph_augmentor.graph_score_document(
                    doc,
                    " ".join(query_terms),
                    district=district,
                    unit=unit,
                )
                graph_score_val = graph_penalty + graph_boost
                boosted += graph_boost
            except Exception:
                pass

        relevance_score_val = 0.0
        if self._relevance_scorer:
            try:
                relevance_raw = self._relevance_scorer.score(" ".join(query_terms), doc, matched_terms=matched_terms)
                relevance_score_val = relevance_raw * max(float(self.relevance_weight), 0.0)
                boosted += relevance_score_val
            except Exception:
                pass

        return boosted, matched_terms, feedback_boost_val, relevance_score_val, graph_score_val, graph_entities

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
        max_per_type = max(1, top_k // 2 + 1)
        for hit in deduped:
            dt = hit.get("doc_type", "参考材料")
            if type_count[dt] < max_per_type:
                diverse.append(hit)
                type_count[dt] += 1
            else:
                fallback.append(hit)

        if len(diverse) < top_k and fallback:
            diverse.extend(fallback[:top_k - len(diverse)])

        return diverse[:top_k]

    def _adaptive_retrieval_profile(
        self,
        query: str,
        query_terms: list[str],
        requested_top_k: int,
        district: str | None = None,
        tag: str | None = None,
        unit: str | None = None,
    ) -> dict[str, Any]:
        if not self.enable_adaptive_retrieval:
            return {
                "strategy": "standard",
                "effective_top_k": requested_top_k,
                "candidate_multiplier": 6,
                "reasons": ["adaptive_disabled"],
            }

        text = str(query or "")
        chinese_tokens = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}", text)
        issue_hits = self.extract_issue_keywords(text)
        punctuation_complexity = len(re.findall(r"[，,；;。.!！?？、]", text))
        has_metadata = bool(district or tag or unit)

        reasons: list[str] = []
        complexity = 0
        if len(text) >= 120 or len(chinese_tokens) >= 18:
            complexity += 2
            reasons.append("long_query")
        elif len(text) >= 60 or len(chinese_tokens) >= 10:
            complexity += 1
            reasons.append("medium_query")
        if len(issue_hits) >= 2:
            complexity += 2
            reasons.append("multi_issue_signal")
        elif len(issue_hits) == 1:
            complexity += 1
            reasons.append("issue_signal")
        if punctuation_complexity >= 4:
            complexity += 1
            reasons.append("multiple_clauses")
        if not has_metadata:
            complexity += 1
            reasons.append("missing_metadata")
        if len(query_terms) >= 18:
            complexity += 1
            reasons.append("many_query_terms")

        if complexity >= 5:
            strategy = "expanded"
            effective_top_k = min(max(requested_top_k + 3, 7), self.adaptive_max_top_k)
            candidate_multiplier = 9
        elif complexity >= 3:
            strategy = "balanced"
            effective_top_k = min(max(requested_top_k + 1, 5), self.adaptive_max_top_k)
            candidate_multiplier = 7
        else:
            strategy = "focused"
            effective_top_k = min(requested_top_k, self.adaptive_max_top_k)
            candidate_multiplier = 5
            if not reasons:
                reasons.append("simple_query")

        return {
            "strategy": strategy,
            "effective_top_k": max(1, effective_top_k),
            "candidate_multiplier": candidate_multiplier,
            "reasons": reasons,
            "complexity_score": complexity,
        }

    @staticmethod
    def _contradiction_penalty(doc: dict[str, Any], query: str) -> float:
        content = doc.get("content", "")
        negation_patterns = ["不存在", "暂未建设", "尚未建设", "未涉及", "不涉及", "无此", "未有"]
        has_negation = any(p in content for p in negation_patterns)
        if not has_negation:
            return 0.0
        affirmation = ["已建成", "已建设", "正在施工", "正在建设", "已运营", "已开放", "已完成"]
        if any(a in query for a in affirmation):
            return -0.12
        return 0.0

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

        cache_vector = None
        if self._semantic_cache is not None:
            cache_vector = self._cache_query_vector(query, query_terms)
            cached_results, cache_info = self._semantic_cache.get(
                query=query,
                query_vector=cache_vector,
                district=district,
                tag=tag,
                unit=unit,
                top_k=top_k,
                corpus_version=self._corpus_version(),
            )
            if cached_results is not None:
                return self._mark_cache_hit(cached_results, cache_info)

        scores, dense_scores, sparse_scores, bm25_scores = self._score_rewritten_queries(rewritten_queries)
        if scores is None:
            return []

        hyde_used = False
        hyde_queries: list[str] = []
        if self._should_use_hyde(scores):
            hyde_queries = self._build_hyde_queries(query, query_terms, district=district, tag=tag, unit=unit)
            if hyde_queries:
                rescored = self._score_rewritten_queries(rewritten_queries + hyde_queries)
                if rescored[0] is not None:
                    scores, dense_scores, sparse_scores, bm25_scores = rescored
                    rewritten_queries = rewritten_queries + hyde_queries
                    hyde_used = True

        adaptive_profile = self._adaptive_retrieval_profile(
            query,
            query_terms,
            int(top_k),
            district=district,
            tag=tag,
            unit=unit,
        )
        effective_top_k = int(adaptive_profile["effective_top_k"])
        candidate_multiplier = int(adaptive_profile["candidate_multiplier"])

        city_scope = {"\u5317\u4eac\u5e02", "\u5168\u5e02", "?????", "???"}
        ranked_items = []
        for idx, (score, doc) in enumerate(zip(scores, self.docs)):
            boosted_score, matched_terms, fb_val, relevance_val, graph_val, graph_entities = self._rerank_score(
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
                elif doc.get("district") in city_scope:
                    district_priority = 1
            ranked_items.append(
                (boosted_score, district_priority, matched_terms, doc, idx, fb_val, relevance_val, graph_val, graph_entities)
            )

        ranked_items.sort(key=lambda item: (item[0], item[1]), reverse=True)

        candidate_count = max(effective_top_k * candidate_multiplier, 20)
        if self._reranker and self.enable_reranker:
            candidate_count = max(effective_top_k * max(candidate_multiplier - 2, 4), 12)

        candidates = []
        seen_keys = set()
        for score, _, matched_terms, doc, idx, fb_val, relevance_val, graph_val, graph_entities in ranked_items[:candidate_count]:
            if score <= 0:
                continue
            if self.enable_post_processing:
                contradiction_penalty = self._contradiction_penalty(doc, query)
                if contradiction_penalty < 0:
                    score = score + contradiction_penalty
                    if score <= 0:
                        continue
            dedup_key = (doc.get("title", ""), doc.get("source", ""))
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            candidates.append((score, doc, matched_terms, idx, fb_val, relevance_val, graph_val, graph_entities))

        if self._reranker and self.enable_reranker and len(candidates) > effective_top_k:
            try:
                stripped = [(s, d, m, i) for s, d, m, i, _, _, _, _ in candidates]
                reranked = self._reranker.rerank(query, stripped, top_k=max(effective_top_k * 2, 10))
                meta_map = {id(d): (fb, rel, graph, graph_entities) for _, d, _, _, fb, rel, graph, graph_entities in candidates}
                candidates = [(s, d, m, i, *meta_map.get(id(d), (0.0, 0.0, 0.0, []))) for s, d, m, i in reranked]
            except Exception:
                pass

        results = []
        for item in candidates[:max(effective_top_k * 3, 15)]:
            graph_val = 0.0
            graph_entities = []
            if len(item) >= 8:
                score, doc, matched_terms, idx, fb, relevance_val, graph_val, graph_entities = item
            elif len(item) >= 6:
                score, doc, matched_terms, idx, fb, relevance_val = item
            elif len(item) >= 5:
                score, doc, matched_terms, idx, fb = item
                relevance_val = 0.0
            elif len(item) == 4:
                score, doc, matched_terms, idx = item
                fb = 0.0
                relevance_val = 0.0
            else:
                continue

            child_context = self._child_context_for_doc(idx, query_terms)
            child = child_context.get("chunk")
            child_snippet = ""
            parent_context = ""
            retrieval_granularity = "document"
            child_chunk_id = ""
            if child is not None:
                child_snippet = child.content[:220] + ("..." if len(child.content) > 220 else "")
                parent_context = self._parent_context(doc.get("content", ""), child=child)
                retrieval_granularity = "parent_child"
                child_chunk_id = child.chunk_id
                if child_context.get("matched_terms"):
                    matched_terms = list(dict.fromkeys((matched_terms or []) + child_context["matched_terms"]))

            snippet_source = child_snippet or doc.get("content", "")
            dense_val = round(float(dense_scores[idx]), 4) if idx < len(dense_scores) else 0.0
            sparse_val = round(float(sparse_scores[idx]), 4) if idx < len(sparse_scores) else 0.0
            bm25_val = round(float(bm25_scores[idx]), 4) if idx < len(bm25_scores) else 0.0
            hit = RetrievalHit(
                doc_id=doc.get("id", ""),
                title=doc.get("title", "\u672a\u547d\u540d\u6750\u6599"),
                doc_type=doc.get("doc_type", "\u53c2\u8003\u6750\u6599"),
                district=doc.get("district", ""),
                source=doc.get("source", "\u672c\u5730\u77e5\u8bc6\u5e93"),
                score=round(score, 4),
                snippet=(snippet_source[:220] + "...") if len(snippet_source) > 220 else snippet_source,
                matched_terms=matched_terms,
                retrieval_backend=self.active_backend,
                rewrite_count=len(rewritten_queries),
                dense_score=dense_val,
                sparse_score=sparse_val,
                bm25_score=bm25_val,
                lexical_score=sparse_val,
                full_content=doc.get("content", ""),
                rerank_score=round(score, 4) if self._reranker else 0.0,
                feedback_boost=round(fb, 4),
                relevance_score=round(relevance_val, 4),
                parent_doc_id=doc.get("id", ""),
                child_chunk_id=child_chunk_id,
                child_snippet=child_snippet,
                parent_context=parent_context,
                retrieval_granularity=retrieval_granularity,
                adaptive_strategy=adaptive_profile.get("strategy", "standard"),
                adaptive_reason=adaptive_profile.get("reasons", []),
                effective_top_k=effective_top_k,
                score_breakdown={
                    "final_score": round(float(score), 4),
                    "dense_score": dense_val,
                    "sparse_score": sparse_val,
                    "bm25_score": bm25_val,
                    "lexical_score": sparse_val,
                    "rerank_score": round(float(score), 4) if self._reranker else 0.0,
                    "feedback_boost": round(float(fb), 4),
                    "relevance_score": round(float(relevance_val), 4),
                    "graph_score": round(float(graph_val), 4),
                    "graph_entities": graph_entities,
                    "matched_terms_count": len(matched_terms or []),
                    "district": doc.get("district", ""),
                    "adaptive_strategy": adaptive_profile.get("strategy", "standard"),
                    "hyde_used": hyde_used,
                    "hyde_query_count": len(hyde_queries),
                },
            )
            results.append(asdict(hit))

        if self.enable_graph_augment and self._graph_augmentor:
            try:
                graph_facts = self._graph_augmentor.query_related_facts(query, district=district, limit=max(1, effective_top_k // 2))
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
                        bm25_score=0.0,
                        lexical_score=0.0,
                        rerank_score=0.0,
                        feedback_boost=0.0,
                        relevance_score=0.0,
                        full_content=fact["content"],
                        adaptive_strategy=adaptive_profile.get("strategy", "standard"),
                        adaptive_reason=adaptive_profile.get("reasons", []),
                        effective_top_k=effective_top_k,
                        score_breakdown={
                            "final_score": round(fact.get("_graph_relevance", 0.5), 4),
                            "dense_score": 0.0,
                            "sparse_score": 0.0,
                            "bm25_score": 0.0,
                            "lexical_score": 0.0,
                            "rerank_score": 0.0,
                            "feedback_boost": 0.0,
                            "relevance_score": 0.0,
                            "graph_score": round(fact.get("_graph_relevance", 0.5), 4),
                            "graph_entities": fact.get("graph_matched_terms", []),
                            "graph_community": fact.get("graph_community", {}),
                            "source": "knowledge_graph",
                        },
                    )
                    results.append(asdict(graph_hit))
            except Exception:
                pass

        results = self._post_process_results(results, query, query_terms, effective_top_k)
        if self._semantic_cache is not None:
            self._semantic_cache.set(
                query=query,
                results=results,
                query_vector=cache_vector,
                district=district,
                tag=tag,
                unit=unit,
                top_k=top_k,
                corpus_version=self._corpus_version(),
            )
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
            "semantic_chunking_enabled": self.enable_semantic_chunking,
            "semantic_chunk_threshold": self.semantic_chunk_threshold,
            "parent_child_enabled": self.enable_parent_child,
            "adaptive_retrieval_enabled": self.enable_adaptive_retrieval,
            "adaptive_max_top_k": self.adaptive_max_top_k,
            "semantic_cache_enabled": self.enable_semantic_cache,
            "semantic_cache": self._semantic_cache.stats() if self._semantic_cache else {},
            "bm25_enabled": self.enable_bm25,
            "bm25_index_available": self.bm25_index is not None,
            "bm25_weight": self.bm25_weight,
            "tfidf_weight": self.tfidf_weight,
            "hyde_enabled": self.enable_hyde,
            "hyde_trigger_threshold": self.hyde_trigger_threshold,
            "hyde_max_queries": self.hyde_max_queries,
            "relevance_scorer_enabled": self.enable_relevance_scorer,
            "relevance_weight": self.relevance_weight,
            "relevance_scorer": self._relevance_scorer.stats() if self._relevance_scorer else {},
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "reranker_enabled": self.enable_reranker,
            "reranker": self._reranker.status() if self._reranker else {},
            "graph_augment_enabled": self.enable_graph_augment,
            "feedback_boost_enabled": self.enable_feedback_boost,
            "post_processing_enabled": self.enable_post_processing,
        }

    def reload(self) -> dict[str, Any]:
        old_count = len(self.docs)
        old_chunk_count = len(self.chunks)
        self.docs = self._load_docs(self.corpus_path)
        if self._semantic_cache:
            self._semantic_cache.clear()
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
