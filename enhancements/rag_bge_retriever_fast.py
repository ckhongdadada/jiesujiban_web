from __future__ import annotations

import json
import os
import re
import hashlib
import threading
from dataclasses import asdict, dataclass
from typing import Any
from functools import lru_cache

from enhancements.data_paths import get_policy_corpus_path, get_policy_corpus_sample_path

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
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


BEIJING_DISTRICTS = {
    "东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区",
    "门头沟区", "房山区", "通州区", "顺义区", "昌平区", "大兴区",
    "怀柔区", "平谷区", "密云区", "延庆区", "全市", "北京市",
}

ISSUE_RULES = [
    ("垃圾", "垃圾清运"), ("清运", "垃圾清运"), ("异味", "异味扰民"),
    ("噪声", "噪声扰民"), ("扰民", "扰民"), ("施工", "施工扰民"),
    ("停车", "停车秩序"), ("违停", "停车秩序"), ("积水", "道路积水"),
    ("物业", "物业服务"), ("消防", "消防通道"), ("占道", "占道经营"),
    ("路灯", "照明设施"), ("排水", "排水设施"), ("扬尘", "空气质量"),
    ("空气", "空气质量"), ("绿化", "园林绿化"),
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


class QueryCache:
    """查询结果缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.stats = {'hits': 0, 'misses': 0}
    
    def _get_key(self, query: str, district: str | None, tag: str | None, 
                 unit: str | None, top_k: int) -> str:
        key_str = f"{query}|{district}|{tag}|{unit}|{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, district: str | None = None, tag: str | None = None,
            unit: str | None = None, top_k: int = 3) -> list[dict] | None:
        key = self._get_key(query, district, tag, unit, top_k)
        
        with self.lock:
            if key in self.cache:
                self.stats['hits'] += 1
                return self.cache[key]
            self.stats['misses'] += 1
            return None
    
    def set(self, query: str, result: list[dict], district: str | None = None,
            tag: str | None = None, unit: str | None = None, top_k: int = 3):
        key = self._get_key(query, district, tag, unit, top_k)
        
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = result
    
    def get_stats(self) -> dict:
        with self.lock:
            total = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate
            }
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.stats = {'hits': 0, 'misses': 0}


class PolicyRetriever:
    """优化的RAG检索器"""
    
    def __init__(self, corpus_path=None, backend=None, enable_cache=True, cache_size=1000):
        default_corpus_path = get_policy_corpus_path()
        if not default_corpus_path.exists():
            default_corpus_path = get_policy_corpus_sample_path()
        self.corpus_path = corpus_path or str(default_corpus_path)
        self.requested_backend = (backend or os.getenv("RAG_BACKEND", "bge")).lower()
        self.active_backend = "empty"
        self.docs = self._load_docs(self.corpus_path)
        self.vectorizer = None
        self.embedding_model = None
        self.doc_vectors = None
        self.faiss_index = None
        
        self.enable_cache = enable_cache
        self.query_cache = QueryCache(max_size=cache_size) if enable_cache else None
        
        self._build_index()

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

        if self.requested_backend == "bge" and SENTENCE_TRANSFORMERS_AVAILABLE and NUMPY_AVAILABLE:
            try:
                use_cuda = os.getenv("USE_CUDA", "true").lower() == "true"
                device = "cuda" if use_cuda else "cpu"
                
                self.embedding_model = SentenceTransformer(
                    "BAAI/bge-small-zh-v1.5",
                    device=device,
                )
                self.doc_vectors = self.embedding_model.encode(
                    corpus,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                
                if FAISS_AVAILABLE and len(self.docs) >= 100:
                    self._build_faiss_index()
                    self.active_backend = "bge_faiss"
                    print(f"[RAG] 使用 BGE + FAISS 向量检索，文档数: {len(self.docs)}")
                else:
                    self.active_backend = "bge"
                    print(f"[RAG] 使用 BGE 向量检索，文档数: {len(self.docs)}")
                
                if self.enable_cache:
                    print(f"[RAG] 查询缓存已启用，缓存大小: {self.query_cache.max_size}")
                return
            except Exception as exc:
                print(f"[RAG] BGE 初始化失败，回退到 TF-IDF: {exc}")

        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            self.doc_vectors = self.vectorizer.fit_transform(corpus)
            self.active_backend = "tfidf"
            print(f"[RAG] 使用 TF-IDF 检索，文档数: {len(self.docs)}")
        else:
            self.active_backend = "empty"
            print("[RAG] 无可用检索后端")

    def _build_faiss_index(self):
        """构建FAISS索引"""
        if not FAISS_AVAILABLE or self.doc_vectors is None:
            return
        
        n_vectors, dim = self.doc_vectors.shape
        
        if n_vectors < 100:
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.doc_vectors)
        else:
            nlist = min(100, n_vectors // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.faiss_index.train(self.doc_vectors)
            self.faiss_index.add(self.doc_vectors)
        
        print(f"[RAG] FAISS索引构建完成，向量数: {n_vectors}")

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

    def _rerank_score(self, base_score, doc, query_terms, district=None, tag=None, 
                      unit=None, preferred_doc_types=None):
        boosted = float(base_score)
        matched_terms = self._matched_terms(doc, query_terms)
        doc_district = doc.get("district", "")

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
            boosted += min(len(matched_terms) * 0.035, 0.20)
        elif district and doc_district not in (district, "北京市", "全市"):
            boosted -= 0.08

        if district and matched_terms:
            locality_terms = [term for term in matched_terms if term == district or term == district[:-1]]
            if locality_terms:
                boosted += 0.04

        return boosted, matched_terms

    def _search_internal(self, query, top_k=3, district=None, tag=None, 
                         unit=None, preferred_doc_types=None):
        """内部检索逻辑"""
        if not query or self.active_backend == "empty":
            return []

        query_terms = self._collect_query_terms(query, district=district, tag=tag, unit=unit)
        retrieval_query = self._build_query_for_vector_search(query, query_terms)
        preferred_doc_types = self._preferred_doc_types(tag, preferred_doc_types)

        if self.active_backend in ("bge", "bge_faiss") and self.embedding_model is not None and NUMPY_AVAILABLE:
            query_vector = self.embedding_model.encode(
                [retrieval_query], 
                normalize_embeddings=True, 
                show_progress_bar=False
            )
            
            if self.active_backend == "bge_faiss" and self.faiss_index is not None:
                scores, indices = self.faiss_index.search(query_vector, min(top_k * 6, len(self.docs)))
                scores = scores[0]
                doc_indices = indices[0]
            else:
                scores = np.dot(query_vector, self.doc_vectors.T)[0]
                doc_indices = np.arange(len(self.docs))
        elif self.active_backend == "tfidf" and self.vectorizer is not None and SKLEARN_AVAILABLE:
            query_vector = self.vectorizer.transform([retrieval_query])
            scores = cosine_similarity(query_vector, self.doc_vectors)[0]
            doc_indices = np.arange(len(self.docs))
        else:
            return []

        exact_or_city = []
        fallback = []
        
        for idx, score in zip(doc_indices, scores):
            if idx >= len(self.docs):
                continue
            doc = self.docs[idx]
            boosted_score, matched_terms = self._rerank_score(
                score, doc, query_terms,
                district=district, tag=tag, unit=unit,
                preferred_doc_types=preferred_doc_types,
            )
            item = (boosted_score, matched_terms, doc)
            if not district or doc.get("district") in (district, "北京市", "全市"):
                exact_or_city.append(item)
            else:
                fallback.append(item)

        exact_or_city.sort(key=lambda item: item[0], reverse=True)
        fallback.sort(key=lambda item: item[0], reverse=True)
        ranked_items = exact_or_city + fallback

        seen_keys = set()
        results = []
        for score, matched_terms, doc in ranked_items[: max(top_k * 6, 20)]:
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
            )
            results.append(asdict(hit))
            if len(results) >= top_k:
                break
        return results

    def search(self, query, top_k=3, district=None, tag=None, unit=None, 
               preferred_doc_types=None, use_cache=True):
        """检索接口（带缓存）"""
        if use_cache and self.enable_cache and self.query_cache:
            cached = self.query_cache.get(query, district, tag, unit, top_k)
            if cached is not None:
                return cached
        
        results = self._search_internal(query, top_k, district, tag, unit, preferred_doc_types)
        
        if use_cache and self.enable_cache and self.query_cache:
            self.query_cache.set(query, results, district, tag, unit, top_k)
        
        return results

    def batch_search(self, queries: list[str], top_k: int = 3, 
                     districts: list[str] | None = None,
                     tags: list[str] | None = None,
                     units: list[str] | None = None) -> list[list[dict]]:
        """批量检索接口"""
        if not queries:
            return []
        
        n = len(queries)
        districts = districts or [None] * n
        tags = tags or [None] * n
        units = units or [None] * n
        
        results = []
        for i, query in enumerate(queries):
            result = self.search(
                query, 
                top_k=top_k,
                district=districts[i],
                tag=tags[i],
                unit=units[i]
            )
            results.append(result)
        
        return results

    def batch_search_optimized(self, queries: list[str], top_k: int = 3,
                                districts: list[str] | None = None,
                                tags: list[str] | None = None,
                                units: list[str] | None = None) -> list[list[dict]]:
        """优化的批量检索（向量化）"""
        if not queries or self.active_backend == "empty":
            return [[] for _ in queries]
        
        n = len(queries)
        districts = districts or [None] * n
        tags = tags or [None] * n
        units = units or [None] * n
        
        if self.active_backend in ("bge", "bge_faiss") and self.embedding_model is not None:
            query_terms_list = [
                self._collect_query_terms(q, d, t, u)
                for q, d, t, u in zip(queries, districts, tags, units)
            ]
            
            retrieval_queries = [
                self._build_query_for_vector_search(q, qt)
                for q, qt in zip(queries, query_terms_list)
            ]
            
            query_vectors = self.embedding_model.encode(
                retrieval_queries,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            if self.active_backend == "bge_faiss" and self.faiss_index is not None:
                k = min(top_k * 6, len(self.docs))
                scores, indices = self.faiss_index.search(query_vectors, k)
            else:
                scores = np.dot(query_vectors, self.doc_vectors.T)
                indices = np.argsort(-scores, axis=1)[:, :top_k * 6]
            
            all_results = []
            for i in range(n):
                query_scores = scores[i] if len(scores.shape) > 1 else scores
                query_indices = indices[i]
                
                preferred_doc_types = self._preferred_doc_types(tags[i], None)
                
                ranked_items = []
                for j, idx in enumerate(query_indices):
                    if idx >= len(self.docs):
                        continue
                    doc = self.docs[idx]
                    score = query_scores[j] if len(query_scores.shape) > 0 else query_scores
                    
                    boosted_score, matched_terms = self._rerank_score(
                        score, doc, query_terms_list[i],
                        district=districts[i],
                        tag=tags[i],
                        unit=units[i],
                        preferred_doc_types=preferred_doc_types
                    )
                    ranked_items.append((boosted_score, matched_terms, doc))
                
                ranked_items.sort(key=lambda x: x[0], reverse=True)
                
                seen_keys = set()
                results = []
                for score, matched_terms, doc in ranked_items[:top_k]:
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
                    )
                    results.append(asdict(hit))
                
                all_results.append(results)
            
            return all_results
        else:
            return self.batch_search(queries, top_k, districts, tags, units)

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        if self.enable_cache and self.query_cache:
            return self.query_cache.get_stats()
        return {'cache_disabled': True}

    def clear_cache(self):
        """清空缓存"""
        if self.enable_cache and self.query_cache:
            self.query_cache.clear()

    def describe(self):
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "document_count": len(self.docs),
            "corpus_path": self.corpus_path,
            "faiss_enabled": self.faiss_index is not None,
            "cache_enabled": self.enable_cache,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "faiss_available": FAISS_AVAILABLE,
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
    
    def batch_retrieve(self, queries: list[str], districts: list[str] | None = None, **kwargs):
        return self.batch_search_optimized(queries, districts=districts, **kwargs)
