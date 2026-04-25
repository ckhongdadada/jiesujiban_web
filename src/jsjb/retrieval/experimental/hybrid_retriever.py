from __future__ import annotations

import json
import os
import re
import hashlib
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any
from collections import defaultdict

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

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


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
    chunk_id: str = ""
    chunk_index: int = 0


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    content: str
    doc_type: str
    district: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    tags: list[str]
    issue_type: str
    unit: str
    applicable_tags: list[str]


class BM25Index:
    """BM25索引实现"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)
        self.doc_lens = []
        self.avgdl = 0
        self.doc_term_freqs = []
        self.n_docs = 0
        self.idf = {}
        self.doc_ids = []
    
    def _tokenize(self, text: str) -> list[str]:
        if JIEBA_AVAILABLE:
            return list(jieba.cut(text))
        return re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text.lower())
    
    def build(self, documents: list[tuple[str, str]]):
        """构建BM25索引
        
        Args:
            documents: list of (doc_id, text)
        """
        self.n_docs = len(documents)
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.doc_term_freqs = []
        self.doc_lens = []
        
        term_doc_freq = defaultdict(int)
        
        for doc_id, text in documents:
            tokens = self._tokenize(text)
            self.doc_lens.append(len(tokens))
            
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            
            self.doc_term_freqs.append(dict(term_freq))
            
            for term in term_freq:
                term_doc_freq[term] += 1
        
        self.avgdl = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 0
        
        for term, df in term_doc_freq.items():
            self.idf[term] = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """搜索
        
        Returns:
            list of (doc_id, score)
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for i in range(self.n_docs):
            score = 0.0
            doc_len = self.doc_lens[i]
            term_freqs = self.doc_term_freqs[i]
            
            for token in query_tokens:
                if token not in term_freqs:
                    continue
                
                tf = term_freqs[token]
                idf = self.idf.get(token, 0)
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
            
            if score > 0:
                scores.append((self.doc_ids[i], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DocumentChunker:
    """文档分块器"""
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def _split_by_sentence(self, text: str) -> list[str]:
        sentences = re.split(r'([。！？\n]+)', text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                result.append(sentences[i])
        if len(sentences) % 2 == 1 and sentences[-1]:
            result.append(sentences[-1])
        return result
    
    def chunk_document(self, doc: dict[str, Any]) -> list[DocumentChunk]:
        """将文档分块"""
        content = doc.get("content", "")
        
        if len(content) <= self.chunk_size:
            return [DocumentChunk(
                chunk_id=f"{doc.get('id', '')}_0",
                doc_id=doc.get("id", ""),
                title=doc.get("title", "未命名材料"),
                content=content,
                doc_type=doc.get("doc_type", "参考材料"),
                district=doc.get("district", "全市"),
                source=doc.get("source", "本地知识库"),
                chunk_index=0,
                start_char=0,
                end_char=len(content),
                tags=doc.get("tags", []),
                issue_type=doc.get("issue_type", ""),
                unit=doc.get("unit", ""),
                applicable_tags=doc.get("applicable_tags", [])
            )]
        
        sentences = self._split_by_sentence(content)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_content = "".join(current_chunk)
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc.get('id', '')}_{chunk_index}",
                        doc_id=doc.get("id", ""),
                        title=doc.get("title", "未命名材料"),
                        content=chunk_content,
                        doc_type=doc.get("doc_type", "参考材料"),
                        district=doc.get("district", "全市"),
                        source=doc.get("source", "本地知识库"),
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(chunk_content),
                        tags=doc.get("tags", []),
                        issue_type=doc.get("issue_type", ""),
                        unit=doc.get("unit", ""),
                        applicable_tags=doc.get("applicable_tags", [])
                    ))
                    chunk_index += 1
                    start_char += len(chunk_content)
                
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_sentences = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunk_content = "".join(current_chunk)
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc.get('id', '')}_{chunk_index}",
                    doc_id=doc.get("id", ""),
                    title=doc.get("title", "未命名材料"),
                    content=chunk_content,
                    doc_type=doc.get("doc_type", "参考材料"),
                    district=doc.get("district", "全市"),
                    source=doc.get("source", "本地知识库"),
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_content),
                    tags=doc.get("tags", []),
                    issue_type=doc.get("issue_type", ""),
                    unit=doc.get("unit", ""),
                    applicable_tags=doc.get("applicable_tags", [])
                ))
        
        return chunks


class Reranker:
    """重排序模型"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def load(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local_path = os.path.join(base_dir, "reranker_models", self.model_name.replace("/", "___"))
            
            if os.path.exists(local_path):
                self.model = AutoModelForSequenceClassification.from_pretrained(local_path)
                self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            return True
        except Exception as e:
            print(f"[重排序] 加载失败: {e}")
            return False
    
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """重排序文档"""
        if not self.model or not documents:
            return documents[:top_k]
        
        try:
            import torch
            
            pairs = [[query, doc.get("snippet", doc.get("content", ""))] for doc in documents]
            
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(-1)
            
            scores = torch.softmax(scores, dim=-1).cpu().numpy()
            
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            result = []
            for doc, score in scored_docs[:top_k]:
                doc_copy = dict(doc)
                doc_copy["rerank_score"] = float(score)
                result.append(doc_copy)
            
            return result
        except Exception as e:
            print(f"[重排序] 失败: {e}")
            return documents[:top_k]


class HybridRetriever:
    """混合检索器（向量+BM25）"""
    
    def __init__(
        self,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60
    ):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
    
    def reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """倒数排名融合"""
        rrf_scores = defaultdict(float)
        
        for rank, (doc_id, _) in enumerate(vector_results):
            rrf_scores[doc_id] += self.vector_weight / (self.rrf_k + rank + 1)
        
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] += self.bm25_weight / (self.rrf_k + rank + 1)
        
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


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


class IncrementalIndex:
    """增量索引管理"""
    
    def __init__(self):
        self.added_docs = []
        self.deleted_doc_ids = set()
        self.lock = threading.Lock()
        self.last_merge_time = time.time()
        self.merge_interval = 3600
    
    def add_document(self, doc: dict[str, Any]):
        with self.lock:
            self.added_docs.append(doc)
    
    def delete_document(self, doc_id: str):
        with self.lock:
            self.deleted_doc_ids.add(doc_id)
    
    def get_pending_changes(self) -> tuple[list[dict], set[str]]:
        with self.lock:
            return list(self.added_docs), set(self.deleted_doc_ids)
    
    def clear_pending(self):
        with self.lock:
            self.added_docs.clear()
            self.deleted_doc_ids.clear()
            self.last_merge_time = time.time()
    
    def should_merge(self) -> bool:
        return (
            time.time() - self.last_merge_time > self.merge_interval or
            len(self.added_docs) > 100 or
            len(self.deleted_doc_ids) > 50
        )


class PolicyRetrieverAdvanced:
    """高级RAG检索器"""
    
    def __init__(
        self,
        corpus_path=None,
        backend=None,
        enable_cache=True,
        cache_size=1000,
        enable_chunking=True,
        chunk_size=300,
        chunk_overlap=50,
        enable_bm25=True,
        enable_reranker=False,
        vector_weight=0.6,
        bm25_weight=0.4
    ):
        default_corpus_path = get_policy_corpus_path()
        if not default_corpus_path.exists():
            default_corpus_path = get_policy_corpusSample_path()
        self.corpus_path = corpus_path or str(default_corpus_path)
        self.requested_backend = (backend or os.getenv("RAG_BACKEND", "bge")).lower()
        self.active_backend = "empty"
        
        self.enable_cache = enable_cache
        self.query_cache = QueryCache(max_size=cache_size) if enable_cache else None
        
        self.enable_chunking = enable_chunking
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        ) if enable_chunking else None
        
        self.docs = []
        self.chunks = []
        self.chunk_index = {}
        
        self.embedding_model = None
        self.chunk_vectors = None
        self.faiss_index = None
        
        self.enable_bm25 = enable_bm25
        self.bm25_index = None
        
        self.enable_reranker = enable_reranker
        self.reranker = Reranker() if enable_reranker else None
        
        self.hybrid_retriever = HybridRetriever(
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        
        self.incremental_index = IncrementalIndex()
        
        self._load_and_build_index()

    def _load_and_build_index(self):
        """加载文档并构建索引"""
        self.docs = self._load_docs(self.corpus_path)
        
        if self.enable_chunking and self.chunker:
            self._build_chunk_index()
        
        self._build_vector_index()
        
        if self.enable_bm25:
            self._build_bm25_index()
        
        if self.enable_reranker and self.reranker:
            self.reranker.load()
        
        print(f"[RAG高级] 初始化完成")
        print(f"  - 文档数: {len(self.docs)}")
        print(f"  - 分块数: {len(self.chunks)}")
        print(f"  - BM25: {'启用' if self.enable_bm25 else '禁用'}")
        print(f"  - 重排序: {'启用' if self.enable_reranker else '禁用'}")

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
        """构建分块索引"""
        self.chunks = []
        self.chunk_index = {}
        
        for doc in self.docs:
            doc_chunks = self.chunker.chunk_document(doc)
            for chunk in doc_chunks:
                self.chunks.append(chunk)
                self.chunk_index[chunk.chunk_id] = chunk
        
        print(f"[RAG高级] 分块完成，共 {len(self.chunks)} 个分块")

    def _build_vector_index(self):
        """构建向量索引"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not NUMPY_AVAILABLE:
            self.active_backend = "empty"
            return
        
        if self.enable_chunking and self.chunks:
            texts = [chunk.content for chunk in self.chunks]
        else:
            texts = [self._compose_doc_text(doc) for doc in self.docs]
        
        try:
            use_cuda = os.getenv("USE_CUDA", "true").lower() == "true"
            device = "cuda" if use_cuda else "cpu"
            
            self.embedding_model = SentenceTransformer(
                "BAAI/bge-small-zh-v1.5",
                device=device,
            )
            
            self.chunk_vectors = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            
            if FAISS_AVAILABLE and len(texts) >= 100:
                self._build_faiss_index()
                self.active_backend = "bge_faiss"
            else:
                self.active_backend = "bge"
            
            print(f"[RAG高级] 向量索引构建完成")
        except Exception as exc:
            print(f"[RAG高级] 向量索引构建失败: {exc}")
            self.active_backend = "empty"

    def _build_faiss_index(self):
        """构建FAISS索引"""
        if not FAISS_AVAILABLE or self.chunk_vectors is None:
            return
        
        n_vectors, dim = self.chunk_vectors.shape
        
        if n_vectors < 100:
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.chunk_vectors)
        else:
            nlist = min(100, n_vectors // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.faiss_index.train(self.chunk_vectors)
            self.faiss_index.add(self.chunk_vectors)
        
        print(f"[RAG高级] FAISS索引构建完成，向量数: {n_vectors}")

    def _build_bm25_index(self):
        """构建BM25索引"""
        if not self.enable_bm25:
            return
        
        self.bm25_index = BM25Index()
        
        if self.enable_chunking and self.chunks:
            documents = [(chunk.chunk_id, chunk.content) for chunk in self.chunks]
        else:
            documents = [(doc.get("id", ""), self._compose_doc_text(doc)) for doc in self.docs]
        
        self.bm25_index.build(documents)
        print(f"[RAG高级] BM25索引构建完成")

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

    def _matched_terms(self, text, query_terms):
        return [term for term in query_terms if term and term in text][:8]

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

    def _vector_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """向量检索"""
        if self.active_backend not in ("bge", "bge_faiss") or self.embedding_model is None:
            return []
        
        query_vector = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        
        if self.active_backend == "bge_faiss" and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vector, min(top_k, len(self.chunks) if self.chunks else len(self.docs)))
            return [(str(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        else:
            scores = np.dot(query_vector, self.chunk_vectors.T)[0]
            top_indices = np.argsort(-scores)[:top_k]
            chunk_ids = [self.chunks[i].chunk_id if self.chunks else self.docs[i].get("id", "") for i in top_indices]
            return [(chunk_ids[i], float(scores[top_indices[i]])) for i in range(len(top_indices))]

    def _bm25_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """BM25检索"""
        if not self.bm25_index:
            return []
        
        return self.bm25_index.search(query, top_k)

    def _hybrid_search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """混合检索"""
        vector_results = self._vector_search(query, top_k=top_k * 2)
        bm25_results = self._bm25_search(query, top_k=top_k * 2)
        
        if not vector_results and not bm25_results:
            return []
        
        if not vector_results:
            return bm25_results[:top_k]
        
        if not bm25_results:
            return vector_results[:top_k]
        
        return self.hybrid_retriever.reciprocal_rank_fusion(
            vector_results, 
            bm25_results, 
            top_k=top_k
        )

    def _get_chunk_or_doc(self, chunk_id: str) -> dict[str, Any] | None:
        """获取分块或文档"""
        if self.enable_chunking and chunk_id in self.chunk_index:
            chunk = self.chunk_index[chunk_id]
            return {
                "id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "content": chunk.content,
                "doc_type": chunk.doc_type,
                "district": chunk.district,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "tags": chunk.tags,
                "issue_type": chunk.issue_type,
                "unit": chunk.unit,
                "applicable_tags": chunk.applicable_tags
            }
        
        for doc in self.docs:
            if doc.get("id") == chunk_id:
                return doc
        
        return None

    def _rerank_score(self, base_score, doc, query_terms, district=None, tag=None, 
                      unit=None, preferred_doc_types=None):
        boosted = float(base_score)
        matched_terms = self._matched_terms(
            doc.get("content", "") + doc.get("title", ""), 
            query_terms
        )
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

        return boosted, matched_terms

    def search(
        self,
        query,
        top_k=3,
        district=None,
        tag=None,
        unit=None,
        preferred_doc_types=None,
        use_cache=True,
        use_reranker=True
    ):
        """检索接口"""
        if use_cache and self.enable_cache and self.query_cache:
            cached = self.query_cache.get(query, district, tag, unit, top_k)
            if cached is not None:
                return cached
        
        if not query or self.active_backend == "empty":
            return []
        
        query_terms = self._collect_query_terms(query, district=district, tag=tag, unit=unit)
        retrieval_query = self._build_query_for_vector_search(query, query_terms)
        preferred_doc_types = self._preferred_doc_types(tag, preferred_doc_types)
        
        if self.enable_bm25:
            search_results = self._hybrid_search(retrieval_query, top_k=top_k * 6)
        else:
            search_results = self._vector_search(retrieval_query, top_k=top_k * 6)
        
        exact_or_city = []
        fallback = []
        
        for chunk_id, score in search_results:
            doc = self._get_chunk_or_doc(chunk_id)
            if not doc:
                continue
            
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
        
        if self.enable_reranker and self.reranker and use_reranker and len(ranked_items) > top_k:
            rerank_docs = [doc for _, _, doc in ranked_items[:top_k * 3]]
            reranked_docs = self.reranker.rerank(query, rerank_docs, top_k=top_k * 2)
            
            final_results = []
            for doc in reranked_docs:
                hit = RetrievalHit(
                    doc_id=doc.get("id", ""),
                    title=doc.get("title", "未命名材料"),
                    doc_type=doc.get("doc_type", "参考材料"),
                    district=doc.get("district", ""),
                    source=doc.get("source", "本地知识库"),
                    score=doc.get("rerank_score", doc.get("score", 0)),
                    snippet=(doc.get("content", "")[:160] + "...") if len(doc.get("content", "")) > 160 else doc.get("content", ""),
                    matched_terms=doc.get("matched_terms", []),
                    chunk_id=doc.get("chunk_id", ""),
                    chunk_index=doc.get("chunk_index", 0)
                )
                final_results.append(asdict(hit))
            
            if use_cache and self.enable_cache and self.query_cache:
                self.query_cache.set(query, final_results[:top_k], district, tag, unit, top_k)
            
            return final_results[:top_k]
        
        seen_keys = set()
        results = []
        for score, matched_terms, doc in ranked_items[: max(top_k * 6, 20)]:
            if score <= 0:
                continue
            dedup_key = (doc.get("title", ""), doc.get("source", ""), doc.get("chunk_id", ""))
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
                chunk_id=doc.get("chunk_id", ""),
                chunk_index=doc.get("chunk_index", 0)
            )
            results.append(asdict(hit))
            if len(results) >= top_k:
                break
        
        if use_cache and self.enable_cache and self.query_cache:
            self.query_cache.set(query, results, district, tag, unit, top_k)
        
        return results

    def add_document(self, doc: dict[str, Any]):
        """添加文档（增量）"""
        self.incremental_index.add_document(doc)
        
        if self.incremental_index.should_merge():
            self._merge_incremental_changes()

    def _merge_incremental_changes(self):
        """合并增量更新"""
        added_docs, deleted_doc_ids = self.incremental_index.get_pending_changes()
        
        if not added_docs and not deleted_doc_ids:
            return
        
        print(f"[RAG高级] 合并增量更新: 添加 {len(added_docs)}, 删除 {len(deleted_doc_ids)}")
        
        self.docs = [doc for doc in self.docs if doc.get("id") not in deleted_doc_ids]
        self.docs.extend(added_docs)
        
        if self.enable_chunking:
            self._build_chunk_index()
        
        self._build_vector_index()
        
        if self.enable_bm25:
            self._build_bm25_index()
        
        self.incremental_index.clear_pending()

    def get_cache_stats(self) -> dict:
        if self.enable_cache and self.query_cache:
            return self.query_cache.get_stats()
        return {'cache_disabled': True}

    def clear_cache(self):
        if self.enable_cache and self.query_cache:
            self.query_cache.clear()

    def describe(self):
        return {
            "active_backend": self.active_backend,
            "document_count": len(self.docs),
            "chunk_count": len(self.chunks),
            "corpus_path": self.corpus_path,
            "faiss_enabled": self.faiss_index is not None,
            "bm25_enabled": self.bm25_index is not None,
            "reranker_enabled": self.reranker is not None and self.reranker.model is not None,
            "cache_enabled": self.enable_cache,
            "chunking_enabled": self.enable_chunking,
        }
