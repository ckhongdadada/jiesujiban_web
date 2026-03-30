from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

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


class PolicyRetriever:
    def __init__(self, corpus_path=None, backend=None):
        project_root = os.path.dirname(os.path.dirname(__file__))
        default_corpus_path = os.path.join(project_root, "data", "policy_case_corpus.jsonl")
        if not os.path.exists(default_corpus_path):
            default_corpus_path = os.path.join(project_root, "data", "policy_case_corpus.sample.jsonl")
        self.corpus_path = corpus_path or default_corpus_path
        self.requested_backend = (backend or os.getenv("RAG_BACKEND", "bge")).lower()
        self.active_backend = "empty"
        self.docs = self._load_docs(self.corpus_path)
        self.vectorizer = None
        self.embedding_model = None
        self.doc_vectors = None
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
        normalized["issue_type"] = normalized.get("issue_type", "")
        normalized["unit"] = normalized.get("unit", "")
        normalized["applicable_tags"] = [str(tag) for tag in normalized.get("applicable_tags", [])]
        return normalized

    def _build_index(self):
        if not self.docs:
            self.active_backend = "empty"
            return

        corpus = [self._compose_doc_text(doc) for doc in self.docs]

        if self.requested_backend == "bge" and SENTENCE_TRANSFORMERS_AVAILABLE and NUMPY_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(
                    "BAAI/bge-small-zh-v1.5",
                    device="cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"
                )
                self.doc_vectors = self.embedding_model.encode(
                    corpus, 
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                self.active_backend = "bge"
                print("[RAG] 使用 BGE 向量检索，文档数:", len(self.docs))
                return
            except Exception as e:
                print("[RAG] BGE 初始化失败，回退到 TF-IDF:", e)

        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=1,
            )
            self.doc_vectors = self.vectorizer.fit_transform(corpus)
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
        rules = [
            ("垃圾", "垃圾清运"),
            ("异味", "异味扰民"),
            ("噪声", "噪声扰民"),
            ("施工", "施工扰民"),
            ("停车", "停车秩序"),
            ("违停", "停车秩序"),
            ("积水", "道路积水"),
            ("物业", "物业服务"),
            ("消防", "消防通道"),
            ("占道", "占道经营"),
            ("路灯", "照明设施"),
            ("排水", "排水设施"),
        ]
        keywords = []
        for token, label in rules:
            if token in text and label not in keywords:
                keywords.append(label)
        return keywords

    def _collect_query_terms(
        self,
        query,
        district=None,
        tag=None,
        unit=None,
    ):
        terms = self.extract_issue_keywords(query)
        if district:
            terms.append(district)
        if tag:
            terms.append(tag)
        if unit:
            terms.append(unit)
        terms.extend(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,10}", query))
        deduped = []
        for term in terms:
            if term and term not in deduped:
                deduped.append(term)
        return deduped

    def _matched_terms(self, doc, query_terms):
        haystack = self._compose_doc_text(doc)
        return [term for term in query_terms if term and term in haystack][:8]

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
        
        if self.active_backend == "bge" and self.embedding_model is not None and NUMPY_AVAILABLE:
            query_vector = self.embedding_model.encode(
                [query], 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            scores = np.dot(query_vector, self.doc_vectors.T)[0]
        elif self.active_backend == "tfidf" and self.vectorizer is not None and SKLEARN_AVAILABLE:
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.doc_vectors)[0]
        else:
            return []

        scored_docs = []
        for score, doc in zip(scores, self.docs):
            boosted_score = float(score)
            if district and doc.get("district") == district:
                boosted_score += 0.15
            if district and doc.get("district") in ("北京市", "全市"):
                boosted_score += 0.05
            if tag and tag in doc.get("applicable_tags", []):
                boosted_score += 0.08
            if unit and unit == doc.get("unit"):
                boosted_score += 0.08
            if preferred_doc_types and doc.get("doc_type") in preferred_doc_types:
                boosted_score += 0.06
            matched_terms = self._matched_terms(doc, query_terms)
            boosted_score += min(len(matched_terms) * 0.02, 0.12)
            scored_docs.append((boosted_score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        seen_keys = set()
        results = []
        for score, doc in scored_docs[: max(top_k * 3, 10)]:
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
                matched_terms=self._matched_terms(doc, query_terms),
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