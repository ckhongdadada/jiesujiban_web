from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    def __init__(self, corpus_path: str | None = None, backend: str | None = None) -> None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        self.corpus_path = corpus_path or os.path.join(project_root, "data", "policy_case_corpus.sample.jsonl")
        self.requested_backend = (backend or os.getenv("RAG_BACKEND", "tfidf")).lower()
        self.active_backend = "empty"
        self.docs = self._load_docs(self.corpus_path)
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_vectors = None
        self._build_index()

    def _load_docs(self, path: str) -> list[dict[str, Any]]:
        if not os.path.exists(path):
            return []

        docs: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(self._normalize_doc(json.loads(line)))
        return docs

    def _normalize_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
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

    def _build_index(self) -> None:
        if not self.docs:
            self.active_backend = "empty"
            return

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
        )
        corpus = [self._compose_doc_text(doc) for doc in self.docs]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        self.active_backend = "tfidf"

    def _compose_doc_text(self, doc: dict[str, Any]) -> str:
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

    def extract_issue_keywords(self, text: str) -> list[str]:
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
        keywords: list[str] = []
        for token, label in rules:
            if token in text and label not in keywords:
                keywords.append(label)
        return keywords

    def _collect_query_terms(
        self,
        query: str,
        district: str | None = None,
        tag: str | None = None,
        unit: str | None = None,
    ) -> list[str]:
        terms = self.extract_issue_keywords(query)
        if district:
            terms.append(district)
        if tag:
            terms.append(tag)
        if unit:
            terms.append(unit)
        terms.extend(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,10}", query))
        deduped: list[str] = []
        for term in terms:
            if term and term not in deduped:
                deduped.append(term)
        return deduped

    def _matched_terms(self, doc: dict[str, Any], query_terms: list[str]) -> list[str]:
        haystack = self._compose_doc_text(doc)
        return [term for term in query_terms if term and term in haystack][:8]

    def search(
        self,
        query: str,
        top_k: int = 3,
        district: str | None = None,
        tag: str | None = None,
        unit: str | None = None,
        preferred_doc_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not self.vectorizer or self.doc_vectors is None:
            return []

        query_terms = self._collect_query_terms(query, district=district, tag=tag, unit=unit)
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.doc_vectors)[0]
        scored_docs: list[tuple[float, dict[str, Any]]] = []
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
        seen_keys: set[tuple[str, str]] = set()
        results: list[dict[str, Any]] = []
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

    def describe(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "document_count": len(self.docs),
            "corpus_path": self.corpus_path,
        }


class RAGRetriever(PolicyRetriever):
    def __init__(self, data_dir: str | None = None, **kwargs):
        if data_dir:
            corpus_path = os.path.join(data_dir, "policy_case_corpus.sample.jsonl")
            if os.path.exists(corpus_path):
                kwargs["corpus_path"] = corpus_path
        super().__init__(**kwargs)

    def retrieve(self, query: str, district: str | None = None, **kwargs) -> list[dict[str, Any]]:
        return self.search(query, district=district, **kwargs)
