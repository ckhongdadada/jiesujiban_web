from __future__ import annotations

import copy
import math
import os
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.jsjb.core.paths import get_feedback_db_path, get_project_root

try:
    from src.jsjb.knowledge.graph import InMemoryGraph
except ImportError:  # pragma: no cover - optional graph integration
    InMemoryGraph = None

POLICY_SECTION_PATTERN = re.compile(
    r"(?:第[一二三四五六七八九十百千]+条|^[一二三四五六七八九十]+[、.．])",
    re.MULTILINE,
)


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
    bm25_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    feedback_boost: float = 0.0
    relevance_score: float = 0.0
    full_content: str = ""
    parent_doc_id: str = ""
    child_chunk_id: str = ""
    child_snippet: str = ""
    parent_context: str = ""
    retrieval_granularity: str = "document"
    adaptive_strategy: str = "standard"
    adaptive_reason: list[str] = field(default_factory=list)
    effective_top_k: int = 0
    cache_hit: bool = False
    cache_similarity: float = 0.0
    score_breakdown: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    content: str
    doc_type: str = "参考材料"
    district: str = "全市"
    source: str = "本地知识库"
    tags: list[str] = field(default_factory=list)
    issue_type: str = ""
    unit: str = ""
    applicable_tags: list[str] = field(default_factory=list)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0


class SemanticChunker:
    """Chunk documents by embedding similarity, with structure/length fallback."""

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        min_chunk_size: int = 80,
        enable_semantic_chunking: bool = False,
        semantic_threshold: float = 0.72,
        min_semantic_units: int = 3,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.enable_semantic_chunking = enable_semantic_chunking
        self.semantic_threshold = semantic_threshold
        self.min_semantic_units = min_semantic_units

    def _split_by_structure(self, text: str) -> list[str]:
        sections = POLICY_SECTION_PATTERN.split(text)
        if len(sections) > 2:
            return [s.strip() for s in sections if s and s.strip()]

        paragraphs = re.split(r"\n{2,}", text)
        if len(paragraphs) > 1:
            return [p.strip() for p in paragraphs if p.strip()]

        sentences = re.split(r"([\u3002\uff01\uff1f\uff1b\n]+)", text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                result.append(sentences[i])
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        return [s for s in result if s.strip()]

    def _split_units_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """Split into sentence/paragraph units while keeping source offsets."""
        units: list[tuple[str, int, int]] = []
        for match in re.finditer(r".+?(?:[\u3002\uff01\uff1f\uff1b\n]+|$)", text, flags=re.S):
            raw = match.group(0)
            unit = raw.strip()
            if not unit:
                continue
            leading_ws = len(raw) - len(raw.lstrip())
            trailing_ws = len(raw) - len(raw.rstrip())
            start = match.start() + leading_ws
            end = match.end() - trailing_ws
            units.append((unit, start, end))
        return units

    @staticmethod
    def _cosine(vec_a: Any, vec_b: Any) -> float:
        try:
            if hasattr(vec_a, "tolist"):
                vec_a = vec_a.tolist()
            if hasattr(vec_b, "tolist"):
                vec_b = vec_b.tolist()
            dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
            norm_a = sum(float(a) * float(a) for a in vec_a) ** 0.5
            norm_b = sum(float(b) * float(b) for b in vec_b) ** 0.5
            if norm_a <= 1e-9 or norm_b <= 1e-9:
                return 0.0
            return dot / (norm_a * norm_b)
        except Exception:
            return 0.0

    @staticmethod
    def _encode_units(embedding_model: Any, units: list[str]) -> list[Any]:
        vectors = embedding_model.encode(
            units,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return list(vectors)

    def _semantic_sections(self, text: str, embedding_model: Any) -> list[tuple[str, int, int]]:
        units = self._split_units_with_offsets(text)
        if len(units) < self.min_semantic_units or embedding_model is None:
            return []

        try:
            vectors = self._encode_units(embedding_model, [unit[0] for unit in units])
        except Exception:
            return []

        if len(vectors) != len(units):
            return []

        sections: list[tuple[str, int, int]] = []
        current_parts = [units[0][0]]
        current_start = units[0][1]
        current_end = units[0][2]
        current_size = len(units[0][0])

        for idx in range(1, len(units)):
            unit_text, unit_start, unit_end = units[idx]
            semantic_similarity = self._cosine(vectors[idx - 1], vectors[idx])
            would_exceed_size = current_size + len(unit_text) > self.chunk_size
            semantic_boundary = (
                current_size >= self.min_chunk_size
                and semantic_similarity < self.semantic_threshold
            )

            if (semantic_boundary or would_exceed_size) and current_parts:
                section_text = "".join(current_parts).strip()
                if section_text:
                    sections.append((section_text, current_start, current_end))
                current_parts = [unit_text]
                current_start = unit_start
                current_end = unit_end
                current_size = len(unit_text)
            else:
                current_parts.append(unit_text)
                current_end = unit_end
                current_size += len(unit_text)

        if current_parts:
            section_text = "".join(current_parts).strip()
            if section_text:
                sections.append((section_text, current_start, current_end))

        return sections

    def _structure_sections(self, text: str) -> list[tuple[str, int, int]]:
        sections: list[tuple[str, int, int]] = []
        cursor = 0
        for section in self._split_by_structure(text):
            start = text.find(section, cursor)
            if start < 0:
                start = cursor
            end = start + len(section)
            sections.append((section, start, end))
            cursor = end
        return sections

    def _make_chunk(
        self,
        doc: dict[str, Any],
        chunk_content: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
    ) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=f"{doc.get('id', '')}_{chunk_index}",
            doc_id=doc.get("id", ""),
            title=doc.get("title", "\u672a\u547d\u540d\u6750\u6599"),
            content=chunk_content,
            doc_type=doc.get("doc_type", "\u53c2\u8003\u6750\u6599"),
            district=doc.get("district", "\u5168\u5e02"),
            source=doc.get("source", "\u672c\u5730\u77e5\u8bc6\u5e93"),
            tags=doc.get("tags", []),
            issue_type=doc.get("issue_type", ""),
            unit=doc.get("unit", ""),
            applicable_tags=doc.get("applicable_tags", []),
            chunk_index=chunk_index,
            start_char=max(0, start_char),
            end_char=max(start_char, end_char),
        )

    def chunk_document(self, doc: dict[str, Any], embedding_model: Any = None) -> list[DocumentChunk]:
        content = str(doc.get("content", "") or "")
        semantic_requested = self.enable_semantic_chunking and embedding_model is not None
        if len(content) <= self.chunk_size and not semantic_requested:
            return [self._make_chunk(doc, content, 0, 0, len(content))]

        sections = []
        used_semantic_sections = False
        if self.enable_semantic_chunking and embedding_model is not None:
            sections = self._semantic_sections(content, embedding_model)
            used_semantic_sections = bool(sections)
        if not sections:
            sections = self._structure_sections(content)

        if used_semantic_sections and len(sections) > 1:
            semantic_chunks = [
                self._make_chunk(doc, section, idx, section_start, section_end)
                for idx, (section, section_start, section_end) in enumerate(sections)
                if section.strip()
            ]
            if semantic_chunks:
                return semantic_chunks

        chunks: list[DocumentChunk] = []
        current_parts: list[str] = []
        current_start = 0
        current_end = 0
        current_size = 0
        chunk_index = 0

        for section, section_start, section_end in sections:
            section_size = len(section)
            if current_size + section_size > self.chunk_size and current_parts:
                chunk_content = "".join(current_parts).strip()
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append(self._make_chunk(doc, chunk_content, chunk_index, current_start, current_end))
                    chunk_index += 1

                if self.chunk_overlap > 0 and chunk_content:
                    overlap_text = chunk_content[-self.chunk_overlap:]
                    current_parts = [overlap_text]
                    current_start = max(0, current_end - len(overlap_text))
                    current_size = len(overlap_text)
                else:
                    current_parts = []
                    current_size = 0
                    current_start = section_start

            if not current_parts:
                current_start = section_start
            current_parts.append(section)
            current_size += section_size
            current_end = section_end

        if current_parts:
            chunk_content = "".join(current_parts).strip()
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(self._make_chunk(doc, chunk_content, chunk_index, current_start, current_end))

        return chunks if chunks else [self._make_chunk(doc, content, 0, 0, len(content))]


class FeedbackScoreCache:
    """反馈驱动的文档评分缓存"""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path
        self._scores: dict[str, float] = {}
        self._last_loaded: float = 0.0
        self._reload_interval: float = 300.0
        self._lock = threading.Lock()

    def _try_load(self) -> None:
        now = time.time()
        if now - self._last_loaded < self._reload_interval:
            return

        db_path = self._db_path
        if not db_path:
            try:
                db_path = str(get_feedback_db_path())
            except Exception:
                with self._lock:
                    self._last_loaded = now
                return

        if not db_path or not os.path.exists(db_path):
            with self._lock:
                self._last_loaded = now
            return

        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA busy_timeout=3000")
            cursor = conn.cursor()

            cursor.execute(
                "SELECT doc_id, SUM(CASE WHEN is_helpful = 1 THEN 1 ELSE -1 END) "
                "FROM doc_feedback GROUP BY doc_id"
            )
            rows = cursor.fetchall()
            conn.close()

            new_scores: dict[str, float] = {}
            for doc_id, net_votes in rows:
                boost = max(min(float(net_votes) * 0.05, 0.30), -0.30)
                new_scores[str(doc_id)] = boost

            with self._lock:
                self._scores = new_scores
                self._last_loaded = now
        except Exception:
            with self._lock:
                self._last_loaded = now

    def get_boost(self, doc_id: str) -> float:
        self._try_load()
        with self._lock:
            return self._scores.get(doc_id, 0.0)

    def get_boost_for_doc(self, doc: dict[str, Any]) -> float:
        self._try_load()
        candidates = [
            str(doc.get("id", "")).strip(),
            str(doc.get("doc_id", "")).strip(),
            f"{doc.get('source', '')}|{doc.get('title', '')}".strip("|"),
            str(doc.get("title", "")).strip(),
        ]
        with self._lock:
            for key in candidates:
                if key and key in self._scores:
                    return self._scores[key]
            return 0.0

    def record_feedback(self, doc_id: str, is_helpful: bool) -> None:
        delta = 0.05 if is_helpful else -0.05
        with self._lock:
            self._scores[doc_id] = self._scores.get(doc_id, 0.0) + delta


class BM25Index:
    """Small dependency-free BM25 index for Chinese-heavy government text."""

    def __init__(self, corpus: list[str] | None = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.doc_len: list[int] = []
        self.avg_doc_len: float = 0.0
        self.idf: dict[str, float] = {}
        if corpus:
            self.fit(corpus)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = str(text or "").lower()
        words = re.findall(r"[\u4e00-\u9fa5]{2,12}|[a-z0-9]{2,24}", text)
        chinese = "".join(re.findall(r"[\u4e00-\u9fa5]", text))
        char_bigrams = [chinese[i : i + 2] for i in range(max(0, len(chinese) - 1))]
        char_trigrams = [chinese[i : i + 3] for i in range(max(0, len(chinese) - 2))]
        return [token for token in words + char_bigrams + char_trigrams if token.strip()]

    def fit(self, corpus: list[str]) -> "BM25Index":
        self.doc_tokens = [self.tokenize(text) for text in corpus]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.doc_freq = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

        doc_count = max(1, len(self.doc_tokens))
        self.idf = {
            term: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in self.doc_freq.items()
        }
        return self

    def score(self, query: str) -> list[float]:
        if not self.doc_tokens:
            return []

        query_terms = self.tokenize(query)
        if not query_terms:
            return [0.0 for _ in self.doc_tokens]

        query_counts = Counter(query_terms)
        scores: list[float] = []
        avg_len = self.avg_doc_len or 1.0

        for tokens, doc_len in zip(self.doc_tokens, self.doc_len):
            tf = Counter(tokens)
            score = 0.0
            for term, query_tf in query_counts.items():
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / avg_len)
                score += idf * (freq * (self.k1 + 1.0) / max(denom, 1e-9)) * min(query_tf, 3)
            scores.append(float(score))
        return scores


class FeedbackRelevanceScorer:
    """Learn lightweight query/document relevance priors from RAG feedback."""

    def __init__(
        self,
        db_path: str | None = None,
        reload_interval: float = 300.0,
        max_adjustment: float = 0.25,
    ):
        self._db_path = db_path
        self._reload_interval = float(reload_interval)
        self.max_adjustment = float(max_adjustment)
        self._doc_scores: dict[str, float] = {}
        self._term_scores: dict[str, dict[str, float]] = defaultdict(dict)
        self._last_loaded = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _doc_keys(doc: dict[str, Any]) -> list[str]:
        return [
            str(doc.get("id", "")).strip(),
            str(doc.get("doc_id", "")).strip(),
            f"{doc.get('source', '')}|{doc.get('title', '')}".strip("|"),
            str(doc.get("title", "")).strip(),
        ]

    @staticmethod
    def _terms(text: str) -> list[str]:
        terms = BM25Index.tokenize(text)
        seen: set[str] = set()
        deduped: list[str] = []
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
            if len(deduped) >= 32:
                break
        return deduped

    def _resolve_db_path(self) -> str:
        if self._db_path:
            return self._db_path
        try:
            return str(get_feedback_db_path())
        except Exception:
            return ""

    @staticmethod
    def _has_doc_feedback_table(cursor: sqlite3.Cursor) -> bool:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='doc_feedback'")
        return cursor.fetchone() is not None

    def _try_load(self) -> None:
        now = time.time()
        if now - self._last_loaded < self._reload_interval:
            return

        db_path = self._resolve_db_path()
        if not db_path or not os.path.exists(db_path):
            with self._lock:
                self._last_loaded = now
            return

        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA busy_timeout=3000")
            cursor = conn.cursor()
            if not self._has_doc_feedback_table(cursor):
                conn.close()
                with self._lock:
                    self._last_loaded = now
                return

            cursor.execute("PRAGMA table_info(doc_feedback)")
            columns = {row[1] for row in cursor.fetchall()}
            query_col = "query" if "query" in columns else "''"
            reason_col = "reason" if "reason" in columns else "''"
            cursor.execute(
                f"SELECT doc_id, is_helpful, {query_col} AS query_text, {reason_col} AS reason_text "
                "FROM doc_feedback"
            )
            rows = cursor.fetchall()
            conn.close()

            doc_votes: Counter[str] = Counter()
            term_votes: dict[str, Counter[str]] = defaultdict(Counter)
            for doc_id, is_helpful, query_text, reason_text in rows:
                key = str(doc_id or "").strip()
                if not key:
                    continue
                delta = 1.0 if int(is_helpful or 0) == 1 else -1.0
                doc_votes[key] += delta
                for term in self._terms(f"{query_text or ''} {reason_text or ''}"):
                    term_votes[key][term] += delta

            doc_scores = {
                key: max(min(vote * 0.04, self.max_adjustment), -self.max_adjustment)
                for key, vote in doc_votes.items()
            }
            term_scores: dict[str, dict[str, float]] = {}
            for key, votes in term_votes.items():
                term_scores[key] = {
                    term: max(min(vote * 0.03, 0.12), -0.12)
                    for term, vote in votes.items()
                }

            with self._lock:
                self._doc_scores = doc_scores
                self._term_scores = defaultdict(dict, term_scores)
                self._last_loaded = now
        except Exception:
            with self._lock:
                self._last_loaded = now

    def score(self, query: str, doc: dict[str, Any], matched_terms: list[str] | None = None) -> float:
        self._try_load()
        query_terms = set(self._terms(query))
        query_terms.update(term for term in (matched_terms or []) if term)

        with self._lock:
            total = 0.0
            for key in self._doc_keys(doc):
                if not key:
                    continue
                total += self._doc_scores.get(key, 0.0)
                term_scores = self._term_scores.get(key, {})
                if term_scores and query_terms:
                    overlap_score = sum(term_scores.get(term, 0.0) for term in query_terms)
                    total += max(min(overlap_score, 0.16), -0.16)
                    break

        return max(min(float(total), self.max_adjustment), -self.max_adjustment)

    def stats(self) -> dict[str, Any]:
        self._try_load()
        with self._lock:
            return {
                "loaded_docs": len(self._doc_scores),
                "loaded_term_profiles": len(self._term_scores),
                "max_adjustment": self.max_adjustment,
                "reload_interval": self._reload_interval,
            }


class SemanticRetrievalCache:
    """Small in-process semantic cache for repeated or near-duplicate RAG queries."""

    def __init__(self, max_size: int = 256, similarity_threshold: float = 0.92, ttl_seconds: int = 1800):
        self.max_size = max(1, int(max_size))
        self.similarity_threshold = float(similarity_threshold)
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._items: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _metadata_key(
        self,
        *,
        district: str | None = None,
        tag: str | None = None,
        unit: str | None = None,
        top_k: int = 3,
        corpus_version: str = "",
    ) -> str:
        return "|".join(
            [
                str(district or ""),
                str(tag or ""),
                str(unit or ""),
                str(top_k or ""),
                str(corpus_version or ""),
            ]
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", "", str(query or "").lower())

    @staticmethod
    def _cosine(vec_a: Any, vec_b: Any) -> float:
        try:
            dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
            norm_a = sum(float(a) * float(a) for a in vec_a) ** 0.5
            norm_b = sum(float(b) * float(b) for b in vec_b) ** 0.5
            if norm_a <= 1e-9 or norm_b <= 1e-9:
                return 0.0
            return dot / (norm_a * norm_b)
        except Exception:
            return 0.0

    def get(
        self,
        *,
        query: str,
        query_vector: Any = None,
        district: str | None = None,
        tag: str | None = None,
        unit: str | None = None,
        top_k: int = 3,
        corpus_version: str = "",
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
        now = time.time()
        metadata_key = self._metadata_key(
            district=district,
            tag=tag,
            unit=unit,
            top_k=top_k,
            corpus_version=corpus_version,
        )
        normalized_query = self._normalize_query(query)

        with self._lock:
            self._items = [item for item in self._items if now - item["created_at"] <= self.ttl_seconds]
            best_item = None
            best_similarity = 0.0
            best_exact = False
            for item in self._items:
                if item["metadata_key"] != metadata_key:
                    continue
                if item["normalized_query"] == normalized_query:
                    best_item = item
                    best_similarity = 1.0
                    best_exact = True
                    break
                if query_vector is not None and item.get("query_vector") is not None:
                    similarity = self._cosine(query_vector, item["query_vector"])
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_item = item

            if best_item and (best_exact or best_similarity >= self.similarity_threshold):
                self._hits += 1
                best_item["last_accessed"] = now
                return copy.deepcopy(best_item["results"]), {
                    "hit": True,
                    "similarity": round(best_similarity, 6),
                    "exact": best_exact,
                }

            self._misses += 1
            return None, {"hit": False, "similarity": round(best_similarity, 6), "exact": False}

    def set(
        self,
        *,
        query: str,
        results: list[dict[str, Any]],
        query_vector: Any = None,
        district: str | None = None,
        tag: str | None = None,
        unit: str | None = None,
        top_k: int = 3,
        corpus_version: str = "",
    ) -> None:
        now = time.time()
        item = {
            "metadata_key": self._metadata_key(
                district=district,
                tag=tag,
                unit=unit,
                top_k=top_k,
                corpus_version=corpus_version,
            ),
            "normalized_query": self._normalize_query(query),
            "query_vector": list(query_vector) if query_vector is not None else None,
            "results": copy.deepcopy(results),
            "created_at": now,
            "last_accessed": now,
        }
        with self._lock:
            self._items.append(item)
            self._items.sort(key=lambda cached: cached.get("last_accessed", cached["created_at"]), reverse=True)
            if len(self._items) > self.max_size:
                self._items = self._items[: self.max_size]

    def clear(self) -> None:
        with self._lock:
            self._items = []

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total else 0.0
            return {
                "size": len(self._items),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl_seconds,
            }


class GraphAugmentor:
    """知识图谱增强查询"""

    def __init__(self, graph_manager=None):
        self._manager = None
        self._graph = None
        self._loaded = False
        self.attach_graph_manager(graph_manager)

    def attach_graph_manager(self, graph_manager=None) -> bool:
        """Attach the runtime graph loaded by the Flask app/model manager."""
        if graph_manager is not None:
            self._manager = graph_manager
            if hasattr(graph_manager, "graph") and graph_manager.graph is not None:
                self._graph = graph_manager.graph
                self._loaded = True
                return True
            if getattr(graph_manager, "use_neo4j", False):
                self._loaded = True
                return True
        return False

    def _ensure_graph(self):
        if self._loaded:
            return
        self._loaded = True

    def augment_query_terms(self, query: str, district: str | None = None) -> list[str]:
        self._ensure_graph()
        if self._graph is None and self._manager is None:
            return []

        extra_terms: list[str] = []
        try:
            if self._manager is not None and hasattr(self._manager, "search_nodes"):
                for item in self._manager.search_nodes(query, district=district or "", limit=6):
                    props = item.get("properties", {})
                    for key in ("name", "responsible_unit", "district", "status"):
                        if props.get(key):
                            extra_terms.append(props[key])
                return list(dict.fromkeys(extra_terms))[:8]

            if district:
                for node_type in ("Organization", "organization", "unit"):
                    nodes = self._graph.find_node(node_type, district=district)
                    for node_id in nodes[:3]:
                        node_data = self._graph.nodes.get(node_id, {})
                        props = node_data.get("properties", {})
                        if props.get("responsible_unit"):
                            extra_terms.append(props["responsible_unit"])
                        if props.get("name"):
                            extra_terms.append(props["name"])

            for node_type in ("Project", "project"):
                nodes = self._graph.find_node(node_type)
                for node_id in nodes[:5]:
                    node_data = self._graph.nodes.get(node_id, {})
                    props = node_data.get("properties", {})
                    name = props.get("name", "")
                    if name and any(kw in query for kw in name[:4]):
                        extra_terms.append(name)
                        if props.get("responsible_unit"):
                            extra_terms.append(props["responsible_unit"])
        except Exception:
            pass

        return extra_terms[:6]

    def graph_score_document(
        self,
        doc: dict[str, Any],
        query: str,
        district: str | None = None,
        unit: str | None = None,
    ) -> tuple[float, list[str]]:
        """Return a bounded GraphRAG boost and matched graph entity names."""
        self._ensure_graph()
        if self._manager is None and self._graph is None:
            return 0.0, []

        try:
            if self._manager is not None and hasattr(self._manager, "search_nodes"):
                nodes = self._manager.search_nodes(
                    f"{query} {doc.get('title', '')} {doc.get('content', '')[:200]}",
                    district=district or doc.get("district", ""),
                    limit=8,
                )
            else:
                nodes = []

            doc_text = " ".join(
                str(doc.get(key, ""))
                for key in ("title", "content", "district", "unit", "responsible_unit", "issue_type")
            )
            matched: list[str] = []
            boost = 0.0
            for item in nodes:
                props = item.get("properties", {})
                name = str(props.get("name", ""))
                if not name:
                    continue
                node_district = str(props.get("district", ""))
                local_boost = 0.0
                if name in doc_text or name in query:
                    local_boost += 0.08
                if props.get("responsible_unit") and str(props["responsible_unit"]) in doc_text:
                    local_boost += 0.04
                if district and node_district == district:
                    local_boost += 0.03
                if unit and (unit == name or unit in name or name in unit):
                    local_boost += 0.03
                if local_boost > 0:
                    matched.append(name)
                    boost += local_boost * max(float(item.get("score", 0.5)), 0.3)
            return min(boost, 0.22), list(dict.fromkeys(matched))[:8]
        except Exception:
            return 0.0, []

    def verify_fact_consistency(self, doc: dict[str, Any], query: str) -> float:
        self._ensure_graph()
        if self._graph is None:
            return 0.0

        try:
            content = doc.get("content", "")
            negation_patterns = ["不存在", "暂未建设", "尚未建设", "未涉及", "不涉及", "无此", "未有"]
            for pattern in negation_patterns:
                if pattern in content:
                    neg_context = content[max(0, content.index(pattern) - 20):content.index(pattern) + 20]
                    affirmative = ["已建成", "已建设", "正在施工", "正在建设", "已运营", "已开放"]
                    for aff in affirmative:
                        if aff in query:
                            return -0.15
        except Exception:
            pass

        return 0.0

    def query_related_facts(self, query: str, district: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
        self._ensure_graph()
        if self._graph is None and self._manager is None:
            return []

        results: list[dict[str, Any]] = []
        try:
            if self._manager is not None and hasattr(self._manager, "build_community_summary"):
                summary = self._manager.build_community_summary(query, district=district or "", limit=max(limit, 4))
                for item in summary.get("nodes", [])[:limit]:
                    props = item.get("properties", {})
                    name = props.get("name", "")
                    if not name:
                        continue
                    content_parts = [f"【{item.get('type', '')}】{name}"]
                    for key in ("status", "demolition_status", "description", "responsible_unit", "district", "type"):
                        if props.get(key):
                            content_parts.append(f"{key}: {props[key]}")
                    results.append(
                        {
                            "id": f"graph_{item['id']}",
                            "title": f"[图谱] {name}",
                            "content": "\n".join(content_parts),
                            "doc_type": "知识图谱",
                            "district": props.get("district", district or ""),
                            "source": "knowledge_graph",
                            "tags": [item.get("type", "")],
                            "unit": props.get("responsible_unit", ""),
                            "issue_type": "",
                            "applicable_tags": [],
                            "_graph_relevance": item.get("score", 0.5),
                            "graph_matched_terms": item.get("matched_terms", []),
                        }
                    )

                if summary.get("summary") and summary.get("relation_count", 0) > 0:
                    results.append(
                        {
                            "id": f"graph_community_{abs(hash((query, district))) % 1000000}",
                            "title": "[图谱社区摘要] 责任链与关联事实",
                            "content": summary["summary"],
                            "doc_type": "知识图谱摘要",
                            "district": district or "",
                            "source": "knowledge_graph_community",
                            "tags": ["community_summary"],
                            "unit": "",
                            "issue_type": "",
                            "applicable_tags": [],
                            "_graph_relevance": min(float(summary.get("score", 0.5)) + 0.08, 0.95),
                            "graph_community": {
                                "node_count": summary.get("node_count", 0),
                                "relation_count": summary.get("relation_count", 0),
                            },
                        }
                    )
                results.sort(key=lambda x: x.get("_graph_relevance", 0), reverse=True)
                return results[:limit]

            query_lower = query.lower()
            visited_ids: set[str] = set()

            for node_id, node_data in list(self._graph.nodes.items()):
                if len(results) >= limit:
                    break
                if node_id in visited_ids:
                    continue
                props = node_data.get("properties", {})
                name = props.get("name", "")
                if not name:
                    continue

                node_type = node_data.get("type", "")
                relevance = 0.0

                if name in query or query in name:
                    relevance = 0.9
                elif any(kw in query for kw in name[:4] if len(kw) >= 2):
                    relevance = 0.5
                else:
                    desc = props.get("description", "")
                    if desc and any(kw in desc for kw in query_lower.split()[:5]):
                        relevance = 0.3

                if relevance <= 0:
                    continue

                if district:
                    node_district = props.get("district", "")
                    if node_district and node_district not in (district, "北京市", "全市", ""):
                        relevance *= 0.5

                content_parts = [f"【{node_type}】{name}"]
                for key in ("status", "description", "responsible_unit", "district", "type"):
                    val = props.get(key, "")
                    if val:
                        content_parts.append(f"{key}: {val}")

                neighbors = self._graph.get_neighbors(node_id) if hasattr(self._graph, "get_neighbors") else []
                for neighbor_id, rel_type, _ in neighbors[:3]:
                    neighbor_data = self._graph.nodes.get(neighbor_id, {})
                    neighbor_name = neighbor_data.get("properties", {}).get("name", "")
                    if neighbor_name:
                        content_parts.append(f"{rel_type}: {neighbor_name}")

                visited_ids.add(node_id)
                results.append({
                    "id": f"graph_{node_id}",
                    "title": f"[图谱] {name}",
                    "content": "\n".join(content_parts),
                    "doc_type": "知识图谱",
                    "district": props.get("district", ""),
                    "source": "knowledge_graph",
                    "tags": [node_type],
                    "unit": props.get("responsible_unit", ""),
                    "issue_type": "",
                    "applicable_tags": [],
                    "_graph_relevance": relevance,
                })
        except Exception:
            pass

        results.sort(key=lambda x: x.get("_graph_relevance", 0), reverse=True)
        return results[:limit]


class Reranker:
    """Cross-Encoder reranker with lazy loading and safe local-path fallback."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        model_path: str | None = None,
        rerank_weight: float = 0.6,
    ):
        self.model_name = model_name
        self.model_path = model_path or ""
        self.rerank_weight = max(0.0, min(float(rerank_weight), 1.0))
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._load_failed = False

    def _try_load(self) -> bool:
        if self._loaded:
            return self.model is not None
        if self._load_failed:
            return False
        self._loaded = True

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            base_dir = str(get_project_root())
            safe_name = self.model_name.replace("/", "___")
            local_candidates = [
                self.model_path,
                os.path.join(base_dir, "checkpoints", "reranker", safe_name),
                os.path.join(base_dir, "reranker_models", safe_name),
            ]
            local_path = next((path for path in local_candidates if path and os.path.exists(path)), "")

            if local_path:
                self.model = AutoModelForSequenceClassification.from_pretrained(local_path)
                self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            return True
        except Exception as exc:
            print(f"[RAG Reranker] load failed, skip cross-encoder rerank: {exc}")
            self._load_failed = True
            return False

    def rerank(
        self,
        query: str,
        candidates: list[tuple[float, dict[str, Any], list[str], int]],
        top_k: int = 5,
    ) -> list[tuple[float, dict[str, Any], list[str], int]]:
        if not self._try_load() or not candidates:
            return candidates[:top_k]

        try:
            import torch

            pairs = []
            for score, doc, matched, idx in candidates:
                snippet = doc.get("content", "")[:512]
                pairs.append([query, snippet])

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)

            if logits.dim() == 0:
                rerank_scores = [float(logits)]
            else:
                rerank_scores = logits.sigmoid().cpu().numpy().tolist()

            scored = []
            for i, (base_score, doc, matched, idx) in enumerate(candidates):
                rs = float(rerank_scores[i]) if i < len(rerank_scores) else 0.0
                combined = (1.0 - self.rerank_weight) * base_score + self.rerank_weight * rs
                scored.append((combined, doc, matched, idx))

            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:top_k]
        except Exception as exc:
            print(f"[RAG Reranker] rerank failed: {exc}")
            return candidates[:top_k]

    def status(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "rerank_weight": self.rerank_weight,
            "loaded": bool(self.model is not None),
            "load_attempted": self._loaded,
            "load_failed": self._load_failed,
        }

