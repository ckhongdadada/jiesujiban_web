from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.jsjb.core.paths import get_feedback_db_path

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
    rerank_score: float = 0.0
    feedback_boost: float = 0.0
    full_content: str = ""


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
    """语义分块器 — 按政策条款/段落结构分割"""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60, min_chunk_size: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _split_by_structure(self, text: str) -> list[str]:
        sections = POLICY_SECTION_PATTERN.split(text)
        if len(sections) > 2:
            return [s.strip() for s in sections if s and s.strip()]

        paragraphs = re.split(r"\n{2,}", text)
        if len(paragraphs) > 1:
            return [p.strip() for p in paragraphs if p.strip()]

        sentences = re.split(r"([。！？；\n]+)", text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                result.append(sentences[i])
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        return [s for s in result if s.strip()]

    def chunk_document(self, doc: dict[str, Any]) -> list[DocumentChunk]:
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
                tags=doc.get("tags", []),
                issue_type=doc.get("issue_type", ""),
                unit=doc.get("unit", ""),
                applicable_tags=doc.get("applicable_tags", []),
                chunk_index=0,
                start_char=0,
                end_char=len(content),
            )]

        sections = self._split_by_structure(content)
        chunks = []
        current_parts = []
        current_size = 0
        chunk_index = 0
        start_char = 0

        for section in sections:
            section_size = len(section)
            if current_size + section_size > self.chunk_size and current_parts:
                chunk_content = "".join(current_parts)
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc.get('id', '')}_{chunk_index}",
                        doc_id=doc.get("id", ""),
                        title=doc.get("title", "未命名材料"),
                        content=chunk_content,
                        doc_type=doc.get("doc_type", "参考材料"),
                        district=doc.get("district", "全市"),
                        source=doc.get("source", "本地知识库"),
                        tags=doc.get("tags", []),
                        issue_type=doc.get("issue_type", ""),
                        unit=doc.get("unit", ""),
                        applicable_tags=doc.get("applicable_tags", []),
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(chunk_content),
                    ))
                    chunk_index += 1
                    start_char += len(chunk_content)

                if self.chunk_overlap > 0 and current_parts:
                    overlap_text = "".join(current_parts)[-self.chunk_overlap:]
                    current_parts = [overlap_text]
                    current_size = len(overlap_text)
                else:
                    current_parts = []
                    current_size = 0

            current_parts.append(section)
            current_size += section_size

        if current_parts:
            chunk_content = "".join(current_parts)
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc.get('id', '')}_{chunk_index}",
                    doc_id=doc.get("id", ""),
                    title=doc.get("title", "未命名材料"),
                    content=chunk_content,
                    doc_type=doc.get("doc_type", "参考材料"),
                    district=doc.get("district", "全市"),
                    source=doc.get("source", "本地知识库"),
                    tags=doc.get("tags", []),
                    issue_type=doc.get("issue_type", ""),
                    unit=doc.get("unit", ""),
                    applicable_tags=doc.get("applicable_tags", []),
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_content),
                ))

        return chunks if chunks else [DocumentChunk(
            chunk_id=f"{doc.get('id', '')}_0",
            doc_id=doc.get("id", ""),
            title=doc.get("title", "未命名材料"),
            content=content,
            doc_type=doc.get("doc_type", "参考材料"),
            district=doc.get("district", "全市"),
            source=doc.get("source", "本地知识库"),
            tags=doc.get("tags", []),
            issue_type=doc.get("issue_type", ""),
            unit=doc.get("unit", ""),
            applicable_tags=doc.get("applicable_tags", []),
            chunk_index=0,
            start_char=0,
            end_char=len(content),
        )]


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
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT doc_id, SUM(CASE WHEN is_helpful = 1 THEN 1 ELSE -1 END) "
                "FROM doc_feedback GROUP BY doc_id"
            )
            rows = cursor.fetchall()
            conn.close()

            new_scores: dict[str, float] = {}
            for doc_id, net_votes in rows:
                new_scores[doc_id] = float(net_votes) * 0.05

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

    def record_feedback(self, doc_id: str, is_helpful: bool) -> None:
        delta = 0.05 if is_helpful else -0.05
        with self._lock:
            self._scores[doc_id] = self._scores.get(doc_id, 0.0) + delta


class GraphAugmentor:
    """知识图谱增强查询"""

    def __init__(self):
        self._graph = None
        self._loaded = False

    def _ensure_graph(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            from src.jsjb.knowledge.graph import KnowledgeGraphManager
            mgr = KnowledgeGraphManager()
            if hasattr(mgr, "graph") and mgr.graph is not None:
                self._graph = mgr.graph
        except Exception:
            try:
                self._graph = InMemoryGraph()
            except Exception:
                pass

    def augment_query_terms(self, query: str, district: str | None = None) -> list[str]:
        self._ensure_graph()
        if self._graph is None:
            return []

        extra_terms: list[str] = []
        try:
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
        if self._graph is None:
            return []

        results: list[dict[str, Any]] = []
        try:
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
    """Cross-Encoder 重排序"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
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
        except Exception as exc:
            print(f"[RAG Reranker] 加载失败，将跳过重排序: {exc}")
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
                combined = 0.4 * base_score + 0.6 * rs
                scored.append((combined, doc, matched, idx))

            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:top_k]
        except Exception as exc:
            print(f"[RAG Reranker] 重排序失败: {exc}")
            return candidates[:top_k]
