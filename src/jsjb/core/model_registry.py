"""Runtime model registry and lightweight in-process cache."""

from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Any

import torch


class ModelCache:
    """Small TTL cache used by runtime model helpers."""

    def __init__(self, max_size: int = 10, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def set(self, key: str, model: Any, metadata: dict | None = None) -> None:
        with self._lock:
            self._clean_expired()
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
                del self._cache[oldest_key]
            self._cache[key] = {
                "model": model,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            if time.time() - item["timestamp"] > self.ttl:
                del self._cache[key]
                return None
            item["timestamp"] = time.time()
            return item["model"]

    def _clean_expired(self) -> None:
        current_time = time.time()
        expired = [
            key
            for key, item in self._cache.items()
            if current_time - item["timestamp"] > self.ttl
        ]
        for key in expired:
            del self._cache[key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        return len(self._cache)


class ModelManager:
    """Preload and expose long-lived runtime components."""

    def __init__(self):
        self.cache = ModelCache(max_size=5, ttl=7200)
        self._preloaded_models: dict[str, Any] = {}
        self._lock = threading.RLock()

    def preload_models(self, config: dict[str, Any]) -> None:
        print("[模型管理] 开始预加载模型...")

        try:
            from src.jsjb.unit_classifier import ClassifierRuntime

            classifier = ClassifierRuntime(
                model_dir=config.get("classifier_model_dir"),
                base_model_dir=config.get("classifier_base_model"),
                device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            )
            self._preloaded_models["classifier"] = classifier
            print("[模型管理] 分类模型预加载完成")
        except Exception as exc:
            print(f"[模型管理] 分类模型预加载失败: {exc}")

        try:
            from src.jsjb.retrieval import RAGRetriever

            rag_retriever = RAGRetriever(
                backend=config.get("rag_backend", "hybrid"),
                enable_query_rewrite=config.get("rag_enable_query_rewrite", True),
                multi_query_count=config.get("rag_multi_query_count", 4),
                dense_weight=config.get("rag_dense_weight", 0.68),
                sparse_weight=config.get("rag_sparse_weight", 0.32),
            )
            self._preloaded_models["rag"] = rag_retriever
            print("[模型管理] RAG检索器预加载完成")
        except Exception as exc:
            print(f"[模型管理] RAG检索器预加载失败: {exc}")

        try:
            from src.jsjb.location import LocationNER

            ner = LocationNER()
            self._preloaded_models["ner"] = ner
            print("[模型管理] 地名识别器预加载完成")
        except Exception as exc:
            print(f"[模型管理] 地名识别器预加载失败: {exc}")

    def get_model(self, model_type: str) -> Any | None:
        return self._preloaded_models.get(model_type)

    @lru_cache(maxsize=100)
    def cached_generation(self, prompt: str, model_config: tuple | str) -> str:
        from src.jsjb.reply_generation.service import generate_simple_reply

        return generate_simple_reply("", prompt, "", "", {})


model_manager = ModelManager()


def get_model_cache() -> ModelCache:
    return model_manager.cache


def init_model_preloading(config: dict[str, Any]) -> None:
    model_manager.preload_models(config)


@lru_cache(maxsize=50)
def cached_retrieval(query: str, district: str | None = None) -> list:
    from src.jsjb.retrieval.bge_retriever import PolicyRetriever

    retriever = PolicyRetriever()
    return retriever.search(query, district=district, top_k=3)
