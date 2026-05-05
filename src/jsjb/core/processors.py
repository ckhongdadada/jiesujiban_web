"""
Enhanced processor wrappers for location NER and unit classification.

Provides caching, metrics, and disambiguation wrappers around the core
NER and classifier components.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Callable


class MetricsCollector:
    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            histograms: dict[str, Any] = {}
            for name, values in self._histograms.items():
                if values:
                    histograms[name] = {
                        "count": len(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }
                else:
                    histograms[name] = {"count": 0, "avg": 0, "min": 0, "max": 0}
            return {
                "counters": dict(self._counters),
                "histograms": histograms,
                "gauges": dict(self._gauges),
            }


class _LRUCache:
    def __init__(self, maxsize: int = 256, ttl: float = 300.0) -> None:
        self._maxsize = max(1, maxsize)
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            if key in self._cache:
                ts, value = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return True, value
                del self._cache[key]
            self._misses += 1
            return False, None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            self._cache[key] = (time.time(), value)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


class EnhancedLocationProcessor:
    def __init__(
        self,
        ner_component: Any,
        enable_disambiguation: bool = False,
        enable_cache: bool = True,
        cache_ttl: float = 300.0,
        cache_local_size: int = 256,
        enable_geo_validation: bool = False,
        enable_context_disambiguation: bool = False,
    ) -> None:
        self._ner = ner_component
        self._enable_disambiguation = enable_disambiguation
        self._cache = _LRUCache(maxsize=cache_local_size, ttl=cache_ttl) if enable_cache else None
        self._metrics = MetricsCollector()

    def process(self, text: str) -> dict[str, Any]:
        self._metrics.increment("total_requests")

        if self._cache is not None:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            hit, cached = self._cache.get(cache_key)
            if hit:
                self._metrics.increment("cache_hits")
                return cached

        result = self._ner.extract_district(text)

        if self._cache is not None:
            self._cache.put(cache_key, result)

        return result

    def get_stats(self) -> dict[str, Any]:
        stats = self._metrics.get_stats()
        total = stats["counters"].get("total_requests", 0)
        cache_hits = stats["counters"].get("cache_hits", 0)
        return {
            "total_requests": total,
            "cache_hits": cache_hits,
        }


class EnhancedClassificationProcessor:
    def __init__(
        self,
        classifier_component: Any,
        enable_cache: bool = True,
        cache_ttl: float = 300.0,
        cache_local_size: int = 256,
    ) -> None:
        self._classifier = classifier_component
        self._cache = _LRUCache(maxsize=cache_local_size, ttl=cache_ttl) if enable_cache else None
        self._metrics = MetricsCollector()

    def process(self, tag: str, title: str, body: str, district: str = "") -> list[dict[str, Any]]:
        self._metrics.increment("total_requests")

        if self._cache is not None:
            cache_key = hashlib.md5(f"{tag}|{title}|{body}|{district}".encode()).hexdigest()
            hit, cached = self._cache.get(cache_key)
            if hit:
                self._metrics.increment("cache_hits")
                return cached

        result = self._classifier.predict(tag, title, body, district=district)

        if self._cache is not None:
            self._cache.put(cache_key, result)

        return result

    def get_stats(self) -> dict[str, Any]:
        stats = self._metrics.get_stats()
        total = stats["counters"].get("total_requests", 0)
        cache_hits = stats["counters"].get("cache_hits", 0)
        return {
            "total_requests": total,
            "cache_hits": cache_hits,
        }


_location_processor: EnhancedLocationProcessor | None = None
_classification_processor: EnhancedClassificationProcessor | None = None
_init_lock = threading.Lock()


def init_enhanced_processors(
    ner_component: Any = None,
    classifier_component: Any = None,
    rag_component: Any = None,
    generation_func: Callable | None = None,
    enable_cache: bool = True,
    enable_disambiguation: bool = False,
    cache_ttl: float = 300.0,
    cache_local_size: int = 256,
    enable_geo_validation: bool = False,
    enable_context_disambiguation: bool = False,
    enable_redis: bool = False,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: str | None = None,
    batch_max_size: int = 32,
    batch_max_workers: int = 4,
) -> None:
    global _location_processor, _classification_processor

    with _init_lock:
        if ner_component is not None and _location_processor is None:
            _location_processor = EnhancedLocationProcessor(
                ner_component=ner_component,
                enable_disambiguation=enable_disambiguation,
                enable_cache=enable_cache,
                cache_ttl=cache_ttl,
                cache_local_size=cache_local_size,
                enable_geo_validation=enable_geo_validation,
                enable_context_disambiguation=enable_context_disambiguation,
            )

        if classifier_component is not None and _classification_processor is None:
            _classification_processor = EnhancedClassificationProcessor(
                classifier_component=classifier_component,
                enable_cache=enable_cache,
                cache_ttl=cache_ttl,
                cache_local_size=cache_local_size,
            )


def get_location_processor() -> EnhancedLocationProcessor | None:
    return _location_processor


def get_classification_processor() -> EnhancedClassificationProcessor | None:
    return _classification_processor
