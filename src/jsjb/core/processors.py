from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

from src.jsjb.core.cache import MultiLevelCache
from src.jsjb.location.disambiguation import (
    DisambiguationResult,
    LocationCandidate,
    LocationDisambiguator,
)


@dataclass
class ProcessingContext:
    request_id: str
    client_ip: str
    start_time: float
    tag: str = ""
    title: str = ""
    body: str = ""
    forced_unit: str = ""
    location_result: dict = field(default_factory=dict)
    classification_result: list = field(default_factory=list)
    retrieval_result: list = field(default_factory=list)
    reply_result: str = ""
    stage_times: dict = field(default_factory=dict)
    cache_hits: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


class EnhancedLocationProcessor:
    def __init__(
        self,
        ner_component: Any,
        enable_disambiguation: bool = True,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        cache_local_size: int = 5000,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str | None = None,
        enable_redis: bool = False,
        enable_geo_validation: bool = False,
        enable_context_disambiguation: bool = True,
    ):
        self.ner = ner_component
        self.enable_disambiguation = enable_disambiguation
        self.enable_cache = enable_cache
        self.cache = None
        self.disambiguator = None

        if enable_disambiguation:
            self.disambiguator = LocationDisambiguator(
                amap_api_key=None,
                enable_context_disambiguation=enable_context_disambiguation,
                enable_geo_validation=enable_geo_validation,
            )

        if enable_cache:
            self.cache = MultiLevelCache(
                local_cache_size=cache_local_size,
                enable_redis=enable_redis,
                redis_host=redis_host,
                redis_port=redis_port,
                redis_db=redis_db,
                redis_password=redis_password,
                default_ttl=cache_ttl,
                prefix="loc:",
            )

        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "disambiguations": 0,
            "total_time": 0.0,
        }
        self._lock = threading.Lock()

    def process(self, text: str, use_cache: bool = True) -> dict:
        with self._lock:
            self.stats["total_requests"] += 1

        start_time = time.time()
        cache_key = self._generate_cache_key(text)

        if use_cache and self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                with self._lock:
                    self.stats["cache_hits"] += 1
                    self.stats["total_time"] += time.time() - start_time
                return cached

        location_result = self.ner.extract_district(text) if self.ner else {}
        if not isinstance(location_result, dict):
            location_result = {}

        if self.enable_disambiguation and self.disambiguator:
            district = location_result.get("district", "")
            if district:
                disambiguated = self._try_disambiguate(text, district)
                if disambiguated:
                    location_result["district"] = disambiguated.district
                    location_result["confidence"] = max(
                        float(location_result.get("confidence", 0) or 0),
                        float(disambiguated.confidence or 0),
                    )
                    location_result["disambiguated"] = True
                    location_result["disambiguation_method"] = disambiguated.disambiguation_method
                    location_result["disambiguation_confidence"] = disambiguated.confidence
                    with self._lock:
                        self.stats["disambiguations"] += 1

        if use_cache and self.cache:
            self.cache.set(cache_key, location_result)

        with self._lock:
            self.stats["total_time"] += time.time() - start_time
        return location_result

    def _try_disambiguate(self, text: str, location: str) -> DisambiguationResult | None:
        try:
            candidates = [
                LocationCandidate(
                    matched_text=location,
                    district=location,
                    source="ner",
                    confidence=0.9,
                )
            ]
            return self.disambiguator.disambiguate(text, location, candidates)
        except Exception:
            return None

    def _generate_cache_key(self, text: str) -> str:
        return f"loc:{hashlib.md5(text.encode('utf-8')).hexdigest()}"

    def get_stats(self) -> dict:
        with self._lock:
            total = self.stats["total_requests"]
            avg_time = self.stats["total_time"] / total if total else 0
            hit_rate = self.stats["cache_hits"] / total if total else 0
            return {**self.stats, "avg_time": avg_time, "cache_hit_rate": hit_rate}


class EnhancedClassificationProcessor:
    def __init__(
        self,
        classifier_component: Any,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        cache_local_size: int = 5000,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str | None = None,
        enable_redis: bool = False,
    ):
        self.classifier = classifier_component
        self.enable_cache = enable_cache
        self.cache = None

        if enable_cache:
            self.cache = MultiLevelCache(
                local_cache_size=cache_local_size,
                enable_redis=enable_redis,
                redis_host=redis_host,
                redis_port=redis_port,
                redis_db=redis_db,
                redis_password=redis_password,
                default_ttl=cache_ttl,
                prefix="cls:",
            )

        self.stats = {"total_requests": 0, "cache_hits": 0, "total_time": 0.0}
        self._lock = threading.Lock()

    def process(
        self,
        tag: str,
        title: str,
        body: str,
        district: str = "",
        use_cache: bool = True,
    ) -> list:
        with self._lock:
            self.stats["total_requests"] += 1

        start_time = time.time()
        cache_key = self._generate_cache_key(tag, title, body, district)

        if use_cache and self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                with self._lock:
                    self.stats["cache_hits"] += 1
                    self.stats["total_time"] += time.time() - start_time
                return cached

        result = self.classifier.predict(tag, title, body, district=district) if self.classifier else []

        if use_cache and self.cache:
            self.cache.set(cache_key, result)

        with self._lock:
            self.stats["total_time"] += time.time() - start_time
        return result

    def _generate_cache_key(self, tag: str, title: str, body: str, district: str) -> str:
        content = f"{tag}|{title}|{body}|{district}"
        return f"cls:{hashlib.md5(content.encode('utf-8')).hexdigest()}"

    def get_stats(self) -> dict:
        with self._lock:
            total = self.stats["total_requests"]
            avg_time = self.stats["total_time"] / total if total else 0
            hit_rate = self.stats["cache_hits"] / total if total else 0
            return {**self.stats, "avg_time": avg_time, "cache_hit_rate": hit_rate}


class RequestTracker:
    def __init__(self):
        self.requests: dict[str, ProcessingContext] = {}
        self._lock = threading.Lock()
        self._max_requests = 10000

    def start_request(self, client_ip: str, tag: str = "", title: str = "", body: str = "") -> str:
        request_id = self._generate_request_id()
        context = ProcessingContext(
            request_id=request_id,
            client_ip=client_ip,
            start_time=time.time(),
            tag=tag,
            title=title,
            body=body,
        )
        with self._lock:
            if len(self.requests) >= self._max_requests:
                oldest_key = next(iter(self.requests))
                del self.requests[oldest_key]
            self.requests[request_id] = context
        return request_id

    def update_context(self, request_id: str, **kwargs):
        with self._lock:
            context = self.requests.get(request_id)
            if context:
                for key, value in kwargs.items():
                    if hasattr(context, key):
                        setattr(context, key, value)

    def record_stage_time(self, request_id: str, stage: str, duration: float):
        with self._lock:
            context = self.requests.get(request_id)
            if context:
                context.stage_times[stage] = duration

    def end_request(self, request_id: str) -> ProcessingContext | None:
        with self._lock:
            return self.requests.pop(request_id, None)

    def get_context(self, request_id: str) -> ProcessingContext | None:
        with self._lock:
            return self.requests.get(request_id)

    def _generate_request_id(self) -> str:
        return f"req_{uuid.uuid4().hex[:12]}_{int(time.time() * 1000)}"


class MetricsCollector:
    def __init__(self):
        self.counters = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self.histograms = {
            "request_duration": [],
            "location_duration": [],
            "classification_duration": [],
            "retrieval_duration": [],
            "generation_duration": [],
        }
        self.gauges = {"active_requests": 0, "memory_usage": 0, "gpu_memory_usage": 0}
        self._lock = threading.Lock()
        self._max_histogram_size = 10000

    def increment(self, metric: str, value: int = 1):
        with self._lock:
            if metric in self.counters:
                self.counters[metric] += value

    def observe(self, metric: str, value: float):
        with self._lock:
            if metric in self.histograms:
                if len(self.histograms[metric]) >= self._max_histogram_size:
                    self.histograms[metric] = self.histograms[metric][-self._max_histogram_size // 2:]
                self.histograms[metric].append(value)

    def set_gauge(self, metric: str, value: float):
        with self._lock:
            if metric in self.gauges:
                self.gauges[metric] = value

    def get_prometheus_metrics(self) -> str:
        lines = []
        with self._lock:
            for name, value in self.counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            for name, values in self.histograms.items():
                if not values:
                    continue
                sorted_values = sorted(values)
                count = len(sorted_values)
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count {count}")
                lines.append(f"{name}_sum {sum(sorted_values):.4f}")
                for p in [0.5, 0.9, 0.95, 0.99]:
                    idx = min(int(count * p), count - 1)
                    lines.append(f'{name}{{quantile="{p}"}} {sorted_values[idx]:.4f}')
            for name, value in self.gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        with self._lock:
            stats = {"counters": dict(self.counters), "gauges": dict(self.gauges), "histograms": {}}
            for name, values in self.histograms.items():
                if values:
                    sorted_values = sorted(values)
                    count = len(sorted_values)
                    stats["histograms"][name] = {
                        "count": count,
                        "min": sorted_values[0],
                        "max": sorted_values[-1],
                        "avg": sum(sorted_values) / count,
                        "p50": sorted_values[min(int(count * 0.5), count - 1)],
                        "p90": sorted_values[min(int(count * 0.9), count - 1)],
                        "p95": sorted_values[min(int(count * 0.95), count - 1)],
                        "p99": sorted_values[min(int(count * 0.99), count - 1)],
                    }
            return stats


def with_retry(max_retries: int = 3, backoff_factor: float = 0.5, exceptions: tuple = (Exception,)):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        time.sleep(backoff_factor * (2 ** attempt))
            raise last_exception
        return wrapper
    return decorator


class BatchProcessor:
    def __init__(
        self,
        location_processor: EnhancedLocationProcessor,
        classification_processor: EnhancedClassificationProcessor,
        rag_component: Any,
        generation_func: Callable,
        max_batch_size: int = 10,
        max_workers: int = 4,
    ):
        from concurrent.futures import ThreadPoolExecutor

        self.location_processor = location_processor
        self.classification_processor = classification_processor
        self.rag = rag_component
        self.generate_reply = generation_func
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.stats = {"total_batches": 0, "total_items": 0, "total_errors": 0, "total_time": 0.0}
        self._lock = threading.Lock()

    def process_batch(self, items: list[dict], use_cache: bool = True) -> list[dict]:
        from concurrent.futures import as_completed

        with self._lock:
            self.stats["total_batches"] += 1
            self.stats["total_items"] += len(items)

        start_time = time.time()
        futures = {}
        for idx, item in enumerate(items[: self.max_batch_size]):
            future = self.executor.submit(self._process_single_item, item, use_cache)
            futures[future] = idx

        results = []
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append((idx, future.result()))
            except Exception as exc:
                with self._lock:
                    self.stats["total_errors"] += 1
                results.append((idx, {"error": str(exc), "status": "error"}))

        results.sort(key=lambda item: item[0])
        with self._lock:
            self.stats["total_time"] += time.time() - start_time
        return [item[1] for item in results]

    def _process_single_item(self, item: dict, use_cache: bool) -> dict:
        tag = item.get("tag", "")
        title = item.get("title", "")
        body = item.get("body", "")
        forced_unit = item.get("_force_unit", "")

        if not body:
            return {"error": "留言正文不能为空", "status": "error"}

        text = f"{title} {body}"
        location_result = self.location_processor.process(text, use_cache)
        units = self.classification_processor.process(
            tag,
            title,
            body,
            district=location_result.get("district", ""),
            use_cache=use_cache,
        )
        primary_unit = forced_unit or (units[0]["unit"] if units else "相关单位")
        docs = self.rag.retrieve(
            text,
            district=location_result.get("district"),
            tag=tag,
            unit=primary_unit,
        )
        reply = self.generate_reply(
            tag=tag,
            title=title,
            body=body,
            unit=primary_unit,
            location_result=location_result,
            retrieval_hits=docs,
        )
        return {
            "status": "ok",
            "location": location_result,
            "units": units,
            "retrieval": docs[:3] if docs else [],
            "reply": reply,
        }

    def get_stats(self) -> dict:
        with self._lock:
            total = self.stats["total_batches"]
            avg_time = self.stats["total_time"] / total if total else 0
            return {**self.stats, "avg_time": avg_time}


enhanced_processors: dict[str, Any] = {}
request_tracker = RequestTracker()
metrics_collector = MetricsCollector()


def init_enhanced_processors(
    ner_component: Any,
    classifier_component: Any,
    rag_component: Any,
    generation_func: Callable,
    enable_cache: bool = True,
    enable_disambiguation: bool = True,
    cache_ttl: int = 3600,
    cache_local_size: int = 5000,
    enable_geo_validation: bool = False,
    enable_context_disambiguation: bool = True,
    enable_redis: bool = False,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: str | None = None,
    batch_max_size: int = 10,
    batch_max_workers: int = 4,
):
    global enhanced_processors
    enhanced_processors["location"] = EnhancedLocationProcessor(
        ner_component=ner_component,
        enable_disambiguation=enable_disambiguation,
        enable_cache=enable_cache,
        cache_ttl=cache_ttl,
        cache_local_size=cache_local_size,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_password=redis_password,
        enable_redis=enable_redis,
        enable_geo_validation=enable_geo_validation,
        enable_context_disambiguation=enable_context_disambiguation,
    )
    enhanced_processors["classification"] = EnhancedClassificationProcessor(
        classifier_component=classifier_component,
        enable_cache=enable_cache,
        cache_ttl=cache_ttl,
        cache_local_size=cache_local_size,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_password=redis_password,
        enable_redis=enable_redis,
    )
    enhanced_processors["batch"] = BatchProcessor(
        location_processor=enhanced_processors["location"],
        classification_processor=enhanced_processors["classification"],
        rag_component=rag_component,
        generation_func=generation_func,
        max_batch_size=batch_max_size,
        max_workers=batch_max_workers,
    )
    return enhanced_processors


def get_location_processor() -> EnhancedLocationProcessor | None:
    return enhanced_processors.get("location")


def get_classification_processor() -> EnhancedClassificationProcessor | None:
    return enhanced_processors.get("classification")


def get_batch_processor() -> BatchProcessor | None:
    return enhanced_processors.get("batch")

