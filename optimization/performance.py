"""
性能优化模块
实现性能分析、缓存优化、批量处理、推理加速
"""

from __future__ import annotations

import time
import json
import threading
import torch
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from functools import wraps, lru_cache
from collections import defaultdict
import hashlib


@dataclass
class PerformanceMetric:
    """性能指标"""
    operation: str
    duration_ms: float
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self._lock = threading.RLock()
        self._operation_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "total_time": 0, "min_time": float('inf'), "max_time": 0}
        )
    
    def record(self, operation: str, duration_ms: float, metadata: Dict[str, Any] = None):
        """记录性能指标"""
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        
        with self._lock:
            self.metrics.append(metric)
            
            stats = self._operation_stats[operation]
            stats["count"] += 1
            stats["total_time"] += duration_ms
            stats["min_time"] = min(stats["min_time"], duration_ms)
            stats["max_time"] = max(stats["max_time"], duration_ms)
    
    def get_stats(self, operation: str = None) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            if operation:
                stats = self._operation_stats.get(operation, {})
                if stats:
                    return {
                        "operation": operation,
                        "count": stats["count"],
                        "avg_time_ms": stats["total_time"] / stats["count"],
                        "min_time_ms": stats["min_time"],
                        "max_time_ms": stats["max_time"]
                    }
                return {}
            
            all_stats = {}
            for op, stats in self._operation_stats.items():
                all_stats[op] = {
                    "count": stats["count"],
                    "avg_time_ms": stats["total_time"] / stats["count"],
                    "min_time_ms": stats["min_time"],
                    "max_time_ms": stats["max_time"]
                }
            return all_stats
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近的指标"""
        with self._lock:
            return [m.to_dict() for m in self.metrics[-limit:]]
    
    def clear(self):
        """清空指标"""
        with self._lock:
            self.metrics.clear()
            self._operation_stats.clear()


def profile(operation: str):
    """
    性能分析装饰器
    
    Args:
        operation: 操作名称
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                profiler.record(operation, duration_ms)
        
        return wrapper
    return decorator


profiler = PerformanceProfiler()


class RequestCache:
    """请求缓存"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存数量
            ttl: 缓存过期时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        with self._lock:
            if key not in self._cache:
                return None
            
            item = self._cache[key]
            
            if time.time() - item["timestamp"] > self.ttl:
                del self._cache[key]
                return None
            
            return item["value"]
    
    def set(self, key: str, value: Any):
        """设置缓存"""
        with self._lock:
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
                del self._cache[oldest_key]
            
            self._cache[key] = {
                "value": value,
                "timestamp": time.time()
            }
    
    def cached_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        缓存函数调用
        
        Args:
            func: 函数
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            函数结果
        """
        key = self._generate_key(*args, **kwargs)
        
        cached = self.get(key)
        if cached is not None:
            return cached
        
        result = func(*args, **kwargs)
        self.set(key, result)
        
        return result
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, batch_size: int = 32, timeout: float = 0.1):
        """
        初始化批量处理器
        
        Args:
            batch_size: 批量大小
            timeout: 等待超时时间（秒）
        """
        self.batch_size = batch_size
        self.timeout = timeout
        
        self._queue: List[Dict[str, Any]] = []
        self._results: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._batch_event = threading.Event()
    
    def add(self, item_id: str, item_data: Dict[str, Any]) -> Any:
        """
        添加项目到批量队列
        
        Args:
            item_id: 项目ID
            item_data: 项目数据
            
        Returns:
            处理结果
        """
        with self._lock:
            self._queue.append({"id": item_id, "data": item_data})
            
            if len(self._queue) >= self.batch_size:
                self._batch_event.set()
        
        while item_id not in self._results:
            time.sleep(0.01)
        
        result = self._results.pop(item_id)
        return result
    
    def process_batch(self, processor: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]):
        """
        处理批量任务
        
        Args:
            processor: 批量处理函数
        """
        while True:
            self._batch_event.wait(timeout=self.timeout)
            
            with self._lock:
                if not self._queue:
                    self._batch_event.clear()
                    continue
                
                batch = self._queue[:self.batch_size]
                self._queue = self._queue[self.batch_size:]
            
            if batch:
                results = processor(batch)
                
                with self._lock:
                    for result in results:
                        self._results[result["id"]] = result["result"]
            
            self._batch_event.clear()


class InferenceOptimizer:
    """推理优化器"""
    
    def __init__(self):
        self.request_cache = RequestCache(max_size=500, ttl=300)
        self._model_cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def optimize_classifier(self, classifier):
        """优化分类器"""
        original_predict = classifier.predict
        
        @wraps(original_predict)
        def optimized_predict(tag, title, body, *args, **kwargs):
            cache_key = self.request_cache._generate_key(tag, title, body)
            
            cached = self.request_cache.get(cache_key)
            if cached is not None:
                return cached
            
            result = original_predict(tag, title, body, *args, **kwargs)
            self.request_cache.set(cache_key, result)
            
            return result
        
        classifier.predict = optimized_predict
        return classifier
    
    def optimize_generator(self, generator):
        """优化生成器"""
        return generator
    
    def preload_models(self, models: Dict[str, Any]):
        """预加载模型"""
        with self._lock:
            for name, model in models.items():
                self._model_cache[name] = model
                print(f"[推理优化] 已预加载模型: {name}")
    
    def get_model(self, name: str) -> Optional[Any]:
        """获取缓存的模型"""
        return self._model_cache.get(name)


class PerformanceOptimizer:
    """性能优化管理器"""
    
    def __init__(self):
        self.profiler = profiler
        self.cache = RequestCache()
        self.optimizer = InferenceOptimizer()
    
    @profile("full_pipeline")
    def process_request(
        self,
        tag: str,
        title: str,
        body: str,
        unit: str = "",
        enable_cache: bool = True
    ) -> Dict[str, Any]:
        """
        处理请求（优化版）
        
        Args:
            tag: 标签
            title: 标题
            body: 正文
            unit: 单位
            enable_cache: 是否启用缓存
            
        Returns:
            处理结果
        """
        if enable_cache:
            cache_key = self.cache._generate_key(tag, title, body, unit)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        result = self._process_internal(tag, title, body, unit)
        
        if enable_cache:
            cache_key = self.cache._generate_key(tag, title, body, unit)
            self.cache.set(cache_key, result)
        
        return result
    
    def _process_internal(self, tag: str, title: str, body: str, unit: str) -> Dict[str, Any]:
        """内部处理逻辑"""
        from src.jsjb.core.config import load_runtime_config
        from src.jsjb.location import LocationNER
        from src.jsjb.retrieval import PolicyRetriever
        from src.jsjb.unit_classifier.runtime import ClassifierRuntime
        from src.jsjb.reply_generation.service import generate_reply_with_context
        config = load_runtime_config()
        
        result = {
            "tag": tag,
            "title": title,
            "body": body,
            "unit": unit,
            "location_result": {},
            "retrieval_hits": [],
            "predictions": [],
            "reply": "",
            "timings": {}
        }
        
        start_time = time.time()
        
        try:
            ner_start = time.time()
            ner = LocationNER()
            result["location_result"] = ner.extract_district(title + " " + body)
            result["timings"]["location_ner"] = (time.time() - ner_start) * 1000
        except Exception as e:
            print(f"[性能优化] 地名识别失败: {e}")
        
        try:
            rag_start = time.time()
            rag = PolicyRetriever(
                backend=config.rag_backend,
                enable_query_rewrite=config.rag_enable_query_rewrite,
                multi_query_count=config.rag_multi_query_count,
                dense_weight=config.rag_dense_weight,
                sparse_weight=config.rag_sparse_weight,
            )
            result["retrieval_hits"] = rag.search(title + " " + body, top_k=5)
            result["timings"]["rag_retrieval"] = (time.time() - rag_start) * 1000
        except Exception as e:
            print(f"[性能优化] RAG检索失败: {e}")
        
        try:
            classify_start = time.time()
            classifier = ClassifierRuntime(
                model_dir=config.classifier_model_dir,
                base_model_dir=config.classifier_base_model,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            result["predictions"] = classifier.predict(tag, title, body)
            result["timings"]["classification"] = (time.time() - classify_start) * 1000
        except Exception as e:
            print(f"[性能优化] 分类预测失败: {e}")
        
        if unit:
            try:
                gen_start = time.time()
                gen_result = generate_reply_with_context(
                    tag=tag,
                    title=title,
                    body=body,
                    unit=unit,
                    location_result=result["location_result"],
                    retrieval_hits=result["retrieval_hits"],
                    base_model_path=config.generator_base_model,
                    lora_path=config.generator_lora_dir,
                    draft_model_path=config.generator_draft_model,
                    enable_assisted_decoding=config.enable_assisted_decoding,
                    max_new_tokens=config.generation_max_tokens,
                    temperature=config.generation_temperature,
                    return_dict=True,
                )
                result["reply"] = gen_result.get("reply", "")
                result["timings"]["generation"] = (time.time() - gen_start) * 1000
            except Exception as e:
                print(f"[性能优化] 回复生成失败: {e}")
        
        result["timings"]["total"] = (time.time() - start_time) * 1000
        
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            "operation_stats": self.profiler.get_stats(),
            "cache_stats": {
                "size": self.cache.size(),
                "max_size": self.cache.max_size
            },
            "recent_metrics": self.profiler.get_recent_metrics(50)
        }


_optimizer_instance: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """获取性能优化器单例"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = PerformanceOptimizer()
    return _optimizer_instance


if __name__ == "__main__":
    print("性能优化模块测试")
    print("=" * 50)
    
    optimizer = get_performance_optimizer()
    
    print("\n1. 测试性能分析装饰器")
    
    @profile("test_operation")
    def test_function():
        time.sleep(0.1)
        return "done"
    
    test_function()
    
    stats = profiler.get_stats("test_operation")
    print(f"统计信息: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    
    print("\n2. 测试请求缓存")
    cache = RequestCache()
    
    def expensive_function(x):
        time.sleep(0.1)
        return x * 2
    
    start = time.time()
    result1 = cache.cached_call(expensive_function, 5)
    time1 = time.time() - start
    
    start = time.time()
    result2 = cache.cached_call(expensive_function, 5)
    time2 = time.time() - start
    
    print(f"第一次调用: {result1}, 耗时: {time1*1000:.2f}ms")
    print(f"第二次调用(缓存): {result2}, 耗时: {time2*1000:.2f}ms")
    
    print("\n3. 性能报告")
    report = optimizer.get_performance_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
