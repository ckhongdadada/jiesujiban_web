from __future__ import annotations

import os
import time
import threading
import functools
from typing import Any, Callable
from collections import OrderedDict


class UnifiedCache:
    """统一缓存管理器"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }
    
    def _get_key(self, *args, **kwargs) -> str:
        import hashlib
        key_str = f"{args}|{sorted(kwargs.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Any | None:
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            if entry['expires_at'] and time.time() > entry['expires_at']:
                del self.cache[key]
                self.stats['expirations'] += 1
                self.stats['misses'] += 1
                return None
            
            self.cache.move_to_end(key)
            self.stats['hits'] += 1
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: int | None = None):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
            
            expires_at = time.time() + (ttl or self.default_ttl) if ttl != 0 else None
            
            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
    
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'expirations': 0
            }
    
    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            total = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total if total > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'expirations': self.stats['expirations'],
                'hit_rate': hit_rate,
                'usage_percent': len(self.cache) / self.max_size * 100
            }
    
    def cleanup_expired(self) -> int:
        """清理过期条目"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry['expires_at'] and current_time > entry['expires_at']
            ]
            
            for key in expired_keys:
                del self.cache[key]
                self.stats['expirations'] += 1
            
            return len(expired_keys)


class ModelManager:
    """统一模型管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.classifier = None
        self.generator = None
        self.location_ner = None
        self.rag_retriever = None
        
        self.classifier_loaded = False
        self.generator_loaded = False
        self.location_ner_loaded = False
        self.rag_retriever_loaded = False
        
        self.unified_cache = UnifiedCache(max_size=10000, default_ttl=3600)
        
        self.load_lock = threading.Lock()
        self.load_status = {
            'classifier': 'not_loaded',
            'generator': 'not_loaded',
            'location_ner': 'not_loaded',
            'rag_retriever': 'not_loaded'
        }
    
    def load_classifier(self, model_dir: str | None = None, 
                        base_model_dir: str | None = None,
                        label_map_path: str | None = None,
                        device: str = "cuda") -> bool:
        """加载分类模型"""
        with self.load_lock:
            if self.classifier_loaded:
                return True
            
            self.load_status['classifier'] = 'loading'
            
            try:
                from enhancements.classifier_runtime import load_classifier_runtime
                import torch
                
                device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
                
                success = load_classifier_runtime(
                    model_dir=model_dir,
                    base_model_dir=base_model_dir,
                    label_map_path=label_map_path,
                    device=device_obj
                )
                
                if success:
                    self.classifier_loaded = True
                    self.load_status['classifier'] = 'loaded'
                    print("[模型管理器] 分类模型加载完成")
                    return True
                else:
                    self.load_status['classifier'] = 'failed'
                    return False
            except Exception as e:
                self.load_status['classifier'] = 'failed'
                print(f"[模型管理器] 分类模型加载失败: {e}")
                return False
    
    def load_generator(self, base_model_path: str | None = None,
                       lora_path: str | None = None) -> bool:
        """加载生成模型"""
        with self.load_lock:
            if self.generator_loaded:
                return True
            
            self.load_status['generator'] = 'loading'
            
            try:
                from enhancements.enhanced_generation_optimized import load_generator
                
                success = load_generator(base_model_path, lora_path)
                
                if success:
                    self.generator_loaded = True
                    self.load_status['generator'] = 'loaded'
                    print("[模型管理器] 生成模型加载完成")
                    return True
                else:
                    self.load_status['generator'] = 'failed'
                    return False
            except Exception as e:
                self.load_status['generator'] = 'failed'
                print(f"[模型管理器] 生成模型加载失败: {e}")
                return False
    
    def load_location_ner(self, alias_path: str | None = None,
                          enable_lac: bool = True,
                          amap_api_key: str | None = None) -> bool:
        """加载地名识别模型"""
        with self.load_lock:
            if self.location_ner_loaded:
                return True
            
            self.load_status['location_ner'] = 'loading'
            
            try:
                from enhancements.location_ner_optimized import BeijingDistrictResolverOptimized
                
                self.location_ner = BeijingDistrictResolverOptimized(
                    alias_path=alias_path,
                    enable_lac=enable_lac,
                    amap_api_key=amap_api_key,
                    enable_cache=True
                )
                
                self.location_ner_loaded = True
                self.load_status['location_ner'] = 'loaded'
                print("[模型管理器] 地名识别模型加载完成")
                return True
            except Exception as e:
                self.load_status['location_ner'] = 'failed'
                print(f"[模型管理器] 地名识别模型加载失败: {e}")
                return False
    
    def load_rag_retriever(self, corpus_path: str | None = None,
                           backend: str = "bge") -> bool:
        """加载RAG检索器"""
        with self.load_lock:
            if self.rag_retriever_loaded:
                return True
            
            self.load_status['rag_retriever'] = 'loading'
            
            try:
                from enhancements.rag_retriever_bge_optimized import PolicyRetriever
                
                self.rag_retriever = PolicyRetriever(
                    corpus_path=corpus_path,
                    backend=backend,
                    enable_cache=True
                )
                
                self.rag_retriever_loaded = True
                self.load_status['rag_retriever'] = 'loaded'
                print("[模型管理器] RAG检索器加载完成")
                return True
            except Exception as e:
                self.load_status['rag_retriever'] = 'failed'
                print(f"[模型管理器] RAG检索器加载失败: {e}")
                return False
    
    def preload_all(self, classifier_config: dict | None = None,
                    generator_config: dict | None = None,
                    location_ner_config: dict | None = None,
                    rag_config: dict | None = None) -> dict[str, bool]:
        """预加载所有模型"""
        print("[模型管理器] 开始预加载所有模型...")
        
        results = {}
        
        if classifier_config:
            results['classifier'] = self.load_classifier(**classifier_config)
        
        if generator_config:
            results['generator'] = self.load_generator(**generator_config)
        
        if location_ner_config:
            results['location_ner'] = self.load_location_ner(**location_ner_config)
        
        if rag_config:
            results['rag_retriever'] = self.load_rag_retriever(**rag_config)
        
        print("[模型管理器] 预加载完成")
        return results
    
    def preload_all_parallel(self, classifier_config: dict | None = None,
                             generator_config: dict | None = None,
                             location_ner_config: dict | None = None,
                             rag_config: dict | None = None) -> dict[str, bool]:
        """并行预加载所有模型"""
        import concurrent.futures
        
        print("[模型管理器] 开始并行预加载所有模型...")
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            if classifier_config:
                futures['classifier'] = executor.submit(
                    self.load_classifier, **classifier_config
                )
            
            if generator_config:
                futures['generator'] = executor.submit(
                    self.load_generator, **generator_config
                )
            
            if location_ner_config:
                futures['location_ner'] = executor.submit(
                    self.load_location_ner, **location_ner_config
                )
            
            if rag_config:
                futures['rag_retriever'] = executor.submit(
                    self.load_rag_retriever, **rag_config
                )
            
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                    status = "成功" if results[name] else "失败"
                    print(f"[模型管理器] {name} 加载{status}")
                except Exception as e:
                    results[name] = False
                    print(f"[模型管理器] {name} 加载异常: {e}")
        
        print("[模型管理器] 并行预加载完成")
        return results
    
    def get_status(self) -> dict[str, Any]:
        """获取模型状态"""
        return {
            'classifier': {
                'loaded': self.classifier_loaded,
                'status': self.load_status['classifier']
            },
            'generator': {
                'loaded': self.generator_loaded,
                'status': self.load_status['generator']
            },
            'location_ner': {
                'loaded': self.location_ner_loaded,
                'status': self.load_status['location_ner']
            },
            'rag_retriever': {
                'loaded': self.rag_retriever_loaded,
                'status': self.load_status['rag_retriever']
            },
            'cache_stats': self.unified_cache.get_stats()
        }
    
    def clear_all_caches(self):
        """清空所有缓存"""
        self.unified_cache.clear()
        
        if self.rag_retriever and hasattr(self.rag_retriever, 'clear_cache'):
            self.rag_retriever.clear_cache()
        
        if self.location_ner and hasattr(self.location_ner, 'clear_cache'):
            self.location_ner.clear_cache()
        
        print("[模型管理器] 所有缓存已清空")


def monitor_performance(func: Callable) -> Callable:
    """性能监控装饰器"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        metrics = {
            'function': func.__name__,
            'duration_ms': round(duration_ms, 2),
            'success': success,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if not success:
            metrics['error'] = error
        
        print(f"[性能监控] {func.__name__}: {duration_ms:.2f}ms, 成功: {success}")
        
        if not success:
            raise Exception(error)
        
        return result
    
    return wrapper


def cache_result(ttl: int = 3600):
    """缓存装饰器"""
    
    def decorator(func: Callable) -> Callable:
        manager = ModelManager()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = manager.unified_cache._get_key(func.__name__, *args, **kwargs)
            
            cached = manager.unified_cache.get(cache_key)
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            
            manager.unified_cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    
    return decorator


def get_model_manager() -> ModelManager:
    """获取模型管理器单例"""
    return ModelManager()
