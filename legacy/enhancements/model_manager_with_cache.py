from __future__ import annotations

import os
import time
import threading
import functools
from typing import Any, Callable

from enhancements.cache_manager import (
    MultiLevelCache,
    get_global_cache,
    init_cache,
    cache_result
)


class ModelManagerAdvanced:
    """高级模型管理器（支持Redis缓存）"""
    
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
        self.location_disambiguator = None
        
        self.classifier_loaded = False
        self.generator_loaded = False
        self.location_ner_loaded = False
        self.rag_retriever_loaded = False
        self.location_disambiguator_loaded = False
        
        self.cache = get_global_cache()
        
        self.load_lock = threading.Lock()
        self.load_status = {
            'classifier': 'not_loaded',
            'generator': 'not_loaded',
            'location_ner': 'not_loaded',
            'rag_retriever': 'not_loaded',
            'location_disambiguator': 'not_loaded'
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
                          amap_api_key: str | None = None,
                          enable_disambiguation: bool = True) -> bool:
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
                    enable_cache=False
                )
                
                if enable_disambiguation:
                    from enhancements.location_disambiguation import LocationDisambiguator
                    self.location_disambiguator = LocationDisambiguator(
                        amap_api_key=amap_api_key,
                        enable_context_disambiguation=True,
                        enable_geo_validation=True
                    )
                    self.location_disambiguator_loaded = True
                
                self.location_ner_loaded = True
                self.load_status['location_ner'] = 'loaded'
                print("[模型管理器] 地名识别模型加载完成")
                return True
            except Exception as e:
                self.load_status['location_ner'] = 'failed'
                print(f"[模型管理器] 地名识别模型加载失败: {e}")
                return False
    
    def load_rag_retriever(self, corpus_path: str | None = None,
                           backend: str = "bge",
                           enable_bm25: bool = True,
                           enable_reranker: bool = False,
                           enable_chunking: bool = True) -> bool:
        """加载RAG检索器"""
        with self.load_lock:
            if self.rag_retriever_loaded:
                return True
            
            self.load_status['rag_retriever'] = 'loading'
            
            try:
                from enhancements.rag_retriever_advanced import PolicyRetrieverAdvanced
                
                self.rag_retriever = PolicyRetrieverAdvanced(
                    corpus_path=corpus_path,
                    backend=backend,
                    enable_cache=False,
                    enable_bm25=enable_bm25,
                    enable_reranker=enable_reranker,
                    enable_chunking=enable_chunking
                )
                
                self.rag_retriever_loaded = True
                self.load_status['rag_retriever'] = 'loaded'
                print("[模型管理器] RAG检索器加载完成")
                return True
            except Exception as e:
                self.load_status['rag_retriever'] = 'failed'
                print(f"[模型管理器] RAG检索器加载失败: {e}")
                return False
    
    def resolve_location(self, text: str, use_cache: bool = True) -> dict[str, Any]:
        """地名识别（带缓存和消歧）"""
        cache_key = f"location:{text}"
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        if not self.location_ner_loaded or self.location_ner is None:
            return {"district": "未识别", "confidence": 0.0, "source": "none"}
        
        result = self.location_ner.resolve(text)
        
        if self.location_disambiguator_loaded and self.location_disambiguator:
            from enhancements.location_disambiguation import LocationCandidate
            
            candidates = [
                LocationCandidate(
                    matched_text=place.get("matched_text", ""),
                    district=place.get("district", ""),
                    source=place.get("source", ""),
                    confidence=place.get("confidence", 1.0)
                )
                for place in result.get("places", [])
            ]
            
            if len(candidates) > 1:
                disambiguated = self.location_disambiguator.disambiguate(
                    text, 
                    result.get("district", ""), 
                    candidates
                )
                
                if disambiguated:
                    result["district"] = disambiguated.district
                    result["confidence"] = disambiguated.confidence
                    result["disambiguation_method"] = disambiguated.disambiguation_method
        
        if use_cache:
            self.cache.set(cache_key, result, ttl=7200)
        
        return result
    
    def search_rag(self, query: str, top_k: int = 3, 
                   district: str | None = None, use_cache: bool = True) -> list[dict]:
        """RAG检索（带缓存）"""
        cache_key = f"rag:{query}:{top_k}:{district}"
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        if not self.rag_retriever_loaded or self.rag_retriever is None:
            return []
        
        results = self.rag_retriever.search(
            query=query,
            top_k=top_k,
            district=district
        )
        
        if use_cache:
            self.cache.set(cache_key, results, ttl=3600)
        
        return results
    
    def generate_reply(
        self,
        tag: str,
        title: str,
        body: str,
        unit: str,
        location_result: dict[str, Any],
        retrieval_hits: list[dict[str, Any]],
        use_cache: bool = True
    ) -> dict[str, Any]:
        """生成回复（带缓存）"""
        cache_key = f"reply:{tag}:{title}:{body[:50]}:{unit}"
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        if not self.generator_loaded:
            from enhancements.enhanced_generation_optimized import fallback_generate_reply
            reply = fallback_generate_reply(tag, title, body, unit, location_result, retrieval_hits)
            return {"reply": reply, "fallback": True}
        
        from enhancements.enhanced_generation_optimized import generate_reply_with_context
        
        result = generate_reply_with_context(
            tag=tag,
            title=title,
            body=body,
            unit=unit,
            location_result=location_result,
            retrieval_hits=retrieval_hits,
            enable_verification=True
        )
        
        if use_cache:
            self.cache.set(cache_key, result, ttl=1800)
        
        return result
    
    def process_message(
        self,
        tag: str,
        title: str,
        body: str,
        use_cache: bool = True
    ) -> dict[str, Any]:
        """处理单条留言（完整流程）"""
        location_result = self.resolve_location(body, use_cache=use_cache)
        district = location_result.get("district", "未识别")
        
        retrieval_hits = self.search_rag(
            query=f"{title} {body}",
            top_k=3,
            district=district,
            use_cache=use_cache
        )
        
        from enhancements.classifier_runtime import predict_units
        import torch
        
        unit_predictions = []
        if self.classifier_loaded:
            unit_predictions = predict_units(
                tag=tag,
                title=title,
                body=body,
                model_dir=None,
                base_model_dir=None,
                label_map_path=None,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                top_k=3,
                district=district
            )
        
        unit = unit_predictions[0]["unit"] if unit_predictions else "相关部门"
        
        reply_result = self.generate_reply(
            tag=tag,
            title=title,
            body=body,
            unit=unit,
            location_result=location_result,
            retrieval_hits=retrieval_hits,
            use_cache=use_cache
        )
        
        return {
            "tag": tag,
            "title": title,
            "location_result": location_result,
            "retrieval_hits": retrieval_hits,
            "unit_predictions": unit_predictions,
            "reply": reply_result.get("reply", ""),
            "verification": reply_result.get("verification"),
            "fallback": reply_result.get("fallback", False)
        }
    
    def preload_all_parallel(
        self,
        classifier_config: dict | None = None,
        generator_config: dict | None = None,
        location_ner_config: dict | None = None,
        rag_config: dict | None = None
    ) -> dict[str, bool]:
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
            'location_disambiguator': {
                'loaded': self.location_disambiguator_loaded
            },
            'cache_stats': self.cache.get_stats()
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        print("[模型管理器] 缓存已清空")
    
    def warmup_cache(self, hot_queries: list[str]):
        """缓存预热"""
        print(f"[模型管理器] 开始缓存预热，共 {len(hot_queries)} 条查询...")
        
        for i, query in enumerate(hot_queries):
            self.search_rag(query, use_cache=True)
            
            if (i + 1) % 10 == 0:
                print(f"[模型管理器] 预热进度: {i + 1}/{len(hot_queries)}")
        
        print("[模型管理器] 缓存预热完成")


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
        
        print(f"[性能监控] {func.__name__}: {duration_ms:.2f}ms, 成功: {success}")
        
        if not success:
            raise Exception(error)
        
        return result
    
    return wrapper


def get_model_manager() -> ModelManagerAdvanced:
    """获取模型管理器单例"""
    return ModelManagerAdvanced()


def init_model_manager(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: str | None = None,
    enable_redis: bool = True
) -> ModelManagerAdvanced:
    """初始化模型管理器"""
    init_cache(
        enable_redis=enable_redis,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_password=redis_password
    )
    
    return get_model_manager()
