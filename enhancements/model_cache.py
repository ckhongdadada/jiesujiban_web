"""
模型缓存机制 - 提升性能
实现LRU缓存和模型预加载，减少重复加载时间
"""

import threading
import time
from functools import lru_cache
from typing import Any, Dict, Optional
import torch


class ModelCache:
    """模型缓存管理器"""
    
    def __init__(self, max_size: int = 10, ttl: int = 3600):
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
    
    def set(self, key: str, model: Any, metadata: Optional[Dict] = None) -> None:
        """设置缓存"""
        with self._lock:
            # 清理过期缓存
            self._clean_expired()
            
            # 如果缓存已满，删除最旧的
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k]['timestamp'])
                del self._cache[oldest_key]
            
            self._cache[key] = {
                'model': model,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存模型"""
        with self._lock:
            if key not in self._cache:
                return None
            
            item = self._cache[key]
            
            # 检查是否过期
            if time.time() - item['timestamp'] > self.ttl:
                del self._cache[key]
                return None
            
            # 更新访问时间
            item['timestamp'] = time.time()
            return item['model']
    
    def _clean_expired(self) -> None:
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, item in self._cache.items()
            if current_time - item['timestamp'] > self.ttl
        ]
        for key in expired_keys:
            del self._cache[key]
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """返回缓存大小"""
        return len(self._cache)


class ModelManager:
    """模型管理器，集成缓存和预加载"""
    
    def __init__(self):
        self.cache = ModelCache(max_size=5, ttl=7200)  # 2小时缓存
        self._preloaded_models = {}
        self._lock = threading.RLock()
    
    def preload_models(self, config: Dict[str, Any]) -> None:
        """预加载关键模型"""
        print("[模型管理] 开始预加载模型...")
        
        # 预加载分类模型
        try:
            from enhancements.classifier_runtime import ClassifierRuntime
            classifier = ClassifierRuntime(
                model_dir=config.get('classifier_model_dir'),
                base_model_dir=config.get('classifier_base_model'),
                device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            )
            self._preloaded_models['classifier'] = classifier
            print("[模型管理] 分类模型预加载完成")
        except Exception as e:
            print(f"[模型管理] 分类模型预加载失败: {e}")
        
        # 预加载RAG检索器
        try:
            from enhancements.rag_retriever_bge import RAGRetriever
            rag_retriever = RAGRetriever()
            self._preloaded_models['rag'] = rag_retriever
            print("[模型管理] RAG检索器预加载完成")
        except Exception as e:
            print(f"[模型管理] RAG检索器预加载失败: {e}")
        
        # 预加载地名识别
        try:
            from enhancements.location_ner import LocationNER
            ner = LocationNER()
            self._preloaded_models['ner'] = ner
            print("[模型管理] 地名识别器预加载完成")
        except Exception as e:
            print(f"[模型管理] 地名识别器预加载失败: {e}")
    
    def get_model(self, model_type: str) -> Optional[Any]:
        """获取预加载的模型"""
        return self._preloaded_models.get(model_type)
    
    @lru_cache(maxsize=100)
    def cached_generation(self, prompt: str, model_config: Dict) -> str:
        """缓存生成结果（基于prompt的缓存）"""
        # 这里可以集成到生成模块中
        from enhancements.enhanced_generation import generate_simple_reply
        return generate_simple_reply(prompt, model_config)


# 全局模型管理器实例
model_manager = ModelManager()


def get_model_cache() -> ModelCache:
    """获取全局缓存实例"""
    return model_manager.cache


def init_model_preloading(config: Dict[str, Any]) -> None:
    """初始化模型预加载"""
    model_manager.preload_models(config)


# 缓存装饰器示例
@lru_cache(maxsize=50)
def cached_retrieval(query: str, district: Optional[str] = None) -> list:
    """缓存检索结果"""
    from enhancements.rag_retriever_bge import PolicyRetriever
    retriever = PolicyRetriever()
    return retriever.search(query, district=district, top_k=3)


if __name__ == "__main__":
    # 测试缓存功能
    cache = ModelCache()
    
    # 测试设置和获取
    cache.set("test_model", {"name": "test"}, {"version": "1.0"})
    model = cache.get("test_model")
    print(f"缓存测试: {model}")
    print(f"缓存大小: {cache.size()}")