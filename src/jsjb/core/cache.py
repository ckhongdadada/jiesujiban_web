from __future__ import annotations

import os
import json
import time
import hashlib
import threading
import pickle
from typing import Any, Callable
from collections import OrderedDict
from functools import wraps

try:
    import redis
    from redis.cluster import RedisCluster
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisError = Exception


class LocalCache:
    """本地内存缓存（L1缓存）"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }
    
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
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
            
            expires_at = time.time() + ttl if ttl else None
            
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
                'type': 'local',
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'expirations': self.stats['expirations'],
                'hit_rate': hit_rate,
                'usage_percent': len(self.cache) / self.max_size * 100
            }


class RedisCacheBackend:
    """Redis缓存后端（L2缓存）"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        cluster_mode: bool = False,
        cluster_nodes: list[dict] | None = None,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        max_connections: int = 50,
        prefix: str = "cache:"
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis库未安装，请运行: pip install redis")
        
        self.prefix = prefix
        self.cluster_mode = cluster_mode
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }
        self.lock = threading.Lock()
        
        try:
            if cluster_mode and cluster_nodes:
                self.client = RedisCluster(
                    startup_nodes=cluster_nodes,
                    decode_responses=False,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                    max_connections=max_connections
                )
            else:
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=False,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                    max_connections=max_connections
                )
            
            self.client.ping()
            self.connected = True
            print(f"[Redis缓存] 连接成功: {host}:{port}")
        except Exception as e:
            print(f"[Redis缓存] 连接失败: {e}")
            self.connected = False
            self.client = None
    
    def _get_full_key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Any | None:
        if not self.connected or self.client is None:
            return None
        
        try:
            full_key = self._get_full_key(key)
            data = self.client.get(full_key)
            
            if data is None:
                with self.lock:
                    self.stats['misses'] += 1
                return None
            
            with self.lock:
                self.stats['hits'] += 1
            
            return pickle.loads(data)
        except Exception as e:
            with self.lock:
                self.stats['errors'] += 1
            print(f"[Redis缓存] 获取失败: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int | None = None):
        if not self.connected or self.client is None:
            return False
        
        try:
            full_key = self._get_full_key(key)
            data = pickle.dumps(value)
            
            if ttl:
                self.client.setex(full_key, ttl, data)
            else:
                self.client.set(full_key, data)
            
            return True
        except Exception as e:
            with self.lock:
                self.stats['errors'] += 1
            print(f"[Redis缓存] 设置失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        if not self.connected or self.client is None:
            return False
        
        try:
            full_key = self._get_full_key(key)
            self.client.delete(full_key)
            return True
        except Exception as e:
            with self.lock:
                self.stats['errors'] += 1
            print(f"[Redis缓存] 删除失败: {e}")
            return False
    
    def clear(self):
        if not self.connected or self.client is None:
            return
        
        try:
            if self.cluster_mode:
                for node in self.client.get_primaries():
                    for key in node.scan_iter(match=f"{self.prefix}*"):
                        node.delete(key)
            else:
                for key in self.client.scan_iter(match=f"{self.prefix}*"):
                    self.client.delete(key)
        except Exception as e:
            print(f"[Redis缓存] 清空失败: {e}")
    
    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            total = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total if total > 0 else 0
            
            result = {
                'type': 'redis',
                'connected': self.connected,
                'cluster_mode': self.cluster_mode,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'errors': self.stats['errors'],
                'hit_rate': hit_rate
            }
            
            if self.connected and self.client is not None:
                try:
                    info = self.client.info('memory')
                    result['used_memory'] = info.get('used_memory_human', 'unknown')
                    result['used_memory_peak'] = info.get('used_memory_peak_human', 'unknown')
                    
                    info_stats = self.client.info('stats')
                    result['keyspace_hits'] = info_stats.get('keyspace_hits', 0)
                    result['keyspace_misses'] = info_stats.get('keyspace_misses', 0)
                except Exception:
                    pass
            
            return result


class MultiLevelCache:
    """多级缓存管理器"""
    
    def __init__(
        self,
        local_cache_size: int = 10000,
        enable_redis: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str | None = None,
        redis_cluster_mode: bool = False,
        redis_cluster_nodes: list[dict] | None = None,
        default_ttl: int = 3600,
        prefix: str = "app:"
    ):
        self.default_ttl = default_ttl
        self.prefix = prefix
        
        self.local_cache = LocalCache(max_size=local_cache_size)
        
        self.redis_cache = None
        self.enable_redis = enable_redis and REDIS_AVAILABLE
        
        if self.enable_redis:
            try:
                self.redis_cache = RedisCacheBackend(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    cluster_mode=redis_cluster_mode,
                    cluster_nodes=redis_cluster_nodes,
                    prefix=prefix
                )
            except Exception as e:
                print(f"[多级缓存] Redis初始化失败: {e}")
                self.enable_redis = False
        
        self.stats = {
            'total_requests': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0
        }
        self.lock = threading.Lock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        key_str = f"{args}|{sorted(kwargs.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Any | None:
        with self.lock:
            self.stats['total_requests'] += 1
        
        value = self.local_cache.get(key)
        if value is not None:
            with self.lock:
                self.stats['l1_hits'] += 1
            return value
        
        if self.enable_redis and self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                with self.lock:
                    self.stats['l2_hits'] += 1
                
                self.local_cache.set(key, value)
                return value
        
        with self.lock:
            self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int | None = None):
        ttl = ttl or self.default_ttl
        
        self.local_cache.set(key, value, ttl)
        
        if self.enable_redis and self.redis_cache:
            self.redis_cache.set(key, value, ttl)
    
    def delete(self, key: str):
        self.local_cache.delete(key)
        
        if self.enable_redis and self.redis_cache:
            self.redis_cache.delete(key)
    
    def clear(self):
        self.local_cache.clear()
        
        if self.enable_redis and self.redis_cache:
            self.redis_cache.clear()
    
    def get_or_set(
        self,
        key: str,
        value_func: Callable[[], Any],
        ttl: int | None = None
    ) -> Any:
        value = self.get(key)
        
        if value is not None:
            return value
        
        value = value_func()
        
        if value is not None:
            self.set(key, value, ttl)
        
        return value
    
    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            total = self.stats['total_requests']
            l1_hit_rate = self.stats['l1_hits'] / total if total > 0 else 0
            l2_hit_rate = self.stats['l2_hits'] / total if total > 0 else 0
            total_hit_rate = (self.stats['l1_hits'] + self.stats['l2_hits']) / total if total > 0 else 0
            
            result = {
                'total_requests': total,
                'l1_hits': self.stats['l1_hits'],
                'l2_hits': self.stats['l2_hits'],
                'misses': self.stats['misses'],
                'l1_hit_rate': l1_hit_rate,
                'l2_hit_rate': l2_hit_rate,
                'total_hit_rate': total_hit_rate,
                'local_cache': self.local_cache.get_stats()
            }
            
            if self.enable_redis and self.redis_cache:
                result['redis_cache'] = self.redis_cache.get_stats()
            
            return result
    
    def warmup(self, data: dict[str, Any], ttl: int | None = None):
        """缓存预热"""
        for key, value in data.items():
            self.set(key, value, ttl)
        
        print(f"[多级缓存] 预热完成，共 {len(data)} 条数据")


def cache_result(
    key_prefix: str = "",
    ttl: int = 3600,
    cache_instance: MultiLevelCache | None = None
):
    """缓存装饰器"""
    
    def decorator(func: Callable) -> Callable:
        _cache = cache_instance
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _cache
            
            if _cache is None:
                _cache = get_global_cache()
            
            cache_key = f"{key_prefix}:{func.__name__}:{_cache._generate_key(*args, **kwargs)}"
            
            result = _cache.get(cache_key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            
            if result is not None:
                _cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator


_global_cache: MultiLevelCache | None = None
_global_cache_lock = threading.Lock()


def get_global_cache() -> MultiLevelCache:
    """获取全局缓存实例"""
    global _global_cache
    
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = MultiLevelCache(
                    enable_redis=os.getenv("REDIS_ENABLED", "true").lower() == "true",
                    redis_host=os.getenv("REDIS_HOST", "localhost"),
                    redis_port=int(os.getenv("REDIS_PORT", "6379")),
                    redis_db=int(os.getenv("REDIS_DB", "0")),
                    redis_password=os.getenv("REDIS_PASSWORD"),
                    redis_cluster_mode=os.getenv("REDIS_CLUSTER_MODE", "false").lower() == "true",
                    default_ttl=int(os.getenv("CACHE_DEFAULT_TTL", "3600")),
                    prefix=os.getenv("CACHE_PREFIX", "app:")
                )
    
    return _global_cache


def init_cache(
    local_cache_size: int = 10000,
    enable_redis: bool = True,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: str | None = None,
    default_ttl: int = 3600,
    prefix: str = "app:"
) -> MultiLevelCache:
    """初始化全局缓存"""
    global _global_cache
    
    with _global_cache_lock:
        _global_cache = MultiLevelCache(
            local_cache_size=local_cache_size,
            enable_redis=enable_redis,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            redis_password=redis_password,
            default_ttl=default_ttl,
            prefix=prefix
        )
    
    return _global_cache
