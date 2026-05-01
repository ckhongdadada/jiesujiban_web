from __future__ import annotations

import time
import unittest

from src.jsjb.core.cache import MultiLevelCache, LocalCache


class TestLocalCache(unittest.TestCase):
    """测试本地缓存"""

    def test_cache_set_get(self):
        """测试缓存设置和获取"""
        cache = LocalCache(max_size=100)
        cache.set("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

    def test_cache_eviction(self):
        """测试缓存淘汰策略"""
        cache = LocalCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # 应该淘汰key1
        
        self.assertIsNone(cache.get("key1"))
        self.assertEqual(cache.get("key4"), "value4")

    def test_cache_clear(self):
        """测试缓存清空"""
        cache = LocalCache(max_size=100)
        cache.set("key1", "value1")
        cache.clear()
        self.assertIsNone(cache.get("key1"))


class TestMultiLevelCache(unittest.TestCase):
    """测试多级缓存"""

    def test_local_cache_only(self):
        """测试仅使用本地缓存"""
        cache = MultiLevelCache(local_cache_size=100, enable_redis=False)
        cache.set("test_key", "test_value")
        self.assertEqual(cache.get("test_key"), "test_value")

    def test_cache_none_value(self):
        """测试None值处理"""
        cache = MultiLevelCache(local_cache_size=100, enable_redis=False)
        cache.set("key_none", None)
        self.assertIsNone(cache.get("key_none"))

    def test_cache_not_found(self):
        """测试不存在的key"""
        cache = MultiLevelCache(local_cache_size=100, enable_redis=False)
        self.assertIsNone(cache.get("nonexistent_key"))


if __name__ == "__main__":
    unittest.main()