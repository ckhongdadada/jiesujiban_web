from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.jsjb.core.config import load_runtime_config, RuntimeConfig


class TestConfigLoading(unittest.TestCase):
    """测试配置加载和验证"""

    def test_load_default_config(self):
        """测试加载默认配置"""
        config = load_runtime_config()
        self.assertIsInstance(config, RuntimeConfig)
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 5000)

    def test_env_override_port(self):
        """测试环境变量覆盖端口配置"""
        original_port = os.environ.get("APP_PORT")
        os.environ["APP_PORT"] = "8080"
        try:
            config = load_runtime_config()
            self.assertEqual(config.port, 8080)
        finally:
            if original_port is None:
                del os.environ["APP_PORT"]
            else:
                os.environ["APP_PORT"] = original_port

    def test_config_default_values(self):
        """测试配置默认值"""
        config = load_runtime_config()
        self.assertEqual(config.cache_ttl, 3600)
        self.assertEqual(config.cache_local_size, 10000)
        self.assertEqual(config.retrieval_top_k, 5)
        self.assertEqual(config.generation_max_tokens, 512)
        self.assertEqual(config.generation_temperature, 0.7)

    def test_env_override_redis(self):
        """测试环境变量覆盖Redis配置"""
        original_host = os.environ.get("REDIS_HOST")
        original_port = os.environ.get("REDIS_PORT")
        os.environ["REDIS_HOST"] = "redis.example.com"
        os.environ["REDIS_PORT"] = "6380"
        try:
            config = load_runtime_config()
            self.assertEqual(config.redis_host, "redis.example.com")
            self.assertEqual(config.redis_port, 6380)
        finally:
            if original_host is None:
                del os.environ["REDIS_HOST"]
            else:
                os.environ["REDIS_HOST"] = original_host
            if original_port is None:
                del os.environ["REDIS_PORT"]
            else:
                os.environ["REDIS_PORT"] = original_port


class TestRuntimeConfig(unittest.TestCase):
    """测试RuntimeConfig数据类"""

    def test_config_dataclass(self):
        """测试配置数据类结构"""
        config = RuntimeConfig(
            classifier_model_dir="/path/to/model",
            classifier_base_model="/path/to/base",
            generator_base_model="/path/to/gen",
        )
        self.assertEqual(config.classifier_model_dir, "/path/to/model")
        self.assertEqual(config.classifier_base_model, "/path/to/base")
        self.assertEqual(config.generator_base_model, "/path/to/gen")
        self.assertEqual(config.port, 5000)  # 默认值
        self.assertEqual(config.debug, False)  # 默认值


if __name__ == "__main__":
    unittest.main()