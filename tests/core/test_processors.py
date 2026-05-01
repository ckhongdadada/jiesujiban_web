from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestEnhancedLocationProcessor(unittest.TestCase):
    """测试位置处理器"""

    def test_process_basic(self):
        """测试基本处理流程"""
        ner_mock = MagicMock()
        ner_mock.extract_district.return_value = {"district": "朝阳区", "confidence": 0.9}
        
        from src.jsjb.core.processors import EnhancedLocationProcessor
        
        processor = EnhancedLocationProcessor(
            ner_component=ner_mock,
            enable_disambiguation=False,
            enable_cache=False,
        )
        
        result = processor.process("朝阳区的问题")
        self.assertEqual(result["district"], "朝阳区")
        self.assertEqual(result["confidence"], 0.9)

    def test_cache_hit(self):
        """测试缓存命中"""
        ner_mock = MagicMock()
        ner_mock.extract_district.return_value = {"district": "朝阳区"}
        
        from src.jsjb.core.processors import EnhancedLocationProcessor
        
        processor = EnhancedLocationProcessor(
            ner_component=ner_mock,
            enable_disambiguation=False,
            enable_cache=True,
            cache_local_size=100,
        )
        
        # 第一次调用
        result1 = processor.process("测试文本")
        # 第二次调用应该命中缓存
        result2 = processor.process("测试文本")
        
        # 验证缓存命中（NER只调用一次）
        self.assertEqual(ner_mock.extract_district.call_count, 1)
        self.assertEqual(result1, result2)

    def test_get_stats(self):
        """测试统计信息"""
        ner_mock = MagicMock()
        ner_mock.extract_district.return_value = {"district": "朝阳区"}
        
        from src.jsjb.core.processors import EnhancedLocationProcessor
        
        processor = EnhancedLocationProcessor(
            ner_component=ner_mock,
            enable_disambiguation=False,
            enable_cache=False,
        )
        
        processor.process("测试1")
        processor.process("测试2")
        
        stats = processor.get_stats()
        self.assertEqual(stats["total_requests"], 2)
        self.assertEqual(stats["cache_hits"], 0)


class TestEnhancedClassificationProcessor(unittest.TestCase):
    """测试分类处理器"""

    def test_process_basic(self):
        """测试基本处理流程"""
        classifier_mock = MagicMock()
        classifier_mock.predict.return_value = [{"unit": "环卫中心", "confidence": 0.9}]
        
        from src.jsjb.core.processors import EnhancedClassificationProcessor
        
        processor = EnhancedClassificationProcessor(
            classifier_component=classifier_mock,
            enable_cache=False,
        )
        
        result = processor.process("投诉", "标题", "正文", "朝阳区")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["unit"], "环卫中心")

    def test_cache_hit(self):
        """测试缓存命中"""
        classifier_mock = MagicMock()
        classifier_mock.predict.return_value = [{"unit": "环卫中心"}]
        
        from src.jsjb.core.processors import EnhancedClassificationProcessor
        
        processor = EnhancedClassificationProcessor(
            classifier_component=classifier_mock,
            enable_cache=True,
            cache_local_size=100,
        )
        
        # 第一次调用
        processor.process("投诉", "标题", "正文", "朝阳区")
        # 第二次调用应该命中缓存
        processor.process("投诉", "标题", "正文", "朝阳区")
        
        # 验证缓存命中（分类器只调用一次）
        self.assertEqual(classifier_mock.predict.call_count, 1)


class TestMetricsCollector(unittest.TestCase):
    """测试指标收集器"""

    def test_increment_counter(self):
        """测试计数器增量"""
        from src.jsjb.core.processors import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment("requests_total")
        collector.increment("requests_total")
        
        stats = collector.get_stats()
        self.assertEqual(stats["counters"]["requests_total"], 2)

    def test_observe_histogram(self):
        """测试直方图观测"""
        from src.jsjb.core.processors import MetricsCollector
        
        collector = MetricsCollector()
        collector.observe("request_duration", 0.5)
        collector.observe("request_duration", 1.0)
        collector.observe("request_duration", 1.5)
        
        stats = collector.get_stats()
        self.assertEqual(stats["histograms"]["request_duration"]["count"], 3)
        self.assertEqual(stats["histograms"]["request_duration"]["avg"], 1.0)

    def test_set_gauge(self):
        """测试仪表盘设置"""
        from src.jsjb.core.processors import MetricsCollector
        
        collector = MetricsCollector()
        collector.set_gauge("active_requests", 5)
        
        stats = collector.get_stats()
        self.assertEqual(stats["gauges"]["active_requests"], 5)


if __name__ == "__main__":
    unittest.main()