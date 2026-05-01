from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.jsjb.core.logging import StructuredLogger, setup_logging, get_logger


class TestStructuredLogger(unittest.TestCase):
    """测试结构化日志记录器"""

    def test_logger_initialization(self):
        """测试日志记录器初始化"""
        logger = StructuredLogger()
        self.assertIsNotNone(logger.logger)
        self.assertIsNotNone(logger._quality_logger)

    def test_log_info(self):
        """测试INFO级别日志"""
        logger = StructuredLogger()
        logger.info("测试信息日志")

    def test_log_error(self):
        """测试ERROR级别日志"""
        logger = StructuredLogger()
        logger.error("测试错误日志")

    def test_log_request(self):
        """测试请求日志"""
        logger = StructuredLogger()
        request_data = {"tag": "投诉", "title": "测试标题", "body": "测试内容"}
        response_data = {"status": "ok", "reply": "测试回复"}
        logger.log_request(request_data, response_data, 1.5, "192.168.1.1")

    def test_log_quality_anomaly(self):
        """测试质量异常日志"""
        logger = StructuredLogger()
        logger.log_quality_anomaly("串区", {"summary": "测试异常", "district": "朝阳区"})

    def test_log_model_loading(self):
        """测试模型加载日志"""
        logger = StructuredLogger()
        logger.log_model_loading("classifier", True, 2.5)
        logger.log_model_loading("generator", False, 0.0, "加载失败")


class TestStructuredFormatter(unittest.TestCase):
    """测试结构化日志格式化器"""

    def test_format_log(self):
        """测试日志格式化"""
        import logging
        from src.jsjb.core.logging import StructuredFormatter
        
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="test message",
            args=(),
            exc_info=None,
        )
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["message"], "test message")
        self.assertIn("timestamp", parsed)


class TestLoggingSetup(unittest.TestCase):
    """测试日志设置"""

    def test_setup_logging(self):
        """测试设置日志级别"""
        setup_logging("DEBUG")
        logger = get_logger()
        self.assertEqual(logger.logger.level, 10)  # DEBUG level


if __name__ == "__main__":
    unittest.main()