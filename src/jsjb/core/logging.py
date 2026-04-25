"""
结构化日志系统
实现分级日志、文件轮转、结构化输出
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


class StructuredLogger:
    """结构化日志管理器"""

    def __init__(self, log_dir: str = "logs", max_bytes: int = 10*1024*1024, backup_count: int = 5):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger("complaint_system")
        self.logger.setLevel(logging.INFO)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        formatter = StructuredFormatter()

        log_file = os.path.join(log_dir, "complaint_system.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self._quality_logger = logging.getLogger("complaint_system.quality")
        self._quality_logger.setLevel(logging.INFO)
        self._quality_logger.propagate = False

        quality_file = os.path.join(log_dir, "quality_anomalies.log")
        quality_handler = logging.handlers.RotatingFileHandler(
            quality_file, maxBytes=max_bytes, backupCount=backup_count
        )
        quality_handler.setFormatter(formatter)
        self._quality_logger.addHandler(quality_handler)

        self._retrieval_quality_logger = logging.getLogger("complaint_system.retrieval_quality")
        self._retrieval_quality_logger.setLevel(logging.INFO)
        self._retrieval_quality_logger.propagate = False

        retrieval_quality_file = os.path.join(log_dir, "retrieval_weak_evidence.log")
        retrieval_quality_handler = logging.handlers.RotatingFileHandler(
            retrieval_quality_file, maxBytes=max_bytes, backupCount=backup_count
        )
        retrieval_quality_handler.setFormatter(formatter)
        self._retrieval_quality_logger.addHandler(retrieval_quality_handler)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """信息级别日志"""
        self.logger.info(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """警告级别日志"""
        self.logger.warning(message, extra=extra or {})
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """错误级别日志"""
        self.logger.error(message, extra=extra or {})
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """调试级别日志"""
        self.logger.debug(message, extra=extra or {})
    
    def log_request(self, request_data: Dict[str, Any], response_data: Dict[str, Any], 
                   processing_time: float, client_ip: str) -> None:
        """记录请求日志"""
        extra = {
            'type': 'request',
            'client_ip': client_ip,
            'processing_time': processing_time,
            'request_data': {
                'tag': request_data.get('tag', ''),
                'title_length': len(request_data.get('title', '')),
                'body_length': len(request_data.get('body', '')),
                'forced_unit': request_data.get('_force_unit', '')
            },
            'response_data': {
                'has_location': 'location' in response_data,
                'has_docs': 'docs' in response_data and len(response_data.get('docs', [])) > 0,
                'has_reply': 'reply' in response_data,
                'has_error': 'error' in response_data
            }
        }
        self.info(f"请求处理完成 - IP: {client_ip}, 耗时: {processing_time:.2f}s", extra)
    
    def log_model_loading(self, model_type: str, success: bool, load_time: float, 
                         error: Optional[str] = None) -> None:
        """记录模型加载日志"""
        extra = {
            'type': 'model_loading',
            'model_type': model_type,
            'success': success,
            'load_time': load_time,
            'error': error
        }
        
        if success:
            self.info(f"模型加载成功 - {model_type}, 耗时: {load_time:.2f}s", extra)
        else:
            self.error(f"模型加载失败 - {model_type}: {error}", extra)
    
    def log_retrieval(self, query: str, district: str, results_count: int, 
                     retrieval_time: float) -> None:
        """记录检索日志"""
        extra = {
            'type': 'retrieval',
            'query': query[:100],
            'district': district,
            'results_count': results_count,
            'retrieval_time': retrieval_time
        }
        self.info(f"检索完成 - 查询: {query[:50]}..., 结果数: {results_count}", extra)

    def log_quality_anomaly(self, anomaly_type: str, details: Dict[str, Any]) -> None:
        """记录生成质量异常到独立日志文件

        anomaly_type: 串区 | 弱证据 | 事实验证告警 | 生成失败
        """
        extra = {
            'type': 'quality_anomaly',
            'anomaly_type': anomaly_type,
            **details,
        }
        self._quality_logger.info(
            f"[质量异常] {anomaly_type}: {details.get('summary', '')}",
            extra=extra,
        )

    def log_weak_evidence(self, query: str, district: str, grounding: str,
                          top_score: float, matched_terms: list) -> None:
        """记录检索弱证据到独立日志文件，方便后续补语料"""
        extra = {
            'type': 'weak_evidence',
            'query': query[:200],
            'district': district,
            'grounding': grounding,
            'top_score': top_score,
            'matched_terms': matched_terms,
        }
        self._retrieval_quality_logger.info(
            f"[弱证据] grounding={grounding}, district={district}, query={query[:80]}",
            extra=extra,
        )


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加额外字段
        if hasattr(record, 'extra') and record.extra:
            log_data.update(record.extra)
        
        return json.dumps(log_data, ensure_ascii=False)


# 全局日志实例
logger = StructuredLogger()


def get_logger() -> StructuredLogger:
    """获取全局日志实例"""
    return logger


def setup_logging(log_level: str = "INFO") -> None:
    """设置日志级别"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.logger.setLevel(level)


# 日志装饰器
def log_execution_time(func_name: str):
    """记录函数执行时间的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                logger.info(
                    f"函数执行完成 - {func_name}",
                    extra={
                        'type': 'function_execution',
                        'function_name': func_name,
                        'processing_time': processing_time,
                        'success': True
                    }
                )
                return result
            except Exception as e:
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                logger.error(
                    f"函数执行失败 - {func_name}: {str(e)}",
                    extra={
                        'type': 'function_execution',
                        'function_name': func_name,
                        'processing_time': processing_time,
                        'success': False,
                        'error': str(e)
                    }
                )
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试日志功能
    logger = StructuredLogger()
    
    # 测试不同级别的日志
    logger.info("系统启动成功")
    logger.warning("内存使用率较高")
    logger.error("数据库连接失败")
    
    # 测试结构化日志
    logger.log_request(
        {"tag": "投诉", "title": "测试标题", "body": "测试内容"},
        {"reply": "测试回复"},
        1.23,
        "192.168.1.1"
    )
    
    print("日志测试完成，请查看 logs/complaint_system.log")