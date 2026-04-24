"""
主动学习模块
提供不确定样本收集、人工标注、增量训练功能
"""

from enhancements.active_learning.sample_collector import SampleCollector
from enhancements.active_learning.annotation_manager import AnnotationManager
from enhancements.active_learning.incremental_trainer import IncrementalTrainer

__all__ = [
    "SampleCollector",
    "AnnotationManager",
    "IncrementalTrainer"
]
