"""Compatibility imports for the active learning module."""

from src.jsjb.active_learning.sample_collector import SampleCollector
from src.jsjb.active_learning.annotation import AnnotationManager
from src.jsjb.active_learning.trainer import IncrementalTrainer

__all__ = ["SampleCollector", "AnnotationManager", "IncrementalTrainer"]
