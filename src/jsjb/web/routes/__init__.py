"""Route registration helpers for the web layer."""

from .analysis import register_analysis_routes
from .feedback import register_feedback_routes
from .knowledge_review import register_knowledge_review_routes
from .system_status import register_system_status_routes

__all__ = [
    "register_analysis_routes",
    "register_feedback_routes",
    "register_knowledge_review_routes",
    "register_system_status_routes",
]
