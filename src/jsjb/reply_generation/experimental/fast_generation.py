"""Experimental fast generation compatibility layer.

The optimized generation branch is kept importable during the project
restructure, while the maintained implementation lives in
``src.jsjb.reply_generation.qwen_lora``.
"""

from __future__ import annotations

from src.jsjb.reply_generation.qwen_lora import (  # noqa: F401
    fallback_generate_reply,
    generate_reply_with_context,
    generate_simple_reply,
    generator_loaded,
    generator_status,
    load_generator,
)


def clear_generation_cache() -> None:
    """Compatibility no-op for the old experimental cache API."""
    return None


def get_generation_stats() -> dict[str, object]:
    """Return a small status snapshot for callers of the experimental API."""
    return {
        "loaded": generator_loaded(),
        "backend": "qwen_lora",
    }
