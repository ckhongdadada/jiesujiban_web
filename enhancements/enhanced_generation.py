from __future__ import annotations

# Compatibility shim for modules that still import enhancements.enhanced_generation.
# The maintained Qwen + LoRA implementation lives in generation_qwen_lora.py.
from enhancements.generation_qwen_lora import (  # noqa: F401
    fallback_generate_reply,
    generate_reply_with_context,
    generate_simple_reply,
    generator_loaded,
    generator_status,
    get_latest_lora_checkpoint,
    load_generator,
    pick_issue_keyword,
)
