from __future__ import annotations

import jieba


def chinese_tokenizer(text: str) -> list[str]:
    """Stable tokenizer for TF-IDF vectorizers saved with joblib."""
    return jieba.lcut(text)


def _chinese_tokenizer(text: str) -> list[str]:
    """Compatibility name used by older training scripts."""
    return chinese_tokenizer(text)
