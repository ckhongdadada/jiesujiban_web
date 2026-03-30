from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


def _get_base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class RuntimeConfig:
    classifier_model_dir: str
    classifier_base_model: str
    generator_base_model: str
    generator_lora_dir: str
    enhanced_port: int
    use_curated_aliases: bool

    @property
    def classifier_label_map(self) -> str:
        return os.path.join(self.classifier_model_dir, "label_map.json")

    def describe(self) -> dict[str, object]:
        return asdict(self)


def load_runtime_config() -> RuntimeConfig:
    base_dir = _get_base_dir()
    
    return RuntimeConfig(
        classifier_model_dir=os.getenv(
            "CLASSIFIER_MODEL_DIR",
            os.path.join(base_dir, "final_model_fgm"),
        ),
        classifier_base_model=os.getenv(
            "CLASSIFIER_BASE_MODEL",
            os.path.join(base_dir, "final_model_fgm"),
        ),
        generator_base_model=os.getenv(
            "GENERATOR_BASE_MODEL",
            os.path.join(base_dir, "qwen_models", "Qwen", "Qwen2___5-1___5B-Instruct"),
        ),
        generator_lora_dir=os.getenv(
            "GENERATOR_LORA_DIR",
            os.path.join(base_dir, "qwen_reply_model"),
        ),
        enhanced_port=int(os.getenv("ENHANCED_APP_PORT", "5000")),
        use_curated_aliases=os.getenv("USE_CURATED_ALIASES", "true").lower() == "true",
    )


def export_runtime_config(path: str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(load_runtime_config().describe(), fp, ensure_ascii=False, indent=2)
