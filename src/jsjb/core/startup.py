from __future__ import annotations

import os
from typing import Any

from src.jsjb.core.artifacts import inspect_classifier_artifacts, inspect_generator_artifacts
from src.jsjb.core.config import RuntimeConfig


def run_startup_checks(config: RuntimeConfig) -> dict[str, Any]:
    project_root = os.path.dirname(os.path.dirname(__file__))
    classifier = inspect_classifier_artifacts(
        config.classifier_model_dir,
        config.classifier_base_model,
    )
    generator = inspect_generator_artifacts(
        config.generator_base_model,
        config.generator_lora_dir,
    )

    return {
        "classifier": classifier,
        "generator": generator,
        "curated_alias_exists": os.path.exists(
            os.path.join(project_root, "data", "beijing_districts_curated.json")
        ),
        "merged_alias_exists": os.path.exists(
            os.path.join(project_root, "data", "beijing_districts_merged.json")
        ),
        "rag_corpus_exists": os.path.exists(
            os.path.join(project_root, "data", "policy_case_corpus.sample.jsonl")
        ) or os.path.exists(
            os.path.join(project_root, "data", "policy_case_corpus.template.jsonl")
        ),
    }


def summarize_readiness(checks: dict[str, Any]) -> dict[str, Any]:
    classifier_strict_ready = checks["classifier"]["strict_runtime_ready"]
    classifier_compatible_ready = checks["classifier"]["compatible_runtime_ready"]
    generator_ready = checks["generator"]["runtime_ready"]

    return {
        "classifier_strict_ready": classifier_strict_ready,
        "classifier_compatible_ready": classifier_compatible_ready,
        "generator_assets_ready": generator_ready,
        "full_model_pipeline_ready": classifier_compatible_ready and generator_ready,
        "service_ready": classifier_compatible_ready or generator_ready,
        "checks": checks,
    }
