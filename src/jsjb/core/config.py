from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


def _get_base_dir() -> str:
    from src.jsjb.core.paths import get_project_root
    return str(get_project_root())


def _first_existing_path(candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ""


def _resolve_config_path(base_dir: str, raw_path: str) -> str:
    if not raw_path:
        return ""
    return raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)


def _load_json_file(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _load_json_config() -> dict:
    base_dir = _get_base_dir()
    config_candidates = [
        os.path.join(base_dir, "configs", "app", "config.json"),
        os.path.join(base_dir, "config.json"),
    ]
    config_path = _first_existing_path(config_candidates)
    config = _load_json_file(config_path)

    # Layered configs override app-level defaults but keep old config.json compatible.
    for relative_path in [
        os.path.join("configs", "generation", "qwen_generation.json"),
        os.path.join("configs", "retrieval", "rag.json"),
        os.path.join("configs", "feedback", "feedback_loop.json"),
    ]:
        config.update(_load_json_file(os.path.join(base_dir, relative_path)))
    return config


def _load_model_registry() -> dict:
    base_dir = _get_base_dir()
    return _load_json_file(os.path.join(base_dir, "configs", "models", "model_registry.json"))


def _registry_artifact_path(component: str) -> str:
    registry = _load_model_registry()
    active_version = (registry.get("active") or {}).get(component, "")
    artifact = ((registry.get("artifacts") or {}).get(component) or {}).get(active_version, {})
    raw_path = artifact.get("path", "")
    if not raw_path:
        return ""
    return _resolve_config_path(_get_base_dir(), raw_path)


def _env_bool(name: str, fallback: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(fallback)
    return value.lower() == "true"


def _json_bool(config: dict, key: str, fallback: bool) -> bool:
    return bool(config.get(key, fallback))


def load_named_config(name: str) -> dict:
    """Load a named config file from configs/<name parts>.json.

    Examples:
        load_named_config("retrieval/rag")
        load_named_config("evaluation/reply_quality")
    """
    base_dir = _get_base_dir()
    safe_name = name.replace("\\", "/").strip("/")
    return _load_json_file(os.path.join(base_dir, "configs", *safe_name.split("/")) + ".json")


def load_reply_quality_config() -> dict:
    return load_named_config("evaluation/reply_quality")


def load_model_registry_config() -> dict:
    return _load_model_registry()


def load_effective_config_dict() -> dict:
    return load_runtime_config().describe()


def export_effective_config(path: str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(load_effective_config_dict(), fp, ensure_ascii=False, indent=2)


@dataclass
class RuntimeConfig:
    classifier_model_dir: str
    classifier_base_model: str
    generator_base_model: str
    generator_lora_dir: str = ""
    generator_draft_model: str = ""
    classifier_route: str = "hybrid_rankaware"
    classifier_enable_legacy_fallback: bool = True
    use_curated_aliases: bool = True
    host: str = "0.0.0.0"
    port: int = 5000
    enhanced_port: int = 5000
    debug: bool = False
    generation_max_tokens: int = 512
    generation_temperature: float = 0.7
    enable_fact_verification: bool = True
    enable_assisted_decoding: bool = False
    retrieval_top_k: int = 5
    rag_backend: str = "hybrid"
    rag_enable_query_rewrite: bool = True
    rag_multi_query_count: int = 4
    rag_dense_weight: float = 0.68
    rag_sparse_weight: float = 0.32
    enable_detailed_logging: bool = False
    enable_geo_validation: bool = False
    enable_context_disambiguation: bool = True
    enable_cache: bool = True
    enable_disambiguation: bool = True
    cache_ttl: int = 3600
    cache_local_size: int = 10000
    batch_max_size: int = 10
    batch_max_workers: int = 4
    rate_limit_seconds: float = 3.0
    rate_limit_max_requests: int = 100
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    enable_active_learning: bool = False
    active_learning_confidence_threshold: float = 70.0
    active_learning_db_path: str = ""
    active_learning_base_data_path: str = ""
    enable_knowledge_graph: bool = False
    knowledge_graph_path: str = ""
    knowledge_graph_top_k: int = 3
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    enable_neo4j: bool = False
    rag_enable_chunking: bool = True
    rag_chunk_size: int = 400
    rag_chunk_overlap: int = 60
    rag_enable_reranker: bool = True
    rag_enable_graph_augment: bool = True
    rag_enable_feedback_boost: bool = True
    rag_enable_post_processing: bool = True
    model_manifest_path: str = ""
    enforce_model_manifest: bool = False

    @property
    def classifier_label_map(self) -> str:
        return os.path.join(self.classifier_model_dir, "label_map.json")

    def describe(self) -> dict[str, object]:
        return asdict(self)


def load_runtime_config() -> RuntimeConfig:
    base_dir = _get_base_dir()
    json_config = _load_json_config()
    
    classifier_model_dir = os.getenv(
        "CLASSIFIER_MODEL_DIR",
        _first_existing_path(
            [
                _registry_artifact_path("classifier"),
                os.path.join(base_dir, "checkpoints", "classifier", "final_model_hybrid_v3_32cls"),
                os.path.join(base_dir, "checkpoints", "classifier", "final_model_fgm"),
                os.path.join(base_dir, "final_model_hybrid_v3_32cls"),
                os.path.join(base_dir, "final_model_fgm"),
            ]
        ),
    )
    classifier_base_model = os.getenv(
        "CLASSIFIER_BASE_MODEL",
        _first_existing_path(
            [
                os.path.join(base_dir, "local_roberta_model"),
                os.path.join(base_dir, "checkpoints", "classifier", "local_roberta_model"),
                r"C:\python\接诉即办\.venv\local_roberta_model",
                r"C:\Users\28414\Desktop\接诉即办\.venv\local_roberta_model",
                classifier_model_dir,
            ]
        ),
    )
    generator_base_model = os.getenv(
        "GENERATOR_BASE_MODEL",
        _first_existing_path(
            [
                _registry_artifact_path("generator_base"),
                os.path.join(base_dir, "qwen_models", "Qwen", "Qwen2___5-1___5B-Instruct"),
                os.path.join(base_dir, "checkpoints", "generator", "base_models", "qwen_models", "Qwen", "Qwen2___5-1___5B-Instruct"),
                os.path.join(base_dir, "qwen_models", "Qwen", "Qwen2.5-0.5B-Instruct"),
                os.path.join(base_dir, "checkpoints", "generator", "base_models", "qwen_models", "Qwen", "Qwen2.5-0.5B-Instruct"),
                os.path.join(base_dir, "Qwen2.5-3B-Instruct"),
                r"C:\python\接诉即办\.venv\Qwen2.5-3B-Instruct",
                r"C:\Users\28414\Desktop\接诉即办\.venv\Qwen2.5-3B-Instruct",
            ]
        ),
    )
    generator_lora_dir = os.getenv(
        "GENERATOR_LORA_DIR",
        _first_existing_path(
            [
                _registry_artifact_path("generator_lora"),
                os.path.join(base_dir, "qwen_reply_model"),
                os.path.join(base_dir, "checkpoints", "generator", "lora", "qwen_reply_model"),
                r"C:\Users\28414\Desktop\qwen_reply_model",
            ]
        ),
    )
    generator_draft_model = os.getenv(
        "GENERATOR_DRAFT_MODEL",
        _first_existing_path(
            [
                os.path.join(base_dir, "qwen_models", "Qwen", "Qwen2.5-0.5B-Instruct"),
                os.path.join(base_dir, "checkpoints", "generator", "base_models", "qwen_models", "Qwen", "Qwen2.5-0.5B-Instruct"),
                "",
            ]
        ),
    )
    return RuntimeConfig(
        classifier_model_dir=classifier_model_dir,
        classifier_base_model=classifier_base_model,
        generator_base_model=generator_base_model,
        generator_lora_dir=generator_lora_dir,
        generator_draft_model=generator_draft_model,
        classifier_route=os.getenv("CLASSIFIER_ROUTE", json_config.get("classifier_route", "hybrid_rankaware")),
        classifier_enable_legacy_fallback=json_config.get("classifier_enable_legacy_fallback", True),
        use_curated_aliases=os.getenv("USE_CURATED_ALIASES", "true").lower() == "true",
        host=os.getenv("APP_HOST", json_config.get("host", "0.0.0.0")),
        port=int(os.getenv("APP_PORT", str(json_config.get("port", 5000)))),
        enhanced_port=int(os.getenv("ENHANCED_APP_PORT", str(json_config.get("enhanced_port", 5000)))),
        debug=_json_bool(json_config, "debug", False),
        generation_max_tokens=int(json_config.get("generation_max_tokens", 512)),
        generation_temperature=float(json_config.get("generation_temperature", 0.7)),
        enable_fact_verification=_json_bool(json_config, "feature_enable_fact_verification", True),
        enable_assisted_decoding=_json_bool(json_config, "feature_enable_assisted_decoding", False),
        retrieval_top_k=int(json_config.get("retrieval_top_k", 5)),
        rag_backend=os.getenv("RAG_BACKEND", json_config.get("rag_backend", "hybrid")),
        rag_enable_query_rewrite=_json_bool(json_config, "rag_enable_query_rewrite", True),
        rag_multi_query_count=int(json_config.get("rag_multi_query_count", 4)),
        rag_dense_weight=float(json_config.get("rag_dense_weight", 0.68)),
        rag_sparse_weight=float(json_config.get("rag_sparse_weight", 0.32)),
        enable_detailed_logging=_json_bool(json_config, "feature_enable_detailed_logging", False),
        enable_geo_validation=_json_bool(json_config, "feature_enable_geo_validation", False),
        enable_context_disambiguation=_json_bool(json_config, "feature_enable_context_disambiguation", True),
        enable_cache=_json_bool(json_config, "feature_enable_cache", True),
        enable_disambiguation=_json_bool(json_config, "feature_enable_disambiguation", True),
        cache_ttl=int(json_config.get("cache_ttl", 3600)),
        cache_local_size=int(json_config.get("cache_local_size", 10000)),
        batch_max_size=int(json_config.get("batch_max_size", 10)),
        batch_max_workers=int(json_config.get("batch_max_workers", 4)),
        rate_limit_seconds=float(json_config.get("rate_limit_seconds", 3.0)),
        rate_limit_max_requests=int(json_config.get("rate_limit_max_requests", 100)),
        enable_redis=_json_bool(json_config, "feature_enable_redis", False),
        redis_host=os.getenv("REDIS_HOST", json_config.get("redis_host", "localhost")),
        redis_port=int(os.getenv("REDIS_PORT", str(json_config.get("redis_port", 6379)))),
        redis_db=int(os.getenv("REDIS_DB", str(json_config.get("redis_db", 0)))),
        redis_password=os.getenv("REDIS_PASSWORD", json_config.get("redis_password", "")),
        enable_active_learning=_json_bool(json_config, "feature_enable_active_learning", False),
        active_learning_confidence_threshold=float(
            json_config.get("active_learning_confidence_threshold", 70.0)
        ),
        active_learning_db_path=_resolve_config_path(
            base_dir,
            json_config.get(
                "active_learning_db_path",
                os.path.join(base_dir, "data", "runtime", "active_learning.db"),
            ),
        ),
        active_learning_base_data_path=_resolve_config_path(
            base_dir,
            json_config.get(
                "active_learning_base_data_path",
                os.path.join(base_dir, "data", "runtime", "active_learning_base.jsonl"),
            ),
        ),
        enable_knowledge_graph=_json_bool(json_config, "feature_enable_knowledge_graph", False),
        knowledge_graph_path=_resolve_config_path(
            base_dir,
            json_config.get(
                "knowledge_graph_path",
                os.path.join(base_dir, "data", "runtime", "knowledge_graph.json"),
            ),
        ),
        knowledge_graph_top_k=int(json_config.get("knowledge_graph_top_k", 3)),
        neo4j_uri=os.getenv("NEO4J_URI", json_config.get("neo4j_uri", "")),
        neo4j_user=os.getenv("NEO4J_USER", json_config.get("neo4j_user", "neo4j")),
        neo4j_password=os.getenv("NEO4J_PASSWORD", json_config.get("neo4j_password", "")),
        enable_neo4j=_json_bool(json_config, "feature_enable_neo4j", False),
        rag_enable_chunking=_env_bool("RAG_ENABLE_CHUNKING", _json_bool(json_config, "rag_enable_chunking", True)),
        rag_chunk_size=int(os.getenv("RAG_CHUNK_SIZE", str(json_config.get("rag_chunk_size", 400)))),
        rag_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", str(json_config.get("rag_chunk_overlap", 60)))),
        rag_enable_reranker=_env_bool("RAG_ENABLE_RERANKER", _json_bool(json_config, "rag_enable_reranker", True)),
        rag_enable_graph_augment=_env_bool("RAG_ENABLE_GRAPH_AUGMENT", _json_bool(json_config, "rag_enable_graph_augment", True)),
        rag_enable_feedback_boost=_env_bool("RAG_ENABLE_FEEDBACK_BOOST", _json_bool(json_config, "rag_enable_feedback_boost", True)),
        rag_enable_post_processing=_env_bool("RAG_ENABLE_POST_PROCESSING", _json_bool(json_config, "rag_enable_post_processing", True)),
        model_manifest_path=_resolve_config_path(
            base_dir,
            os.getenv(
                "MODEL_MANIFEST_PATH",
                json_config.get("model_manifest_path", os.path.join("data", "runtime", "model_manifest.json")),
            ),
        ),
        enforce_model_manifest=os.getenv(
            "ENFORCE_MODEL_MANIFEST",
            str(json_config.get("feature_enforce_model_manifest", False)),
        ).lower() == "true",
    )


def export_runtime_config(path: str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(load_runtime_config().describe(), fp, ensure_ascii=False, indent=2)
