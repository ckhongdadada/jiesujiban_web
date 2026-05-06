from __future__ import annotations

import json

from src.jsjb.core import config as runtime_config


def test_layered_config_overrides_app_defaults(tmp_path, monkeypatch):
    project = tmp_path
    monkeypatch.setattr(runtime_config, "_get_base_dir", lambda: str(project))
    (project / "configs" / "app").mkdir(parents=True)
    (project / "configs" / "retrieval").mkdir(parents=True)
    (project / "configs" / "generation").mkdir(parents=True)
    (project / "configs" / "feedback").mkdir(parents=True)
    (project / "configs" / "models").mkdir(parents=True)

    (project / "configs" / "app" / "config.json").write_text(
        json.dumps({"retrieval_top_k": 3, "generation_temperature": 0.2}, ensure_ascii=False),
        encoding="utf-8",
    )
    (project / "configs" / "retrieval" / "rag.json").write_text(
        json.dumps({"retrieval_top_k": 9, "rag_dense_weight": 0.7}, ensure_ascii=False),
        encoding="utf-8",
    )
    (project / "configs" / "generation" / "qwen_generation.json").write_text(
        json.dumps({"generation_temperature": 0.6}, ensure_ascii=False),
        encoding="utf-8",
    )
    (project / "configs" / "feedback" / "feedback_loop.json").write_text("{}", encoding="utf-8")

    loaded = runtime_config._load_json_config()

    assert loaded["retrieval_top_k"] == 9
    assert loaded["generation_temperature"] == 0.6
    assert loaded["rag_dense_weight"] == 0.7


def test_runtime_config_prefers_model_registry_paths(tmp_path, monkeypatch):
    project = tmp_path
    monkeypatch.setattr(runtime_config, "_get_base_dir", lambda: str(project))
    for rel in [
        "configs/app",
        "configs/models",
        "configs/retrieval",
        "configs/generation",
        "configs/feedback",
        "checkpoints/classifier/demo",
        "checkpoints/generator/base",
        "checkpoints/generator/lora",
    ]:
        (project / rel).mkdir(parents=True)
    for path in [
        project / "checkpoints" / "classifier" / "demo" / "label_map.json",
        project / "checkpoints" / "generator" / "base" / "config.json",
        project / "checkpoints" / "generator" / "lora" / "adapter_config.json",
    ]:
        path.write_text("{}", encoding="utf-8")
    (project / "configs" / "app" / "config.json").write_text("{}", encoding="utf-8")
    (project / "configs" / "retrieval" / "rag.json").write_text("{}", encoding="utf-8")
    (project / "configs" / "generation" / "qwen_generation.json").write_text("{}", encoding="utf-8")
    (project / "configs" / "feedback" / "feedback_loop.json").write_text("{}", encoding="utf-8")
    registry = {
        "schema_version": 1,
        "active": {
            "classifier": "demo",
            "generator_base": "base",
            "generator_lora": "lora",
        },
        "artifacts": {
            "classifier": {"demo": {"path": "checkpoints/classifier/demo"}},
            "generator_base": {"base": {"path": "checkpoints/generator/base"}},
            "generator_lora": {"lora": {"path": "checkpoints/generator/lora"}},
        },
    }
    (project / "configs" / "models" / "model_registry.json").write_text(json.dumps(registry), encoding="utf-8")

    cfg = runtime_config.load_runtime_config()

    assert cfg.classifier_model_dir.endswith("checkpoints\\classifier\\demo") or cfg.classifier_model_dir.endswith("checkpoints/classifier/demo")
    assert cfg.generator_base_model.endswith("checkpoints\\generator\\base") or cfg.generator_base_model.endswith("checkpoints/generator/base")
    assert cfg.generator_lora_dir.endswith("checkpoints\\generator\\lora") or cfg.generator_lora_dir.endswith("checkpoints/generator/lora")
