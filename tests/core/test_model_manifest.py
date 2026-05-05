from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.jsjb.core.model_manifest import (
    build_model_manifest,
    validate_model_manifest,
    write_model_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_config(tmp_path: Path) -> SimpleNamespace:
    classifier_dir = tmp_path / "classifier"
    base_model = tmp_path / "base_model"
    generator_base = tmp_path / "generator_base"
    lora_dir = tmp_path / "lora"

    classifier_dir.mkdir()
    base_model.mkdir()
    generator_base.mkdir()
    lora_dir.mkdir()

    _write_json(classifier_dir / "label_map.json", {"0": "城管委", "1": "住建委"})
    _write_json(
        classifier_dir / "model_meta.json",
        {
            "architecture": "bert_cnn_attention_tfidf",
            "classifier_route": "hybrid_rankaware",
            "use_tfidf": True,
            "tfidf_dim": 3000,
            "label_signature": "demo-signature",
        },
    )
    (classifier_dir / "pytorch_model.bin").write_bytes(b"classifier-weights")
    (classifier_dir / "tfidf_vectorizer.joblib").write_bytes(b"tfidf")
    _write_json(generator_base / "config.json", {"model_type": "qwen2"})
    _write_json(generator_base / "tokenizer_config.json", {"tokenizer_class": "demo"})
    _write_json(lora_dir / "adapter_config.json", {"peft_type": "LORA"})
    (lora_dir / "adapter_model.safetensors").write_bytes(b"lora")

    return SimpleNamespace(
        classifier_model_dir=str(classifier_dir),
        classifier_base_model=str(base_model),
        classifier_route="hybrid_rankaware",
        generator_base_model=str(generator_base),
        generator_lora_dir=str(lora_dir),
        generator_draft_model="",
    )


def test_build_model_manifest_records_label_signature(tmp_path):
    config = _make_config(tmp_path)
    manifest = build_model_manifest(config)

    classifier = manifest["components"]["classifier"]
    assert classifier["label_map"]["num_labels"] == 2
    assert len(classifier["label_map"]["signature"]) == 64
    assert classifier["files"]["weights"]["size"] == len(b"classifier-weights")


def test_validate_model_manifest_accepts_unchanged_artifacts(tmp_path):
    config = _make_config(tmp_path)
    manifest = build_model_manifest(config)

    status = validate_model_manifest(config, manifest)

    assert status["ok"] is True
    assert status["errors"] == []


def test_validate_model_manifest_detects_label_drift(tmp_path):
    config = _make_config(tmp_path)
    manifest = build_model_manifest(config)
    label_path = Path(config.classifier_model_dir) / "label_map.json"
    _write_json(label_path, {"0": "城管委", "1": "住建委", "2": "交通委"})

    status = validate_model_manifest(config, manifest)

    assert status["ok"] is False
    assert any("label_map signature changed" in error for error in status["errors"])
    assert any("label count changed" in error for error in status["errors"])


def test_write_model_manifest_roundtrip(tmp_path):
    config = _make_config(tmp_path)
    output = tmp_path / "runtime" / "model_manifest.json"

    manifest = write_model_manifest(output, config)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == manifest["schema_version"]
