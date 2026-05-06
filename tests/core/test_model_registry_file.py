from __future__ import annotations

import json

from src.jsjb.core.model_registry_file import (
    promote_artifact,
    save_model_registry,
    validate_registry,
)


def test_validate_registry_accepts_minimal_classifier_artifact(tmp_path, monkeypatch):
    project = tmp_path
    monkeypatch.setattr("src.jsjb.core.model_registry_file.get_project_root", lambda: project)
    model_dir = project / "checkpoints" / "classifier" / "demo"
    model_dir.mkdir(parents=True)
    (model_dir / "pytorch_model.bin").write_bytes(b"weights")
    (model_dir / "label_map.json").write_text(json.dumps({"0": "政府"}, ensure_ascii=False), encoding="utf-8")
    (model_dir / "model_meta.json").write_text(json.dumps({"num_classes": 1}, ensure_ascii=False), encoding="utf-8")
    registry = {
        "schema_version": 1,
        "active": {"classifier": "demo"},
        "artifacts": {
            "classifier": {
                "demo": {
                    "path": "checkpoints/classifier/demo",
                    "artifact_type": "classifier",
                }
            }
        },
    }
    registry_path = project / "configs" / "models" / "model_registry.json"
    save_model_registry(registry, registry_path)

    result = validate_registry(registry_path)

    assert result["ok"] is True
    assert not result["errors"]


def test_promote_artifact_updates_active_version(tmp_path, monkeypatch):
    project = tmp_path
    monkeypatch.setattr("src.jsjb.core.model_registry_file.get_project_root", lambda: project)
    for version in ["old", "new"]:
        model_dir = project / "checkpoints" / "classifier" / version
        model_dir.mkdir(parents=True)
        (model_dir / "pytorch_model.bin").write_bytes(b"weights")
        (model_dir / "label_map.json").write_text(json.dumps({"0": "政府"}, ensure_ascii=False), encoding="utf-8")
        (model_dir / "model_meta.json").write_text(json.dumps({"num_classes": 1}, ensure_ascii=False), encoding="utf-8")
    registry = {
        "schema_version": 1,
        "active": {"classifier": "old"},
        "artifacts": {
            "classifier": {
                "old": {"path": "checkpoints/classifier/old", "artifact_type": "classifier"},
                "new": {"path": "checkpoints/classifier/new", "artifact_type": "classifier"},
            }
        },
    }
    registry_path = project / "configs" / "models" / "model_registry.json"
    save_model_registry(registry, registry_path)

    promote_artifact(component="classifier", version="new", registry_path=registry_path)
    updated = json.loads(registry_path.read_text(encoding="utf-8"))

    assert updated["active"]["classifier"] == "new"
