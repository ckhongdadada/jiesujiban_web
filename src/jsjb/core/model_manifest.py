"""Model artifact manifest generation and validation.

The manifest is a lightweight lock file for runtime model artifacts.  It keeps
the classifier label signature, architecture metadata, and key file sizes in
one place so we can detect accidental model/label/vectorizer drift before a
demo or deployment.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: str | os.PathLike[str], *, include_hashes: bool = False) -> dict[str, Any]:
    file_path = Path(path)
    entry: dict[str, Any] = {
        "path": str(file_path),
        "exists": file_path.exists(),
        "size": None,
    }
    if file_path.exists() and file_path.is_file():
        stat = file_path.stat()
        entry["size"] = int(stat.st_size)
        entry["modified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime))
        if include_hashes:
            entry["sha256"] = _sha256_file(file_path)
    return entry


def _first_existing_file(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0] if paths else Path("")


def _label_signature(label_map: dict[str, Any]) -> str:
    canonical = json.dumps(label_map, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _label_map_summary(path: Path) -> dict[str, Any]:
    label_map = _load_json(path)
    return {
        "path": str(path),
        "exists": bool(label_map),
        "num_labels": len(label_map),
        "signature": _label_signature(label_map) if label_map else "",
    }


def _classifier_manifest(config: Any, *, include_hashes: bool) -> dict[str, Any]:
    model_dir = Path(str(config.classifier_model_dir or ""))
    label_map_path = model_dir / "label_map.json"
    meta_path = model_dir / "model_meta.json"
    meta = _load_json(meta_path)
    weights_path = _first_existing_file([model_dir / "pytorch_model.bin", model_dir / "model.safetensors"])
    tfidf_path = model_dir / str(meta.get("tfidf_vectorizer") or "tfidf_vectorizer.joblib")

    return {
        "model_dir": str(model_dir),
        "base_model": str(config.classifier_base_model or ""),
        "route": str(getattr(config, "classifier_route", "")),
        "label_map": _label_map_summary(label_map_path),
        "model_meta": {
            "path": str(meta_path),
            "exists": bool(meta),
            "content": meta,
        },
        "files": {
            "weights": _file_entry(weights_path, include_hashes=include_hashes),
            "label_map": _file_entry(label_map_path, include_hashes=include_hashes),
            "model_meta": _file_entry(meta_path, include_hashes=include_hashes),
            "tfidf_vectorizer": _file_entry(tfidf_path, include_hashes=include_hashes),
        },
    }


def _generator_manifest(config: Any, *, include_hashes: bool) -> dict[str, Any]:
    base_model = Path(str(config.generator_base_model or ""))
    lora_dir = Path(str(getattr(config, "generator_lora_dir", "") or ""))
    draft_model = str(getattr(config, "generator_draft_model", "") or "")

    return {
        "base_model": str(base_model),
        "lora_dir": str(lora_dir),
        "draft_model": draft_model,
        "files": {
            "base_config": _file_entry(base_model / "config.json", include_hashes=include_hashes),
            "base_tokenizer": _file_entry(base_model / "tokenizer_config.json", include_hashes=include_hashes),
            "lora_config": _file_entry(lora_dir / "adapter_config.json", include_hashes=include_hashes),
            "lora_weights_safetensors": _file_entry(lora_dir / "adapter_model.safetensors", include_hashes=include_hashes),
            "lora_weights_bin": _file_entry(lora_dir / "adapter_model.bin", include_hashes=include_hashes),
        },
    }


def build_model_manifest(config: Any, *, include_hashes: bool = False, created_by: str = "runtime") -> dict[str, Any]:
    """Build a manifest from the currently resolved runtime configuration."""

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "created_by": created_by,
        "include_hashes": bool(include_hashes),
        "components": {
            "classifier": _classifier_manifest(config, include_hashes=include_hashes),
            "generator": _generator_manifest(config, include_hashes=include_hashes),
        },
    }


def load_model_manifest(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    payload = _load_json(path)
    return payload or None


def _compare_file(
    name: str,
    expected: dict[str, Any],
    current: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if expected.get("exists") and not current.get("exists"):
        errors.append(f"{name} missing: {current.get('path')}")
        return
    if expected.get("size") is not None and current.get("size") != expected.get("size"):
        errors.append(f"{name} size changed: expected {expected.get('size')}, got {current.get('size')}")
    expected_hash = expected.get("sha256")
    current_hash = current.get("sha256")
    if expected_hash and current_hash and expected_hash != current_hash:
        errors.append(f"{name} sha256 changed")
    if not expected.get("exists") and current.get("exists"):
        warnings.append(f"{name} exists now but was absent when manifest was created")


def validate_model_manifest(config: Any, manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the current artifact set against a manifest payload."""

    result: dict[str, Any] = {
        "enabled": bool(manifest),
        "ok": True,
        "errors": [],
        "warnings": [],
    }
    if not manifest:
        result["warnings"].append("model manifest not configured or not found")
        return result

    if int(manifest.get("schema_version", 0)) != MANIFEST_SCHEMA_VERSION:
        result["errors"].append(
            f"unsupported manifest schema_version: {manifest.get('schema_version')}"
        )

    include_hashes = bool(manifest.get("include_hashes", False))
    current = build_model_manifest(config, include_hashes=include_hashes, created_by="validator")
    expected_components = manifest.get("components", {})
    current_components = current.get("components", {})

    expected_classifier = expected_components.get("classifier", {})
    current_classifier = current_components.get("classifier", {})
    expected_label = expected_classifier.get("label_map", {})
    current_label = current_classifier.get("label_map", {})
    if expected_label.get("signature") and expected_label.get("signature") != current_label.get("signature"):
        result["errors"].append("classifier label_map signature changed")
    if expected_label.get("num_labels") is not None and expected_label.get("num_labels") != current_label.get("num_labels"):
        result["errors"].append(
            f"classifier label count changed: expected {expected_label.get('num_labels')}, got {current_label.get('num_labels')}"
        )

    expected_meta = expected_classifier.get("model_meta", {}).get("content", {})
    current_meta = current_classifier.get("model_meta", {}).get("content", {})
    for key in ["architecture", "classifier_route", "use_tfidf", "tfidf_dim", "label_signature"]:
        if key in expected_meta and expected_meta.get(key) != current_meta.get(key):
            result["errors"].append(
                f"classifier model_meta.{key} changed: expected {expected_meta.get(key)!r}, got {current_meta.get(key)!r}"
            )

    expected_files = expected_classifier.get("files", {})
    current_files = current_classifier.get("files", {})
    for name in ["weights", "label_map", "model_meta", "tfidf_vectorizer"]:
        _compare_file(
            f"classifier.{name}",
            expected_files.get(name, {}),
            current_files.get(name, {}),
            result["errors"],
            result["warnings"],
        )

    expected_generator = expected_components.get("generator", {})
    current_generator = current_components.get("generator", {})
    expected_gen_files = expected_generator.get("files", {})
    current_gen_files = current_generator.get("files", {})
    for name in ["base_config", "base_tokenizer", "lora_config", "lora_weights_safetensors", "lora_weights_bin"]:
        _compare_file(
            f"generator.{name}",
            expected_gen_files.get(name, {}),
            current_gen_files.get(name, {}),
            result["errors"],
            result["warnings"],
        )

    result["ok"] = not result["errors"]
    return result


def validate_model_manifest_file(config: Any, path: str | os.PathLike[str]) -> dict[str, Any]:
    return validate_model_manifest(config, load_model_manifest(path))


def write_model_manifest(
    path: str | os.PathLike[str],
    config: Any,
    *,
    include_hashes: bool = False,
    created_by: str = "generate_model_manifest",
) -> dict[str, Any]:
    manifest = build_model_manifest(config, include_hashes=include_hashes, created_by=created_by)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    return manifest
