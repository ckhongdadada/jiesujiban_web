"""File-based model artifact registry and validation helpers."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.jsjb.core.paths import get_project_root


REGISTRY_SCHEMA_VERSION = 1


ARTIFACT_REQUIREMENTS = {
    "classifier": {
        "required_any": [["pytorch_model.bin", "model.safetensors"]],
        "required": ["label_map.json", "model_meta.json"],
        "optional": ["tfidf_vectorizer.joblib", "tokenizer.json", "tokenizer_config.json", "config.json"],
    },
    "qwen_lora": {
        "required": ["adapter_config.json"],
        "required_any": [["adapter_model.safetensors", "adapter_model.bin"]],
        "optional": ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md"],
    },
    "qwen_base": {
        "required": ["config.json"],
        "required_any": [["tokenizer.json", "tokenizer.model"]],
        "required_glob_any": [["*.safetensors", "*.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"]],
        "optional": ["tokenizer_config.json", "generation_config.json"],
    },
}


@dataclass
class RegistryValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    artifacts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "artifacts": self.artifacts,
        }


def get_default_registry_path() -> Path:
    return get_project_root() / "configs" / "models" / "model_registry.json"


def resolve_project_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return get_project_root() / candidate


def load_model_registry(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path else get_default_registry_path()
    if not registry_path.exists():
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "active": {},
            "artifacts": {},
        }
    with registry_path.open("r", encoding="utf-8-sig") as fp:
        payload = json.load(fp)
    return payload if isinstance(payload, dict) else {}


def save_model_registry(registry: dict[str, Any], path: str | os.PathLike[str] | None = None) -> Path:
    registry_path = Path(path) if path else get_default_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as fp:
        json.dump(registry, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    return registry_path


def get_active_artifact(
    component: str,
    registry: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    registry = registry or load_model_registry()
    active_version = (registry.get("active") or {}).get(component, "")
    artifact = ((registry.get("artifacts") or {}).get(component) or {}).get(active_version, {})
    return active_version, artifact


def validate_registry(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    registry = load_model_registry(path)
    errors: list[str] = []
    warnings: list[str] = []
    artifacts_report: dict[str, Any] = {}

    if int(registry.get("schema_version", 0)) != REGISTRY_SCHEMA_VERSION:
        errors.append(f"unsupported registry schema_version: {registry.get('schema_version')}")

    active = registry.get("active") or {}
    artifacts = registry.get("artifacts") or {}
    for component, version in active.items():
        component_artifacts = artifacts.get(component) or {}
        artifact = component_artifacts.get(version)
        if not artifact:
            errors.append(f"active {component} version not found in artifacts: {version}")
            continue
        report = validate_artifact(version=version, artifact=artifact)
        artifacts_report[f"{component}:{version}"] = report
        errors.extend(f"{component}:{version}: {item}" for item in report.get("errors", []))
        warnings.extend(f"{component}:{version}: {item}" for item in report.get("warnings", []))

    return RegistryValidationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        artifacts=artifacts_report,
    ).to_dict()


def validate_artifact(version: str, artifact: dict[str, Any]) -> dict[str, Any]:
    artifact_type = artifact.get("artifact_type", "")
    artifact_path = resolve_project_path(artifact.get("path", ""))
    requirements = ARTIFACT_REQUIREMENTS.get(artifact_type, {})
    errors: list[str] = []
    warnings: list[str] = []
    files: dict[str, Any] = {}

    if not artifact_path.exists():
        errors.append(f"artifact path does not exist: {artifact_path}")
        return {
            "version": version,
            "artifact_type": artifact_type,
            "path": str(artifact_path),
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "files": files,
        }
    if not artifact_type:
        warnings.append("artifact_type is empty")

    for filename in requirements.get("required", []):
        file_path = artifact_path / filename
        files[filename] = _file_status(file_path)
        if not file_path.exists():
            errors.append(f"required file missing: {filename}")

    for group in requirements.get("required_any", []):
        statuses = [_file_status(artifact_path / filename) for filename in group]
        for filename, status in zip(group, statuses):
            files[filename] = status
        if not any(status["exists"] for status in statuses):
            errors.append(f"one of required files missing: {group}")

    for group in requirements.get("required_glob_any", []):
        matched = []
        for pattern in group:
            matched.extend(artifact_path.glob(pattern))
        if not matched:
            errors.append(f"one of required file patterns missing: {group}")
        else:
            files[f"matched:{','.join(group)}"] = {
                "exists": True,
                "matches": [str(path) for path in matched[:10]],
                "count": len(matched),
            }

    for filename in requirements.get("optional", []):
        files.setdefault(filename, _file_status(artifact_path / filename))

    meta_path = artifact_path / "model_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            meta = {}
            errors.append("model_meta.json is not valid JSON")
        if artifact_type == "classifier":
            label_map = artifact_path / "label_map.json"
            num_labels = _label_count(label_map)
            meta_num = int(meta.get("num_classes") or 0)
            if meta_num and num_labels and meta_num != num_labels:
                errors.append(f"num_classes mismatch: model_meta={meta_num}, label_map={num_labels}")

    return {
        "version": version,
        "artifact_type": artifact_type,
        "path": str(artifact_path),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "files": files,
    }


def promote_artifact(
    *,
    component: str,
    version: str,
    registry_path: str | os.PathLike[str] | None = None,
    copy_to_current: bool = False,
    current_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Promote a registered artifact version to active, optionally copying it to a current directory."""
    registry = load_model_registry(registry_path)
    component_artifacts = (registry.get("artifacts") or {}).get(component) or {}
    artifact = component_artifacts.get(version)
    if not artifact:
        raise ValueError(f"artifact not found: component={component}, version={version}")
    validation = validate_artifact(version=version, artifact=artifact)
    if not validation.get("ok"):
        raise ValueError(f"artifact validation failed: {validation.get('errors')}")

    promoted_path = artifact.get("path", "")
    if copy_to_current:
        src = resolve_project_path(artifact.get("path", ""))
        dst = resolve_project_path(current_path or f"checkpoints/{component}/current")
        if dst.exists():
            backup = dst.with_name(f"{dst.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}")
            shutil.move(str(dst), str(backup))
        shutil.copytree(src, dst)
        promoted_path = str(dst.relative_to(get_project_root())) if dst.is_relative_to(get_project_root()) else str(dst)

    registry.setdefault("active", {})[component] = version
    registry["artifacts"][component][version]["promoted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if copy_to_current:
        registry["artifacts"][component][version]["current_path"] = promoted_path
    save_model_registry(registry, registry_path)
    return {
        "status": "ok",
        "component": component,
        "version": version,
        "copy_to_current": copy_to_current,
        "current_path": promoted_path if copy_to_current else "",
        "registry_path": str(Path(registry_path) if registry_path else get_default_registry_path()),
    }


def _file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size": 0}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
    }


def _label_count(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return len(payload) if isinstance(payload, dict) else 0
    except Exception:
        return 0
