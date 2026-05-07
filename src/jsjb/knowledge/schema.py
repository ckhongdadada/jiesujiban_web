"""Canonical schema helpers for the local government knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ENTITY_TYPES = {
    "Project",
    "Location",
    "Organization",
    "Policy",
    "Case",
    "Resource",
    "Fact",
}

RELATION_TYPES = {
    "LOCATED_IN",
    "RESPONSIBLE_FOR",
    "REPORTS_TO",
    "BASED_ON",
    "STATUS_CHANGED",
    "SIMILAR_TO",
    "HAS_RESOURCE",
    "EVIDENCED_BY",
    "RELATED_TO",
}

FACT_TO_ENTITY_TYPE = {
    "project_status": "Project",
    "demolition_status": "Project",
    "responsible_unit": "Organization",
    "unit_mapping": "Organization",
    "public_resource": "Resource",
}


@dataclass(frozen=True)
class EntitySchema:
    entity_type: str
    required_fields: tuple[str, ...] = ("name",)
    optional_fields: tuple[str, ...] = field(default_factory=tuple)


SCHEMA: dict[str, EntitySchema] = {
    "Project": EntitySchema(
        "Project",
        optional_fields=(
            "status",
            "demolition_status",
            "district",
            "street",
            "responsible_unit",
            "description",
            "source_text",
            "updated_from",
        ),
    ),
    "Location": EntitySchema(
        "Location",
        optional_fields=("district", "street", "type", "coordinates", "updated_from"),
    ),
    "Organization": EntitySchema(
        "Organization",
        optional_fields=("district", "street", "type", "level", "responsibilities", "updated_from"),
    ),
    "Resource": EntitySchema(
        "Resource",
        optional_fields=("district", "resource_type", "status", "address", "phone", "updated_from"),
    ),
    "Fact": EntitySchema(
        "Fact",
        optional_fields=("fact_type", "district", "source_text", "confidence", "updated_from"),
    ),
}


def canonical_entity_type(entity_type: str) -> str:
    text = (entity_type or "").strip()
    aliases = {
        "organization": "Organization",
        "unit": "Organization",
        "org": "Organization",
        "project": "Project",
        "location": "Location",
        "resource": "Resource",
        "fact": "Fact",
    }
    return aliases.get(text, aliases.get(text.lower(), text if text in ENTITY_TYPES else "Fact"))


def canonical_relation_type(relation_type: str) -> str:
    text = (relation_type or "RELATED_TO").strip().upper()
    return text if text in RELATION_TYPES else "RELATED_TO"


def normalize_properties(entity_type: str, name: str, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    entity_type = canonical_entity_type(entity_type)
    props = dict(properties or {})
    props["name"] = str(name or props.get("name") or "").strip()
    if not props["name"]:
        raise ValueError(f"{entity_type} entity requires non-empty name")

    schema = SCHEMA.get(entity_type)
    if schema:
        allowed = set(schema.required_fields) | set(schema.optional_fields)
        normalized = {key: value for key, value in props.items() if key in allowed or key.startswith("_")}
    else:
        normalized = props
    normalized["entity_type"] = entity_type
    return normalized


def fact_entity_type(fact_type: str) -> str:
    return FACT_TO_ENTITY_TYPE.get((fact_type or "").strip(), "Fact")


def fact_subject_name(fact_type: str, content: dict[str, Any]) -> str:
    if fact_type in {"project_status", "demolition_status", "responsible_unit", "unit_mapping"}:
        return str(content.get("project") or content.get("name") or "").strip()
    if fact_type == "public_resource":
        return str(content.get("name") or content.get("resource") or "").strip()
    return str(content.get("name") or content.get("project") or content.get("source_text") or fact_type).strip()
