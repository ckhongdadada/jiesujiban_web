"""Import manually reviewed facts into the structured KB and graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.jsjb.knowledge.graph import KnowledgeGraphManager
from src.jsjb.knowledge.structured_kb import StructuredKnowledgeBase, get_knowledge_base


def build_kb_fact_from_candidate(candidate: Dict[str, Any], district: str = "") -> Dict[str, Any]:
    """Convert a review-queue row into the structured-KB fact shape."""
    fact_type = candidate.get("fact_type", "")
    content = candidate.get("fact_content", {}) or {}
    kb_fact = {
        "type": fact_type,
        "district": content.get("district", district),
        "street": content.get("street", ""),
        "source": "reviewed_feedback",
        "confidence": 0.9,
    }

    if fact_type in {"project_status", "demolition_status"}:
        kb_fact["project"] = content.get("project", "")
        kb_fact["status"] = content.get("status", "")
        kb_fact["demolition_status"] = content.get("demolition_status", "")
        kb_fact["current_phase"] = content.get("current_phase", "")
        kb_fact["expected_completion"] = content.get("expected_completion", "")
        kb_fact["responsible_unit"] = content.get("responsible_unit", "")
        kb_fact["source_text"] = content.get("source_text", "")
    elif fact_type in {"responsible_unit", "unit_mapping"}:
        kb_fact["type"] = "responsible_unit"
        kb_fact["project"] = content.get("project", "")
        kb_fact["unit"] = content.get("unit", content.get("responsible_unit", ""))
        kb_fact["source_text"] = content.get("source_text", "")
    elif fact_type == "public_resource":
        kb_fact["name"] = content.get("name", "")
        kb_fact["resource_type"] = content.get("resource_type", "")
        kb_fact["status"] = content.get("status", "")
        kb_fact["count"] = content.get("count", 0)
        kb_fact["facilities"] = content.get("facilities", [])
        kb_fact["address"] = content.get("address", "")
        kb_fact["phone"] = content.get("phone", "")
    else:
        kb_fact.update(content)

    return kb_fact


def import_reviewed_fact(
    candidate: Dict[str, Any],
    *,
    feedback: Dict[str, Any] | None = None,
    graph_manager: KnowledgeGraphManager | None = None,
    graph_path: str = "",
    knowledge_base: StructuredKnowledgeBase | None = None,
) -> Dict[str, Any]:
    """Persist an approved fact into durable project knowledge stores."""
    feedback = feedback or {}
    district = feedback.get("district", "")
    content = candidate.get("fact_content", {}) or {}
    fact_type = candidate.get("fact_type", "")
    kb = knowledge_base or get_knowledge_base()
    kb_fact = build_kb_fact_from_candidate(candidate, district=district)

    result: Dict[str, Any] = {
        "fact_type": fact_type,
        "structured_kb_updated": False,
        "graph_updated": False,
        "graph_saved": False,
        "created_nodes": [],
        "created_relations": [],
        "kb_fact": kb_fact,
    }

    if kb.add_fact(kb_fact):
        kb.save()
        result["structured_kb_updated"] = True

    if graph_manager is not None:
        _import_to_graph(graph_manager, fact_type, content, district, result)
        result["graph_updated"] = bool(result["created_nodes"] or result["created_relations"])
        if graph_path and not graph_manager.use_neo4j:
            Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
            graph_manager.save(graph_path)
            result["graph_saved"] = True

    return result


def _import_to_graph(
    manager: KnowledgeGraphManager,
    fact_type: str,
    content: Dict[str, Any],
    district: str,
    result: Dict[str, Any],
) -> None:
    if fact_type == "project_status":
        project_name = content.get("project", "")
        if not project_name:
            raise ValueError("project_status fact missing project name")
        project_id = manager.merge_entity(
            "Project",
            project_name,
            {
                "status": content.get("status", ""),
                "district": district,
                "responsible_unit": content.get("responsible_unit", ""),
                "description": content.get("source_text", ""),
                "source_text": content.get("source_text", ""),
                "updated_from": "reviewed_feedback",
            },
        )
        result["created_nodes"].append({"type": "Project", "id": project_id, "name": project_name})
        _attach_project_context(manager, project_id, project_name, content, district, result)
        return

    if fact_type == "demolition_status":
        project_name = content.get("project", "")
        if not project_name:
            raise ValueError("demolition_status fact missing project name")
        project_id = manager.merge_entity(
            "Project",
            project_name,
            {
                "demolition_status": content.get("demolition_status", ""),
                "district": district,
                "description": content.get("source_text", ""),
                "source_text": content.get("source_text", ""),
                "updated_from": "reviewed_feedback",
            },
        )
        result["created_nodes"].append({"type": "Project", "id": project_id, "name": project_name})
        _attach_project_context(manager, project_id, project_name, content, district, result)
        return

    if fact_type in {"responsible_unit", "unit_mapping"}:
        project_name = content.get("project", "")
        unit_name = content.get("unit", content.get("responsible_unit", ""))
        if not project_name or not unit_name:
            raise ValueError("responsible_unit fact missing project or unit")
        project_id = manager.merge_entity(
            "Project",
            project_name,
            {"district": district, "responsible_unit": unit_name, "updated_from": "reviewed_feedback"},
        )
        org_id = manager.merge_entity(
            "Organization",
            unit_name,
            {"district": district, "responsibilities": content.get("source_text", ""), "updated_from": "reviewed_feedback"},
        )
        manager.create_relationship(org_id, project_id, "RESPONSIBLE_FOR")
        result["created_nodes"].extend(
            [
                {"type": "Project", "id": project_id, "name": project_name},
                {"type": "Organization", "id": org_id, "name": unit_name},
            ]
        )
        result["created_relations"].append({"from": unit_name, "to": project_name, "type": "RESPONSIBLE_FOR"})
        _attach_project_context(manager, project_id, project_name, content, district, result, include_unit=False)
        return

    if fact_type == "public_resource":
        resource_name = content.get("name", "")
        if not resource_name:
            raise ValueError("public_resource fact missing resource name")
        resource_id = manager.merge_entity(
            "Resource",
            resource_name,
            {
                "status": content.get("status", ""),
                "resource_type": content.get("resource_type", ""),
                "district": district,
                "address": content.get("address", ""),
                "phone": content.get("phone", ""),
                "source_text": content.get("source_text", ""),
                "updated_from": "reviewed_feedback",
            },
        )
        result["created_nodes"].append({"type": "Resource", "id": resource_id, "name": resource_name})
        if district:
            location_id = manager.merge_entity("Location", district, {"district": district, "type": "区县"})
            manager.create_relationship(location_id, resource_id, "HAS_RESOURCE")
            result["created_nodes"].append({"type": "Location", "id": location_id, "name": district})
            result["created_relations"].append({"from": district, "to": resource_name, "type": "HAS_RESOURCE"})
        return

    raise ValueError(f"unsupported fact type: {fact_type}")


def _attach_project_context(
    manager: KnowledgeGraphManager,
    project_id: str,
    project_name: str,
    content: Dict[str, Any],
    district: str,
    result: Dict[str, Any],
    *,
    include_unit: bool = True,
) -> None:
    """Attach standard Project -> Location and Org -> Project context."""
    if district:
        location_id = manager.merge_entity(
            "Location",
            district,
            {"district": district, "type": "区县", "updated_from": "reviewed_feedback"},
        )
        manager.create_relationship(project_id, location_id, "LOCATED_IN")
        result["created_nodes"].append({"type": "Location", "id": location_id, "name": district})
        result["created_relations"].append({"from": project_name, "to": district, "type": "LOCATED_IN"})

    if include_unit:
        unit_name = content.get("unit") or content.get("responsible_unit")
        if unit_name:
            org_id = manager.merge_entity(
                "Organization",
                unit_name,
                {"district": district, "updated_from": "reviewed_feedback"},
            )
            manager.create_relationship(org_id, project_id, "RESPONSIBLE_FOR")
            result["created_nodes"].append({"type": "Organization", "id": org_id, "name": unit_name})
            result["created_relations"].append({"from": unit_name, "to": project_name, "type": "RESPONSIBLE_FOR"})
