"""
知识图谱主流程 Demo
完整演示：留言输入 -> 事实提取 -> 图谱查询 -> 审核验证 -> 知识入库

主用途定位：事实结构化 + 审核验证
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from src.jsjb.knowledge.entity_extractor import EntityExtractor
from src.jsjb.knowledge.graph import KnowledgeGraphManager
from src.jsjb.knowledge.query import GraphQueryEngine
from src.jsjb.knowledge.structured_kb import StructuredKnowledgeBase
from src.jsjb.knowledge.fusion import InformationFusion


SEPARATOR = "=" * 60


def _print_stage(stage_num: int, title: str):
    print(f"\n{SEPARATOR}")
    print(f"  阶段 {stage_num}：{title}")
    print(SEPARATOR)


def _print_json(data: Any, indent: int = 2):
    if isinstance(data, (dict, list)):
        print(json.dumps(data, ensure_ascii=False, indent=indent))
    else:
        print(data)


def step1_extract_entities(complaint_text: str, district: str = "") -> Dict[str, Any]:
    """阶段1：从留言文本中提取实体"""
    _print_stage(1, "事实提取")

    extractor = EntityExtractor(use_ner=False)

    doc = {
        "title": complaint_text[:50],
        "content": complaint_text,
        "district": district,
    }

    entities = extractor.extract_from_document(doc)

    print(f"\n[输入文本] {complaint_text[:80]}...")
    print(f"\n[提取结果]")
    print(f"  项目: {[p['name'] for p in entities['projects']]}")
    print(f"  地点: {[l['name'] for l in entities['locations']]}")
    print(f"  单位: {[o['name'] for o in entities['organizations']]}")
    print(f"  政策: {[p['title'] for p in entities['policies']]}")
    print(f"  日期: {entities['dates']}")
    print(f"  状态: {entities['statuses']}")
    print(f"  联系方式: {entities['contacts']}")
    print(f"  金额: {entities['amounts']}")
    print(f"  关系数: {len(entities['relationships'])}")

    return entities


def step2_query_graph(
    entities: Dict[str, Any],
    graph_manager: KnowledgeGraphManager,
    district: str = "",
) -> Dict[str, Any]:
    """阶段2：查询图谱获取上下文"""
    _print_stage(2, "图谱查询")

    query_engine = GraphQueryEngine(graph_manager)
    context: Dict[str, Any] = {
        "responsibility_chain": [],
        "similar_cases": [],
        "related_policies": [],
        "known_projects": [],
    }

    org_names = [o["name"] for o in entities.get("organizations", [])]
    for org_name in org_names[:3]:
        chain = query_engine.query_responsibility_chain(org_name)
        if chain:
            context["responsibility_chain"].append({
                "org": org_name,
                "chain": [c.get("properties", {}).get("name", c.get("id", "")) for c in chain],
            })

    for project in entities.get("projects", [])[:3]:
        info = query_engine.query_project_info(project["name"])
        if "error" not in info:
            context["known_projects"].append(info)

    project_names = [p["name"] for p in entities.get("projects", [])]
    for pname in project_names[:3]:
        similar = query_engine.query_similar_cases(pname)
        if similar:
            context["similar_cases"].extend(similar)

    district_projects = query_engine.query_projects_by_district(district) if district else []
    if district_projects:
        context["district_projects"] = district_projects[:5]

    print(f"\n[查询结果]")
    print(f"  责任链: {len(context['responsibility_chain'])} 条")
    for rc in context["responsibility_chain"]:
        print(f"    {rc['org']} -> {' -> '.join(rc['chain'])}")

    print(f"  已知项目: {len(context['known_projects'])} 个")
    for kp in context["known_projects"]:
        proj = kp.get("project", {})
        print(f"    {proj.get('properties', {}).get('name', '未知')} - 状态: {proj.get('properties', {}).get('status', '未知')}")

    print(f"  相似案例: {len(context['similar_cases'])} 条")
    print(f"  同区项目: {len(context.get('district_projects', []))} 个")

    if not any(context.values()):
        print("  (图谱为空，跳过查询 — 首次运行时正常)")

    return context


def step3_audit_facts(
    entities: Dict[str, Any],
    graph_context: Dict[str, Any],
    kb: StructuredKnowledgeBase,
) -> Dict[str, Any]:
    """阶段3：审核验证提取的事实"""
    _print_stage(3, "审核验证")

    fusion = InformationFusion()
    audit_results: Dict[str, Any] = {
        "approved": [],
        "rejected": [],
        "needs_review": [],
        "conflicts": [],
    }

    sources: Dict[str, List[Dict[str, Any]]] = {"extracted": []}

    for project in entities.get("projects", []):
        sources["extracted"].append({
            "project": project["name"],
            "status": project.get("status", ""),
            "responsible_unit": "",
            "district": project.get("district", ""),
        })

    for org in entities.get("organizations", []):
        sources["extracted"].append({
            "name": org["name"],
            "status": "",
            "responsible_unit": org["name"],
            "district": org.get("district", ""),
        })

    known_facts: List[Dict[str, Any]] = []
    for project in entities.get("projects", []):
        existing = kb.get_project_status(project["name"])
        if existing:
            known_facts.append({
                "project": project["name"],
                "status": existing.status,
                "responsible_unit": existing.responsible_unit,
                "district": existing.district,
                "source": "knowledge_base",
            })

    if known_facts:
        sources["knowledge_base"] = known_facts

    if len(sources) > 1:
        fused = fusion.fuse_multi_source_info(sources)
        for fact_key, fact_info in fused.get("fused_facts", {}).items():
            if fact_info.get("has_conflict"):
                audit_results["conflicts"].append(fact_info)
                audit_results["needs_review"].append(fact_info)
            else:
                audit_results["approved"].append(fact_info)
    else:
        for item in sources.get("extracted", []):
            item_copy = dict(item)
            item_copy["confidence"] = 0.7
            item_copy["sources"] = ["extracted"]
            item_copy["has_conflict"] = False
            audit_results["approved"].append(item_copy)

    CONFIDENCE_THRESHOLD = 0.5
    final_approved = []
    for item in audit_results["approved"]:
        if item.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
            final_approved.append(item)
        else:
            item["reject_reason"] = "置信度过低"
            audit_results["rejected"].append(item)
    audit_results["approved"] = final_approved

    print(f"\n[审核结果]")
    print(f"  通过: {len(audit_results['approved'])} 条")
    print(f"  拒绝: {len(audit_results['rejected'])} 条")
    print(f"  待人工审核: {len(audit_results['needs_review'])} 条")
    print(f"  冲突: {len(audit_results['conflicts'])} 条")

    for item in audit_results["approved"][:5]:
        name = item.get("entity_name") or item.get("project") or item.get("name", "未知")
        conf = item.get("confidence", "N/A")
        print(f"    [通过] {name} (置信度: {conf})")

    for item in audit_results["rejected"][:3]:
        name = item.get("entity_name") or item.get("project") or item.get("name", "未知")
        reason = item.get("reject_reason", "")
        print(f"    [拒绝] {name} ({reason})")

    for item in audit_results["conflicts"][:3]:
        name = item.get("entity_name", "未知")
        print(f"    [冲突] {name}: {item.get('value')} vs 已有知识")

    return audit_results


def step4_write_to_kb(
    audit_results: Dict[str, Any],
    entities: Dict[str, Any],
    graph_manager: KnowledgeGraphManager,
    kb: StructuredKnowledgeBase,
) -> Dict[str, Any]:
    """阶段4：审核通过的事实入库"""
    _print_stage(4, "知识入库")

    write_report: Dict[str, Any] = {
        "graph_nodes_added": 0,
        "graph_edges_added": 0,
        "kb_facts_added": 0,
        "details": [],
    }

    entity_map: Dict[str, str] = {}

    for loc in entities.get("locations", []):
        props = {k: v for k, v in loc.items() if k != "name"}
        node_id = graph_manager.create_location(loc["name"], **props)
        entity_map[("Location", loc["name"])] = node_id
        write_report["graph_nodes_added"] += 1
        write_report["details"].append(f"地点节点: {loc['name']}")

    for org in entities.get("organizations", []):
        props = {k: v for k, v in org.items() if k != "name"}
        node_id = graph_manager.create_organization(org["name"], **props)
        entity_map[("Organization", org["name"])] = node_id
        write_report["graph_nodes_added"] += 1
        write_report["details"].append(f"单位节点: {org['name']}")

    for project in entities.get("projects", []):
        props = {k: v for k, v in project.items() if k != "name"}
        node_id = graph_manager.create_project(project["name"], **props)
        entity_map[("Project", project["name"])] = node_id
        write_report["graph_nodes_added"] += 1
        write_report["details"].append(f"项目节点: {project['name']}")

    for policy in entities.get("policies", []):
        props = {k: v for k, v in policy.items() if k != "title"}
        node_id = graph_manager.create_policy(policy["title"], **props)
        entity_map[("Policy", policy["title"])] = node_id
        write_report["graph_nodes_added"] += 1
        write_report["details"].append(f"政策节点: {policy['title']}")

    for rel in entities.get("relationships", []):
        from_key = (rel["from_type"], rel["from"])
        to_key = (rel["to_type"], rel["to"])
        if from_key in entity_map and to_key in entity_map:
            graph_manager.create_relationship(
                entity_map[from_key],
                entity_map[to_key],
                rel["rel_type"],
            )
            write_report["graph_edges_added"] += 1

    for org_name, loc_names in _iter_org_location_pairs(entities):
        org_key = ("Organization", org_name)
        loc_key = ("Location", loc_names)
        if org_key in entity_map and loc_key in entity_map:
            graph_manager.add_located_in(entity_map[org_key], entity_map[loc_key])
            write_report["graph_edges_added"] += 1

    for project in entities.get("projects", []):
        project_key = ("Project", project["name"])
        district = project.get("district", "")
        loc_key = ("Location", district)
        if project_key in entity_map and loc_key in entity_map:
            graph_manager.add_located_in(entity_map[project_key], entity_map[loc_key])
            write_report["graph_edges_added"] += 1

        for org in entities.get("organizations", []):
            org_key = ("Organization", org["name"])
            if project_key in entity_map and org_key in entity_map:
                graph_manager.add_responsible_for(entity_map[org_key], entity_map[project_key])
                write_report["graph_edges_added"] += 1
                break

    for item in audit_results.get("approved", []):
        project_name = item.get("entity_name") or item.get("project") or item.get("name", "")
        if not project_name:
            continue

        status = item.get("value", "") if "fact_type" in item else item.get("status", "")
        unit = item.get("responsible_unit", "")
        district = item.get("district", "")

        fact = {
            "type": "project_status",
            "project": project_name,
            "status": status,
            "responsible_unit": unit,
            "district": district,
            "source": "demo_pipeline",
            "confidence": item.get("confidence", 0.7),
        }
        if kb.add_fact(fact):
            write_report["kb_facts_added"] += 1
            write_report["details"].append(f"KB事实: {project_name} ({status})")

    print(f"\n[入库结果]")
    print(f"  图谱新增节点: {write_report['graph_nodes_added']} 个")
    print(f"  图谱新增关系: {write_report['graph_edges_added']} 条")
    print(f"  知识库新增事实: {write_report['kb_facts_added']} 条")

    if write_report["details"]:
        print(f"\n  [详情]")
        for detail in write_report["details"][:10]:
            print(f"    + {detail}")

    return write_report


def _iter_org_location_pairs(entities: Dict[str, Any]):
    orgs = entities.get("organizations", [])
    locs = [l["name"] for l in entities.get("locations", []) if l["type"] == "区县"]
    for org in orgs:
        org_district = org.get("district", "")
        if org_district and org_district in locs:
            yield org["name"], org_district


def run_demo_pipeline(
    complaint_text: str,
    district: str = "",
    graph_save_path: str | None = None,
    kb_save_path: str | None = None,
) -> Dict[str, Any]:
    """
    运行完整的知识图谱 Demo 流程

    Args:
        complaint_text: 市民留言文本
        district: 所在区县（可选）
        graph_save_path: 图谱保存路径（可选）
        kb_save_path: 知识库保存路径（可选）

    Returns:
        完整流程报告
    """
    print(f"\n{'#' * 60}")
    print(f"  知识图谱主流程 Demo")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    report: Dict[str, Any] = {
        "input": {
            "text": complaint_text[:100],
            "district": district,
        },
        "start_time": datetime.now().isoformat(),
    }

    entities = step1_extract_entities(complaint_text, district)
    report["entities"] = {
        "projects": len(entities.get("projects", [])),
        "locations": len(entities.get("locations", [])),
        "organizations": len(entities.get("organizations", [])),
        "policies": len(entities.get("policies", [])),
        "relationships": len(entities.get("relationships", [])),
    }

    graph_manager = KnowledgeGraphManager()
    kb = StructuredKnowledgeBase(kb_save_path)

    context = step2_query_graph(entities, graph_manager, district)
    report["graph_context"] = {
        "responsibility_chains": len(context.get("responsibility_chain", [])),
        "known_projects": len(context.get("known_projects", [])),
        "similar_cases": len(context.get("similar_cases", [])),
    }

    audit_results = step3_audit_facts(entities, context, kb)
    report["audit"] = {
        "approved": len(audit_results.get("approved", [])),
        "rejected": len(audit_results.get("rejected", [])),
        "needs_review": len(audit_results.get("needs_review", [])),
        "conflicts": len(audit_results.get("conflicts", [])),
    }

    write_report = step4_write_to_kb(audit_results, entities, graph_manager, kb)
    report["write"] = write_report

    if graph_save_path:
        graph_manager.graph.save_to_file(graph_save_path)
        print(f"\n[保存] 图谱已保存到: {graph_save_path}")

    kb.save()
    print(f"[保存] 知识库已保存")

    report["end_time"] = datetime.now().isoformat()

    _print_summary(report)

    graph_manager.close()

    return report


def _print_summary(report: Dict[str, Any]):
    _print_stage(5, "流程总结")
    print(f"\n  输入文本: {report['input']['text'][:60]}...")
    print(f"  区县: {report['input'].get('district') or '未指定'}")
    print(f"\n  提取实体: {sum(v for k, v in report['entities'].items() if isinstance(v, int))} 个")
    print(f"  图谱查询: {sum(report['graph_context'].values())} 条上下文")
    print(f"  审核通过: {report['audit']['approved']} 条")
    print(f"  冲突/待审: {report['audit']['conflicts']} 条")
    print(f"  入库节点: {report['write']['graph_nodes_added']} 个")
    print(f"  入库关系: {report['write']['graph_edges_added']} 条")
    print(f"  入库事实: {report['write']['kb_facts_added']} 条")
    print(f"\n{SEPARATOR}")
    print(f"  Demo 流程完成")
    print(f"{SEPARATOR}\n")


DEMO_CASES = [
    {
        "name": "老旧小区改造",
        "text": (
            "朝阳区望京街道花家地西里小区居民反映，小区内多栋楼外墙脱落严重，"
            "存在安全隐患。该小区建于1998年，属于老旧小区综合整治范围。"
            "朝阳区住建委已将该项目列入2026年改造计划，预计2026年6月开工，"
            "由朝阳区房管局负责实施，预算约3500万元。"
        ),
        "district": "朝阳区",
    },
    {
        "name": "道路积水投诉",
        "text": (
            "海淀区中关村南大街与四通桥交叉口，每逢大雨必积水，"
            "严重影响周边居民出行。海淀区水务局应当排查该路段排水设施，"
            "联系养护单位紧急修复。根据《北京市排水条例》相关规定，"
            "市水务局负责全市排水设施的监督管理工作。"
        ),
        "district": "海淀区",
    },
    {
        "name": "广场舞噪声扰民",
        "text": (
            "丰台区方庄街道芳古园小区居民投诉，每天晚上7点到9点，"
            "小区广场有人跳广场舞，音量过大，影响周围居民休息。"
            "丰台区公安分局已多次出警劝导，但效果不佳。"
            "建议协调社区居委会制定文明公约，限制活动时间和音量。"
        ),
        "district": "丰台区",
    },
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="知识图谱主流程 Demo")
    parser.add_argument(
        "--case", type=int, default=0, choices=[0, 1, 2],
        help="选择 demo 案例 (0=老旧小区改造, 1=道路积水, 2=广场舞噪声)",
    )
    parser.add_argument("--text", type=str, default="", help="自定义输入文本")
    parser.add_argument("--district", type=str, default="", help="区县")
    parser.add_argument("--graph-save", type=str, default="", help="图谱保存路径")
    parser.add_argument("--kb-save", type=str, default="", help="知识库保存路径")

    args = parser.parse_args()

    if args.text:
        text = args.text
        district = args.district
    else:
        case = DEMO_CASES[args.case]
        text = case["text"]
        district = case["district"]
        print(f"\n[选择案例] {case['name']}")

    report = run_demo_pipeline(
        complaint_text=text,
        district=district,
        graph_save_path=args.graph_save or None,
        kb_save_path=args.kb_save or None,
    )

    return report


if __name__ == "__main__":
    main()
