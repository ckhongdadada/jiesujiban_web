"""
知识图谱固定 Demo 流程

演示从一条留言中提取事实 -> 进入审核候选 -> 审核通过后入库的完整流程。
同时包含两个可视化解释示例。

用法:
    python -m scripts.demo_knowledge_graph
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.jsjb.knowledge.graph import KnowledgeGraphManager
from src.jsjb.knowledge.query import GraphQueryEngine


def step1_build_demo_graph() -> KnowledgeGraphManager:
    print("=" * 60)
    print("Step 1: 构建示例知识图谱")
    print("=" * 60)

    manager = KnowledgeGraphManager()

    chaoyang = manager.create_location("朝阳区", district="朝阳区", type="区县")
    wangjing = manager.create_location("望京街道", district="朝阳区", type="街道")
    manager.add_located_in(wangjing, chaoyang)

    huanwei = manager.create_organization("朝阳区环卫中心", type="区级单位", level="区级")
    chengguan = manager.create_organization("朝阳区城管执法局", type="区级单位", level="区级")
    manager.add_reports_to(huanwei, chengguan)

    project1 = manager.create_project(
        "望京垃圾清运项目",
        type="环卫",
        status="进行中",
        district="朝阳区",
        description="望京街道垃圾清运不及时整改项目",
    )
    manager.add_located_in(project1, wangjing)
    manager.add_responsible_for(huanwei, project1)

    project2 = manager.create_project(
        "望京停车秩序整治",
        type="交通",
        status="已完工",
        district="朝阳区",
        description="望京CBD区域占道停车整治",
    )
    manager.add_located_in(project2, wangjing)
    manager.add_responsible_for(chengguan, project2)

    policy1 = manager.create_policy(
        "北京市生活垃圾管理条例",
        doc_number="京政发〔2020〕15号",
        content="规范生活垃圾投放、收集、运输和处理",
    )
    manager.add_based_on(project1, policy1)

    case1 = manager.create_case(
        "望京西园四区垃圾堆积投诉",
        issue_type="垃圾清运",
        district="朝阳区",
        resolution="已由环卫中心增加清运频次",
    )
    manager.add_similar_to(case1, project1, similarity=0.85)

    resource1 = manager.create_resource(
        "望京环卫站",
        type="环卫设施",
        status="运营中",
        location="望京街道",
    )
    manager.add_has_resource(wangjing, resource1)

    print(f"  创建节点: {len(manager.graph.nodes)} 个")
    print(f"  创建关系: {len(manager.graph.edges)} 条")
    print()
    return manager


def step2_extract_facts_from_message():
    print("=" * 60)
    print("Step 2: 从留言中提取事实 (模拟)")
    print("=" * 60)

    sample_message = {
        "tag": "投诉",
        "title": "望京西园四区垃圾堆积如山",
        "body": "朝阳区望京西园四区已经一周没有清运垃圾了，严重影响居民生活。",
        "unit": "环卫中心",
        "district": "朝阳区",
    }

    extracted_facts = [
        {
            "fact_type": "project_status",
            "fact_content": {
                "project": "望京垃圾清运项目",
                "status": "问题未解决",
                "source_text": sample_message["body"],
            },
            "confidence": 0.82,
        },
        {
            "fact_type": "responsible_unit",
            "fact_content": {
                "project": "望京垃圾清运项目",
                "unit": "朝阳区环卫中心",
                "source_text": sample_message["body"],
            },
            "confidence": 0.90,
        },
    ]

    print(f"  留言标题: {sample_message['title']}")
    print(f"  提取事实数: {len(extracted_facts)}")
    for fact in extracted_facts:
        print(f"    - 类型: {fact['fact_type']}, 置信度: {fact['confidence']}")
    print()
    return extracted_facts


def step3_review_and_import(manager: KnowledgeGraphManager, facts: list[dict]):
    print("=" * 60)
    print("Step 3: 事实审核与入库")
    print("=" * 60)

    for fact in facts:
        fact_type = fact["fact_type"]
        content = fact["fact_content"]
        confidence = fact["confidence"]

        approved = confidence >= 0.75
        status = "通过" if approved else "待人工审核"
        print(f"  事实类型: {fact_type}")
        print(f"  审核结果: {status} (置信度: {confidence})")

        if approved:
            if fact_type == "project_status":
                project_name = content.get("project", "")
                projects = manager.find_by_name("Project", project_name)
                if projects:
                    print(f"    -> 更新项目状态: {project_name} -> {content.get('status')}")
                else:
                    new_project = manager.create_project(
                        project_name,
                        type="环卫",
                        status=content.get("status", ""),
                        district="朝阳区",
                        description=content.get("source_text", ""),
                    )
                    print(f"    -> 新建项目节点: {project_name}")

            elif fact_type == "responsible_unit":
                project_name = content.get("project", "")
                unit_name = content.get("unit", "")
                projects = manager.find_by_name("Project", project_name)
                orgs = manager.find_by_name("Organization", unit_name)
                if projects and orgs:
                    manager.add_responsible_for(orgs[0]["id"], projects[0]["id"])
                    print(f"    -> 建立责任关系: {unit_name} -> {project_name}")
    print()


def step4_visualization_example_1(manager: KnowledgeGraphManager):
    print("=" * 60)
    print("可视化解释示例 1: 项目全链路追溯")
    print("=" * 60)

    engine = GraphQueryEngine(manager)
    summary = engine.generate_summary("望京垃圾清运项目")
    print(summary)

    chain = engine.query_responsibility_chain("望京垃圾清运项目")
    if chain:
        print("【责任链】")
        for idx, org in enumerate(chain):
            props = org.get("properties", org)
            name = props.get("name", str(org))
            level = props.get("level", "")
            indent = "  " * idx
            print(f"{indent}└─ {name} (级别: {level})")
    print()


def step5_visualization_example_2(manager: KnowledgeGraphManager):
    print("=" * 60)
    print("可视化解释示例 2: 区域资源与项目概览")
    print("=" * 60)

    engine = GraphQueryEngine(manager)

    print("【朝阳区项目列表】")
    projects = engine.query_projects_by_district("朝阳区")
    for project in projects:
        props = project.get("properties", project)
        name = props.get("name", str(project))
        status = props.get("status", "")
        print(f"  - {name} (状态: {status})")

    print()
    print("【望京街道公共资源】")
    resources = engine.query_location_resources("望京街道")
    for resource in resources:
        props = resource.get("properties", resource)
        name = props.get("name", str(resource))
        rtype = props.get("type", "")
        rstatus = props.get("status", "")
        print(f"  - {name} (类型: {rtype}, 状态: {rstatus})")
    print()


def step6_save_graph(manager: KnowledgeGraphManager):
    print("=" * 60)
    print("Step 6: 保存知识图谱")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "runtime")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "knowledge_graph_demo.json")
    manager.save(output_path)
    print(f"  图谱已保存到: {output_path}")
    print(f"  节点数: {len(manager.graph.nodes)}")
    print(f"  关系数: {len(manager.graph.edges)}")
    print()


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         知识图谱 Demo — 从留言到知识入库                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    manager = step1_build_demo_graph()
    facts = step2_extract_facts_from_message()
    step3_review_and_import(manager, facts)
    step4_visualization_example_1(manager)
    step5_visualization_example_2(manager)
    step6_save_graph(manager)

    print("Demo 完成！")


if __name__ == "__main__":
    main()
