"""
知识图谱构建脚本
从语料库构建知识图谱
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.jsjb.knowledge.graph import KnowledgeGraphManager
from src.jsjb.knowledge.entity_extractor import EntityExtractor
from src.jsjb.knowledge.query import GraphQueryEngine


def build_graph_from_corpus(corpus_path: str, graph_manager: KnowledgeGraphManager):
    """从语料库构建图谱"""
    
    extractor = EntityExtractor()
    
    print(f"[构建图谱] 读取语料库: {corpus_path}")
    
    doc_count = 0
    entity_count = 0
    relation_count = 0
    
    # 用于去重的实体映射
    entity_map = {}  # {(type, name): node_id}
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_count += 1
            
            if doc_count % 100 == 0:
                print(f"[构建图谱] 已处理 {doc_count} 篇文档...")
            
            # 提取实体
            entities = extractor.extract_from_document(doc)
            
            # 创建节点
            current_entities = {}
            
            # 创建项目节点
            for project in entities.get("projects", []):
                key = ("Project", project["name"])
                if key not in entity_map:
                    node_id = graph_manager.create_project(project["name"], **project)
                    entity_map[key] = node_id
                    entity_count += 1
                current_entities[project["name"]] = entity_map[key]
            
            # 创建地点节点
            for location in entities.get("locations", []):
                key = ("Location", location["name"])
                if key not in entity_map:
                    node_id = graph_manager.create_location(location["name"], **location)
                    entity_map[key] = node_id
                    entity_count += 1
                current_entities[location["name"]] = entity_map[key]
            
            # 创建单位节点
            for org in entities.get("organizations", []):
                key = ("Organization", org["name"])
                if key not in entity_map:
                    node_id = graph_manager.create_organization(org["name"], **org)
                    entity_map[key] = node_id
                    entity_count += 1
                current_entities[org["name"]] = entity_map[key]
            
            # 创建政策节点
            for policy in entities.get("policies", []):
                key = ("Policy", policy["title"])
                if key not in entity_map:
                    node_id = graph_manager.create_policy(policy["title"], **policy)
                    entity_map[key] = node_id
                    entity_count += 1
                current_entities[policy["title"]] = entity_map[key]
            
            # 提取并创建关系
            text = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('snippet', '')}"
            relationships = extractor.extract_relationships(text, entities)
            
            for rel in relationships:
                from_key = (rel["from_type"], rel["from"])
                to_key = (rel["to_type"], rel["to"])
                
                if from_key in entity_map and to_key in entity_map:
                    graph_manager.create_relationship(
                        entity_map[from_key],
                        entity_map[to_key],
                        rel["rel_type"]
                    )
                    relation_count += 1
    
    print(f"\n[构建图谱] 完成!")
    print(f"  - 处理文档: {doc_count} 篇")
    print(f"  - 创建实体: {entity_count} 个")
    print(f"  - 创建关系: {relation_count} 条")
    
    return entity_count, relation_count


def main():
    parser = argparse.ArgumentParser(description="构建知识图谱")
    parser.add_argument("--corpus", type=str, required=True, help="语料库路径")
    parser.add_argument("--neo4j-uri", type=str, help="Neo4j URI (可选)")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j用户名")
    parser.add_argument("--neo4j-password", type=str, help="Neo4j密码")
    parser.add_argument("--save-path", type=str, help="内存图谱保存路径")
    parser.add_argument("--query-test", action="store_true", help="构建后进行查询测试")
    
    args = parser.parse_args()
    
    # 初始化图谱管理器
    graph_manager = KnowledgeGraphManager(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )
    
    # 构建图谱
    entity_count, relation_count = build_graph_from_corpus(args.corpus, graph_manager)
    
    # 保存内存图谱
    if not graph_manager.use_neo4j and args.save_path:
        graph_manager.save(args.save_path)
        print(f"\n[构建图谱] 已保存到: {args.save_path}")
    
    # 查询测试
    if args.query_test:
        print("\n[查询测试] 开始测试...")
        query_engine = GraphQueryEngine(graph_manager)
        
        # 测试查询
        print("\n1. 查询项目信息:")
        projects = graph_manager.find_by_name("Project", "")
        if projects:
            project_name = projects[0]["properties"]["name"]
            info = query_engine.query_project_info(project_name)
            print(query_engine._format_project_summary(info))
        
        print("\n2. 查询地点资源:")
        locations = graph_manager.find_by_name("Location", "")
        if locations:
            location_name = locations[0]["properties"]["name"]
            resources = query_engine.query_location_resources(location_name)
            print(f"地点: {location_name}")
            print(f"资源数量: {len(resources)}")
    
    # 关闭连接
    graph_manager.close()
    
    print("\n[构建图谱] 全部完成!")


if __name__ == "__main__":
    main()
