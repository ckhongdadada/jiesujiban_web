"""
知识图谱管理器
负责图谱的创建、更新、查询
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.jsjb.knowledge.schema import canonical_entity_type, canonical_relation_type, normalize_properties

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("[知识图谱] Neo4j驱动未安装，将使用内存图谱")


class InMemoryGraph:
    """内存图谱（Neo4j不可用时的备选方案）"""
    
    def __init__(self):
        self.nodes = {}  # {node_id: {type, properties}}
        self.edges = []  # [{from, to, type, properties}]
        self.node_counter = 0
        
    def add_node(self, node_type: str, properties: Dict) -> str:
        """添加节点"""
        node_type = canonical_entity_type(node_type)
        node_id = f"{node_type}_{self.node_counter}"
        self.node_counter += 1
        self.nodes[node_id] = {
            "type": node_type,
            "properties": properties,
            "created_at": datetime.now().isoformat()
        }
        return node_id
    
    def add_edge(self, from_id: str, to_id: str, edge_type: str, properties: Dict = None):
        """添加边"""
        edge_type = canonical_relation_type(edge_type)
        properties = properties or {}
        for edge in self.edges:
            if edge["from"] == from_id and edge["to"] == to_id and edge["type"] == edge_type:
                edge["properties"].update(properties)
                edge["updated_at"] = datetime.now().isoformat()
                return
        self.edges.append({
            "from": from_id,
            "to": to_id,
            "type": edge_type,
            "properties": properties,
            "created_at": datetime.now().isoformat()
        })
    
    def find_node(self, node_type: str, **filters) -> List[str]:
        """查找节点"""
        node_type = canonical_entity_type(node_type)
        results = []
        for node_id, node_data in self.nodes.items():
            if node_data["type"] != node_type:
                continue
            match = True
            for key, value in filters.items():
                if node_data["properties"].get(key) != value:
                    match = False
                    break
            if match:
                results.append(node_id)
        return results
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str, edge_type: str = None) -> List[Tuple[str, str, Dict]]:
        """获取邻居节点"""
        edge_type = canonical_relation_type(edge_type) if edge_type else None
        neighbors = []
        for edge in self.edges:
            if edge["from"] == node_id:
                if edge_type is None or edge["type"] == edge_type:
                    neighbors.append((edge["to"], edge["type"], edge["properties"]))
            elif edge["to"] == node_id:
                if edge_type is None or edge["type"] == edge_type:
                    props = dict(edge["properties"])
                    props["_direction"] = "reverse"
                    neighbors.append((edge["from"], edge["type"], props))
        return neighbors
    
    def save_to_file(self, filepath: str):
        """保存到文件"""
        data = {
            "nodes": self.nodes,
            "edges": self.edges,
            "node_counter": self.node_counter
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filepath: str):
        """从文件加载"""
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.nodes = data.get("nodes", {})
        self.edges = data.get("edges", [])
        self.node_counter = data.get("node_counter", 0)


class KnowledgeGraphManager:
    """知识图谱管理器"""
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        self.use_neo4j = NEO4J_AVAILABLE and neo4j_uri
        
        if self.use_neo4j:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            print(f"[知识图谱] 使用Neo4j: {neo4j_uri}")
        else:
            self.graph = InMemoryGraph()
            print("[知识图谱] 使用内存图谱")
    
    def close(self):
        """关闭连接"""
        if self.use_neo4j and hasattr(self, 'driver'):
            self.driver.close()
    
    # ==================== 节点操作 ====================
    
    def create_project(self, name: str, **properties) -> str:
        """创建项目节点"""
        props = {
            "name": name,
            "type": properties.get("type", ""),
            "status": properties.get("status", ""),
            "district": properties.get("district", ""),
            "start_date": properties.get("start_date", ""),
            "end_date": properties.get("end_date", ""),
            "budget": properties.get("budget", ""),
            "description": properties.get("description", "")
        }
        
        if self.use_neo4j:
            return self._neo4j_create_node("Project", props)
        else:
            return self.graph.add_node("Project", props)
    
    def create_location(self, name: str, **properties) -> str:
        """创建地点节点"""
        props = {
            "name": name,
            "type": properties.get("type", ""),
            "district": properties.get("district", ""),
            "street": properties.get("street", ""),
            "coordinates": properties.get("coordinates", "")
        }
        
        if self.use_neo4j:
            return self._neo4j_create_node("Location", props)
        else:
            return self.graph.add_node("Location", props)
    
    def create_organization(self, name: str, **properties) -> str:
        """创建单位节点"""
        props = {
            "name": name,
            "type": properties.get("type", ""),
            "level": properties.get("level", ""),
            "contact": properties.get("contact", ""),
            "responsibilities": properties.get("responsibilities", "")
        }
        
        if self.use_neo4j:
            return self._neo4j_create_node("Organization", props)
        else:
            return self.graph.add_node("Organization", props)
    
    def create_policy(self, title: str, **properties) -> str:
        """创建政策节点"""
        props = {
            "title": title,
            "doc_number": properties.get("doc_number", ""),
            "publish_date": properties.get("publish_date", ""),
            "effective_date": properties.get("effective_date", ""),
            "content": properties.get("content", ""),
            "source": properties.get("source", "")
        }
        
        if self.use_neo4j:
            return self._neo4j_create_node("Policy", props)
        else:
            return self.graph.add_node("Policy", props)
    
    def create_case(self, title: str, **properties) -> str:
        """创建案例节点"""
        props = {
            "title": title,
            "case_id": properties.get("case_id", ""),
            "date": properties.get("date", ""),
            "issue_type": properties.get("issue_type", ""),
            "resolution": properties.get("resolution", ""),
            "district": properties.get("district", "")
        }
        
        if self.use_neo4j:
            return self._neo4j_create_node("Case", props)
        else:
            return self.graph.add_node("Case", props)
    
    def create_resource(self, name: str, **properties) -> str:
        """创建资源节点"""
        props = {
            "name": name,
            "type": properties.get("type", ""),
            "status": properties.get("status", ""),
            "capacity": properties.get("capacity", ""),
            "location": properties.get("location", "")
        }
        
        if self.use_neo4j:
            return self._neo4j_create_node("Resource", props)
        else:
            return self.graph.add_node("Resource", props)
    
    # ==================== 关系操作 ====================
    
    def create_relationship(
        self, 
        from_id: str, 
        to_id: str, 
        rel_type: str, 
        properties: Dict = None
    ):
        """创建关系"""
        rel_type = canonical_relation_type(rel_type)
        if self.use_neo4j:
            self._neo4j_create_relationship(from_id, to_id, rel_type, properties or {})
        else:
            self.graph.add_edge(from_id, to_id, rel_type, properties)
    
    def add_located_in(self, entity_id: str, location_id: str):
        """添加"位于"关系"""
        self.create_relationship(entity_id, location_id, "LOCATED_IN")
    
    def add_responsible_for(self, org_id: str, entity_id: str):
        """添加"负责"关系"""
        self.create_relationship(org_id, entity_id, "RESPONSIBLE_FOR")
    
    def add_reports_to(self, sub_org_id: str, parent_org_id: str):
        """添加"上级单位"关系"""
        self.create_relationship(sub_org_id, parent_org_id, "REPORTS_TO")
    
    def add_based_on(self, entity_id: str, policy_id: str):
        """添加"依据政策"关系"""
        self.create_relationship(entity_id, policy_id, "BASED_ON")
    
    def add_status_change(self, project_id: str, from_status: str, to_status: str, date: str):
        """添加"状态变更"关系"""
        props = {"from": from_status, "to": to_status, "date": date}
        self.create_relationship(project_id, project_id, "STATUS_CHANGED", props)
    
    def add_similar_to(self, case_id1: str, case_id2: str, similarity: float):
        """添加"相似案例"关系"""
        props = {"similarity": similarity}
        self.create_relationship(case_id1, case_id2, "SIMILAR_TO", props)
    
    def add_has_resource(self, location_id: str, resource_id: str):
        """添加"包含资源"关系"""
        self.create_relationship(location_id, resource_id, "HAS_RESOURCE")
    
    # ==================== 查询操作 ====================
    
    def find_by_name(self, node_type: str, name: str) -> List[Dict]:
        """根据名称查找节点"""
        if self.use_neo4j:
            return self._neo4j_find_nodes(node_type, {"name": name})
        else:
            node_ids = self.graph.find_node(node_type, name=name)
            return [{"id": nid, **self.graph.get_node(nid)} for nid in node_ids]
    
    def get_responsible_org(self, entity_id: str) -> Optional[Dict]:
        """获取负责单位"""
        if self.use_neo4j:
            return self._neo4j_get_related(entity_id, "RESPONSIBLE_FOR", reverse=True)
        else:
            for edge in self.graph.edges:
                if edge["to"] == entity_id and edge["type"] == "RESPONSIBLE_FOR":
                    nid = edge["from"]
                    return {"id": nid, **self.graph.get_node(nid)}
            return None
    
    def get_parent_org(self, org_id: str) -> Optional[Dict]:
        """获取上级单位"""
        if self.use_neo4j:
            return self._neo4j_get_related(org_id, "REPORTS_TO")
        else:
            neighbors = self.graph.get_neighbors(org_id, "REPORTS_TO")
            if neighbors:
                nid = neighbors[0][0]
                return {"id": nid, **self.graph.get_node(nid)}
            return None
    
    def get_location(self, entity_id: str) -> Optional[Dict]:
        """获取位置"""
        if self.use_neo4j:
            return self._neo4j_get_related(entity_id, "LOCATED_IN")
        else:
            neighbors = self.graph.get_neighbors(entity_id, "LOCATED_IN")
            if neighbors:
                nid = neighbors[0][0]
                return {"id": nid, **self.graph.get_node(nid)}
            return None
    
    def get_policies(self, entity_id: str) -> List[Dict]:
        """获取相关政策"""
        if self.use_neo4j:
            return self._neo4j_get_all_related(entity_id, "BASED_ON")
        else:
            neighbors = self.graph.get_neighbors(entity_id, "BASED_ON")
            return [self.graph.get_node(nid) for nid, _, _ in neighbors]
    
    def get_status_history(self, project_id: str) -> List[Dict]:
        """获取状态变更历史"""
        if self.use_neo4j:
            return self._neo4j_get_status_history(project_id)
        else:
            history = []
            for edge in self.graph.edges:
                if edge["from"] == project_id and edge["type"] == "STATUS_CHANGED":
                    history.append(edge["properties"])
            return sorted(history, key=lambda x: x.get("date", ""))
    
    def get_similar_cases(self, case_id: str, min_similarity: float = 0.7) -> List[Tuple[Dict, float]]:
        """获取相似案例"""
        if self.use_neo4j:
            return self._neo4j_get_similar_cases(case_id, min_similarity)
        else:
            similar = []
            neighbors = self.graph.get_neighbors(case_id, "SIMILAR_TO")
            for nid, _, props in neighbors:
                similarity = props.get("similarity", 0)
                if similarity >= min_similarity:
                    similar.append((self.graph.get_node(nid), similarity))
            return sorted(similar, key=lambda x: x[1], reverse=True)
    
    # ==================== Neo4j实现 ====================
    
    def _neo4j_create_node(self, label: str, properties: Dict) -> str:
        """Neo4j创建节点"""
        with self.driver.session() as session:
            props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
            query = f"CREATE (n:{label} {{{props_str}}}) RETURN id(n) as node_id"
            result = session.run(query, **properties)
            return str(result.single()["node_id"])
    
    def _neo4j_create_relationship(self, from_id: str, to_id: str, rel_type: str, properties: Dict):
        """Neo4j创建关系"""
        with self.driver.session() as session:
            props_str = ", ".join([f"r.{k} = ${k}" for k in properties.keys()])
            set_clause = f"SET {props_str}" if properties else ""
            query = f"""
            MATCH (a), (b)
            WHERE id(a) = $from_id AND id(b) = $to_id
            CREATE (a)-[r:{rel_type}]->(b)
            {set_clause}
            RETURN r
            """
            session.run(query, from_id=int(from_id), to_id=int(to_id), **properties)
    
    def _neo4j_find_nodes(self, label: str, filters: Dict) -> List[Dict]:
        """Neo4j查找节点"""
        with self.driver.session() as session:
            where_clauses = [f"n.{k} = ${k}" for k in filters.keys()]
            where_str = " AND ".join(where_clauses) if where_clauses else "true"
            query = f"MATCH (n:{label}) WHERE {where_str} RETURN id(n) as id, n"
            result = session.run(query, **filters)
            return [{"id": str(record["id"]), **dict(record["n"])} for record in result]
    
    def _neo4j_get_related(self, node_id: str, rel_type: str, reverse: bool = False) -> Optional[Dict]:
        """Neo4j获取相关节点"""
        with self.driver.session() as session:
            if reverse:
                query = f"""
                MATCH (n)-[:{rel_type}]->(m)
                WHERE id(m) = $node_id
                RETURN id(n) as id, n
                LIMIT 1
                """
            else:
                query = f"""
                MATCH (n)-[:{rel_type}]->(m)
                WHERE id(n) = $node_id
                RETURN id(m) as id, m
                LIMIT 1
                """
            result = session.run(query, node_id=int(node_id))
            record = result.single()
            if record:
                return {"id": str(record["id"]), **dict(record["n"] if reverse else record["m"])}
            return None
    
    def _neo4j_get_all_related(self, node_id: str, rel_type: str) -> List[Dict]:
        """Neo4j获取所有相关节点"""
        with self.driver.session() as session:
            query = f"""
            MATCH (n)-[:{rel_type}]->(m)
            WHERE id(n) = $node_id
            RETURN id(m) as id, m
            """
            result = session.run(query, node_id=int(node_id))
            return [{"id": str(record["id"]), **dict(record["m"])} for record in result]
    
    def _neo4j_get_status_history(self, project_id: str) -> List[Dict]:
        """Neo4j获取状态历史"""
        with self.driver.session() as session:
            query = """
            MATCH (p)-[r:STATUS_CHANGED]->(p)
            WHERE id(p) = $project_id
            RETURN r.from as from_status, r.to as to_status, r.date as date
            ORDER BY r.date
            """
            result = session.run(query, project_id=int(project_id))
            return [dict(record) for record in result]
    
    def _neo4j_get_similar_cases(self, case_id: str, min_similarity: float) -> List[Tuple[Dict, float]]:
        """Neo4j获取相似案例"""
        with self.driver.session() as session:
            query = """
            MATCH (c1)-[r:SIMILAR_TO]->(c2)
            WHERE id(c1) = $case_id AND r.similarity >= $min_similarity
            RETURN id(c2) as id, c2, r.similarity as similarity
            ORDER BY r.similarity DESC
            """
            result = session.run(query, case_id=int(case_id), min_similarity=min_similarity)
            return [({"id": str(record["id"]), **dict(record["c2"])}, record["similarity"]) for record in result]
    
    # ==================== 持久化 ====================
    
    def save(self, filepath: str):
        """保存图谱（仅内存模式）"""
        if not self.use_neo4j:
            self.graph.save_to_file(filepath)
    
    def load(self, filepath: str):
        """加载图谱（仅内存模式）"""
        if not self.use_neo4j:
            self.graph.load_from_file(filepath)
    
    # ==================== 增量更新 ====================
    
    def import_from_ownthink(self, data_path: str, max_entities: int = None) -> Dict[str, int]:
        """
        从OwnThink数据导入知识图谱
        
        Args:
            data_path: OwnThink数据文件路径（JSONL格式）
            max_entities: 最大导入实体数
            
        Returns:
            导入统计信息
        """
        import json
        
        stats = {
            "entities_imported": 0,
            "relations_imported": 0,
            "skipped": 0
        }
        
        if not os.path.exists(data_path):
            print(f"[知识图谱] 数据文件不存在: {data_path}")
            return stats
        
        entity_name_to_id = {}
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_entities and stats["entities_imported"] >= max_entities:
                    break
                
                if not line.strip():
                    continue
                
                try:
                    entity_data = json.loads(line)
                except json.JSONDecodeError:
                    stats["skipped"] += 1
                    continue
                
                entity_name = entity_data.get("name", "")
                if not entity_name:
                    stats["skipped"] += 1
                    continue
                
                entity_type = entity_data.get("type", "Entity")
                properties = entity_data.get("properties", {})
                relations = entity_data.get("relations", [])
                
                if entity_name in entity_name_to_id:
                    entity_id = entity_name_to_id[entity_name]
                else:
                    entity_id = self._create_entity_with_type(entity_type, entity_name, properties)
                    entity_name_to_id[entity_name] = entity_id
                    stats["entities_imported"] += 1
                
                for rel in relations:
                    rel_type = rel.get("type", "RELATED_TO")
                    target_name = rel.get("value", "")
                    
                    if not target_name:
                        continue
                    
                    if target_name not in entity_name_to_id:
                        target_id = self._create_entity_with_type("Entity", target_name, {})
                        entity_name_to_id[target_name] = target_id
                        stats["entities_imported"] += 1
                    else:
                        target_id = entity_name_to_id[target_name]
                    
                    self.create_relationship(entity_id, target_id, rel_type)
                    stats["relations_imported"] += 1
        
        print(f"[知识图谱] 导入完成: {stats['entities_imported']} 实体, {stats['relations_imported']} 关系")
        return stats
    
    def _create_entity_with_type(self, entity_type: str, name: str, properties: Dict) -> str:
        """根据类型创建实体"""
        type_mapping = {
            "Location": self.create_location,
            "Organization": self.create_organization,
            "Project": self.create_project,
            "Policy": self.create_policy,
            "Case": self.create_case,
            "Resource": self.create_resource,
        }
        
        creator = type_mapping.get(entity_type, self._create_generic_entity)
        return creator(name, **properties)
    
    def _create_generic_entity(self, name: str, **properties) -> str:
        """创建通用实体"""
        props = {"name": name, **properties}
        
        if self.use_neo4j:
            return self._neo4j_create_node("Entity", props)
        else:
            return self.graph.add_node("Entity", props)
    
    def merge_entity(self, entity_type: str, name: str, properties: Dict = None) -> str:
        """
        合并实体（如果存在则更新，不存在则创建）
        
        Args:
            entity_type: 实体类型
            name: 实体名称
            properties: 实体属性
            
        Returns:
            实体ID
        """
        entity_type = canonical_entity_type(entity_type)
        properties = normalize_properties(entity_type, name, properties)
        
        if self.use_neo4j:
            return self._neo4j_merge_node(entity_type, properties)
        else:
            existing = self.graph.find_node(entity_type, name=name)
            if existing:
                node_id = existing[0]
                self.graph.nodes[node_id]["properties"].update(properties)
                return node_id
            else:
                return self.graph.add_node(entity_type, properties)

    def search_nodes(
        self,
        query: str,
        *,
        district: str = "",
        entity_types: List[str] | None = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Search graph nodes by lexical overlap for GraphRAG retrieval."""
        query = str(query or "")
        if not query.strip():
            return []
        types = [canonical_entity_type(t) for t in entity_types] if entity_types else []
        if self.use_neo4j:
            return self._neo4j_search_nodes(query, district=district, entity_types=types, limit=limit)

        query_terms = self._term_set(query)
        results: List[Dict[str, Any]] = []
        for node_id, node_data in self.graph.nodes.items():
            node_type = node_data.get("type", "")
            if types and node_type not in types:
                continue
            props = node_data.get("properties", {})
            node_district = str(props.get("district", ""))
            if district and node_district and node_district not in {district, "北京市", "全市"}:
                district_factor = 0.6
            else:
                district_factor = 1.0

            text = " ".join(str(v) for v in props.values() if v)
            terms = self._term_set(text)
            overlap = query_terms & terms
            name = str(props.get("name", ""))
            score = 0.0
            if name and (name in query or query in name):
                score += 0.75
            score += min(len(overlap) * 0.08, 0.4)
            if district and node_district == district:
                score += 0.12
            if score <= 0:
                continue
            results.append(
                {
                    "id": node_id,
                    "type": node_type,
                    "properties": props,
                    "score": round(score * district_factor, 4),
                    "matched_terms": sorted(overlap)[:12],
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def build_community_summary(
        self,
        query: str,
        *,
        district: str = "",
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Build a compact community summary around matched graph nodes."""
        seeds = self.search_nodes(query, district=district, limit=limit)
        if not seeds:
            return {"summary": "", "nodes": [], "relations": [], "score": 0.0}

        relations: List[Dict[str, Any]] = []
        seen_nodes = {item["id"] for item in seeds}
        if not self.use_neo4j:
            for item in seeds:
                for neighbor_id, rel_type, rel_props in self.graph.get_neighbors(item["id"]):
                    neighbor = self.graph.get_node(neighbor_id)
                    if not neighbor:
                        continue
                    seen_nodes.add(neighbor_id)
                    relations.append(
                        {
                            "from": item["id"],
                            "to": neighbor_id,
                            "type": rel_type,
                            "properties": rel_props,
                            "neighbor": neighbor,
                        }
                    )
                    if len(relations) >= limit * 3:
                        break

        node_lines = []
        for item in seeds:
            props = item["properties"]
            fields = []
            for key in ("status", "demolition_status", "district", "responsible_unit", "resource_type"):
                if props.get(key):
                    fields.append(f"{key}:{props[key]}")
            node_lines.append(f"{item['type']}:{props.get('name', '')}" + (f" ({' | '.join(fields)})" if fields else ""))

        rel_lines = []
        for rel in relations[: limit * 2]:
            neighbor_props = rel.get("neighbor", {}).get("properties", {})
            rel_lines.append(f"{rel['type']} -> {neighbor_props.get('name', rel['to'])}")

        summary_parts = []
        if node_lines:
            summary_parts.append("相关实体：" + "；".join(node_lines[:limit]))
        if rel_lines:
            summary_parts.append("关联关系：" + "；".join(rel_lines[: limit * 2]))
        return {
            "summary": "\n".join(summary_parts),
            "nodes": seeds,
            "relations": relations,
            "score": round(max(item["score"] for item in seeds), 4),
            "node_count": len(seen_nodes),
            "relation_count": len(relations),
        }

    @staticmethod
    def _term_set(text: str) -> set[str]:
        terms: set[str] = set()
        for token in re.findall(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fa5]{2,}", str(text or "")):
            token = token.strip()
            if not token:
                continue
            if re.fullmatch(r"[\u4e00-\u9fa5]{2,}", token):
                for n in (2, 3, 4):
                    if len(token) >= n:
                        for i in range(len(token) - n + 1):
                            terms.add(token[i : i + n])
            else:
                terms.add(token.lower())
        return terms

    def _neo4j_search_nodes(
        self,
        query: str,
        *,
        district: str = "",
        entity_types: List[str] | None = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        labels = entity_types or ["Project", "Organization", "Location", "Resource", "Fact"]
        results: List[Dict[str, Any]] = []
        for label in labels:
            with self.driver.session() as session:
                cypher = f"""
                MATCH (n:{label})
                WHERE any(value IN [n.name, n.description, n.source_text, n.district, n.responsible_unit]
                          WHERE value IS NOT NULL AND $query CONTAINS toString(value))
                   OR any(value IN [n.name, n.description, n.source_text, n.district, n.responsible_unit]
                          WHERE value IS NOT NULL AND toString(value) CONTAINS $query)
                RETURN id(n) as id, n
                LIMIT $limit
                """
                for record in session.run(cypher, query=query, limit=limit):
                    props = dict(record["n"])
                    node_district = props.get("district", "")
                    if district and node_district and node_district not in {district, "北京市", "全市"}:
                        score = 0.45
                    else:
                        score = 0.7
                    results.append({"id": str(record["id"]), "type": label, "properties": props, "score": score})
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]
    
    def _neo4j_merge_node(self, label: str, properties: Dict) -> str:
        """Neo4j合并节点"""
        with self.driver.session() as session:
            set_props = ", ".join([f"n.{k} = ${k}" for k in properties.keys() if k != "name"])
            query = f"""
            MERGE (n:{label} {{name: $name}})
            ON CREATE SET n.created_at = datetime()
            {f'ON MATCH SET {set_props}' if set_props else ''}
            RETURN id(n) as node_id
            """
            result = session.run(query, **properties)
            return str(result.single()["node_id"])
    
    def batch_create_entities(self, entities: List[Dict]) -> Dict[str, str]:
        """
        批量创建实体
        
        Args:
            entities: 实体列表，每项包含 {type, name, properties}
            
        Returns:
            {name: entity_id} 映射
        """
        name_to_id = {}
        
        for entity in entities:
            entity_type = entity.get("type", "Entity")
            name = entity.get("name", "")
            properties = entity.get("properties", {})
            
            if not name:
                continue
            
            entity_id = self.merge_entity(entity_type, name, properties)
            name_to_id[name] = entity_id
        
        return name_to_id
    
    def batch_create_relations(self, relations: List[Dict], name_to_id: Dict[str, str]):
        """
        批量创建关系
        
        Args:
            relations: 关系列表，每项包含 {from_name, to_name, type, properties}
            name_to_id: 实体名称到ID的映射
        """
        for rel in relations:
            from_name = rel.get("from_name")
            to_name = rel.get("to_name")
            rel_type = rel.get("type", "RELATED_TO")
            properties = rel.get("properties", {})
            
            from_id = name_to_id.get(from_name)
            to_id = name_to_id.get(to_name)
            
            if from_id and to_id:
                self.create_relationship(from_id, to_id, rel_type, properties)
    
    def clear(self):
        """清空图谱"""
        if self.use_neo4j:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        else:
            self.graph = InMemoryGraph()
        
        print("[知识图谱] 图谱已清空")
    
    def get_stats(self) -> Dict[str, int]:
        """获取图谱统计信息"""
        if self.use_neo4j:
            with self.driver.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                return {
                    "nodes": node_count,
                    "relations": rel_count,
                    "storage": "neo4j"
                }
        else:
            return {
                "nodes": len(self.graph.nodes),
                "relations": len(self.graph.edges),
                "storage": "memory"
            }
