"""
图查询引擎
提供高级查询和推理功能
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from src.jsjb.knowledge.graph import KnowledgeGraphManager


class GraphQueryEngine:
    """图查询引擎"""
    
    def __init__(self, graph_manager: KnowledgeGraphManager):
        self.graph = graph_manager
    
    def query_project_info(self, project_name: str) -> Dict[str, Any]:
        """
        查询项目完整信息
        
        Returns:
            {
                "project": {...},
                "location": {...},
                "responsible_org": {...},
                "parent_org": {...},
                "policies": [...],
                "status_history": [...]
            }
        """
        # 查找项目
        projects = self.graph.find_by_name("Project", project_name)
        if not projects:
            return {"error": "项目不存在"}
        
        project = projects[0]
        project_id = project["id"]
        
        # 查询关联信息
        location = self.graph.get_location(project_id)
        responsible_org = self.graph.get_responsible_org(project_id)
        parent_org = None
        if responsible_org:
            parent_org = self.graph.get_parent_org(responsible_org["id"])
        
        policies = self.graph.get_policies(project_id)
        status_history = self.graph.get_status_history(project_id)
        
        return {
            "project": project,
            "location": location,
            "responsible_org": responsible_org,
            "parent_org": parent_org,
            "policies": policies,
            "status_history": status_history
        }
    
    def query_responsibility_chain(self, entity_name: str) -> List[Dict]:
        """
        查询责任链（从基层到上级）
        
        Returns:
            [街道办, 区级单位, 市级单位]
        """
        # 查找实体
        entities = self.graph.find_by_name("Project", entity_name)
        if not entities:
            entities = self.graph.find_by_name("Location", entity_name)
        if not entities:
            return []
        
        entity_id = entities[0]["id"]
        
        # 查找负责单位
        org = self.graph.get_responsible_org(entity_id)
        if not org:
            return []
        
        # 沿着上级关系向上查询
        chain = [org]
        current_org = org
        max_depth = 5  # 防止循环
        
        for _ in range(max_depth):
            parent = self.graph.get_parent_org(current_org["id"])
            if not parent:
                break
            chain.append(parent)
            current_org = parent
        
        return chain
    
    def query_similar_cases(self, case_title: str, min_similarity: float = 0.7) -> List[Dict]:
        """查询相似案例"""
        cases = self.graph.find_by_name("Case", case_title)
        if not cases:
            return []
        
        case_id = cases[0]["id"]
        similar = self.graph.get_similar_cases(case_id, min_similarity)
        
        return [{"case": case, "similarity": sim} for case, sim in similar]
    
    def query_location_resources(self, location_name: str) -> List[Dict]:
        """查询地点的公共资源"""
        locations = self.graph.find_by_name("Location", location_name)
        if not locations:
            return []
        
        location_id = locations[0]["id"]
        
        # 查询包含的资源
        if self.graph.use_neo4j:
            with self.graph.driver.session() as session:
                query = """
                MATCH (l)-[:HAS_RESOURCE]->(r:Resource)
                WHERE id(l) = $location_id
                RETURN id(r) as id, r
                """
                result = session.run(query, location_id=int(location_id))
                return [{"id": str(record["id"]), **dict(record["r"])} for record in result]
        else:
            neighbors = self.graph.graph.get_neighbors(location_id, "HAS_RESOURCE")
            return [self.graph.graph.get_node(nid) for nid, _, _ in neighbors]
    
    def query_projects_by_district(self, district: str, status: str = None) -> List[Dict]:
        """查询某区的所有项目"""
        if self.graph.use_neo4j:
            with self.graph.driver.session() as session:
                where_clause = "p.district = $district"
                if status:
                    where_clause += " AND p.status = $status"
                
                query = f"""
                MATCH (p:Project)
                WHERE {where_clause}
                RETURN id(p) as id, p
                """
                result = session.run(query, district=district, status=status)
                return [{"id": str(record["id"]), **dict(record["p"])} for record in result]
        else:
            # 内存图谱查询
            projects = []
            for node_id, node_data in self.graph.graph.nodes.items():
                if node_data["type"] != "Project":
                    continue
                props = node_data["properties"]
                if props.get("district") == district:
                    if status is None or props.get("status") == status:
                        projects.append({"id": node_id, **node_data})
            return projects
    
    def query_org_projects(self, org_name: str) -> List[Dict]:
        """查询某单位负责的所有项目"""
        orgs = self.graph.find_by_name("Organization", org_name)
        if not orgs:
            return []
        
        org_id = orgs[0]["id"]
        
        if self.graph.use_neo4j:
            with self.graph.driver.session() as session:
                query = """
                MATCH (o)-[:RESPONSIBLE_FOR]->(p:Project)
                WHERE id(o) = $org_id
                RETURN id(p) as id, p
                """
                result = session.run(query, org_id=int(org_id))
                return [{"id": str(record["id"]), **dict(record["p"])} for record in result]
        else:
            neighbors = self.graph.graph.get_neighbors(org_id, "RESPONSIBLE_FOR")
            return [self.graph.graph.get_node(nid) for nid, _, _ in neighbors]
    
    def multi_hop_query(self, start_entity: str, path: List[str]) -> List[Dict]:
        """
        多跳查询
        
        Args:
            start_entity: 起始实体名称
            path: 关系路径，如 ["RESPONSIBLE_FOR", "REPORTS_TO"]
            
        Returns:
            查询结果列表
        """
        # 查找起始实体
        start_nodes = []
        for node_type in ["Project", "Organization", "Location"]:
            nodes = self.graph.find_by_name(node_type, start_entity)
            if nodes:
                start_nodes = nodes
                break
        
        if not start_nodes:
            return []
        
        current_ids = [start_nodes[0]["id"]]
        
        # 沿着路径查询
        for rel_type in path:
            next_ids = []
            for node_id in current_ids:
                if self.graph.use_neo4j:
                    related = self.graph._neo4j_get_all_related(node_id, rel_type)
                else:
                    neighbors = self.graph.graph.get_neighbors(node_id, rel_type)
                    related = [self.graph.graph.get_node(nid) for nid, _, _ in neighbors]
                
                next_ids.extend([r["id"] for r in related])
            
            current_ids = next_ids
            if not current_ids:
                return []
        
        # 获取最终节点的完整信息
        results = []
        for node_id in current_ids:
            if self.graph.use_neo4j:
                with self.graph.driver.session() as session:
                    query = "MATCH (n) WHERE id(n) = $node_id RETURN id(n) as id, n"
                    result = session.run(query, node_id=int(node_id))
                    record = result.single()
                    if record:
                        results.append({"id": str(record["id"]), **dict(record["n"])})
            else:
                node = self.graph.graph.get_node(node_id)
                if node:
                    results.append({"id": node_id, **node})
        
        return results
    
    def generate_summary(self, entity_name: str) -> str:
        """生成实体摘要"""
        # 尝试查找项目
        projects = self.graph.find_by_name("Project", entity_name)
        if projects:
            info = self.query_project_info(entity_name)
            return self._format_project_summary(info)
        
        # 尝试查找地点
        locations = self.graph.find_by_name("Location", entity_name)
        if locations:
            resources = self.query_location_resources(entity_name)
            return self._format_location_summary(locations[0], resources)
        
        return f"未找到实体：{entity_name}"
    
    def _format_project_summary(self, info: Dict) -> str:
        """格式化项目摘要"""
        if "error" in info:
            return info["error"]
        
        project = info["project"]["properties"]
        location = info.get("location", {}).get("properties", {})
        org = info.get("responsible_org", {}).get("properties", {})
        
        summary = f"【项目信息】{project.get('name', '')}\n"
        summary += f"类型：{project.get('type', '')}\n"
        summary += f"状态：{project.get('status', '')}\n"
        
        if location:
            summary += f"位置：{location.get('name', '')}\n"
        
        if org:
            summary += f"负责单位：{org.get('name', '')}\n"
            parent = info.get("parent_org", {}).get("properties", {})
            if parent:
                summary += f"上级单位：{parent.get('name', '')}\n"
        
        if info.get("status_history"):
            summary += "\n【状态变更历史】\n"
            for status in info["status_history"]:
                summary += f"{status.get('date', '')}: {status.get('from', '')} → {status.get('to', '')}\n"
        
        return summary
    
    def _format_location_summary(self, location: Dict, resources: List[Dict]) -> str:
        """格式化地点摘要"""
        props = location.get("properties", {})
        summary = f"【地点信息】{props.get('name', '')}\n"
        summary += f"类型：{props.get('type', '')}\n"
        summary += f"行政区：{props.get('district', '')}\n"
        
        if resources:
            summary += "\n【公共资源】\n"
            for resource in resources:
                r_props = resource.get("properties", {})
                summary += f"- {r_props.get('name', '')} ({r_props.get('type', '')})\n"
        
        return summary
