"""
知识图谱构建与查询自动化测试
覆盖：InMemoryGraph、KnowledgeGraphManager、GraphQueryEngine
"""

import json
import os
import tempfile
import unittest

from src.jsjb.knowledge.graph import InMemoryGraph, KnowledgeGraphManager
from src.jsjb.knowledge.query import GraphQueryEngine


class TestInMemoryGraph(unittest.TestCase):
    """内存图谱基础操作测试"""

    def setUp(self):
        self.graph = InMemoryGraph()

    def test_add_node(self):
        node_id = self.graph.add_node("Project", {"name": "测试项目"})
        self.assertIn(node_id, self.graph.nodes)
        self.assertEqual(self.graph.nodes[node_id]["type"], "Project")
        self.assertEqual(self.graph.nodes[node_id]["properties"]["name"], "测试项目")

    def test_node_counter_auto_increment(self):
        id1 = self.graph.add_node("Project", {"name": "A"})
        id2 = self.graph.add_node("Project", {"name": "B"})
        self.assertNotEqual(id1, id2)

    def test_add_edge(self):
        n1 = self.graph.add_node("Organization", {"name": "住建委"})
        n2 = self.graph.add_node("Project", {"name": "道路工程"})
        self.graph.add_edge(n1, n2, "RESPONSIBLE_FOR")
        self.assertEqual(len(self.graph.edges), 1)
        self.assertEqual(self.graph.edges[0]["type"], "RESPONSIBLE_FOR")

    def test_find_node_by_type(self):
        self.graph.add_node("Project", {"name": "A", "district": "朝阳区"})
        self.graph.add_node("Project", {"name": "B", "district": "海淀区"})
        self.graph.add_node("Location", {"name": "朝阳区"})

        projects = self.graph.find_node("Project")
        self.assertEqual(len(projects), 2)

        chaoyang = self.graph.find_node("Project", district="朝阳区")
        self.assertEqual(len(chaoyang), 1)

    def test_get_node(self):
        node_id = self.graph.add_node("Project", {"name": "测试"})
        node = self.graph.get_node(node_id)
        self.assertIsNotNone(node)
        self.assertEqual(node["properties"]["name"], "测试")

    def test_get_node_not_found(self):
        self.assertIsNone(self.graph.get_node("nonexistent"))

    def test_get_neighbors(self):
        org = self.graph.add_node("Organization", {"name": "住建委"})
        proj = self.graph.add_node("Project", {"name": "道路"})
        self.graph.add_edge(org, proj, "RESPONSIBLE_FOR")
        self.graph.add_edge(org, proj, "LOCATED_IN")

        neighbors = self.graph.get_neighbors(org, "RESPONSIBLE_FOR")
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0][0], proj)

        all_neighbors = self.graph.get_neighbors(org)
        self.assertEqual(len(all_neighbors), 2)

    def test_save_and_load(self):
        self.graph.add_node("Project", {"name": "持久化项目"})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            self.graph.save_to_file(path)
            loaded = InMemoryGraph()
            loaded.load_from_file(path)
            self.assertEqual(len(loaded.nodes), 1)
            proj = list(loaded.nodes.values())[0]
            self.assertEqual(proj["properties"]["name"], "持久化项目")
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        graph = InMemoryGraph()
        graph.load_from_file("/nonexistent/path.json")
        self.assertEqual(len(graph.nodes), 0)


class TestKnowledgeGraphManager(unittest.TestCase):
    """图谱管理器测试"""

    def setUp(self):
        self.gm = KnowledgeGraphManager()

    def tearDown(self):
        self.gm.close()

    def test_create_project(self):
        nid = self.gm.create_project("道路改造", type="道路工程", status="进行中")
        self.assertIsNotNone(nid)
        node = self.gm.graph.get_node(nid)
        self.assertEqual(node["properties"]["name"], "道路改造")
        self.assertEqual(node["properties"]["status"], "进行中")

    def test_create_location(self):
        nid = self.gm.create_location("朝阳区", type="区县", district="朝阳区")
        self.assertIsNotNone(nid)
        node = self.gm.graph.get_node(nid)
        self.assertEqual(node["properties"]["name"], "朝阳区")

    def test_create_organization(self):
        nid = self.gm.create_organization("住建委", type="建设单位", level="区级")
        self.assertIsNotNone(nid)
        node = self.gm.graph.get_node(nid)
        self.assertEqual(node["properties"]["name"], "住建委")

    def test_create_policy(self):
        nid = self.gm.create_policy("排水条例", doc_number="BJ-2024-001")
        self.assertIsNotNone(nid)
        node = self.gm.graph.get_node(nid)
        self.assertEqual(node["properties"]["title"], "排水条例")

    def test_create_case(self):
        nid = self.gm.create_case("积水投诉案例", issue_type="道路", district="海淀区")
        self.assertIsNotNone(nid)

    def test_create_resource(self):
        nid = self.gm.create_resource("社区图书馆", type="library", status="运营中")
        self.assertIsNotNone(nid)

    def test_relationship_located_in(self):
        proj = self.gm.create_project("项目A")
        loc = self.gm.create_location("朝阳区")
        self.gm.add_located_in(proj, loc)

        found = self.gm.get_location(proj)
        self.assertIsNotNone(found)
        self.assertEqual(found["properties"]["name"], "朝阳区")

    def test_relationship_responsible_for(self):
        org = self.gm.create_organization("住建委")
        proj = self.gm.create_project("道路工程")
        self.gm.add_responsible_for(org, proj)

        found = self.gm.get_responsible_org(proj)
        self.assertIsNotNone(found)
        self.assertEqual(found["properties"]["name"], "住建委")

    def test_relationship_reports_to(self):
        sub = self.gm.create_organization("望京街道办")
        parent = self.gm.create_organization("朝阳区政府")
        self.gm.add_reports_to(sub, parent)

        found = self.gm.get_parent_org(sub)
        self.assertIsNotNone(found)
        self.assertEqual(found["properties"]["name"], "朝阳区政府")

    def test_relationship_based_on(self):
        proj = self.gm.create_project("排水工程")
        policy = self.gm.create_policy("排水条例")
        self.gm.add_based_on(proj, policy)

        policies = self.gm.get_policies(proj)
        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0]["properties"]["title"], "排水条例")

    def test_status_change(self):
        proj = self.gm.create_project("项目A", status="规划中")
        self.gm.add_status_change(proj, "规划中", "进行中", "2026-01-01")

        history = self.gm.get_status_history(proj)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["from"], "规划中")
        self.assertEqual(history[0]["to"], "进行中")

    def test_similar_cases(self):
        c1 = self.gm.create_case("案例A")
        c2 = self.gm.create_case("案例B")
        self.gm.add_similar_to(c1, c2, 0.85)

        similar = self.gm.get_similar_cases(c1)
        self.assertEqual(len(similar), 1)
        self.assertAlmostEqual(similar[0][1], 0.85)

    def test_find_by_name(self):
        self.gm.create_project("朝阳区道路改造")
        self.gm.create_project("海淀区排水工程")

        results = self.gm.find_by_name("Project", "朝阳区道路改造")
        self.assertEqual(len(results), 1)

        not_found = self.gm.find_by_name("Project", "不存在的项目")
        self.assertEqual(len(not_found), 0)

    def test_get_responsible_org_not_found(self):
        proj = self.gm.create_project("独立项目")
        result = self.gm.get_responsible_org(proj)
        self.assertIsNone(result)


class TestGraphQueryEngine(unittest.TestCase):
    """图查询引擎测试"""

    def setUp(self):
        self.gm = KnowledgeGraphManager()
        self._build_test_graph()
        self.engine = GraphQueryEngine(self.gm)

    def tearDown(self):
        self.gm.close()

    def _build_test_graph(self):
        loc = self.gm.create_location("朝阳区", type="区县", district="朝阳区")
        org = self.gm.create_organization("朝阳区住建委", type="建设单位", level="区级")
        parent_org = self.gm.create_organization("北京市住建委", type="建设单位", level="市级")

        proj = self.gm.create_project(
            "望京老旧小区改造",
            type="老旧小区",
            status="进行中",
            district="朝阳区",
        )

        self.gm.add_located_in(proj, loc)
        self.gm.add_responsible_for(org, proj)
        self.gm.add_reports_to(org, parent_org)
        self.gm.add_status_change(proj, "规划中", "进行中", "2026-01-15")

        policy = self.gm.create_policy("老旧小区整治办法", source="市住建委")
        self.gm.add_based_on(proj, policy)

        self.loc_id = loc
        self.org_id = org
        self.parent_org_id = parent_org
        self.proj_id = proj

    def test_query_project_info(self):
        info = self.engine.query_project_info("望京老旧小区改造")
        self.assertNotIn("error", info)
        self.assertIsNotNone(info["project"])
        self.assertIsNotNone(info["location"])
        self.assertIsNotNone(info["responsible_org"])
        self.assertEqual(len(info["status_history"]), 1)
        self.assertEqual(len(info["policies"]), 1)

    def test_query_project_info_not_found(self):
        info = self.engine.query_project_info("不存在的项目")
        self.assertIn("error", info)

    def test_query_responsibility_chain(self):
        chain = self.engine.query_responsibility_chain("望京老旧小区改造")
        self.assertTrue(len(chain) >= 2)
        org_names = [c.get("properties", {}).get("name", "") for c in chain]
        self.assertTrue(any("住建委" in n for n in org_names))

    def test_query_projects_by_district(self):
        results = self.engine.query_projects_by_district("朝阳区")
        self.assertTrue(len(results) >= 1)

    def test_query_projects_by_district_empty(self):
        results = self.engine.query_projects_by_district("不存在的区")
        self.assertEqual(len(results), 0)

    def test_query_org_projects(self):
        results = self.engine.query_org_projects("朝阳区住建委")
        self.assertTrue(len(results) >= 1)

    def test_multi_hop_query(self):
        results = self.engine.multi_hop_query("朝阳区住建委", ["RESPONSIBLE_FOR", "LOCATED_IN"])
        self.assertTrue(len(results) >= 1)

    def test_generate_summary_project(self):
        summary = self.engine.generate_summary("望京老旧小区改造")
        self.assertIn("望京老旧小区改造", summary)
        self.assertIn("进行中", summary)

    def test_generate_summary_not_found(self):
        summary = self.engine.generate_summary("不存在的实体")
        self.assertIn("未找到", summary)


if __name__ == "__main__":
    unittest.main()
