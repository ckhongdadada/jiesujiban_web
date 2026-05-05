"""
主价值链集成测试
模拟完整流程：留言输入 -> 事实提取 -> 图谱查询 -> 审核验证 -> 知识入库
"""

import unittest
from pathlib import Path

from tests import BaseTestCase
from src.jsjb.knowledge.entity_extractor import EntityExtractor
from src.jsjb.knowledge.graph import KnowledgeGraphManager
from src.jsjb.knowledge.query import GraphQueryEngine
from src.jsjb.knowledge.fusion import InformationFusion
from src.jsjb.knowledge.structured_kb import StructuredKnowledgeBase


class TestPipelineIntegration(BaseTestCase):
    """主价值链集成测试"""

    def setUp(self):
        self.extractor = EntityExtractor(use_ner=False)
        self.graph = KnowledgeGraphManager()
        self.fusion = InformationFusion()
        self.kb_path = self.test_data_dir / "integration_kb.json"
        self.kb = StructuredKnowledgeBase(str(self.kb_path))

    def tearDown(self):
        self.graph.close()

    def _run_pipeline(self, text: str, district: str = "") -> dict:
        doc = {"title": text[:50], "content": text, "district": district}
        entities = self.extractor.extract_from_document(doc)

        entity_map = {}
        for loc in entities.get("locations", [])[:5]:
            props = {k: v for k, v in loc.items() if k != "name"}
            nid = self.graph.create_location(loc["name"], **props)
            entity_map[("Location", loc["name"])] = nid
        for org in entities.get("organizations", [])[:5]:
            props = {k: v for k, v in org.items() if k != "name"}
            nid = self.graph.create_organization(org["name"], **props)
            entity_map[("Organization", org["name"])] = nid
        for proj in entities.get("projects", [])[:5]:
            props = {k: v for k, v in proj.items() if k != "name"}
            nid = self.graph.create_project(proj["name"], **props)
            entity_map[("Project", proj["name"])] = nid
        for policy in entities.get("policies", [])[:5]:
            props = {k: v for k, v in policy.items() if k != "title"}
            nid = self.graph.create_policy(policy["title"], **props)
            entity_map[("Policy", policy["title"])] = nid

        for rel in entities.get("relationships", []):
            from_key = (rel["from_type"], rel["from"])
            to_key = (rel["to_type"], rel["to"])
            if from_key in entity_map and to_key in entity_map:
                self.graph.create_relationship(
                    entity_map[from_key], entity_map[to_key], rel["rel_type"]
                )

        sources = {"extracted": []}
        for proj in entities.get("projects", []):
            sources["extracted"].append({
                "project": proj["name"],
                "status": proj.get("status", ""),
                "responsible_unit": "",
                "district": proj.get("district", district),
            })
        fusion_result = self.fusion.fuse_multi_source_info(sources)

        approved = []
        for fact_key, fact_info in fusion_result.get("fused_facts", {}).items():
            if fact_info.get("confidence", 0) >= 0.3:
                approved.append(fact_info)

        facts_written = 0
        for item in approved:
            name = item.get("entity_name", "")
            if name:
                success = self.kb.add_fact({
                    "type": "project_status",
                    "project": name,
                    "status": item.get("value", ""),
                    "district": item.get("district", district),
                    "source": "integration_test",
                    "confidence": item.get("confidence", 0.5),
                })
                if success:
                    facts_written += 1

        return {
            "entities": entities,
            "fusion_result": fusion_result,
            "approved_count": len(approved),
            "facts_written": facts_written,
            "graph_nodes": len(self.graph.graph.nodes),
            "graph_edges": len(self.graph.graph.edges),
        }

    def test_full_pipeline_old_community(self):
        text = (
            "朝阳区望京街道花家地西里小区居民反映，小区内多栋楼外墙脱落严重。"
            "朝阳区住建委已将该项目列入2026年改造计划，预算约3500万元。"
        )
        result = self._run_pipeline(text, "朝阳区")

        self.assertGreater(len(result["entities"]["locations"]), 0)
        self.assertGreater(len(result["entities"]["organizations"]), 0)
        self.assertGreater(len(result["entities"]["projects"]), 0)
        self.assertGreater(result["graph_nodes"], 0)
        self.assertGreater(result["approved_count"], 0)

    def test_full_pipeline_road_water(self):
        text = (
            "海淀区中关村南大街与四通桥交叉口，每逢大雨必积水。"
            "海淀区水务局应当排查该路段排水设施。根据《北京市排水条例》相关规定处理。"
        )
        result = self._run_pipeline(text, "海淀区")

        self.assertTrue(len(result["entities"]["policies"]) > 0)
        self.assertTrue(any("排水条例" in p["title"] for p in result["entities"]["policies"]))

    def test_full_pipeline_noise_complaint(self):
        text = (
            "丰台区方庄街道芳古园小区居民投诉，每天晚上7点到9点，"
            "小区广场有人跳广场舞，音量过大。丰台区公安分局已多次出警劝导。"
        )
        result = self._run_pipeline(text, "丰台区")

        self.assertGreater(result["graph_nodes"], 0)

    def test_pipeline_end_to_end_persistence(self):
        text = "朝阳区住建委负责望京老旧小区改造工程，该项目位于朝阳区望京街道。"
        self._run_pipeline(text, "朝阳区")
        self.kb.save()

        kb2 = StructuredKnowledgeBase(str(self.kb_path))
        self.assertTrue(len(kb2.project_status) > 0)


class TestPipelineWithPreExistingGraph(BaseTestCase):
    """预置图谱下的集成测试"""

    def setUp(self):
        self.graph = KnowledgeGraphManager()
        self.kb_path = self.test_data_dir / "preexist_kb.json"
        self.kb = StructuredKnowledgeBase(str(self.kb_path))
        self._seed_graph()

    def tearDown(self):
        self.graph.close()

    def _seed_graph(self):
        self.gm = self.graph
        self.loc = self.gm.create_location("朝阳区", type="区县", district="朝阳区")
        self.org = self.gm.create_organization("朝阳区住建委", level="区级")
        self.parent_org = self.gm.create_organization("北京市住建委", level="市级")
        self.proj = self.gm.create_project("望京老旧小区改造", status="规划中", district="朝阳区")

        self.gm.add_located_in(self.proj, self.loc)
        self.gm.add_responsible_for(self.org, self.proj)
        self.gm.add_reports_to(self.org, self.parent_org)

    def test_query_responsibility_chain_with_graph(self):
        engine = GraphQueryEngine(self.graph)
        chain = engine.query_responsibility_chain("望京老旧小区改造")
        self.assertTrue(len(chain) >= 2)

    def test_query_project_info_with_graph(self):
        engine = GraphQueryEngine(self.graph)
        info = engine.query_project_info("望京老旧小区改造")
        self.assertNotIn("error", info)
        self.assertEqual(info["project"]["properties"]["status"], "规划中")

    def test_query_org_projects_with_graph(self):
        engine = GraphQueryEngine(self.graph)
        projects = engine.query_org_projects("朝阳区住建委")
        self.assertTrue(len(projects) >= 1)

    def test_audit_with_preexisting_facts(self):
        self.kb.add_fact({
            "type": "project_status",
            "project": "望京老旧小区改造",
            "status": "规划中",
            "district": "朝阳区",
        })

        fusion = InformationFusion()
        sources = {
            "knowledge_base": [
                {"project": "望京老旧小区改造", "status": "规划中"}
            ],
            "user_verified": [
                {"project": "望京老旧小区改造", "status": "进行中"}
            ]
        }
        result = fusion.fuse_multi_source_info(sources)
        fact = result["fused_facts"]["望京老旧小区改造_status"]
        self.assertTrue(fact["has_conflict"])


if __name__ == "__main__":
    unittest.main()
