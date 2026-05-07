
from __future__ import annotations

import json

from flask import Flask

from src.jsjb.web.routes.evaluation import register_evaluation_routes


class FakeRag:
    def search(self, query, top_k=5, district=None, tag=None, unit=None):
        return [
            {
                "doc_id": "doc-1",
                "title": "\u5783\u573e\u6e05\u8fd0\u6848\u4f8b",
                "doc_type": "case",
                "district": district or "\u5927\u5174\u533a",
                "source": "test",
                "score": 0.88,
                "dense_score": 0.8,
                "sparse_score": 0.5,
                "feedback_boost": 0.05,
                "matched_terms": ["\u5783\u573e"],
                "child_chunk_id": "doc-1_0",
                "child_snippet": "\u5783\u573e\u6876\u7ad9\u6e05\u8fd0\u6848\u4f8b",
                "parent_context": "\u7236\u6587\u6863\u4e0a\u4e0b\u6587",
                "score_breakdown": {"final_score": 0.88, "dense_score": 0.8},
            }
        ]

    def describe(self):
        return {"active_backend": "hybrid", "chunk_count": 1}


def test_evaluation_latest_route_reads_report(tmp_path):
    latest = tmp_path / "eval_latest.json"
    latest.write_text(json.dumps({"status": "ok", "summary": {"sample_count": 1}}), encoding="utf-8")
    app = Flask(__name__)
    register_evaluation_routes(app, ensure_components=lambda: None, get_rag=lambda: FakeRag(), latest_path=str(latest))

    client = app.test_client()
    response = client.get("/api/evaluation/latest")

    assert response.status_code == 200
    assert response.get_json()["summary"]["sample_count"] == 1


def test_chunk_inspector_exposes_score_breakdown(tmp_path):
    app = Flask(__name__)
    register_evaluation_routes(app, ensure_components=lambda: None, get_rag=lambda: FakeRag(), latest_path=str(tmp_path / "missing.json"))

    client = app.test_client()
    response = client.get("/api/rag/chunk-inspector?query=%E5%9E%83%E5%9C%BE%E6%B8%85%E8%BF%90&district=%E5%A4%A7%E5%85%B4%E5%8C%BA")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["hits"][0]["child_chunk_id"] == "doc-1_0"
    assert payload["hits"][0]["score_breakdown"]["final_score"] == 0.88
