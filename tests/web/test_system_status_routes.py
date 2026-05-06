from __future__ import annotations

from flask import Flask

from src.jsjb.web.routes.system_status import register_system_status_routes


def test_system_status_route_returns_snapshot():
    app = Flask(__name__)
    register_system_status_routes(
        app,
        build_system_status_snapshot=lambda: {
            "status": "ok",
            "classifier": {"ready": True},
            "rag": {"doc_count": 3},
        },
    )

    client = app.test_client()
    response = client.get("/api/system/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["classifier"]["ready"] is True
    assert payload["rag"]["doc_count"] == 3
