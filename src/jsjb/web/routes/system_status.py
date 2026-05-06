"""System observability routes for demos and operations."""

from __future__ import annotations

from flask import jsonify


def register_system_status_routes(app, *, build_system_status_snapshot):
    """Register observability endpoints that explain runtime readiness."""

    @app.route("/api/system/status", methods=["GET"])
    def system_status():
        return jsonify(build_system_status_snapshot())
