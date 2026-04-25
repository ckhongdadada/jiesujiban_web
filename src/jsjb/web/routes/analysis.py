"""Analysis-related route registration."""

from __future__ import annotations

from flask import jsonify, render_template


def register_analysis_routes(app, *, build_health_snapshot):
    """Register page and health-check routes."""

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health", methods=["GET"])
    def health():
        snapshot = build_health_snapshot()
        return jsonify(
            {
                "status": "ok",
                "classifier_ready": snapshot["classifier"]["compatible_runtime_ready"],
                "generator_ready": snapshot["generator"]["runtime_ready"],
                "classifier_loaded": snapshot["classifier_loaded"],
                "generator_loaded": snapshot["generator_loaded"],
                "ner_ready": snapshot["ner_ready"],
                "rag_ready": snapshot["rag_ready"],
                "device": snapshot["device"],
                "note": "health endpoint checks runtime artifacts and in-memory loading status",
            }
        )

    @app.route("/api/health/live", methods=["GET"])
    def live():
        return jsonify({"status": "alive"})

    @app.route("/api/health/ready", methods=["GET"])
    def ready():
        snapshot = build_health_snapshot()
        cls_ready = snapshot["classifier"]["compatible_runtime_ready"]
        gen_ready = snapshot["generator"]["runtime_ready"]
        rag_ready = snapshot["rag_ready"]
        ner_ready = snapshot["ner_ready"]
        if cls_ready and gen_ready and rag_ready and ner_ready:
            return jsonify(
                {
                    "status": "ready",
                    "classifier_loaded": snapshot["classifier_loaded"],
                    "generator_loaded": snapshot["generator_loaded"],
                }
            )
        return (
            jsonify(
                {
                    "status": "loading",
                    "classifier": "ready" if cls_ready else "wait",
                    "generator": "ready" if gen_ready else "wait",
                    "rag": "ready" if rag_ready else "wait",
                    "ner": "ready" if ner_ready else "wait",
                }
            ),
            503,
        )
