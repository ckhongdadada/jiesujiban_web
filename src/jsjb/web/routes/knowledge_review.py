"""Knowledge-graph review route registration."""

from __future__ import annotations

import json

from flask import jsonify, request


def register_knowledge_review_routes(
    app,
    *,
    feedback_db,
    import_reviewed_fact_to_graph,
    logger,
):
    """Register fact-review routes for knowledge-graph import."""

    @app.route("/api/knowledge-graph/review-candidates", methods=["GET"])
    def knowledge_graph_review_candidates():
        try:
            status = request.args.get("status", "pending")
            limit = request.args.get("limit", 50, type=int)
            candidates = feedback_db.list_knowledge_graph_fact_queue(status=status, limit=limit)
            return jsonify({"status": "ok", "candidates": candidates, "count": len(candidates)})
        except Exception as e:
            logger.error(f"[knowledge_graph_review_candidates] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/knowledge-graph/review/<int:candidate_id>", methods=["POST"])
    def knowledge_graph_review_candidate(candidate_id: int):
        try:
            payload = request.get_json(force=True) or {}
            action = payload.get("action", "").strip().lower()
            reviewer = payload.get("reviewer", "")
            review_notes = payload.get("review_notes", "")

            if action not in {"approve", "reject"}:
                return jsonify({"status": "error", "message": "action must be approve or reject"}), 400

            candidate = feedback_db.get_knowledge_graph_fact_candidate(candidate_id)
            if not candidate:
                return jsonify({"status": "not_found", "message": "candidate not found"}), 404

            imported_to_graph = False
            import_result = ""
            if action == "approve":
                import_result_payload = import_reviewed_fact_to_graph(candidate)
                imported_to_graph = True
                import_result = json.dumps(import_result_payload, ensure_ascii=False)

            feedback_db.review_knowledge_graph_fact_candidate(
                candidate_id=candidate_id,
                action=action,
                reviewer=reviewer,
                review_notes=review_notes,
                imported_to_graph=imported_to_graph,
                import_result=import_result,
            )
            return jsonify(
                {
                    "status": "ok",
                    "candidate_id": candidate_id,
                    "action": action,
                    "imported_to_graph": imported_to_graph,
                    "import_result": json.loads(import_result) if import_result else None,
                }
            )
        except Exception as e:
            logger.error(f"[knowledge_graph_review_candidate] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
