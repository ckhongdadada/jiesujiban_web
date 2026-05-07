from __future__ import annotations

from flask import jsonify, request


def register_rag_active_learning_routes(app, *, feedback_db, logger):
    """Register RAG active-learning review routes."""

    @app.route("/api/rag/active-learning/candidates", methods=["GET"])
    def rag_active_learning_candidates():
        try:
            status = request.args.get("status", "pending_review")
            limit = request.args.get("limit", 50, type=int)
            candidates = feedback_db.list_rag_active_learning_candidates(status=status, limit=limit)
            return jsonify({"status": "ok", "candidates": candidates, "count": len(candidates)})
        except Exception as exc:
            logger.error(f"[rag_active_learning_candidates] {exc}")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/rag/active-learning/stats", methods=["GET"])
    def rag_active_learning_stats():
        try:
            return jsonify({"status": "ok", **feedback_db.get_rag_active_learning_stats()})
        except Exception as exc:
            logger.error(f"[rag_active_learning_stats] {exc}")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/rag/active-learning/review/<int:candidate_id>", methods=["POST"])
    def rag_active_learning_review(candidate_id: int):
        try:
            data = request.get_json(force=True) or {}
            action = data.get("action", "approve")
            ok = feedback_db.review_rag_active_learning_candidate(
                candidate_id=candidate_id,
                action=action,
                reviewer=data.get("reviewer", ""),
                review_label=data.get("review_label", ""),
                review_notes=data.get("review_notes", data.get("notes", "")),
            )
            if not ok:
                return jsonify({"status": "not_found", "message": "candidate not found"}), 404
            return jsonify({"status": "ok", "candidate_id": candidate_id, "action": action})
        except Exception as exc:
            logger.error(f"[rag_active_learning_review] {exc}")
            return jsonify({"status": "error", "message": str(exc)}), 500
