
"""Evaluation and RAG observability routes."""

from __future__ import annotations

import json
from pathlib import Path

from flask import jsonify, request

from src.jsjb.core.paths import get_evaluation_outputs_dir
from src.jsjb.evaluation.rag_generation import extract_hit_observability


def register_evaluation_routes(app, *, ensure_components, get_rag, logger=None, latest_path: str | None = None):
    latest = Path(latest_path) if latest_path else get_evaluation_outputs_dir() / "eval_latest.json"

    @app.route("/api/evaluation/latest", methods=["GET"])
    def evaluation_latest():
        if not latest.exists():
            return jsonify({
                "status": "missing",
                "message": "No Phase 1 evaluation report has been generated yet.",
                "expected_path": str(latest),
            }), 404
        try:
            return jsonify(json.loads(latest.read_text(encoding="utf-8-sig")))
        except Exception as exc:
            if logger:
                logger.error(f"[evaluation_latest] {exc}")
            return jsonify({"status": "error", "message": str(exc), "path": str(latest)}), 500

    @app.route("/api/rag/chunk-inspector", methods=["GET", "POST"])
    def rag_chunk_inspector():
        try:
            payload = request.get_json(silent=True) or {}
            query = payload.get("query") or request.args.get("query") or ""
            district = payload.get("district") or request.args.get("district") or None
            tag = payload.get("tag") or request.args.get("tag") or None
            unit = payload.get("unit") or request.args.get("unit") or None
            top_k = int(payload.get("top_k") or request.args.get("top_k") or 5)
            if not query.strip():
                return jsonify({"status": "error", "message": "query is required"}), 400

            ensure_components()
            rag = get_rag()
            if rag is None:
                return jsonify({"status": "error", "message": "RAG component is not initialized"}), 503

            hits = rag.search(query, top_k=top_k, district=district, tag=tag, unit=unit)
            inspected = [extract_hit_observability(hit, rank=index + 1) for index, hit in enumerate(hits)]
            return jsonify({
                "status": "ok",
                "query": query,
                "district": district or "",
                "tag": tag or "",
                "unit": unit or "",
                "rag_status": rag.describe() if hasattr(rag, "describe") else {},
                "hit_count": len(inspected),
                "hits": inspected,
            })
        except Exception as exc:
            if logger:
                logger.error(f"[rag_chunk_inspector] {exc}")
            return jsonify({"status": "error", "message": str(exc)}), 500
