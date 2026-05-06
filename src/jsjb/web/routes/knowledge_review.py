"""Knowledge-graph review route registration."""

from __future__ import annotations

import json

from flask import jsonify, request

from src.jsjb.knowledge.demo_pipeline import run_demo_pipeline


DEMO_CASES = [
    {
        "name": "老旧小区改造",
        "text": (
            "朝阳区望京街道花家地西里小区居民反映，小区内多栋楼外墙脱落严重，"
            "存在安全隐患。该小区建于1998年，属于老旧小区综合整治范围。"
            "朝阳区住建委已将该项目列入2026年改造计划，预计2026年6月开工，"
            "由朝阳区房管局负责实施，预算约3500万元。"
        ),
        "district": "朝阳区",
    },
    {
        "name": "道路积水投诉",
        "text": (
            "海淀区中关村南大街与四通桥交叉口，每逢大雨必积水，"
            "严重影响周边居民出行。海淀区水务局应当排查该路段排水设施，"
            "联系养护单位紧急修复。根据《北京市排水条例》相关规定，"
            "市水务局负责全市排水设施的监督管理工作。"
        ),
        "district": "海淀区",
    },
    {
        "name": "广场舞噪声扰民",
        "text": (
            "丰台区方庄街道芳古园小区居民投诉，每天晚上7点到9点，"
            "小区广场有人跳广场舞，音量过大，影响周围居民休息。"
            "丰台区公安分局已多次出警劝导，但效果不佳。"
            "建议协调社区居委会制定文明公约，限制活动时间和音量。"
        ),
        "district": "丰台区",
    },
]


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
                imported_to_graph = bool(import_result_payload.get("graph_updated"))
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

    @app.route("/api/knowledge-graph/demo", methods=["POST"])
    def knowledge_graph_demo():
        try:
            payload = request.get_json(force=True) or {}
            text = payload.get("text", "").strip()
            district = payload.get("district", "").strip()
            case_idx = payload.get("case_index", -1)

            if not text and 0 <= case_idx < len(DEMO_CASES):
                case = DEMO_CASES[case_idx]
                text = case["text"]
                district = district or case["district"]

            if not text:
                return jsonify({"status": "error", "message": "请提供文本或选择预设案例"}), 400

            report = run_demo_pipeline(
                complaint_text=text,
                district=district,
            )

            serializable = _make_serializable(report)
            return jsonify({"status": "ok", "report": serializable})
        except Exception as e:
            logger.error(f"[knowledge_graph_demo] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/knowledge-graph/demo-cases", methods=["GET"])
    def knowledge_graph_demo_cases():
        cases = [
            {"index": i, "name": c["name"], "district": c["district"], "preview": c["text"][:60] + "..."}
            for i, c in enumerate(DEMO_CASES)
        ]
        return jsonify({"status": "ok", "cases": cases})


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)
