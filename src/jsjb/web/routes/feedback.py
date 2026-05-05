"""Feedback route registration."""

from __future__ import annotations

import time

from flask import jsonify, request


def register_feedback_routes(
    app,
    *,
    feedback_db,
    reply_error_extractor,
    get_active_learning_state,
    logger,
):
    """Register feedback and feedback-query routes."""

    @app.route("/api/feedback", methods=["POST"])
    def feedback():
        try:
            data = request.get_json(force=True)

            feedback_record = {
                "timestamp": data.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
                "tag": data.get("tag", ""),
                "title": data.get("title", ""),
                "body": data.get("body", ""),
                "reply": data.get("reply", ""),
                "unit": data.get("unit", ""),
                "district": data.get("district", ""),
                "is_helpful": data.get("is_helpful", False),
                "feedback_type": data.get("feedback_type", ""),
                "comments": data.get("comments", ""),
                "client_ip": request.remote_addr,
                "user_agent": request.headers.get("User-Agent", ""),
                "processing_time": data.get("processing_time", 0),
            }

            feedback_id = feedback_db.add_feedback(feedback_record)
            reply_error_analysis = None
            reply_error_analysis_id = None
            active_learning_collected = False

            reference_reply = data.get("reference_reply", "")
            retrieval_hits = data.get("retrieval", []) or []
            location_result = data.get("location", {}) or {
                "district": data.get("district", ""),
                "street": data.get("street", ""),
                "subdistrict": data.get("subdistrict", ""),
            }

            if (not feedback_record["is_helpful"]) and reference_reply.strip():
                reply_error_analysis = reply_error_extractor.analyze(
                    generated_reply=feedback_record["reply"],
                    reference_reply=reference_reply,
                    retrieval_hits=retrieval_hits,
                    location_result=location_result,
                )
                reply_error_analysis_id = feedback_db.save_reply_error_analysis(
                    feedback_id=feedback_id,
                    analysis=reply_error_analysis,
                    reference_reply=reference_reply,
                )
                queued_fact_count = feedback_db.queue_knowledge_graph_facts(
                    feedback_id=feedback_id,
                    analysis_id=reply_error_analysis_id,
                    analysis=reply_error_analysis,
                    source_reply_type="reference",
                )
            else:
                queued_fact_count = 0

            active_learning_state = get_active_learning_state() or {}
            sample_collector = active_learning_state.get("collector")
            if sample_collector is not None and not feedback_record["is_helpful"]:
                units = data.get("units", []) or []
                prediction_probs = {
                    unit.get("unit", ""): unit.get("confidence", 0)
                    for unit in units[:5]
                    if unit.get("unit")
                }
                top_confidence = units[0].get("confidence", 0.0) if units else 0.0
                trigger_reason = data.get("feedback_type", "") or data.get("comments", "")[:120]
                active_learning_collected = sample_collector.add_sample(
                    sample_id=f"feedback_{feedback_id}",
                    tag=feedback_record["tag"],
                    title=feedback_record["title"],
                    body=feedback_record["body"],
                    district=feedback_record["district"],
                    predicted_unit=feedback_record["unit"],
                    confidence=top_confidence,
                    prediction_probs=prediction_probs,
                    user_feedback=feedback_record["comments"] or feedback_record["feedback_type"],
                    sample_source="user_negative_feedback",
                    trigger_reason=trigger_reason or "negative user feedback",
                    risk_flags={
                        "feedback_type": data.get("feedback_type", ""),
                        "has_reference_reply": bool(reference_reply.strip()),
                        "reply_error_types": (reply_error_analysis or {}).get("error_types", []),
                        "reply_error_severity": (reply_error_analysis or {}).get("severity", ""),
                    },
                )

            logger.info(
                f"[feedback] stored record id={feedback_id}, helpful={feedback_record['is_helpful']}"
            )

            return jsonify(
                {
                    "status": "ok",
                    "message": "feedback saved",
                    "feedback_id": feedback_id,
                    "reply_error_analysis_id": reply_error_analysis_id,
                    "reply_error_analysis": reply_error_analysis,
                    "queued_fact_count": queued_fact_count,
                    "active_learning_collected": active_learning_collected,
                }
            )

        except Exception as e:
            logger.error(f"[feedback] save failed: {e}", extra={"error": str(e)})
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/stats", methods=["GET"])
    def feedback_stats():
        try:
            stats = feedback_db.get_statistics()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"[feedback_stats] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/error-analysis/<int:feedback_id>", methods=["GET"])
    def feedback_error_analysis(feedback_id: int):
        try:
            analysis = feedback_db.get_reply_error_analysis(feedback_id)
            if not analysis:
                return jsonify({"status": "not_found", "message": "analysis not found"}), 404
            return jsonify({"status": "ok", "analysis": analysis})
        except Exception as e:
            logger.error(f"[feedback_error_analysis] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/list", methods=["GET"])
    def feedback_list():
        try:
            limit = request.args.get("limit", 100, type=int)
            offset = request.args.get("offset", 0, type=int)

            records = feedback_db.get_feedback(limit=limit, offset=offset)
            return jsonify({"records": records, "count": len(records)})
        except Exception as e:
            logger.error(f"[feedback_list] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/search", methods=["GET"])
    def feedback_search():
        try:
            keyword = request.args.get("keyword", "")
            tag = request.args.get("tag", "")
            district = request.args.get("district", "")
            feedback_type = request.args.get("feedback_type", "")
            is_helpful = request.args.get("is_helpful", type=lambda x: x.lower() == "true")
            start_date = request.args.get("start_date", "")
            end_date = request.args.get("end_date", "")
            limit = request.args.get("limit", 100, type=int)
            offset = request.args.get("offset", 0, type=int)

            results = feedback_db.search_feedback(
                keyword=keyword if keyword else None,
                tag=tag if tag else None,
                district=district if district else None,
                feedback_type=feedback_type if feedback_type else None,
                is_helpful=is_helpful,
                start_date=start_date if start_date else None,
                end_date=end_date if end_date else None,
                limit=limit,
                offset=offset,
            )

            return jsonify({"results": results, "count": len(results)})
        except Exception as e:
            logger.error(f"[feedback_search] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/doc", methods=["POST"])
    def doc_feedback():
        try:
            data = request.get_json(force=True)
            doc_id = data.get("doc_id", "")
            is_helpful = data.get("is_helpful", True)
            query = data.get("query", "")

            if not doc_id:
                return jsonify({"status": "error", "message": "doc_id is required"}), 400

            feedback_db.record_doc_feedback(doc_id, is_helpful, query)
            return jsonify({"status": "ok", "doc_id": doc_id})
        except Exception as e:
            logger.error(f"[doc_feedback] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
