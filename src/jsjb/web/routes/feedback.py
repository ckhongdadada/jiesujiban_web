"""Feedback route registration."""

from __future__ import annotations

import time

from flask import jsonify, request

from src.jsjb.feedback.automation import FeedbackAutomationRouter


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
            active_learning_state = get_active_learning_state() or {}
            router = FeedbackAutomationRouter(
                feedback_db=feedback_db,
                reply_error_extractor=reply_error_extractor,
                sample_collector=active_learning_state.get("collector"),
            )
            automation = router.run(
                feedback_id=feedback_id,
                feedback_record=feedback_record,
                request_data=data,
            )
            automation_payload = automation.to_dict()

            logger.info(
                f"[feedback] stored record id={feedback_id}, helpful={feedback_record['is_helpful']}"
            )

            return jsonify(
                {
                    "status": "ok",
                    "message": "feedback saved",
                    "feedback_id": feedback_id,
                    "reply_error_analysis_id": automation.reply_error_analysis_id,
                    "reply_error_analysis": automation.quality_attribution,
                    "queued_fact_count": automation.queued_fact_count,
                    "active_learning_collected": automation.active_learning_collected,
                    "doc_feedback_recorded": automation.doc_feedback_recorded,
                    "feedback_automation": automation_payload,
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
