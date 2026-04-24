from __future__ import annotations

import os
import json
import threading
import time
import uuid
from os import path

import torch
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS

from enhancements.runtime_config import load_runtime_config
from enhancements.data_paths import (
    get_runtime_catalog_path,
    get_runtime_district_file,
    get_policy_corpus_path,
    get_policy_corpus_sample_path,
)
from enhancements.location_ner import LocationNER
from enhancements.rag_retriever_bge import RAGRetriever as BGERetriever
from enhancements.classifier_runtime import ClassifierRuntime, classifier_status
from enhancements.enhanced_generation import (
    generator_loaded,
    generator_status,
    load_generator,
    generate_simple_reply,
    generate_reply_with_context,
)
from enhancements.model_cache import model_manager, init_model_preloading
from enhancements.structured_logger import StructuredLogger, setup_logging
from enhancements.input_validation import (
    validate_and_clean_input,
    set_error_handler,
    ErrorHandler
)
from enhancements.feedback_db import get_feedback_database
from enhancements.enhanced_processors import (
    init_enhanced_processors,
    get_location_processor,
    get_classification_processor,
    get_batch_processor,
    request_tracker,
    metrics_collector,
    with_retry,
    ProcessingContext
)
from enhancements.active_learning.sample_collector import SampleCollector
from enhancements.active_learning.annotation_manager import AnnotationManager
from enhancements.active_learning.incremental_trainer import IncrementalTrainer
from enhancements.active_learning.api_integration import (
    create_active_learning_blueprint,
    add_sample_collection_middleware,
)
from enhancements.knowledge_graph.graph_manager import KnowledgeGraphManager
from enhancements.knowledge_graph.graph_query import GraphQueryEngine

RATE_LIMIT_LOCK = threading.Lock()
IP_REQUEST_TIMES = {}
RATE_LIMIT_SECONDS = 3.0


def create_app():
    app = Flask(__name__)
    CORS(app)
    
    config = load_runtime_config()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    logger = StructuredLogger()
    setup_logging("INFO")
    
    error_handler = ErrorHandler(logger)
    set_error_handler(error_handler)
    
    logger.info("鍒濆鍖栫敤鎴峰弽棣堟暟鎹簱...")
    feedback_db = get_feedback_database()
    logger.info("?????????")
    
    logger.info("寮€濮嬮鍔犺浇妯″瀷...")
    init_model_preloading(config.__dict__)
    
    _components = {
        "ner": None,
        "rag": None,
        "classifier": None,
        "knowledge_graph": None,
        "active_learning": None,
    }
    _lock = threading.Lock()
    _enhanced_initialized = False

    def init_optional_extensions():
        if config.enable_knowledge_graph and _components["knowledge_graph"] is None:
            neo4j_uri = config.neo4j_uri if config.enable_neo4j else None
            neo4j_user = config.neo4j_user if config.enable_neo4j else None
            neo4j_password = config.neo4j_password if config.enable_neo4j else None
            
            graph_manager = KnowledgeGraphManager(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password
            )
            if config.knowledge_graph_path and os.path.exists(config.knowledge_graph_path):
                graph_manager.load(config.knowledge_graph_path)
            _components["knowledge_graph"] = {
                "manager": graph_manager,
                "query": GraphQueryEngine(graph_manager),
            }
            graph_size = 0
            if not graph_manager.use_neo4j:
                graph_size = len(graph_manager.graph.nodes)
            logger.info(f"[knowledge_graph] loaded nodes: {graph_size}, neo4j: {graph_manager.use_neo4j}")

        if config.enable_active_learning and _components["active_learning"] is None:
            sample_collector = SampleCollector(db_path=config.active_learning_db_path)
            annotation_manager = AnnotationManager(sample_collector)
            incremental_trainer = IncrementalTrainer(
                sample_collector=sample_collector,
                base_data_path=config.active_learning_base_data_path,
                model_dir=config.classifier_model_dir,
            )
            _components["active_learning"] = {
                "collector": sample_collector,
                "annotation_manager": annotation_manager,
                "trainer": incremental_trainer,
            }
            al_blueprint = create_active_learning_blueprint(
                sample_collector,
                annotation_manager,
                incremental_trainer,
            )
            app.register_blueprint(al_blueprint)
            add_sample_collection_middleware(
                app,
                sample_collector,
                confidence_threshold=config.active_learning_confidence_threshold,
            )
            logger.info("[active_learning] enabled and blueprint registered")

    def query_knowledge_graph(location: dict, unit: str) -> dict:
        kg = _components.get("knowledge_graph")
        if not kg:
            return {"enabled": False, "matches": []}

        manager = kg["manager"]
        query_engine = kg["query"]
        matches = []
        district = location.get("district", "") if location else ""
        
        if district:
            for item in query_engine.query_projects_by_district(district)[: config.knowledge_graph_top_k]:
                matches.append({"type": "project", "data": item})
            for item in manager.find_by_name("Location", district)[: config.knowledge_graph_top_k]:
                matches.append({"type": "location", "data": item})

        if unit:
            for item in query_engine.query_org_projects(unit)[: config.knowledge_graph_top_k]:
                matches.append({"type": "organization_project", "data": item})

        graph_size = 0
        if not manager.use_neo4j:
            graph_size = len(manager.graph.nodes)
        
        return {
            "enabled": True,
            "matches": matches,
            "graph_size": graph_size
        }

    init_optional_extensions()

    def init_components():
        nonlocal _enhanced_initialized
        with _lock:
            start_time = time.time()
            
            if _components["ner"] is None:
                _components["ner"] = model_manager.get_model("ner")
                if _components["ner"] is None:
                    logger.info("姝ｅ湪鍔犺浇鍦板悕璇嗗埆搴?..")
                    _components["ner"] = LocationNER(data_dir=os.path.join(BASE_DIR, "data"))
            
            if _components["rag"] is None:
                _components["rag"] = model_manager.get_model("rag")
                if _components["rag"] is None:
                    logger.info("姝ｅ湪鍔犺浇 RAG 妫€绱㈢储寮?..")
                    _components["rag"] = BGERetriever(
                        data_dir=os.path.join(BASE_DIR, "data"),
                        backend=config.rag_backend,
                        enable_query_rewrite=config.rag_enable_query_rewrite,
                        multi_query_count=config.rag_multi_query_count,
                        dense_weight=config.rag_dense_weight,
                        sparse_weight=config.rag_sparse_weight,
                    )
            
            if _components["classifier"] is None:
                _components["classifier"] = model_manager.get_model("classifier")
                if _components["classifier"] is None:
                    logger.info("姝ｅ湪璁剧疆鍒嗙被杩愯鏃舵帴鍙?..")
                    _components["classifier"] = ClassifierRuntime(
                        model_dir=config.classifier_model_dir,
                        base_model_dir=config.classifier_base_model,
                        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    )
            
            if not _enhanced_initialized:
                init_enhanced_processors(
                    ner_component=_components["ner"],
                    classifier_component=_components["classifier"],
                    rag_component=_components["rag"],
                    generation_func=_create_generation_func(config),
                    enable_cache=config.enable_cache,
                    enable_disambiguation=config.enable_disambiguation,
                    cache_ttl=config.cache_ttl,
                    cache_local_size=config.cache_local_size,
                    enable_geo_validation=config.enable_geo_validation,
                    enable_context_disambiguation=config.enable_context_disambiguation,
                    enable_redis=config.enable_redis,
                    redis_host=config.redis_host,
                    redis_port=config.redis_port,
                    redis_db=config.redis_db,
                    redis_password=config.redis_password or None,
                    batch_max_size=config.batch_max_size,
                    batch_max_workers=config.batch_max_workers,
                )
                _enhanced_initialized = True
            
            load_time = time.time() - start_time
            logger.info(f"??????????: {load_time:.2f}?")

    def _create_generation_func(config):
        def generation_func(tag, title, body, unit, location_result, retrieval_hits):
            if retrieval_hits:
                return generate_reply_with_context(
                    tag=tag,
                    title=title,
                    body=body,
                    unit=unit,
                    location_result=location_result,
                    retrieval_hits=retrieval_hits,
                    base_model_path=config.generator_base_model,
                    lora_path=config.generator_lora_dir,
                    draft_model_path=config.generator_draft_model,
                    enable_verification=config.enable_fact_verification,
                    enable_assisted_decoding=config.enable_assisted_decoding,
                    max_new_tokens=config.generation_max_tokens,
                    temperature=config.generation_temperature,
                )
            else:
                return generate_simple_reply(
                    tag=tag,
                    title=title,
                    body=body,
                    unit=unit,
                    location_result=location_result,
                    base_model_path=config.generator_base_model,
                    lora_path=config.generator_lora_dir,
                    draft_model_path=config.generator_draft_model,
                    enable_assisted_decoding=config.enable_assisted_decoding,
                    max_new_tokens=config.generation_max_tokens,
                    temperature=config.generation_temperature,
                )
        return generation_func

    def build_health_snapshot() -> dict[str, object]:
        ner_ready = any(
            path.exists()
            for path in [
                get_runtime_district_file("beijing_districts_merged.json"),
                get_runtime_district_file("beijing_districts_curated.json"),
                get_runtime_district_file("beijing_districts_extra.json"),
                get_runtime_catalog_path(),
            ]
        )
        rag_ready = any(
            path.exists()
            for path in [
                get_policy_corpus_path(),
                get_policy_corpus_sample_path(),
            ]
        )
        classifier_info = classifier_status(
            model_dir=config.classifier_model_dir,
            base_model_dir=config.classifier_base_model,
        )
        generator_info = generator_status(
            base_model_path=config.generator_base_model,
            lora_path=config.generator_lora_dir,
            draft_model_path=config.generator_draft_model,
            enable_assisted_decoding=config.enable_assisted_decoding,
        )
        return {
            "ner_ready": ner_ready,
            "rag_ready": rag_ready,
            "classifier": classifier_info,
            "generator": generator_info,
            "classifier_loaded": _components["classifier"] is not None and _components["classifier"].model is not None,
            "generator_loaded": generator_loaded(),
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        }

    @app.before_request
    def before_request():
        g.request_id = request.headers.get('X-Request-ID', f"req_{uuid.uuid4().hex[:12]}")
        g.start_time = time.time()
        metrics_collector.increment('requests_total')
        metrics_collector.set_gauge('active_requests', 
            metrics_collector.gauges.get('active_requests', 0) + 1)
    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            metrics_collector.observe('request_duration', duration)
        
        metrics_collector.set_gauge('active_requests',
            max(0, metrics_collector.gauges.get('active_requests', 0) - 1))
        
        response.headers['X-Request-ID'] = getattr(g, 'request_id', '')
        return response

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/annotation")
    def annotation_page():
        return render_template("annotation.html")

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        start_time = time.time()
        client_ip = request.remote_addr
        request_id = getattr(g, 'request_id', '')
        
        with RATE_LIMIT_LOCK:
            curr_time = time.time()
            if curr_time - IP_REQUEST_TIMES.get(client_ip, 0) < RATE_LIMIT_SECONDS:
                error_response = error_handler.handle_rate_limit(client_ip)
                return jsonify(error_response), 429
            IP_REQUEST_TIMES[client_ip] = curr_time

        try:
            data = request.get_json(force=True)
            cleaned_data, validation_error = validate_and_clean_input(data)
            
            if validation_error:
                metrics_collector.increment('requests_error')
                return jsonify(validation_error), 400
            
            tag = cleaned_data.get("tag", "")
            title = cleaned_data.get("title", "")
            body = cleaned_data.get("body", "")
            forced_unit = cleaned_data.get("_force_unit", "")

            if not body:
                metrics_collector.increment('requests_error')
                return jsonify({"error": "鐣欒█姝ｆ枃涓嶈兘涓虹┖"}), 400

            logger.info(f"寮€濮嬪鐞嗗垎鏋愯姹?- ID: {request_id}, IP: {client_ip}")
            
            init_components()

            location_processor = get_location_processor()
            classification_processor = get_classification_processor()
            
            if location_processor and classification_processor:
                location_start = time.time()
                location = location_processor.process(f"{title} {body}")
                location_time = time.time() - location_start
                metrics_collector.observe('location_duration', location_time)
                
                classification_start = time.time()
                units = classification_processor.process(
                    tag, title, body,
                    district=location.get("district", "")
                )
                classification_time = time.time() - classification_start
                metrics_collector.observe('classification_duration', classification_time)
            else:
                location_start = time.time()
                location = _components["ner"].extract_district(f"{title} {body}")
                location_time = time.time() - location_start
                
                classification_start = time.time()
                units = _components["classifier"].predict(tag, title, body, district=location.get("district", ""))
                classification_time = time.time() - classification_start
            
            logger.info(f"?????????: {classification_time:.2f}?")
            
            primary_unit = forced_unit or (units[0]["unit"] if units else "鐩稿叧鍗曚綅")

            retrieval_start = time.time()
            docs = _components["rag"].retrieve(
                f"{title} {body}",
                district=location.get("district"),
                tag=tag,
                unit=primary_unit,
                top_k=config.retrieval_top_k,
            )
            retrieval_time = time.time() - retrieval_start
            metrics_collector.observe('retrieval_duration', retrieval_time)
            logger.log_retrieval(f"{title} {body}", location.get("district", ""), len(docs), retrieval_time)
            knowledge_graph = query_knowledge_graph(location, primary_unit)

            generation_start = time.time()
            if docs:
                gen_result = generate_reply_with_context(
                    tag=tag,
                    title=title,
                    body=body,
                    unit=primary_unit,
                    location_result=location,
                    retrieval_hits=docs,
                    base_model_path=config.generator_base_model,
                    lora_path=config.generator_lora_dir,
                    draft_model_path=config.generator_draft_model,
                    enable_verification=config.enable_fact_verification,
                    enable_assisted_decoding=config.enable_assisted_decoding,
                    max_new_tokens=config.generation_max_tokens,
                    temperature=config.generation_temperature,
                    return_dict=True,
                )
                reply = gen_result.get("reply", "")
                verification = gen_result.get("verification")
            else:
                reply = generate_simple_reply(
                    tag=tag,
                    title=title,
                    body=body,
                    unit=primary_unit,
                    location_result=location,
                    base_model_path=config.generator_base_model,
                    lora_path=config.generator_lora_dir,
                    draft_model_path=config.generator_draft_model,
                    enable_assisted_decoding=config.enable_assisted_decoding,
                    max_new_tokens=config.generation_max_tokens,
                    temperature=config.generation_temperature,
                )
                verification = None
            generation_time = time.time() - generation_start
            metrics_collector.observe('generation_duration', generation_time)
            logger.info(f"?????????: {generation_time:.2f}?")

            total_time = time.time() - start_time
            
            response_data = {
                "status": "ok",
                "request_id": request_id,
                "location": location,
                "units": units,
                "retrieval": docs[:3] if docs else [],
                "knowledge_graph": knowledge_graph,
                "reply": reply,
                "verification": verification,
                "needs_review": bool(verification and verification.get("needs_review")),
                "processing_time": {
                    "total": round(total_time, 3),
                    "location": round(location_time, 3),
                    "classification": round(classification_time, 3),
                    "retrieval": round(retrieval_time, 3),
                    "generation": round(generation_time, 3)
                }
            }
            
            metrics_collector.increment('requests_success')
            logger.log_request(cleaned_data, response_data, total_time, client_ip)

            return jsonify(response_data)

        except Exception as e:
            metrics_collector.increment('requests_error')
            error_response = error_handler.handle_unexpected_error(e)
            error_response['request_id'] = request_id
            return jsonify(error_response), 500

    @app.route("/api/batch", methods=["POST"])
    def batch_analyze():
        client_ip = request.remote_addr
        request_id = getattr(g, 'request_id', '')
        
        try:
            data = request.get_json(force=True)
            items = data.get('items', [])
            
            if not items:
                return jsonify({"error": "璇锋彁渚涜澶勭悊鐨勭暀瑷€鍒楄〃"}), 400
            
            if len(items) > config.batch_max_size:
                return jsonify({"error": "鍗曟鎵归噺澶勭悊鏈€澶氭敮鎸?0鏉＄暀瑷€"}), 400
            
            logger.info(f"寮€濮嬫壒閲忓鐞?- ID: {request_id}, 鏁伴噺: {len(items)}")
            
            init_components()
            
            batch_processor = get_batch_processor()
            if not batch_processor:
                return jsonify({"error": "?????????"}), 500
            
            start_time = time.time()
            results = batch_processor.process_batch(items, use_cache=True)
            total_time = time.time() - start_time
            
            success_count = sum(1 for r in results if r.get('status') == 'ok')
            
            response_data = {
                "status": "ok",
                "request_id": request_id,
                "total": len(items),
                "success": success_count,
                "failed": len(items) - success_count,
                "processing_time": round(total_time, 3),
                "results": results
            }
            
            logger.info(f"?????? - ??: {success_count}/{len(items)}, ??: {total_time:.2f}?")
            
            return jsonify(response_data)
            
        except Exception as e:
            metrics_collector.increment('requests_error')
            error_response = error_handler.handle_unexpected_error(e)
            error_response['request_id'] = request_id
            return jsonify(error_response), 500

    @app.route("/api/health", methods=["GET"])
    def health():
        snapshot = build_health_snapshot()
        return jsonify({
            "status": "ok",
            "classifier_ready": snapshot["classifier"]["compatible_runtime_ready"],
            "generator_ready": snapshot["generator"]["runtime_ready"],
            "classifier_loaded": snapshot["classifier_loaded"],
            "generator_loaded": snapshot["generator_loaded"],
            "ner_ready": snapshot["ner_ready"],
            "rag_ready": snapshot["rag_ready"],
            "device": snapshot["device"],
            "note": "??????????????????????????",
        })

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
            return jsonify({
                "status": "ready",
                "classifier_loaded": snapshot["classifier_loaded"],
                "generator_loaded": snapshot["generator_loaded"],
            })
        return jsonify({
            "status": "loading",
            "classifier": "ready" if cls_ready else "wait",
            "generator": "ready" if gen_ready else "wait",
            "rag": "ready" if rag_ready else "wait",
            "ner": "ready" if ner_ready else "wait",
        }), 503

    @app.route("/api/metrics", methods=["GET"])
    def metrics():
        return metrics_collector.get_prometheus_metrics(), 200, {
            'Content-Type': 'text/plain; charset=utf-8'
        }

    @app.route("/api/metrics/json", methods=["GET"])
    def metrics_json():
        return jsonify(metrics_collector.get_stats())

    @app.route("/api/stats", methods=["GET"])
    def stats():
        location_processor = get_location_processor()
        classification_processor = get_classification_processor()
        batch_processor = get_batch_processor()
        
        stats_data = {
            "metrics": metrics_collector.get_stats(),
            "location_processor": location_processor.get_stats() if location_processor else None,
            "classification_processor": classification_processor.get_stats() if classification_processor else None,
            "batch_processor": batch_processor.get_stats() if batch_processor else None,
        }
        
        return jsonify(stats_data)

    @app.route("/api/feedback", methods=["POST"])
    def feedback():
        try:
            data = request.get_json(force=True)
            
            feedback_record = {
                'timestamp': data.get('timestamp', time.strftime("%Y-%m-%dT%H:%M:%S")),
                'tag': data.get('tag', ''),
                'title': data.get('title', ''),
                'body': data.get('body', ''),
                'reply': data.get('reply', ''),
                'unit': data.get('unit', ''),
                'district': data.get('district', ''),
                'is_helpful': data.get('is_helpful', False),
                'feedback_type': data.get('feedback_type', ''),
                'comments': data.get('comments', ''),
                'client_ip': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', ''),
                'processing_time': data.get('processing_time', 0),
                'request_id': data.get('request_id', '')
            }
            
            feedback_id = feedback_db.add_feedback(feedback_record)
            
            logger.info(f"[鍙嶉] 宸茶褰曞弽棣堝埌鏁版嵁搴擄細ID={feedback_id}, helpful={feedback_record['is_helpful']}")
            
            return jsonify({
                "status": "ok", 
                "message": "?????",
                "feedback_id": feedback_id
            })
            
        except Exception as e:
            logger.error(f"[??] ????: {e}", extra={"error": str(e)})
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/stats", methods=["GET"])
    def feedback_stats():
        try:
            stats = feedback_db.get_statistics()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"[??] ??????: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/list", methods=["GET"])
    def feedback_list():
        try:
            limit = request.args.get('limit', 100, type=int)
            offset = request.args.get('offset', 0, type=int)
            
            records = feedback_db.get_feedback(limit=limit, offset=offset)
            return jsonify({"records": records, "count": len(records)})
        except Exception as e:
            logger.error(f"[??] ??????: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/cache/clear", methods=["POST"])
    def clear_cache():
        try:
            location_processor = get_location_processor()
            classification_processor = get_classification_processor()
            
            cleared = {}
            
            if location_processor and location_processor.cache:
                location_processor.cache.clear()
                cleared['location'] = True
            
            if classification_processor and classification_processor.cache:
                classification_processor.cache.clear()
                cleared['classification'] = True
            
            logger.info(f"[缂撳瓨] 宸叉竻闄ょ紦瀛? {cleared}")
            
            return jsonify({
                "status": "ok",
                "message": "?????",
                "cleared": cleared
            })
        except Exception as e:
            logger.error(f"[??] ????: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/cache/stats", methods=["GET"])
    def cache_stats():
        try:
            location_processor = get_location_processor()
            classification_processor = get_classification_processor()
            
            stats = {}
            
            if location_processor and location_processor.cache:
                stats['location'] = location_processor.cache.get_stats()
            
            if classification_processor and classification_processor.cache:
                stats['classification'] = classification_processor.cache.get_stats()
            
            return jsonify(stats)
        except Exception as e:
            logger.error(f"[??] ??????: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    return app


if __name__ == "__main__":
    print("=" * 55)
    print("  ?????????? (???)")
    print("=" * 55)
    
    config = load_runtime_config()
    app = create_app()
    
    print(f"\n鏈嶅姟鍦板潃: http://{config.host}:{config.port}")
    print(f"API鏂囨。: http://{config.host}:{config.port}/api/health")
    print(f"鐩戞帶鎸囨爣: http://{config.host}:{config.port}/api/metrics")
    print("\n鎸?Ctrl+C 鍋滄鏈嶅姟\n")
    
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
        threaded=True
    )
