"""
?????????? - Flask ???
"""

from __future__ import annotations

import os
import json
import re
import threading
import time

import torch
from flask import Flask, request, jsonify, render_template
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
from enhancements.reply_error_extractor import ReplyErrorExtractor
from enhancements.enhanced_processors import (
    init_enhanced_processors,
    get_location_processor,
    get_classification_processor,
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
    
    # 閸掓繂顫愰崠鏍波閺嬪嫬瀵查弮銉ョ箶
    logger = StructuredLogger()
    setup_logging("INFO")
    
    # 閸掓繂顫愰崠鏍晩鐠囶垰顦╅悶鍡楁珤
    error_handler = ErrorHandler(logger)
    set_error_handler(error_handler)
    
    # 閸掓繂顫愰崠鏍у冀妫ｅ牊鏆熼幑顔肩氨
    logger.info("閸掓繂顫愰崠鏍暏閹村嘲寮芥＃鍫熸殶閹诡喖绨?..")
    feedback_db = get_feedback_database()
    reply_error_extractor = ReplyErrorExtractor()
    logger.info("?????????")
    
    logger.info("???????...")
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

    def init_optional_extensions():
        if config.enable_knowledge_graph and _components["knowledge_graph"] is None:
            graph_manager = KnowledgeGraphManager(
                neo4j_uri=config.neo4j_uri if config.enable_neo4j else None,
                neo4j_user=config.neo4j_user,
                neo4j_password=config.neo4j_password,
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
            logger.info(f"[knowledge_graph] loaded nodes: {graph_size}")

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
                "annotation": annotation_manager,
                "trainer": incremental_trainer,
            }
            app.register_blueprint(
                create_active_learning_blueprint(
                    sample_collector,
                    annotation_manager,
                    incremental_trainer,
                )
            )
            add_sample_collection_middleware(
                app,
                sample_collector,
                confidence_threshold=config.active_learning_confidence_threshold,
            )
            logger.info("[active_learning] enabled")

    def query_knowledge_graph(location: dict, unit: str) -> dict:
        kg = _components.get("knowledge_graph")
        if not kg:
            return {"enabled": False, "matches": []}

        manager = kg["manager"]
        query_engine = kg["query"]
        matches = []
        district = location.get("district") if location else ""

        if district:
            for item in query_engine.query_projects_by_district(district)[: config.knowledge_graph_top_k]:
                matches.append({"type": "project", "data": item})
            for item in manager.find_by_name("Location", district)[: config.knowledge_graph_top_k]:
                matches.append({"type": "location", "data": item})

        if unit:
            for item in query_engine.query_org_projects(unit)[: config.knowledge_graph_top_k]:
                matches.append({"type": "organization_project", "data": item})

        graph_size = None
        if not manager.use_neo4j:
            graph_size = len(manager.graph.nodes)
        return {"enabled": True, "loaded_nodes": graph_size, "matches": matches[: config.knowledge_graph_top_k]}

    def import_reviewed_fact_to_graph(candidate: dict) -> dict:
        kg = _components.get("knowledge_graph")
        if not kg:
            raise RuntimeError("knowledge graph is not enabled")

        manager = kg["manager"]
        fact_type = candidate.get("fact_type", "")
        content = candidate.get("fact_content", {}) or {}
        feedback = feedback_db.get_feedback_by_id(candidate.get("feedback_id"))
        district = (feedback or {}).get("district", "")

        import_result = {"fact_type": fact_type, "created_nodes": [], "created_relations": []}

        if fact_type == "project_status":
            project_name = content.get("project", "")
            if not project_name:
                raise ValueError("project_status fact missing project name")
            project_id = manager.merge_entity(
                "Project",
                project_name,
                {
                    "status": content.get("status", ""),
                    "district": district,
                    "description": content.get("source_text", ""),
                },
            )
            import_result["created_nodes"].append({"type": "Project", "id": project_id, "name": project_name})
            return import_result

        if fact_type == "demolition_status":
            project_name = content.get("project", "")
            if not project_name:
                raise ValueError("demolition_status fact missing project name")
            project_id = manager.merge_entity(
                "Project",
                project_name,
                {
                    "demolition_status": content.get("demolition_status", ""),
                    "district": district,
                    "description": content.get("source_text", ""),
                },
            )
            import_result["created_nodes"].append({"type": "Project", "id": project_id, "name": project_name})
            return import_result

        if fact_type == "responsible_unit":
            project_name = content.get("project", "")
            unit_name = content.get("unit", "")
            if not project_name or not unit_name:
                raise ValueError("responsible_unit fact missing project or unit")
            project_id = manager.merge_entity("Project", project_name, {"district": district})
            org_id = manager.merge_entity("Organization", unit_name, {"district": district})
            manager.create_relationship(org_id, project_id, "RESPONSIBLE_FOR")
            import_result["created_nodes"].extend(
                [
                    {"type": "Project", "id": project_id, "name": project_name},
                    {"type": "Organization", "id": org_id, "name": unit_name},
                ]
            )
            import_result["created_relations"].append({"from": unit_name, "to": project_name, "type": "RESPONSIBLE_FOR"})
            return import_result

        if fact_type == "public_resource":
            resource_name = content.get("name", "")
            if not resource_name:
                raise ValueError("public_resource fact missing resource name")
            resource_id = manager.merge_entity(
                "Resource",
                resource_name,
                {
                    "status": content.get("status", ""),
                    "resource_type": content.get("resource_type", ""),
                    "address": content.get("address", ""),
                    "phone": content.get("phone", ""),
                },
            )
            import_result["created_nodes"].append({"type": "Resource", "id": resource_id, "name": resource_name})
            if district:
                location_id = manager.merge_entity("Location", district, {"district": district, "type": "区县"})
                manager.create_relationship(location_id, resource_id, "HAS_RESOURCE")
                import_result["created_nodes"].append({"type": "Location", "id": location_id, "name": district})
                import_result["created_relations"].append({"from": district, "to": resource_name, "type": "HAS_RESOURCE"})
            return import_result

        raise ValueError(f"unsupported fact type: {fact_type}")

    init_optional_extensions()

    def init_components():
        nonlocal _enhanced_initialized
        with _lock:
            start_time = time.time()
            
            # 鐏忔繆鐦禒搴暕閸旂姾娴囧Ο鈥崇€烽懢宄板絿
            if _components["ner"] is None:
                _components["ner"] = model_manager.get_model("ner")
                if _components["ner"] is None:
                    logger.info("濮濓絽婀崝鐘烘祰閸︽澘鎮曠拠鍡楀焼鎼?..")
                    _components["ner"] = LocationNER(data_dir=os.path.join(BASE_DIR, "data"))
            
            if _components["rag"] is None:
                _components["rag"] = model_manager.get_model("rag")
                if _components["rag"] is None:
                    logger.info("濮濓絽婀崝鐘烘祰 RAG 濡偓缁便垻鍌ㄥ?..")
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
                    logger.info("濮濓絽婀拋鍓х枂閸掑棛琚潻鎰攽閺冭埖甯撮崣?..")
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

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        start_time = time.time()
        client_ip = request.remote_addr
        
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
                return jsonify(validation_error), 400
            
            tag = cleaned_data.get("tag", "")
            title = cleaned_data.get("title", "")
            body = cleaned_data.get("body", "")
            forced_unit = cleaned_data.get("_force_unit", "")

            if not body:
                return jsonify({"error": "閻ｆ瑨鈻堝锝嗘瀮娑撳秷鍏樻稉铏光敄"}), 400

            logger.info(f"???????? - IP: {client_ip}")
            
            init_components()

            # 閸︽澘鎮曠拠鍡楀焼
            location_start = time.time()
            location_processor = get_location_processor()
            if location_processor:
                location = location_processor.process(f"{title} {body}")
            else:
                location = _components["ner"].extract_district(f"{title} {body}")
            location_time = time.time() - location_start
            logger.log_retrieval(f"{title} {body}", location.get("district", ""), 0, location_time)

            # 閸楁洑缍呴崚鍡欒
            classification_start = time.time()
            classification_processor = get_classification_processor()
            if classification_processor:
                units = classification_processor.process(
                    tag,
                    title,
                    body,
                    district=location.get("district", ""),
                )
            else:
                units = _components["classifier"].predict(tag, title, body, district=location.get("district", ""))
            classification_time = time.time() - classification_start
            logger.info(f"?????????: {classification_time:.2f}?")
            
            primary_unit = forced_unit or (units[0]["unit"] if units else "閻╃鍙ч崡鏇氱秴")

            retrieval_start = time.time()
            docs = _components["rag"].retrieve(
                f"{title} {body}",
                district=location.get("district"),
                tag=tag,
                unit=primary_unit,
                top_k=config.retrieval_top_k,
            )
            retrieval_time = time.time() - retrieval_start
            logger.log_retrieval(f"{title} {body}", location.get("district", ""), len(docs), retrieval_time)
            knowledge_graph = query_knowledge_graph(location, primary_unit)

            # 閻㈢喐鍨氶崶鐐差槻
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
            logger.info(f"?????????: {generation_time:.2f}?")

            total_time = time.time() - start_time
            
            # 鐠佹澘缍嶇拠閿嬬湴鐎瑰本鍨?
            response_data = {
                "status": "ok",
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
                    "generation": round(generation_time, 3),
                },
            }
            
            logger.log_request(
                cleaned_data, 
                response_data, 
                total_time, 
                client_ip
            )

            return jsonify(response_data)

        except Exception as e:
            # 婢跺嫮鎮婇張顏嗙叀闁挎瑨顕?
            error_response = error_handler.handle_unexpected_error(e)
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

    @app.route("/api/feedback", methods=["POST"])
    def feedback():
        """??????????? SQLite?"""
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
                'processing_time': data.get('processing_time', 0)
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

            active_learning_state = _components.get("active_learning") or {}
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

            logger.info(f"[閸欏秹顩璢 瀹歌尪顔囪ぐ鏇炲冀妫ｅ牆鍩岄弫鐗堝祦鎼存搫绱癐D={feedback_id}, helpful={feedback_record['is_helpful']}")

            return jsonify({
                "status": "ok",
                "message": "?????",
                "feedback_id": feedback_id,
                "reply_error_analysis_id": reply_error_analysis_id,
                "reply_error_analysis": reply_error_analysis,
                "queued_fact_count": queued_fact_count,
                "active_learning_collected": active_learning_collected,
            })

        except Exception as e:
            logger.error(f"[??] ????: {e}", extra={"error": str(e)})
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/stats", methods=["GET"])
    def feedback_stats():
        """閼惧嘲褰囬崣宥夘洯缂佺喕顓搁弫鐗堝祦"""
        try:
            stats = feedback_db.get_statistics()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"[??] ??????: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/error-analysis/<int:feedback_id>", methods=["GET"])
    def feedback_error_analysis(feedback_id: int):
        """Return structured reply error analysis for one feedback record."""
        try:
            analysis = feedback_db.get_reply_error_analysis(feedback_id)
            if not analysis:
                return jsonify({"status": "not_found", "message": "analysis not found"}), 404
            return jsonify({"status": "ok", "analysis": analysis})
        except Exception as e:
            logger.error(f"[feedback_error_analysis] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/knowledge-graph/review-candidates", methods=["GET"])
    def knowledge_graph_review_candidates():
        """List queued fact candidates waiting for review before graph import."""
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
        """Approve or reject one queued fact candidate."""
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

    @app.route("/api/feedback/list", methods=["GET"])
    def feedback_list():
        """List feedback records."""
        try:
            limit = request.args.get('limit', 100, type=int)
            offset = request.args.get('offset', 0, type=int)
            
            records = feedback_db.get_feedback(limit=limit, offset=offset)
            return jsonify({"records": records, "count": len(records)})
        except Exception as e:
            logger.error(f"[??] ??????: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/search", methods=["GET"])
    def feedback_search():
        """Search feedback records."""
        try:
            keyword = request.args.get('keyword', '')
            tag = request.args.get('tag', '')
            district = request.args.get('district', '')
            feedback_type = request.args.get('feedback_type', '')
            is_helpful = request.args.get('is_helpful', type=lambda x: x.lower() == 'true')
            start_date = request.args.get('start_date', '')
            end_date = request.args.get('end_date', '')
            limit = request.args.get('limit', 100, type=int)
            offset = request.args.get('offset', 0, type=int)
            
            results = feedback_db.search_feedback(
                keyword=keyword if keyword else None,
                tag=tag if tag else None,
                district=district if district else None,
                feedback_type=feedback_type if feedback_type else None,
                is_helpful=is_helpful,
                start_date=start_date if start_date else None,
                end_date=end_date if end_date else None,
                limit=limit,
                offset=offset
            )
            
            return jsonify({"results": results, "count": len(results)})
        except Exception as e:
            logger.error(f"[??] ????: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    return app


if __name__ == "__main__":
    print("=" * 55)
    print("  接诉即办智能服务系统")
    print("=" * 55)

    config = load_runtime_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  运行设备: {device}")

    print("  正在检查模型产物...")
    gen_state = generator_status(
        config.generator_base_model,
        config.generator_lora_dir,
        config.generator_draft_model,
        enable_assisted_decoding=config.enable_assisted_decoding,
    )
    if gen_state["runtime_ready"]:
        print("  生成模型产物已就绪 ✓")
    else:
        print(f"  生成模型未就绪: {', '.join(gen_state['missing']) or '缺少必要文件'}")

    print(f"  访问地址: http://127.0.0.1:{config.enhanced_port}")
    print("=" * 55)

    app = create_app()
    app.run(host="0.0.0.0", port=config.enhanced_port, debug=False)

