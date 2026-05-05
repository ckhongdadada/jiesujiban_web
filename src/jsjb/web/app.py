"""Flask application factory for the 接诉即办 service."""

from __future__ import annotations

import os
import re
import threading
import time

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.jsjb.core.config import load_runtime_config
from src.jsjb.core.model_manifest import validate_model_manifest_file
from src.jsjb.core.paths import (
    get_runtime_catalog_path,
    get_runtime_district_file,
    get_policy_corpus_path,
    get_policy_corpus_sample_path,
)
from src.jsjb.location import LocationNER
from src.jsjb.retrieval import RAGRetriever as BGERetriever
from src.jsjb.unit_classifier import ClassifierRuntime, classifier_status
from src.jsjb.reply_generation.service import (
    generator_loaded,
    generator_status,
    load_generator,
    generate_simple_reply,
    generate_reply_with_context,
)
from src.jsjb.reply_generation.qwen_lora import _grounding_strength
from src.jsjb.core.model_registry import model_manager, init_model_preloading
from src.jsjb.core.logging import StructuredLogger, setup_logging
from src.jsjb.core.validation import (
    validate_and_clean_input, 
    set_error_handler, 
    ErrorHandler
)
from src.jsjb.web.routes import (
    register_analysis_routes,
    register_feedback_routes,
    register_knowledge_review_routes,
)
from src.jsjb.feedback import get_feedback_database
from src.jsjb.reply_generation.error_analysis import ReplyErrorExtractor
from src.jsjb.core.processors import (
    init_enhanced_processors,
    get_location_processor,
    get_classification_processor,
)
from src.jsjb.active_learning import SampleCollector, AnnotationManager, IncrementalTrainer
from src.jsjb.active_learning.api import (
    create_active_learning_blueprint,
    add_sample_collection_middleware,
)
from src.jsjb.knowledge import KnowledgeGraphManager, GraphQueryEngine

RATE_LIMIT_LOCK = threading.Lock()
IP_REQUEST_TIMES = {}
RATE_LIMIT_SECONDS = 3.0

from src.jsjb.core.paths import get_project_root
PROJECT_ROOT = str(get_project_root())


def create_app():
    app = Flask(__name__, template_folder=os.path.join(PROJECT_ROOT, "templates"))
    CORS(app)
    
    config = load_runtime_config()
    BASE_DIR = PROJECT_ROOT
    
    # 閸掓繂顫愰崠鏍波閺嬪嫬瀵查弮銉ョ箶
    logger = StructuredLogger()
    setup_logging("INFO")
    
    error_handler = ErrorHandler(logger)
    set_error_handler(error_handler)
    
    logger.info("初始化用户反馈数据库...")
    feedback_db = get_feedback_database()
    reply_error_extractor = ReplyErrorExtractor()
    logger.info("用户反馈数据库就绪")
    
    logger.info("开始预加载模型...")
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
                    logger.info("NER 模型未命中缓存，改为按需初始化。")
                    _components["ner"] = LocationNER(data_dir=os.path.join(BASE_DIR, "data"))
            
            if _components["rag"] is None:
                _components["rag"] = model_manager.get_model("rag")
                if _components["rag"] is None:
                    logger.info("RAG 检索器未命中缓存，改为按需初始化。")
                    _components["rag"] = BGERetriever(
                        data_dir=os.path.join(BASE_DIR, "data"),
                        backend=config.rag_backend,
                        enable_query_rewrite=config.rag_enable_query_rewrite,
                        multi_query_count=config.rag_multi_query_count,
                        dense_weight=config.rag_dense_weight,
                        sparse_weight=config.rag_sparse_weight,
                        enable_chunking=config.rag_enable_chunking,
                        chunk_size=config.rag_chunk_size,
                        chunk_overlap=config.rag_chunk_overlap,
                        enable_reranker=config.rag_enable_reranker,
                        enable_graph_augment=config.rag_enable_graph_augment,
                        enable_feedback_boost=config.rag_enable_feedback_boost,
                        enable_post_processing=config.rag_enable_post_processing,
                    )
            
            if _components["classifier"] is None:
                _components["classifier"] = model_manager.get_model("classifier")
                if _components["classifier"] is None:
                    logger.info("分类模型未命中缓存，改为按需初始化。")
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
            logger.info(f"组件初始化完成，耗时: {load_time:.2f}秒")

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
        manifest_info = validate_model_manifest_file(config, config.model_manifest_path)
        return {
            "ner_ready": ner_ready,
            "rag_ready": rag_ready,
            "classifier": classifier_info,
            "generator": generator_info,
            "model_manifest": manifest_info,
            "model_manifest_strict": config.enforce_model_manifest,
            "classifier_loaded": _components["classifier"] is not None and _components["classifier"].model is not None,
            "generator_loaded": generator_loaded(),
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        }

    register_analysis_routes(app, build_health_snapshot=build_health_snapshot)

    @app.route("/api/rag/reload", methods=["POST"])
    def rag_reload():
        try:
            rag = _components.get("rag")
            if rag is None:
                return jsonify({"status": "error", "message": "RAG component not initialized"}), 503
            result = rag.reload()
            logger.info(f"[rag_reload] {result}")
            return jsonify(result)
        except Exception as e:
            logger.error(f"[rag_reload] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

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
            debug_mode = cleaned_data.get("_debug", False)

            if not body:
                return jsonify({"error": "留言正文不能为空"}), 400

            logger.info(f"开始处理分析请求 - IP: {client_ip}")
            
            init_components()

            location_start = time.time()
            location_processor = get_location_processor()
            if location_processor:
                location = location_processor.process(f"{title} {body}")
            else:
                location = _components["ner"].extract_district(f"{title} {body}")
            location_time = time.time() - location_start
            logger.log_retrieval(f"{title} {body}", location.get("district", ""), 0, location_time)

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
            logger.info(f"单位分类完成，耗时: {classification_time:.2f}秒")
            
            primary_unit = forced_unit or (units[0]["unit"] if units else "相关部门")

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
            logger.info(f"回复生成完成，耗时: {generation_time:.2f}秒")

            total_time = time.time() - start_time
            
            grounding = _grounding_strength(docs, district=location.get("district")) if docs else "none"
            is_forced_unit = bool(forced_unit)
            reply_mode = "保守兜底生成" if grounding in {"none", "weak"} else "RAG增强生成"
            if is_forced_unit:
                reply_mode += " (用户指定单位)"

            if grounding in {"none", "weak"}:
                top_score = docs[0].get("score", 0.0) if docs else 0.0
                matched_terms = [d.get("title", "") for d in docs[:3]] if docs else []
                logger.log_weak_evidence(
                    query=f"{title} {body}",
                    district=location.get("district", ""),
                    grounding=grounding,
                    top_score=top_score,
                    matched_terms=matched_terms,
                )

            if verification and verification.get("needs_review"):
                logger.log_quality_anomaly("事实验证告警", {
                    "summary": f"回复可能包含未验证事实",
                    "title": title[:80],
                    "unit": primary_unit,
                })

            response_data = {
                "status": "ok",
                "location": location,
                "units": units,
                "unit_explanation": {
                    "classifier_route": config.classifier_route,
                    "top1_meaning": "模型预测最可能的承办单位",
                    "top3_meaning": "模型预测的前3个候选承办单位，按置信度降序",
                    "unit_source": "用户手动指定" if is_forced_unit else "模型预测",
                },
                "retrieval": docs[:3] if docs else [],
                "evidence_strength": grounding,
                "knowledge_graph": knowledge_graph,
                "reply": reply,
                "reply_mode": reply_mode,
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

            if debug_mode:
                response_data["debug"] = {
                    "retrieval_full": docs[:10] if docs else [],
                    "retrieval_count": len(docs),
                    "location_raw": location,
                    "units_full": units,
                    "grounding": grounding,
                    "reply_mode": reply_mode,
                    "classifier_route": config.classifier_route,
                    "retrieval_top_k": config.retrieval_top_k,
                    "generation_max_tokens": config.generation_max_tokens,
                    "generation_temperature": config.generation_temperature,
                    "enable_fact_verification": config.enable_fact_verification,
                }
            
            logger.log_request(
                cleaned_data, 
                response_data, 
                total_time, 
                client_ip
            )

            return jsonify(response_data)

        except Exception as e:
            error_response = error_handler.handle_unexpected_error(e)
            return jsonify(error_response), 500

    register_feedback_routes(
        app,
        feedback_db=feedback_db,
        reply_error_extractor=reply_error_extractor,
        get_active_learning_state=lambda: _components.get("active_learning") or {},
        logger=logger,
    )

    register_knowledge_review_routes(
        app,
        feedback_db=feedback_db,
        import_reviewed_fact_to_graph=import_reviewed_fact_to_graph,
        logger=logger,
    )

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

