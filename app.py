"""
接诉即办智能服务系统 - Flask 后端
集成分类模型（RoBERTa）、生成模型（Qwen+LoRA）、地名识别、政策检索功能。
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
from enhancements.location_ner import LocationNER
from enhancements.rag_retriever_bge import RAGRetriever as BGERetriever
from enhancements.classifier_runtime import ClassifierRuntime
from enhancements.enhanced_generation import (
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

RATE_LIMIT_LOCK = threading.Lock()
IP_REQUEST_TIMES = {}
RATE_LIMIT_SECONDS = 3.0


def create_app():
    app = Flask(__name__)
    CORS(app)
    
    config = load_runtime_config()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化结构化日志
    logger = StructuredLogger()
    setup_logging("INFO")
    
    # 初始化错误处理器
    error_handler = ErrorHandler(logger)
    set_error_handler(error_handler)
    
    # 初始化反馈数据库
    logger.info("初始化用户反馈数据库...")
    feedback_db = get_feedback_database()
    logger.info("用户反馈数据库就绪")
    
    # 预加载模型（提升性能）
    logger.info("开始预加载模型...")
    init_model_preloading(config.__dict__)
    
    _components = {
        "ner": None,
        "rag": None,
        "classifier": None,
    }
    _lock = threading.Lock()

    def init_components():
        with _lock:
            start_time = time.time()
            
            # 尝试从预加载模型获取
            if _components["ner"] is None:
                _components["ner"] = model_manager.get_model("ner")
                if _components["ner"] is None:
                    logger.info("正在加载地名识别库...")
                    _components["ner"] = LocationNER(data_dir=os.path.join(BASE_DIR, "data"))
            
            if _components["rag"] is None:
                _components["rag"] = model_manager.get_model("rag")
                if _components["rag"] is None:
                    logger.info("正在加载 RAG 检索索引...")
                    _components["rag"] = BGERetriever(data_dir=os.path.join(BASE_DIR, "data"))
            
            if _components["classifier"] is None:
                _components["classifier"] = model_manager.get_model("classifier")
                if _components["classifier"] is None:
                    logger.info("正在设置分类运行时接口...")
                    _components["classifier"] = ClassifierRuntime(
                        model_dir=config.classifier_model_dir,
                        base_model_dir=config.classifier_base_model,
                        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    )
            
            load_time = time.time() - start_time
            logger.info(f"组件初始化完成，耗时: {load_time:.2f}秒")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        start_time = time.time()
        client_ip = request.remote_addr
        
        # 频率限制检查
        with RATE_LIMIT_LOCK:
            curr_time = time.time()
            if curr_time - IP_REQUEST_TIMES.get(client_ip, 0) < RATE_LIMIT_SECONDS:
                error_response = error_handler.handle_rate_limit(client_ip)
                return jsonify(error_response), 429
            IP_REQUEST_TIMES[client_ip] = curr_time

        try:
            # 输入验证和清理
            data = request.get_json(force=True)
            cleaned_data, validation_error = validate_and_clean_input(data)
            
            if validation_error:
                return jsonify(validation_error), 400
            
            tag = cleaned_data.get("tag", "")
            title = cleaned_data.get("title", "")
            body = cleaned_data.get("body", "")
            forced_unit = cleaned_data.get("_force_unit", "")

            if not body:
                return jsonify({"error": "留言正文不能为空"}), 400

            # 记录请求开始
            logger.info(f"开始处理分析请求 - IP: {client_ip}")
            
            init_components()

            # 地名识别
            location_start = time.time()
            location = _components["ner"].extract_district(f"{title} {body}")
            location_time = time.time() - location_start
            logger.log_retrieval(f"{title} {body}", location.get("district", ""), 0, location_time)

            # RAG检索
            retrieval_start = time.time()
            docs = _components["rag"].retrieve(f"{title} {body}", district=location.get("district"))
            retrieval_time = time.time() - retrieval_start
            logger.log_retrieval(f"{title} {body}", location.get("district", ""), len(docs), retrieval_time)

            # 单位分类
            classification_start = time.time()
            units = _components["classifier"].predict(tag, title, body)
            classification_time = time.time() - classification_start
            logger.info(f"单位分类完成，耗时: {classification_time:.2f}秒")
            
            primary_unit = forced_unit or (units[0]["unit"] if units else "相关单位")

            # 生成回复
            generation_start = time.time()
            if docs:
                reply = generate_reply_with_context(
                    tag=tag,
                    title=title,
                    body=body,
                    unit=primary_unit,
                    location_result=location,
                    retrieval_hits=docs,
                    base_model_path=config.generator_base_model,
                    lora_path=config.generator_lora_dir,
                )
            else:
                reply = generate_simple_reply(
                    tag=tag,
                    title=title,
                    body=body,
                    unit=primary_unit,
                    location_result=location,  # ✅ 修复：传递地区信息
                    base_model_path=config.generator_base_model,
                    lora_path=config.generator_lora_dir,
                )

            generation_time = time.time() - generation_start
            logger.info(f"回复生成完成，耗时: {generation_time:.2f}秒")

            # 计算总处理时间
            total_time = time.time() - start_time
            
            # 记录请求完成
            response_data = {
                "status": "ok",
                "location": location,
                "units": units,
                "retrieval": docs[:3] if docs else [],
                "reply": reply,
            }
            
            logger.log_request(
                cleaned_data, 
                response_data, 
                total_time, 
                client_ip
            )

            return jsonify(response_data)

        except Exception as e:
            # 处理未知错误
            error_response = error_handler.handle_unexpected_error(e)
            return jsonify(error_response), 500

    @app.route("/api/health", methods=["GET"])
    def health():
        init_components()
        cls_ready = _components["classifier"] is not None and _components["classifier"].model is not None
        gen_ready = load_generator(config.generator_base_model, config.generator_lora_dir)
        device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return jsonify({
            "classifier_ready": cls_ready,
            "generator_ready": gen_ready,
            "device": device,
            "note": "模型在服务启动时预加载，服务可用即代表模型就绪",
        })

    @app.route("/api/health/live", methods=["GET"])
    def live():
        return jsonify({"status": "alive"})

    @app.route("/api/health/ready", methods=["GET"])
    def ready():
        init_components()
        cls_ready = _components["classifier"] is not None and _components["classifier"].model is not None
        gen_ready = load_generator(config.generator_base_model, config.generator_lora_dir)
        if cls_ready and gen_ready:
            return jsonify({"status": "ready"})
        return jsonify({
            "status": "loading",
            "classifier": "ready" if cls_ready else "wait",
            "generator": "ready" if gen_ready else "wait"
        }), 503

    @app.route("/api/feedback", methods=["POST"])
    def feedback():
        """收集用户反馈 - SQLite 持久化存储"""
        try:
            data = request.get_json(force=True)
            
            # 构建反馈记录
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
            
            # 存储到 SQLite 数据库
            feedback_id = feedback_db.add_feedback(feedback_record)
            
            logger.info(f"[反馈] 已记录反馈到数据库：ID={feedback_id}, helpful={feedback_record['is_helpful']}")
            
            return jsonify({
                "status": "ok", 
                "message": "反馈已记录",
                "feedback_id": feedback_id
            })
            
        except Exception as e:
            logger.error(f"[反馈] 记录失败：{e}", extra={"error": str(e)})
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/stats", methods=["GET"])
    def feedback_stats():
        """获取反馈统计数据"""
        try:
            stats = feedback_db.get_statistics()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"[反馈] 获取统计失败：{e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/list", methods=["GET"])
    def feedback_list():
        """获取反馈记录列表"""
        try:
            limit = request.args.get('limit', 100, type=int)
            offset = request.args.get('offset', 0, type=int)
            
            records = feedback_db.get_feedback(limit=limit, offset=offset)
            return jsonify({"records": records, "count": len(records)})
        except Exception as e:
            logger.error(f"[反馈] 获取列表失败：{e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/feedback/search", methods=["GET"])
    def feedback_search():
        """搜索反馈记录"""
        try:
            keyword = request.args.get('keyword', '')
            tag = request.args.get('tag', '')
            district = request.args.get('district', '')
            is_helpful = request.args.get('is_helpful', type=lambda x: x.lower() == 'true')
            start_date = request.args.get('start_date', '')
            end_date = request.args.get('end_date', '')
            limit = request.args.get('limit', 100, type=int)
            offset = request.args.get('offset', 0, type=int)
            
            results = feedback_db.search_feedback(
                keyword=keyword if keyword else None,
                tag=tag if tag else None,
                district=district if district else None,
                is_helpful=is_helpful,
                start_date=start_date if start_date else None,
                end_date=end_date if end_date else None,
                limit=limit,
                offset=offset
            )
            
            return jsonify({"results": results, "count": len(results)})
        except Exception as e:
            logger.error(f"[反馈] 搜索失败：{e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    return app


if __name__ == "__main__":
    print("=" * 55)
    print("  接诉即办智能服务系统")
    print("=" * 55)
    
    config = load_runtime_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  运行设备: {device}")
    
    print("  正在预加载模型...")
    cls_loaded = load_generator(config.generator_base_model, config.generator_lora_dir)
    
    if cls_loaded:
        print("  模型预加载完成 ✓")
    else:
        print("  警告: 部分模型加载失败，请检查模型文件")
    
    print(f"  访问地址: http://127.0.0.1:{config.enhanced_port}")
    print("=" * 55)
    
    app = create_app()
    app.run(host="0.0.0.0", port=config.enhanced_port, debug=False)
