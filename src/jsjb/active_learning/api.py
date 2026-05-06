"""
主动学习API集成
提供Flask路由和中间件
"""

from __future__ import annotations

from flask import Blueprint, request, jsonify, g
from typing import Dict, Any
import time

from src.jsjb.active_learning.sample_collector import SampleCollector
from src.jsjb.active_learning.annotation import AnnotationManager
from src.jsjb.active_learning.automation import ActiveLearningAutomationService
from src.jsjb.active_learning.trainer import IncrementalTrainer


def create_active_learning_blueprint(
    sample_collector: SampleCollector,
    annotation_manager: AnnotationManager,
    incremental_trainer: IncrementalTrainer
) -> Blueprint:
    """创建主动学习蓝图"""
    
    bp = Blueprint('active_learning', __name__, url_prefix='/api/active-learning')
    automation_service = ActiveLearningAutomationService(sample_collector)
    
    @bp.route('/samples/pending', methods=['GET'])
    def get_pending_samples():
        """获取待标注样本"""
        limit = request.args.get('limit', 20, type=int)
        strategy = request.args.get('strategy', 'priority')
        
        try:
            samples = annotation_manager.get_next_batch(limit, strategy)
            formatted = [annotation_manager.format_for_annotation(s) for s in samples]
            
            return jsonify({
                "status": "ok",
                "samples": formatted,
                "count": len(formatted)
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @bp.route('/samples/annotate', methods=['POST'])
    def annotate_sample():
        """标注单个样本"""
        data = request.get_json()
        
        sample_id = data.get('sample_id')
        correct_unit = data.get('correct_unit')
        annotator = data.get('annotator', 'unknown')
        notes = data.get('notes', '')
        
        if not sample_id or not correct_unit:
            return jsonify({
                "status": "error",
                "message": "缺少必要参数"
            }), 400
        
        result = annotation_manager.submit_annotation(
            sample_id, correct_unit, annotator, notes
        )
        
        return jsonify(result)
    
    @bp.route('/samples/annotate-batch', methods=['POST'])
    def annotate_batch():
        """批量标注"""
        data = request.get_json()
        
        annotations = data.get('annotations', [])
        annotator = data.get('annotator', 'unknown')
        
        if not annotations:
            return jsonify({
                "status": "error",
                "message": "标注列表为空"
            }), 400
        
        result = annotation_manager.submit_batch_annotations(annotations, annotator)
        
        return jsonify({
            "status": "ok",
            **result
        })
    
    @bp.route('/progress', methods=['GET'])
    def get_progress():
        """获取标注进度"""
        progress = annotation_manager.get_annotation_progress()
        return jsonify({
            "status": "ok",
            **progress
        })
    
    @bp.route('/statistics', methods=['GET'])
    def get_statistics():
        """获取统计信息"""
        stats = sample_collector.get_statistics()
        return jsonify({
            "status": "ok",
            **stats
        })
    
    @bp.route('/quality-report', methods=['GET'])
    def get_quality_report():
        """获取标注质量报告"""
        report = annotation_manager.get_annotation_quality_report()
        return jsonify({
            "status": "ok",
            **report
        })
    
    @bp.route('/suggest-action', methods=['GET'])
    def suggest_action():
        """建议下一步操作"""
        suggestion = annotation_manager.suggest_next_action()
        return jsonify({
            "status": "ok",
            **suggestion
        })
    
    @bp.route('/review-plan', methods=['GET'])
    def get_review_plan():
        """Build a diversified annotation plan from pending samples."""
        limit = request.args.get('limit', 30, type=int)
        try:
            return jsonify(automation_service.build_review_plan(limit=limit))
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route('/training-package', methods=['POST'])
    def prepare_training_package():
        """Export annotated active-learning samples into a safe training package."""
        data = request.get_json() or {}
        result = automation_service.prepare_training_package(
            output_dir=data.get('output_dir'),
            min_samples=data.get('min_samples', 1),
            unused_only=data.get('unused_only', True),
        )
        status_code = 200 if result.status == "ok" else 400
        return jsonify(result.to_dict()), status_code

    @bp.route('/train', methods=['POST'])
    def trigger_training():
        """触发增量训练"""
        data = request.get_json() or {}
        
        epochs = data.get('epochs', 3)
        batch_size = data.get('batch_size', 16)
        learning_rate = data.get('learning_rate', 2e-5)
        dry_run = data.get('dry_run', False)
        
        # 检查是否有足够的样本
        if not incremental_trainer.should_trigger_training(threshold=20):
            return jsonify({
                "status": "error",
                "message": "标注样本不足，建议至少20条"
            }), 400
        
        # 执行训练
        result = incremental_trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dry_run=dry_run
        )
        
        return jsonify(result)
    
    @bp.route('/training-history', methods=['GET'])
    def get_training_history():
        """获取训练历史"""
        limit = request.args.get('limit', 10, type=int)
        history = incremental_trainer.get_training_history(limit)
        
        return jsonify({
            "status": "ok",
            "history": history
        })
    
    @bp.route('/export', methods=['POST'])
    def export_data():
        """导出训练数据"""
        data = request.get_json() or {}
        output_path = data.get('output_path', 'data/runtime/exported_training_data.jsonl')
        unused_only = data.get('unused_only', True)
        
        try:
            sample_collector.export_training_data(output_path, unused_only)
            return jsonify({
                "status": "ok",
                "message": "导出成功",
                "path": output_path
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    
    return bp


_sample_collector_state = {
    "collector": None,
    "threshold": 70.0,
    "collected_hashes": set(),
    "max_cache_size": 10000
}


def add_sample_collection_middleware(
    app,
    sample_collector: SampleCollector,
    confidence_threshold: float = 70.0,
):
    """
    添加样本收集中间件
    
    在每次预测后自动收集低置信度样本
    """
    
    _sample_collector_state["collector"] = sample_collector
    _sample_collector_state["threshold"] = confidence_threshold
    
    @app.before_request
    def before_analyze_request():
        if request.path == '/api/analyze' and request.method == 'POST':
            g._analyze_start_time = time.time()
    
    @app.after_request
    def collect_uncertain_samples(response):
        """收集不确定样本"""
        if request.path != '/api/analyze' or request.method != 'POST':
            return response
        
        if response.status_code != 200:
            return response
        
        collector = _sample_collector_state["collector"]
        if collector is None:
            return response
        
        try:
            response_data = response.get_json()
            if not response_data:
                return response
            
            request_data = request.get_json(silent=True)
            if not request_data:
                return response
            
            units = response_data.get('units', [])
            if not units:
                return response
            
            top_unit = units[0]
            confidence = top_unit.get('confidence', 1.0)
            threshold = _sample_collector_state["threshold"]
            
            if confidence <= 1 and threshold > 1:
                threshold = threshold / 100
            elif confidence > 1 and threshold <= 1:
                threshold = threshold * 100
            
            title = request_data.get('title', '')
            body = request_data.get('body', '')[:500]
            
            import hashlib
            collected = _sample_collector_state["collected_hashes"]
            prediction_probs = {u['unit']: u['confidence'] for u in units[:5]}

            def _maybe_collect(source: str, reason: str, risk_flags: Dict[str, Any] | None = None):
                sample_id = hashlib.md5(f"{title}{body}|{source}".encode()).hexdigest()[:16]
                if sample_id in collected:
                    return
                if len(collected) >= _sample_collector_state["max_cache_size"]:
                    collected.clear()
                collected.add(sample_id)
                collector.add_sample(
                    sample_id=sample_id,
                    tag=request_data.get('tag', ''),
                    title=title,
                    body=body,
                    district=response_data.get('location', {}).get('district', ''),
                    predicted_unit=top_unit['unit'],
                    confidence=confidence,
                    prediction_probs=prediction_probs,
                    sample_source=source,
                    trigger_reason=reason,
                    risk_flags=risk_flags,
                )

            if confidence < threshold:
                _maybe_collect(
                    "classifier_low_confidence",
                    f"top1_confidence={confidence:.4f} below threshold={threshold:.4f}",
                )

            verification = response_data.get("verification") or {}
            if response_data.get("needs_review"):
                _maybe_collect(
                    "generation_verification",
                    verification.get("summary", "generation verification flagged sample"),
                    verification,
                )
        
        except Exception as e:
            print(f"[样本收集] 收集样本失败: {e}")
        
        return response
    
    return app
