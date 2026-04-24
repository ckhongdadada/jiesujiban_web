"""
知识更新器
实现从用户反馈中自动更新知识库
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from enhancements.feedback_db import FeedbackDatabase, get_feedback_database
from enhancements.fact_extractor import FactExtractor, extract_facts_from_feedback
from enhancements.structured_kb import StructuredKnowledgeBase, get_knowledge_base, ProjectInfo, PublicResource, UnitMapping


class KnowledgeUpdater:
    """知识库自动更新器"""
    
    def __init__(
        self,
        feedback_db: Optional[FeedbackDatabase] = None,
        knowledge_base: Optional[StructuredKnowledgeBase] = None
    ):
        self.feedback_db = feedback_db or get_feedback_database()
        self.kb = knowledge_base or get_knowledge_base()
        self.fact_extractor = FactExtractor()
    
    def run_daily_update(self) -> Dict[str, Any]:
        """
        执行每日自动更新
        
        Returns:
            更新报告
        """
        report = {
            "update_time": datetime.now().isoformat(),
            "processed_count": 0,
            "error_count": 0,
            "correct_count": 0,
            "facts_extracted": 0,
            "kb_updated": False,
            "updates": []
        }
        
        feedbacks = self.feedback_db.get_recent_feedback(days=1)
        report["processed_count"] = len(feedbacks)
        
        for feedback in feedbacks:
            feedback_id = feedback.get("id")
            is_helpful = feedback.get("is_helpful")
            
            if is_helpful == 0:
                self._process_error(feedback, report)
                report["error_count"] += 1
            elif is_helpful == 1:
                self._process_correct(feedback, report)
                report["correct_count"] += 1
        
        self._update_rag_corpus()
        report["kb_updated"] = True
        
        return report
    
    def _process_error(self, feedback: Dict[str, Any], report: Dict[str, Any]) -> None:
        """处理错误样本"""
        feedback_id = feedback.get("id")
        
        analysis_result = self.feedback_db.analyze_feedback(feedback_id)
        
        if analysis_result.get("error_type") and analysis_result.get("error_type") != "未知错误":
            correct_facts = analysis_result.get("correct_facts", {})
            
            if correct_facts:
                for key, value in correct_facts.items():
                    self._add_correction_fact(feedback, key, value, report)
    
    def _process_correct(self, feedback: Dict[str, Any], report: Dict[str, Any]) -> None:
        """处理正确样本"""
        reply = feedback.get("reply", "")
        
        if not reply:
            return
        
        facts = self.fact_extractor.extract_facts_from_reply(
            reply=reply,
            feedback_data=feedback
        )
        
        for fact in facts:
            self._add_fact_to_kb(fact, report)
            report["facts_extracted"] += 1
    
    def _add_correction_fact(
        self,
        feedback: Dict[str, Any],
        fact_key: str,
        fact_value: str,
        report: Dict[str, Any]
    ) -> None:
        """添加修正事实"""
        feedback_id = feedback.get("id")
        
        self.feedback_db.add_extracted_fact(
            feedback_id=feedback_id,
            fact_type="correction",
            fact_content=json.dumps({fact_key: fact_value}, ensure_ascii=False),
            confidence=0.9,
            source="user_correction"
        )
        
        update_record = {
            "type": "correction",
            "key": fact_key,
            "value": fact_value,
            "feedback_id": feedback_id
        }
        report["updates"].append(update_record)
    
    def _add_fact_to_kb(self, fact: Any, report: Dict[str, Any]) -> None:
        """将事实添加到知识库"""
        try:
            fact_dict = self.fact_extractor.facts_to_kb_format([fact])[0]
            success = self.kb.add_fact(fact_dict)
            
            if success:
                update_record = {
                    "type": fact.fact_type,
                    "content": fact.content,
                    "success": True
                }
                report["updates"].append(update_record)
        except Exception as e:
            print(f"添加事实到知识库失败: {e}")
    
    def _update_rag_corpus(self) -> None:
        """更新RAG语料库"""
        try:
            rag_documents = self.kb.export_to_rag_format()
            
            rag_path = self._get_rag_corpus_path()
            
            with open(rag_path, 'w', encoding='utf-8') as f:
                for doc in rag_documents:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            
            print(f"[知识更新] RAG语料库已更新，共{len(rag_documents)}条记录")
        except Exception as e:
            print(f"[知识更新] 更新RAG语料库失败: {e}")
    
    def _get_rag_corpus_path(self) -> Path:
        """获取RAG语料库路径"""
        base_dir = Path(__file__).resolve().parents[1]
        return base_dir / "data" / "runtime" / "policy_case_corpus.jsonl"
    
    def batch_update(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量更新知识库
        
        Args:
            facts: 事实列表
            
        Returns:
            更新报告
        """
        report = {
            "total": len(facts),
            "success": 0,
            "failed": 0,
            "details": []
        }
        
        for fact in facts:
            try:
                success = self.kb.add_fact(fact)
                if success:
                    report["success"] += 1
                    report["details"].append({
                        "fact": fact,
                        "status": "success"
                    })
                else:
                    report["failed"] += 1
                    report["details"].append({
                        "fact": fact,
                        "status": "failed"
                    })
            except Exception as e:
                report["failed"] += 1
                report["details"].append({
                    "fact": fact,
                    "status": "error",
                    "error": str(e)
                })
        
        if report["success"] > 0:
            self.kb.save()
            self._update_rag_corpus()
        
        return report
    
    def get_key_projects(self) -> List[Dict[str, Any]]:
        """获取重点项目列表"""
        projects = []
        
        for name, info in self.kb.project_status.items():
            projects.append({
                "name": name,
                "status": info.status,
                "district": info.district,
                "update_time": info.update_time
            })
        
        projects.sort(key=lambda x: x.get("update_time", ""), reverse=True)
        
        return projects[:20]
    
    def update_project_status(self, update: Dict[str, Any]) -> bool:
        """
        更新项目状态
        
        Args:
            update: 更新信息
            
        Returns:
            是否更新成功
        """
        project_name = update.get("project")
        if not project_name:
            return False
        
        existing_project = self.kb.get_project_status(project_name)
        
        if existing_project:
            if update.get("new_status"):
                existing_project.status = update["new_status"]
            if update.get("demolition_status"):
                existing_project.demolition_status = update["demolition_status"]
            if update.get("responsible_unit"):
                existing_project.responsible_unit = update["responsible_unit"]
            
            self.kb.update_project_status(existing_project)
        else:
            new_project = ProjectInfo(
                name=project_name,
                status=update.get("new_status", ""),
                demolition_status=update.get("demolition_status", ""),
                responsible_unit=update.get("responsible_unit", ""),
                district=update.get("district", ""),
                source=update.get("source", "auto_update"),
                confidence=update.get("confidence", 0.7)
            )
            self.kb.update_project_status(new_project)
        
        return True
    
    def generate_update_report(self) -> Dict[str, Any]:
        """生成更新报告"""
        kb_stats = self.kb.get_statistics()
        error_stats = self.feedback_db.get_error_statistics()
        feedback_stats = self.feedback_db.get_statistics()
        
        return {
            "generated_at": datetime.now().isoformat(),
            "knowledge_base": kb_stats,
            "error_analysis": error_stats,
            "feedback_stats": {
                "total_count": feedback_stats.get("total_count", 0),
                "helpful_rate": feedback_stats.get("helpful_rate", 0)
            }
        }


def run_daily_knowledge_update() -> Dict[str, Any]:
    """
    执行每日知识更新的便捷函数
    
    Returns:
        更新报告
    """
    updater = KnowledgeUpdater()
    return updater.run_daily_update()


if __name__ == "__main__":
    updater = KnowledgeUpdater()
    
    print("执行每日知识更新...")
    report = updater.run_daily_update()
    
    print("\n更新报告:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    
    print("\n知识库统计:")
    stats = updater.kb.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    print("\n重点项目:")
    projects = updater.get_key_projects()
    for project in projects[:5]:
        print(f"  - {project['name']}: {project['status']}")
