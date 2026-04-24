"""
更新调度器
实现定时任务调度，自动更新知识库
"""

from __future__ import annotations

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

from enhancements.knowledge_updater import KnowledgeUpdater
from enhancements.info_fusion import InformationFusion
from enhancements.structured_kb import get_knowledge_base


@dataclass
class UpdateTask:
    """更新任务"""
    task_id: str
    task_name: str
    task_type: str
    schedule: str
    last_run: str = ""
    next_run: str = ""
    status: str = "pending"
    result: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.result is None:
            self.result = {}


class UpdateScheduler:
    """更新调度器"""
    
    def __init__(self, enable_apscheduler: bool = True):
        self.knowledge_updater = KnowledgeUpdater()
        self.info_fusion = InformationFusion()
        self.kb = get_knowledge_base()
        
        self.scheduler = None
        self.tasks: Dict[str, UpdateTask] = {}
        self.task_history: List[Dict[str, Any]] = []
        
        if enable_apscheduler and APSCHEDULER_AVAILABLE:
            self.scheduler = BackgroundScheduler()
            self._setup_default_tasks()
        elif not APSCHEDULER_AVAILABLE:
            print("[更新调度器] 警告: apscheduler 未安装，定时任务功能不可用")
            print("[更新调度器] 可通过 pip install apscheduler 安装")
    
    def _setup_default_tasks(self) -> None:
        """设置默认任务"""
        self.tasks["daily_gov_update"] = UpdateTask(
            task_id="daily_gov_update",
            task_name="每日政务数据更新",
            task_type="government_data",
            schedule="cron:hour=2,minute=0"
        )
        
        self.tasks["hourly_news_update"] = UpdateTask(
            task_id="hourly_news_update",
            task_name="每小时新闻更新",
            task_type="news_data",
            schedule="interval:hours=4"
        )
        
        self.tasks["weekly_kb_update"] = UpdateTask(
            task_id="weekly_kb_update",
            task_name="每周知识库全面更新",
            task_type="knowledge_base",
            schedule="cron:day_of_week=sun,hour=3,minute=0"
        )
        
        self.tasks["daily_feedback_learning"] = UpdateTask(
            task_id="daily_feedback_learning",
            task_name="每日反馈学习",
            task_type="feedback_learning",
            schedule="cron:hour=1,minute=0"
        )
    
    def start(self) -> None:
        """启动调度器"""
        if not self.scheduler:
            print("[更新调度器] 调度器未初始化，无法启动")
            return
        
        for task_id, task in self.tasks.items():
            self._add_task_to_scheduler(task)
        
        self.scheduler.start()
        print("[更新调度器] 调度器已启动")
    
    def _add_task_to_scheduler(self, task: UpdateTask) -> None:
        """添加任务到调度器"""
        if not self.scheduler:
            return
        
        schedule_parts = task.schedule.split(":")
        schedule_type = schedule_parts[0]
        
        if schedule_type == "cron":
            trigger = self._parse_cron_schedule(":".join(schedule_parts[1:]))
            self.scheduler.add_job(
                self._execute_task,
                trigger=trigger,
                args=[task.task_id],
                id=task.task_id,
                name=task.task_name
            )
        elif schedule_type == "interval":
            trigger = self._parse_interval_schedule(":".join(schedule_parts[1:]))
            self.scheduler.add_job(
                self._execute_task,
                trigger=trigger,
                args=[task.task_id],
                id=task.task_id,
                name=task.task_name
            )
    
    def _parse_cron_schedule(self, cron_expr: str) -> CronTrigger:
        """解析cron表达式"""
        params = {}
        for part in cron_expr.split(","):
            key, value = part.split("=")
            params[key.strip()] = int(value.strip())
        
        return CronTrigger(**params)
    
    def _parse_interval_schedule(self, interval_expr: str) -> IntervalTrigger:
        """解析interval表达式"""
        params = {}
        for part in interval_expr.split(","):
            key, value = part.split("=")
            params[key.strip()] = int(value.strip())
        
        return IntervalTrigger(**params)
    
    def _execute_task(self, task_id: str) -> None:
        """执行任务"""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        task.status = "running"
        task.last_run = datetime.now().isoformat()
        
        print(f"[更新调度器] 开始执行任务: {task.task_name}")
        
        try:
            if task.task_type == "government_data":
                result = self._run_gov_update()
            elif task.task_type == "news_data":
                result = self._run_news_update()
            elif task.task_type == "knowledge_base":
                result = self._run_kb_update()
            elif task.task_type == "feedback_learning":
                result = self._run_feedback_learning()
            else:
                result = {"error": f"未知任务类型: {task.task_type}"}
            
            task.result = result
            task.status = "completed"
            
            self.task_history.append({
                "task_id": task_id,
                "task_name": task.task_name,
                "execution_time": task.last_run,
                "status": "success",
                "result": result
            })
            
            print(f"[更新调度器] 任务完成: {task.task_name}")
            
        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
            
            self.task_history.append({
                "task_id": task_id,
                "task_name": task.task_name,
                "execution_time": task.last_run,
                "status": "failed",
                "error": str(e)
            })
            
            print(f"[更新调度器] 任务失败: {task.task_name}, 错误: {e}")
    
    def _run_gov_update(self) -> Dict[str, Any]:
        """执行政务数据更新"""
        try:
            from crawlers.government_data_crawler import crawl_government_data
            
            project_infos = crawl_government_data()
            
            if project_infos:
                self.knowledge_updater.batch_update(project_infos)
            
            return {
                "projects_crawled": len(project_infos),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def _run_news_update(self) -> Dict[str, Any]:
        """执行新闻更新"""
        try:
            from crawlers.news_api_connector import NewsAPIConnector
            
            connector = NewsAPIConnector()
            
            key_projects = self.knowledge_updater.get_key_projects()
            
            updates_count = 0
            for project in key_projects[:10]:
                news_list = connector.search_project_news(
                    project["name"],
                    project.get("district", "")
                )
                
                updates = connector.extract_project_updates(news_list)
                
                for update in updates:
                    self.knowledge_updater.update_project_status({
                        "project": update.project,
                        "new_status": update.new_status,
                        "source": update.source,
                        "confidence": update.confidence
                    })
                    updates_count += 1
            
            return {
                "projects_checked": len(key_projects[:10]),
                "updates_applied": updates_count,
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def _run_kb_update(self) -> Dict[str, Any]:
        """执行知识库全面更新"""
        try:
            report = self.knowledge_updater.run_daily_update()
            
            return {
                "report": report,
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def _run_feedback_learning(self) -> Dict[str, Any]:
        """执行反馈学习"""
        try:
            report = self.knowledge_updater.run_daily_update()
            
            return {
                "processed_count": report.get("processed_count", 0),
                "facts_extracted": report.get("facts_extracted", 0),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def stop(self) -> None:
        """停止调度器"""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            print("[更新调度器] 调度器已停止")
    
    def add_custom_task(
        self,
        task_id: str,
        task_name: str,
        task_type: str,
        schedule: str,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        添加自定义任务
        
        Args:
            task_id: 任务ID
            task_name: 任务名称
            task_type: 任务类型
            schedule: 调度表达式
            callback: 回调函数
            
        Returns:
            是否添加成功
        """
        if task_id in self.tasks:
            return False
        
        task = UpdateTask(
            task_id=task_id,
            task_name=task_name,
            task_type=task_type,
            schedule=schedule
        )
        
        self.tasks[task_id] = task
        
        if self.scheduler:
            self._add_task_to_scheduler(task)
        
        return True
    
    def remove_task(self, task_id: str) -> bool:
        """移除任务"""
        if task_id not in self.tasks:
            return False
        
        if self.scheduler:
            try:
                self.scheduler.remove_job(task_id)
            except Exception:
                pass
        
        del self.tasks[task_id]
        return True
    
    def run_task_now(self, task_id: str) -> Dict[str, Any]:
        """立即执行任务"""
        if task_id not in self.tasks:
            return {"error": f"任务不存在: {task_id}"}
        
        self._execute_task(task_id)
        
        return self.tasks[task_id].result or {}
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "schedule": task.schedule,
            "last_run": task.last_run,
            "status": task.status
        }
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务"""
        return [self.get_task_status(task_id) for task_id in self.tasks]
    
    def get_task_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取任务历史"""
        return self.task_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_tasks": len(self.tasks),
            "running_tasks": sum(1 for t in self.tasks.values() if t.status == "running"),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.status == "completed"),
            "failed_tasks": sum(1 for t in self.tasks.values() if t.status == "failed"),
            "history_count": len(self.task_history)
        }


_scheduler_instance: Optional[UpdateScheduler] = None


def get_scheduler() -> UpdateScheduler:
    """获取调度器单例"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = UpdateScheduler()
    return _scheduler_instance


def start_scheduler() -> None:
    """启动调度器"""
    scheduler = get_scheduler()
    scheduler.start()


if __name__ == "__main__":
    print("更新调度器测试")
    print("=" * 50)
    
    scheduler = UpdateScheduler(enable_apscheduler=APSCHEDULER_AVAILABLE)
    
    print("\n已注册任务:")
    for task_info in scheduler.get_all_tasks():
        print(f"  - {task_info['task_name']} ({task_info['task_type']})")
        print(f"    调度: {task_info['schedule']}")
    
    print("\n立即执行反馈学习任务...")
    result = scheduler.run_task_now("daily_feedback_learning")
    print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    print("\n统计信息:")
    stats = scheduler.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    if APSCHEDULER_AVAILABLE:
        print("\n启动调度器...")
        scheduler.start()
        
        print("调度器运行中，按 Ctrl+C 停止...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()
            print("\n调度器已停止")
