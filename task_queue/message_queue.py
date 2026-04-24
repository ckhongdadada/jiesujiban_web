"""
消息队列模块
实现异步任务处理、任务队列管理、后台工作线程
"""

from __future__ import annotations

import json
import time
import threading
import queue
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20


@dataclass
class Task:
    """任务"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int
    status: str
    created_at: str
    started_at: str = ""
    completed_at: str = ""
    result: Dict[str, Any] = None
    error: str = ""
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.result is None:
            self.result = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


class TaskQueue:
    """任务队列"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化任务队列
        
        Args:
            max_size: 队列最大容量
        """
        self.max_size = max_size
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_size)
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.RLock()
        self._task_counter = 0
    
    def put(self, task: Task) -> bool:
        """
        添加任务到队列
        
        Args:
            task: 任务对象
            
        Returns:
            是否添加成功
        """
        with self._lock:
            if self._queue.qsize() >= self.max_size:
                return False
            
            self._task_counter += 1
            priority_tuple = (-task.priority, self._task_counter, task.task_id)
            self._queue.put(priority_tuple)
            self._tasks[task.task_id] = task
            
            return True
    
    def get(self, timeout: float = 1.0) -> Optional[Task]:
        """
        从队列获取任务
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            任务对象，如果队列为空则返回None
        """
        try:
            priority_tuple = self._queue.get(timeout=timeout)
            task_id = priority_tuple[2]
            return self._tasks.get(task_id)
        except queue.Empty:
            return None
    
    def task_done(self):
        """标记任务完成"""
        self._queue.task_done()
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        return self._tasks.get(task_id)
    
    def update_task(self, task: Task):
        """更新任务状态"""
        with self._lock:
            self._tasks[task.task_id] = task
    
    def size(self) -> int:
        """获取队列大小"""
        return self._queue.qsize()
    
    def clear(self):
        """清空队列"""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._tasks.clear()


class WorkerPool:
    """工作线程池"""
    
    def __init__(
        self,
        task_queue: TaskQueue,
        num_workers: int = 4,
        task_handlers: Optional[Dict[str, Callable]] = None
    ):
        """
        初始化工作线程池
        
        Args:
            task_queue: 任务队列
            num_workers: 工作线程数量
            task_handlers: 任务处理器映射
        """
        self.task_queue = task_queue
        self.num_workers = num_workers
        self.task_handlers = task_handlers or {}
        
        self._workers: List[threading.Thread] = []
        self._running = False
        self._lock = threading.RLock()
    
    def register_handler(self, task_type: str, handler: Callable):
        """
        注册任务处理器
        
        Args:
            task_type: 任务类型
            handler: 处理函数
        """
        self.task_handlers[task_type] = handler
    
    def start(self):
        """启动工作线程池"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            
            for i in range(self.num_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"Worker-{i}",
                    daemon=True
                )
                worker.start()
                self._workers.append(worker)
            
            print(f"[工作线程池] 已启动 {self.num_workers} 个工作线程")
    
    def stop(self):
        """停止工作线程池"""
        with self._lock:
            self._running = False
    
    def _worker_loop(self):
        """工作线程主循环"""
        while self._running:
            task = self.task_queue.get(timeout=1.0)
            
            if task is None:
                continue
            
            self._process_task(task)
            self.task_queue.task_done()
    
    def _process_task(self, task: Task):
        """处理任务"""
        task.status = TaskStatus.RUNNING.value
        task.started_at = datetime.now().isoformat()
        self.task_queue.update_task(task)
        
        handler = self.task_handlers.get(task.task_type)
        
        if handler is None:
            task.status = TaskStatus.FAILED.value
            task.error = f"未找到任务处理器: {task.task_type}"
            task.completed_at = datetime.now().isoformat()
            self.task_queue.update_task(task)
            return
        
        try:
            result = handler(task.payload)
            
            task.status = TaskStatus.COMPLETED.value
            task.result = result if isinstance(result, dict) else {"data": result}
            task.completed_at = datetime.now().isoformat()
            
        except Exception as e:
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING.value
                self.task_queue.put(task)
                print(f"[工作线程] 任务 {task.task_id} 失败，正在重试 ({task.retry_count}/{task.max_retries})")
            else:
                task.status = TaskStatus.FAILED.value
                task.error = str(e)
                task.completed_at = datetime.now().isoformat()
                print(f"[工作线程] 任务 {task.task_id} 失败: {e}")
        
        self.task_queue.update_task(task)


class MessageQueue:
    """消息队列管理器"""
    
    def __init__(self, num_workers: int = 4, max_queue_size: int = 1000):
        """
        初始化消息队列
        
        Args:
            num_workers: 工作线程数量
            max_queue_size: 队列最大容量
        """
        self.task_queue = TaskQueue(max_size=max_queue_size)
        self.worker_pool = WorkerPool(self.task_queue, num_workers)
        
        self._task_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认处理器"""
        self.register_handler("predict", self._handle_predict)
        self.register_handler("generate", self._handle_generate)
        self.register_handler("crawl", self._handle_crawl)
        self.register_handler("update_kb", self._handle_update_kb)
    
    def register_handler(self, task_type: str, handler: Callable):
        """注册任务处理器"""
        self._task_handlers[task_type] = handler
        self.worker_pool.register_handler(task_type, handler)
    
    def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = TaskPriority.NORMAL.value,
        max_retries: int = 3
    ) -> str:
        """
        提交任务
        
        Args:
            task_type: 任务类型
            payload: 任务数据
            priority: 优先级
            max_retries: 最大重试次数
            
        Returns:
            任务ID
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            status=TaskStatus.PENDING.value,
            created_at=datetime.now().isoformat(),
            max_retries=max_retries
        )
        
        self.task_queue.put(task)
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        task = self.task_queue.get_task(task_id)
        
        if task is None:
            return None
        
        return task.to_dict()
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.task_queue.size()
    
    def start(self):
        """启动消息队列"""
        self.worker_pool.start()
    
    def stop(self):
        """停止消息队列"""
        self.worker_pool.stop()
    
    def _handle_predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理预测任务"""
        from enhancements.classifier_runtime import ClassifierRuntime
        
        tag = payload.get("tag", "")
        title = payload.get("title", "")
        body = payload.get("body", "")
        
        classifier = ClassifierRuntime()
        predictions = classifier.predict(tag, title, body)
        
        return {"predictions": predictions}
    
    def _handle_generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理生成任务"""
        from enhancements.enhanced_generation import generate_reply_with_context
        
        result = generate_reply_with_context(
            tag=payload.get("tag", ""),
            title=payload.get("title", ""),
            body=payload.get("body", ""),
            unit=payload.get("unit", ""),
            location_result=payload.get("location_result", {}),
            retrieval_hits=payload.get("retrieval_hits", [])
        )
        
        return result
    
    def _handle_crawl(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理爬虫任务"""
        crawl_type = payload.get("crawl_type", "government")
        
        if crawl_type == "government":
            from crawlers.government_data_crawler import crawl_government_data
            results = crawl_government_data(
                districts=payload.get("districts"),
                max_pages=payload.get("max_pages", 5)
            )
            return {"announcements": len(results)}
        
        elif crawl_type == "news":
            from crawlers.news_api_connector import NewsAPIConnector
            connector = NewsAPIConnector()
            news = connector.search_project_news(
                project_name=payload.get("project_name", ""),
                district=payload.get("district", "")
            )
            return {"news_count": len(news)}
        
        return {"error": "未知的爬虫类型"}
    
    def _handle_update_kb(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理知识库更新任务"""
        from enhancements.knowledge_updater import run_daily_knowledge_update
        
        report = run_daily_knowledge_update()
        
        return report


_mq_instance: Optional[MessageQueue] = None


def get_message_queue() -> MessageQueue:
    """获取消息队列单例"""
    global _mq_instance
    if _mq_instance is None:
        _mq_instance = MessageQueue()
    return _mq_instance


def submit_async_task(
    task_type: str,
    payload: Dict[str, Any],
    priority: int = TaskPriority.NORMAL.value
) -> str:
    """
    提交异步任务
    
    Args:
        task_type: 任务类型
        payload: 任务数据
        priority: 优先级
        
    Returns:
        任务ID
    """
    mq = get_message_queue()
    return mq.submit_task(task_type, payload, priority)


if __name__ == "__main__":
    print("消息队列模块测试")
    print("=" * 50)
    
    mq = MessageQueue(num_workers=2)
    mq.start()
    
    print("\n1. 提交预测任务")
    task_id = mq.submit_task(
        "predict",
        {
            "tag": "投诉",
            "title": "测试标题",
            "body": "测试内容"
        },
        priority=TaskPriority.HIGH.value
    )
    print(f"任务ID: {task_id}")
    
    print("\n2. 查询任务状态")
    time.sleep(2)
    status = mq.get_task_status(task_id)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    
    print("\n3. 队列大小")
    print(f"当前队列大小: {mq.get_queue_size()}")
    
    mq.stop()
    print("\n消息队列已停止")
