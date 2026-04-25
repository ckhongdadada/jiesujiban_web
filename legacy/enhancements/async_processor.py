from __future__ import annotations

import os
import time
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import defaultdict


@dataclass(order=True)
class PriorityTask:
    """优先级任务"""
    priority: int
    task_id: str = field(compare=False)
    data: dict = field(compare=False)
    callback: Callable | None = field(default=None, compare=False)
    submit_time: float = field(default_factory=time.time, compare=False)


class AsyncTaskProcessor:
    """异步任务处理器"""
    
    def __init__(
        self,
        max_workers: int = 4,
        model_manager: Any = None
    ):
        self.max_workers = max_workers
        self.model_manager = model_manager
        
        self.task_queue: queue.PriorityQueue[PriorityTask] = queue.PriorityQueue()
        self.result_queue: queue.Queue[dict] = queue.Queue()
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: dict[str, Future] = {}
        
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_time': 0.0,
            'stage_times': defaultdict(float),
            'stage_counts': defaultdict(int)
        }
        
        self._running = True
        self._lock = threading.Lock()
        
        self._start_workers()
    
    def _start_workers(self):
        """启动工作线程"""
        for i in range(self.max_workers):
            threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            ).start()
        
        print(f"[异步处理器] 启动 {self.max_workers} 个工作线程")
    
    def _worker_loop(self, worker_id: int):
        """工作线程循环"""
        while self._running:
            try:
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self._process_task(task, worker_id)
                self.task_queue.task_done()
            
            except Exception as e:
                print(f"[Worker {worker_id}] 异常: {e}")
    
    def _process_task(self, task: PriorityTask, worker_id: int):
        """处理任务"""
        start_time = time.time()
        
        try:
            data = task.data
            
            stage_times = {}
            
            stage_start = time.time()
            location_result = self._stage_location(data)
            stage_times['location'] = time.time() - stage_start
            
            stage_start = time.time()
            rag_results = self._stage_rag(data, location_result)
            stage_times['rag'] = time.time() - stage_start
            
            stage_start = time.time()
            unit_result = self._stage_classify(data, location_result)
            stage_times['classify'] = time.time() - stage_start
            
            stage_start = time.time()
            reply_result = self._stage_generate(data, location_result, rag_results, unit_result)
            stage_times['generate'] = time.time() - stage_start
            
            result = {
                'task_id': task.task_id,
                'status': 'success',
                'location': location_result,
                'rag_results': rag_results,
                'unit': unit_result,
                'reply': reply_result,
                'stage_times': stage_times,
                'total_time': time.time() - start_time,
                'worker_id': worker_id
            }
            
            with self._lock:
                self.stats['total_completed'] += 1
                self.stats['total_time'] += result['total_time']
                for stage, t in stage_times.items():
                    self.stats['stage_times'][stage] += t
                    self.stats['stage_counts'][stage] += 1
            
            self.result_queue.put(result)
            
            if task.callback:
                try:
                    task.callback(result)
                except Exception as e:
                    print(f"[Worker {worker_id}] 回调失败: {e}")
        
        except Exception as e:
            with self._lock:
                self.stats['total_failed'] += 1
            
            self.result_queue.put({
                'task_id': task.task_id,
                'status': 'error',
                'error': str(e),
                'worker_id': worker_id
            })
    
    def _stage_location(self, data: dict) -> dict:
        """阶段1：地名识别"""
        if self.model_manager:
            return self.model_manager.resolve_location(data.get('body', ''))
        return {'district': '未识别', 'confidence': 0.0}
    
    def _stage_rag(self, data: dict, location_result: dict) -> list:
        """阶段2：RAG检索"""
        if self.model_manager:
            return self.model_manager.search_rag(
                f"{data.get('title', '')} {data.get('body', '')}",
                top_k=3,
                district=location_result.get('district')
            )
        return []
    
    def _stage_classify(self, data: dict, location_result: dict) -> dict:
        """阶段3：分类预测"""
        return {'unit': data.get('unit', '相关部门'), 'confidence': 0.8}
    
    def _stage_generate(
        self,
        data: dict,
        location_result: dict,
        rag_results: list,
        unit_result: dict
    ) -> dict:
        """阶段4：文本生成"""
        if self.model_manager:
            return self.model_manager.generate_reply(
                tag=data.get('tag', ''),
                title=data.get('title', ''),
                body=data.get('body', ''),
                unit=unit_result.get('unit', '相关部门'),
                location_result=location_result,
                retrieval_hits=rag_results
            )
        return {'reply': '模拟回复'}
    
    def submit(
        self,
        task_id: str,
        tag: str,
        title: str,
        body: str,
        priority: int = 0,
        callback: Callable | None = None
    ) -> str:
        """提交任务"""
        task = PriorityTask(
            priority=priority,
            task_id=task_id,
            data={
                'tag': tag,
                'title': title,
                'body': body
            },
            callback=callback
        )
        
        self.task_queue.put(task)
        
        with self._lock:
            self.stats['total_submitted'] += 1
        
        return task_id
    
    def get_result(self, timeout: float | None = None) -> dict | None:
        """获取结果"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            stats = self.stats.copy()
            
            if stats['total_completed'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['total_completed']
            
            for stage in stats['stage_times']:
                count = stats['stage_counts'][stage]
                if count > 0:
                    stats[f'avg_{stage}_time'] = stats['stage_times'][stage] / count
            
            stats['queue_size'] = self.task_queue.qsize()
            stats['result_queue_size'] = self.result_queue.qsize()
            
            return stats
    
    def shutdown(self):
        """关闭处理器"""
        self._running = False
        self.executor.shutdown(wait=False)
        print("[异步处理器] 已关闭")


class PipelineProcessor:
    """流水线处理器"""
    
    def __init__(self, model_manager: Any = None, buffer_size: int = 100):
        self.model_manager = model_manager
        self.buffer_size = buffer_size
        
        self.stage1_queue: queue.Queue[dict] = queue.Queue(maxsize=buffer_size)
        self.stage2_queue: queue.Queue[dict] = queue.Queue(maxsize=buffer_size)
        self.stage3_queue: queue.Queue[dict] = queue.Queue(maxsize=buffer_size)
        self.stage4_queue: queue.Queue[dict] = queue.Queue(maxsize=buffer_size)
        self.result_queue: queue.Queue[dict] = queue.Queue()
        
        self.stats = {
            'total_submitted': 0,
            'stage1_processed': 0,
            'stage2_processed': 0,
            'stage3_processed': 0,
            'stage4_processed': 0
        }
        
        self._running = True
        self._lock = threading.Lock()
        
        self._start_pipeline()
    
    def _start_pipeline(self):
        """启动流水线"""
        threading.Thread(target=self._stage1_worker, daemon=True).start()
        threading.Thread(target=self._stage2_worker, daemon=True).start()
        threading.Thread(target=self._stage3_worker, daemon=True).start()
        threading.Thread(target=self._stage4_worker, daemon=True).start()
        
        print("[流水线处理器] 启动4阶段流水线")
    
    def _stage1_worker(self):
        """阶段1：地名识别"""
        while self._running:
            try:
                task = self.stage1_queue.get(timeout=1.0)
                
                start = time.time()
                location_result = self._process_location(task)
                
                task['location_result'] = location_result
                task['stage1_time'] = time.time() - start
                
                self.stage2_queue.put(task)
                
                with self._lock:
                    self.stats['stage1_processed'] += 1
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Pipeline Stage1] 错误: {e}")
    
    def _stage2_worker(self):
        """阶段2：RAG检索"""
        while self._running:
            try:
                task = self.stage2_queue.get(timeout=1.0)
                
                start = time.time()
                rag_results = self._process_rag(task)
                
                task['rag_results'] = rag_results
                task['stage2_time'] = time.time() - start
                
                self.stage3_queue.put(task)
                
                with self._lock:
                    self.stats['stage2_processed'] += 1
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Pipeline Stage2] 错误: {e}")
    
    def _stage3_worker(self):
        """阶段3：分类预测"""
        while self._running:
            try:
                task = self.stage3_queue.get(timeout=1.0)
                
                start = time.time()
                unit_result = self._process_classify(task)
                
                task['unit_result'] = unit_result
                task['stage3_time'] = time.time() - start
                
                self.stage4_queue.put(task)
                
                with self._lock:
                    self.stats['stage3_processed'] += 1
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Pipeline Stage3] 错误: {e}")
    
    def _stage4_worker(self):
        """阶段4：文本生成"""
        while self._running:
            try:
                task = self.stage4_queue.get(timeout=1.0)
                
                start = time.time()
                reply_result = self._process_generate(task)
                
                task['reply_result'] = reply_result
                task['stage4_time'] = time.time() - start
                task['total_time'] = time.time() - task.get('submit_time', time.time())
                
                self.result_queue.put(task)
                
                with self._lock:
                    self.stats['stage4_processed'] += 1
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Pipeline Stage4] 错误: {e}")
    
    def _process_location(self, task: dict) -> dict:
        if self.model_manager:
            return self.model_manager.resolve_location(task.get('body', ''))
        return {'district': '未识别'}
    
    def _process_rag(self, task: dict) -> list:
        if self.model_manager:
            return self.model_manager.search_rag(
                f"{task.get('title', '')} {task.get('body', '')}",
                top_k=3,
                district=task.get('location_result', {}).get('district')
            )
        return []
    
    def _process_classify(self, task: dict) -> dict:
        return {'unit': '相关部门', 'confidence': 0.8}
    
    def _process_generate(self, task: dict) -> dict:
        if self.model_manager:
            return self.model_manager.generate_reply(
                tag=task.get('tag', ''),
                title=task.get('title', ''),
                body=task.get('body', ''),
                unit=task.get('unit_result', {}).get('unit', '相关部门'),
                location_result=task.get('location_result', {}),
                retrieval_hits=task.get('rag_results', [])
            )
        return {'reply': '模拟回复'}
    
    def submit(self, task_id: str, tag: str, title: str, body: str):
        """提交任务"""
        task = {
            'task_id': task_id,
            'tag': tag,
            'title': title,
            'body': body,
            'submit_time': time.time()
        }
        
        self.stage1_queue.put(task)
        
        with self._lock:
            self.stats['total_submitted'] += 1
    
    def get_result(self, timeout: float | None = None) -> dict | None:
        """获取结果"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            stats = self.stats.copy()
            stats['stage1_queue_size'] = self.stage1_queue.qsize()
            stats['stage2_queue_size'] = self.stage2_queue.qsize()
            stats['stage3_queue_size'] = self.stage3_queue.qsize()
            stats['stage4_queue_size'] = self.stage4_queue.qsize()
            return stats
    
    def shutdown(self):
        """关闭流水线"""
        self._running = False
        print("[流水线处理器] 已关闭")


_processor_instance: AsyncTaskProcessor | None = None
_pipeline_instance: PipelineProcessor | None = None
_processor_lock = threading.Lock()


def get_async_processor(
    max_workers: int = 4,
    model_manager: Any = None
) -> AsyncTaskProcessor:
    """获取异步处理器单例"""
    global _processor_instance
    
    with _processor_lock:
        if _processor_instance is None:
            _processor_instance = AsyncTaskProcessor(
                max_workers=max_workers,
                model_manager=model_manager
            )
        return _processor_instance


def get_pipeline_processor(
    model_manager: Any = None
) -> PipelineProcessor:
    """获取流水线处理器单例"""
    global _pipeline_instance
    
    with _processor_lock:
        if _pipeline_instance is None:
            _pipeline_instance = PipelineProcessor(model_manager=model_manager)
        return _pipeline_instance
