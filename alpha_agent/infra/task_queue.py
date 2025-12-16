"""
异步任务队列 - Celery

功能:
1. 因子生成任务
2. 回测任务
3. 模型训练任务
4. 任务状态查询

参考: https://docs.celeryq.dev/
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
import uuid
import logging

logger = logging.getLogger(__name__)

# Celery可选
try:
    from celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logger.info("Celery未安装，使用本地任务队列")


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REVOKED = "revoked"


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: str = None
    created_at: str = None
    finished_at: str = None


# Celery应用配置
def create_celery_app(
    broker_url: str = "redis://localhost:6379/0",
    result_backend: str = "redis://localhost:6379/1",
) -> "Celery":
    """创建Celery应用"""
    if not CELERY_AVAILABLE:
        return None
    
    app = Celery(
        'alpha_agent',
        broker=broker_url,
        backend=result_backend,
    )
    
    app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='Asia/Shanghai',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=3600,  # 1小时超时
        worker_prefetch_multiplier=1,
    )
    
    return app


# 本地任务队列 (无Celery时使用)
class LocalTaskQueue:
    """本地任务队列"""
    
    def __init__(self):
        self.tasks: Dict[str, TaskResult] = {}
        self.handlers: Dict[str, Callable] = {}
    
    def register(self, name: str, handler: Callable):
        """注册任务处理器"""
        self.handlers[name] = handler
    
    def submit(self, name: str, *args, **kwargs) -> str:
        """提交任务"""
        task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat(),
        )
        
        # 同步执行
        handler = self.handlers.get(name)
        if handler is None:
            self.tasks[task_id].status = TaskStatus.FAILED
            self.tasks[task_id].error = f"未知任务: {name}"
            return task_id
        
        try:
            self.tasks[task_id].status = TaskStatus.RUNNING
            result = handler(*args, **kwargs)
            self.tasks[task_id].status = TaskStatus.SUCCESS
            self.tasks[task_id].result = result
        except Exception as e:
            self.tasks[task_id].status = TaskStatus.FAILED
            self.tasks[task_id].error = str(e)
        finally:
            self.tasks[task_id].finished_at = datetime.now().isoformat()
        
        return task_id
    
    def get_status(self, task_id: str) -> TaskResult:
        """获取任务状态"""
        return self.tasks.get(task_id)
    
    def revoke(self, task_id: str):
        """取消任务"""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.REVOKED


# 全局队列
_local_queue = LocalTaskQueue()


class TaskManager:
    """任务管理器"""
    
    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        use_celery: bool = True,
    ):
        self.use_celery = use_celery and CELERY_AVAILABLE
        
        if self.use_celery:
            self.celery_app = create_celery_app(broker_url)
        else:
            self.celery_app = None
            self.local_queue = _local_queue
    
    def task(self, name: str = None):
        """任务装饰器"""
        def decorator(func):
            task_name = name or func.__name__
            
            if self.use_celery:
                # 注册Celery任务
                return self.celery_app.task(name=task_name)(func)
            else:
                # 注册本地任务
                self.local_queue.register(task_name, func)
                
                def wrapper(*args, **kwargs):
                    return self.local_queue.submit(task_name, *args, **kwargs)
                
                wrapper.delay = lambda *a, **k: wrapper(*a, **k)
                wrapper.apply_async = lambda args=(), kwargs={}: wrapper(*args, **kwargs)
                return wrapper
        
        return decorator
    
    def submit(self, task_name: str, *args, **kwargs) -> str:
        """提交任务"""
        if self.use_celery:
            result = self.celery_app.send_task(task_name, args=args, kwargs=kwargs)
            return result.id
        else:
            return self.local_queue.submit(task_name, *args, **kwargs)
    
    def get_status(self, task_id: str) -> TaskResult:
        """获取任务状态"""
        if self.use_celery:
            result = AsyncResult(task_id, app=self.celery_app)
            status_map = {
                'PENDING': TaskStatus.PENDING,
                'STARTED': TaskStatus.RUNNING,
                'SUCCESS': TaskStatus.SUCCESS,
                'FAILURE': TaskStatus.FAILED,
                'REVOKED': TaskStatus.REVOKED,
            }
            return TaskResult(
                task_id=task_id,
                status=status_map.get(result.status, TaskStatus.PENDING),
                result=result.result if result.successful() else None,
                error=str(result.result) if result.failed() else None,
            )
        else:
            return self.local_queue.get_status(task_id)
    
    def revoke(self, task_id: str):
        """取消任务"""
        if self.use_celery:
            self.celery_app.control.revoke(task_id, terminate=True)
        else:
            self.local_queue.revoke(task_id)


# 全局任务管理器
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """获取任务管理器"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


# Celery应用实例 (用于worker)
celery_app = create_celery_app() if CELERY_AVAILABLE else None


def async_task(name: str = None):
    """异步任务装饰器"""
    return get_task_manager().task(name)


# ==================== 预定义任务 ====================

@async_task("generate_factor")
def generate_factor_task(prompt: str, config: dict = None) -> dict:
    """因子生成任务"""
    from ..agents import MiningAgent
    
    agent = MiningAgent(**(config or {}))
    result = agent.run(prompt)
    
    return {
        'status': 'success',
        'factors': result.generated_factors if result else [],
    }


@async_task("run_backtest")
def backtest_task(factor_code: str, config: dict = None) -> dict:
    """回测任务"""
    from ..mining import run_backtest
    
    # 执行回测
    result = run_backtest(factor_code, **(config or {}))
    
    return {
        'status': 'success',
        'sharpe': result.sharpe_ratio,
        'return': result.total_return,
        'max_drawdown': result.max_drawdown,
    }


@async_task("train_model")
def train_model_task(model_type: str, data_config: dict, model_config: dict = None) -> dict:
    """模型训练任务"""
    # TODO: 实现模型训练
    return {'status': 'success'}
