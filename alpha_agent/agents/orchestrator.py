"""
Agent协调器 - 管理多Agent协作
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from ..core.base import BaseAgent, AgentResult, AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """任务定义"""
    task_id: str
    task_type: str  # mining, modeling, analysis
    description: str
    status: str = "pending"
    assigned_agent: str = ""
    result: Optional[AgentResult] = None


class Orchestrator:
    """多Agent协调器"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: List[Task] = []
        self.task_queue: List[Task] = []
    
    def register_agent(self, name: str, agent: BaseAgent):
        """注册Agent"""
        self.agents[name] = agent
        logger.info(f"Agent已注册: {name}")
    
    def create_task(
        self,
        task_type: str,
        description: str,
    ) -> Task:
        """创建任务"""
        task = Task(
            task_id=f"task_{len(self.tasks)+1}",
            task_type=task_type,
            description=description,
        )
        self.tasks.append(task)
        self.task_queue.append(task)
        return task
    
    def assign_task(self, task: Task) -> Optional[str]:
        """分配任务给合适的Agent"""
        # 简单的任务分配逻辑
        agent_mapping = {
            'mining': 'MiningAgent',
            'modeling': 'ModelingAgent',
            'analysis': 'AnalysisAgent',
        }
        
        agent_name = agent_mapping.get(task.task_type)
        if agent_name and agent_name in self.agents:
            task.assigned_agent = agent_name
            return agent_name
        
        # 默认使用第一个可用Agent
        if self.agents:
            first_agent = list(self.agents.keys())[0]
            task.assigned_agent = first_agent
            return first_agent
        
        return None
    
    def execute_task(self, task: Task) -> AgentResult:
        """执行任务"""
        if not task.assigned_agent:
            self.assign_task(task)
        
        if task.assigned_agent not in self.agents:
            raise ValueError(f"Agent不存在: {task.assigned_agent}")
        
        agent = self.agents[task.assigned_agent]
        task.status = "running"
        
        try:
            result = agent.run(task.description)
            task.status = "completed"
            task.result = result
            return result
        except Exception as e:
            task.status = "failed"
            logger.error(f"任务执行失败: {e}")
            raise
    
    def run_pipeline(self, tasks: List[Dict]) -> List[AgentResult]:
        """运行任务管道"""
        results = []
        
        for task_def in tasks:
            task = self.create_task(
                task_type=task_def.get('type', 'mining'),
                description=task_def.get('description', ''),
            )
            
            result = self.execute_task(task)
            results.append(result)
            
            # 如果任务失败，可以选择停止或继续
            if result.status == AgentStatus.FAILED:
                logger.warning(f"任务失败: {task.task_id}")
        
        return results
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'agents': list(self.agents.keys()),
            'total_tasks': len(self.tasks),
            'pending': len([t for t in self.tasks if t.status == 'pending']),
            'running': len([t for t in self.tasks if t.status == 'running']),
            'completed': len([t for t in self.tasks if t.status == 'completed']),
            'failed': len([t for t in self.tasks if t.status == 'failed']),
        }
