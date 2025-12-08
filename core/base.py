"""
Agent基类和数据结构定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import pandas as pd


class AgentStatus(Enum):
    """Agent状态"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    WAITING = "waiting"


class FactorStatus(Enum):
    """因子状态"""
    PENDING = "pending"
    EXECUTING = "executing"
    EVALUATED = "evaluated"
    EXCELLENT = "excellent"  # IC > 0.05
    GOOD = "good"            # IC > 0.03
    MARGINAL = "marginal"    # IC > 0.02
    POOR = "poor"            # IC <= 0.02
    FAILED = "failed"


@dataclass
class FactorResult:
    """因子结果"""
    name: str
    code: str
    factor_type: str = "unknown"
    
    # 评估指标
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    
    # 元信息
    status: FactorStatus = FactorStatus.PENDING
    error: str = ""
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 因子数据
    values: Optional[pd.Series] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'code': self.code,
            'factor_type': self.factor_type,
            'ic': self.ic,
            'icir': self.icir,
            'rank_ic': self.rank_ic,
            'sharpe': self.sharpe,
            'status': self.status.value,
            'error': self.error,
            'timestamp': self.timestamp,
        }
    
    def is_valid(self) -> bool:
        """是否为有效因子"""
        return self.status in [FactorStatus.EXCELLENT, FactorStatus.GOOD, FactorStatus.MARGINAL]


@dataclass
class AgentResult:
    """Agent执行结果"""
    agent_name: str
    status: AgentStatus = AgentStatus.IDLE
    
    # 结果
    factors: List[FactorResult] = field(default_factory=list)
    best_factor: Optional[FactorResult] = None
    
    # 统计
    total_generated: int = 0
    total_valid: int = 0
    total_excellent: int = 0
    
    # 执行信息
    start_time: str = ""
    end_time: str = ""
    duration: float = 0.0
    error: str = ""
    
    # 日志
    logs: List[str] = field(default_factory=list)
    
    def add_factor(self, factor: FactorResult):
        """添加因子结果"""
        self.factors.append(factor)
        self.total_generated += 1
        
        if factor.is_valid():
            self.total_valid += 1
        if factor.status == FactorStatus.EXCELLENT:
            self.total_excellent += 1
        
        # 更新最佳因子
        if self.best_factor is None or abs(factor.ic) > abs(self.best_factor.ic):
            if factor.is_valid():
                self.best_factor = factor
    
    def log(self, message: str):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")


class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.status = AgentStatus.IDLE
        self.result = AgentResult(agent_name=name)
    
    @abstractmethod
    def run(self, *args, **kwargs) -> AgentResult:
        """运行Agent"""
        pass
    
    @abstractmethod
    def step(self, *args, **kwargs) -> FactorResult:
        """执行单步"""
        pass
    
    def reset(self):
        """重置状态"""
        self.status = AgentStatus.IDLE
        self.result = AgentResult(agent_name=self.name)
    
    def log(self, message: str):
        """记录日志"""
        self.result.log(message)
        print(f"[{self.name}] {message}")


class FactorPool:
    """因子池 - 管理所有生成的因子"""
    
    def __init__(self):
        self.factors: Dict[str, FactorResult] = {}
        self.history: List[FactorResult] = []
    
    def add(self, factor: FactorResult) -> bool:
        """添加因子"""
        if factor.name in self.factors:
            return False
        
        self.factors[factor.name] = factor
        self.history.append(factor)
        return True
    
    def get(self, name: str) -> Optional[FactorResult]:
        """获取因子"""
        return self.factors.get(name)
    
    def get_valid_factors(self) -> List[FactorResult]:
        """获取所有有效因子"""
        return [f for f in self.factors.values() if f.is_valid()]
    
    def get_excellent_factors(self) -> List[FactorResult]:
        """获取优秀因子"""
        return [f for f in self.factors.values() if f.status == FactorStatus.EXCELLENT]
    
    def get_top_factors(self, n: int = 10, metric: str = 'ic') -> List[FactorResult]:
        """获取Top N因子"""
        valid = self.get_valid_factors()
        sorted_factors = sorted(valid, key=lambda x: abs(getattr(x, metric, 0)), reverse=True)
        return sorted_factors[:n]
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        records = [f.to_dict() for f in self.factors.values()]
        return pd.DataFrame(records)
    
    def __len__(self) -> int:
        return len(self.factors)
    
    def __contains__(self, name: str) -> bool:
        return name in self.factors
