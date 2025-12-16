"""
核心模块 - 基础组件

包含:
- base: Agent基类和接口定义
- llm: LLM生成器
- sandbox: 安全沙箱执行
- evaluator: 因子评估
"""

from .base import BaseAgent, AgentResult, FactorResult
from .llm import LLMGenerator
from .sandbox import Sandbox, execute_code
from .evaluator import FactorEvaluator, evaluate_factor

__all__ = [
    'BaseAgent', 'AgentResult', 'FactorResult',
    'LLMGenerator',
    'Sandbox', 'execute_code',
    'FactorEvaluator', 'evaluate_factor',
]
