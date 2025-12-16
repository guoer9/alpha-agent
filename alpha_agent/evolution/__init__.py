"""
进化式因子生成引擎

核心思想:
1. LLM生成多个变体（种群初始化）
2. 并行评估所有变体
3. 选择精英 + 生成诊断报告
4. LLM基于反馈改进 → 新变体
5. 迭代直到满足条件
"""

from .config import EvolutionConfig
from .individual import Individual, EvolutionHistory
from .engine import EvolutionaryEngine

__all__ = [
    'EvolutionConfig',
    'Individual',
    'EvolutionHistory',
    'EvolutionaryEngine',
]
