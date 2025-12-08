"""
Prompt组装系统

分层Prompt架构:
1. System Prompt: 定义角色与边界（永恒不变）
2. Schema Context: 数据字典与算子列表（硬约束）
3. RAG Context: 动态检索的高分因子示例（软引导）
4. History/Feedback: 上一轮的错误与回测报告（进化压力）
5. Task Instruction: 当前的具体指令
"""

from .composer import PromptComposer
from .templates import (
    SystemPrompts,
    TaskTemplates,
)

__all__ = [
    'PromptComposer',
    'SystemPrompts',
    'TaskTemplates',
]
