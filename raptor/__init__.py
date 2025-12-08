"""
RAPTOR - 递归抽象处理与层次化检索

Recursive Abstractive Processing for Tree-Organized Retrieval

四层金字塔结构:
- L0: 原始因子（叶子节点）
- L1: 因子簇（相似因子聚合）
- L2: 策略类型（动量/反转/价值等）
- L3: 全局洞察（跨策略知识）

检索策略:
- Top-Down: 从高层概念向下
- Traversal: 遍历相关路径
"""

from .tree import RaptorTree, TreeNode
from .retriever import RaptorRetriever, RetrievalConfig
from .builder import RaptorBuilder, BuildConfig

__all__ = [
    'RaptorTree',
    'TreeNode',
    'RaptorRetriever',
    'RetrievalConfig',
    'RaptorBuilder',
    'BuildConfig',
]
