"""
GraphRAG - 因子知识图谱

基于图结构存储因子、反思、市场状态及其关系
支持多跳推理和关系检索
"""

from .schema import (
    NodeType, EdgeType,
    FactorNode, ReflectionNode, RegimeNode, ConceptNode,
    GraphEdge,
)
from .store import GraphStore
from .retriever import GraphRetriever

__all__ = [
    'NodeType', 'EdgeType',
    'FactorNode', 'ReflectionNode', 'RegimeNode', 'ConceptNode',
    'GraphEdge',
    'GraphStore',
    'GraphRetriever',
]
