"""
GraphRetriever - 图检索器

支持多种检索模式:
1. 相似因子检索
2. 多跳推理
3. 子图提取
4. 关系路径检索
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .schema import (
    NodeType, EdgeType,
    FactorNode, ReflectionNode, RegimeNode, ConceptNode,
    GraphEdge,
)
from .store import BaseGraphStore, InMemoryGraphStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    nodes: List[Any]
    edges: List[GraphEdge]
    paths: List[List[str]]
    score: float = 0.0
    explanation: str = ""


class GraphRetriever:
    """图检索器"""
    
    def __init__(self, store: BaseGraphStore):
        self.store = store
    
    # ============================================================
    # 因子检索
    # ============================================================
    
    def find_similar_factors(
        self,
        factor_id: str = None,
        category: str = None,
        tags: List[str] = None,
        min_ic: float = None,
        max_turnover: float = None,
        limit: int = 10,
    ) -> List[FactorNode]:
        """
        查找相似因子
        
        Args:
            factor_id: 基准因子ID（找与它相关的）
            category: 类别过滤
            tags: 标签过滤
            min_ic: 最小IC
            max_turnover: 最大换手率
            limit: 返回数量
        """
        # 如果指定了factor_id，先找相关因子
        if factor_id:
            neighbors = self.store.get_neighbors(
                factor_id, 
                edge_type=EdgeType.SIMILAR_TO,
                direction="both"
            )
            factors = [n for n in neighbors if isinstance(n, FactorNode)]
        else:
            # 全量查询
            factors = self.store.query_nodes(node_type=NodeType.FACTOR)
        
        # 过滤
        result = []
        for f in factors:
            if category and f.category != category:
                continue
            if tags and not any(t in f.tags for t in tags):
                continue
            if min_ic and f.ic < min_ic:
                continue
            if max_turnover and f.turnover > max_turnover:
                continue
            result.append(f)
        
        # 按IC排序
        result.sort(key=lambda x: x.ic, reverse=True)
        
        return result[:limit]
    
    def find_factors_by_concept(
        self,
        concept_id: str,
        include_subconcepts: bool = True,
    ) -> List[FactorNode]:
        """查找某概念下的所有因子"""
        concept_ids = {concept_id}
        
        # 包含子概念
        if include_subconcepts:
            for node in self.store.query_nodes(node_type=NodeType.CONCEPT):
                if isinstance(node, ConceptNode) and node.parent_concept == concept_id:
                    concept_ids.add(node.id)
        
        # 找属于这些概念的因子
        factors = []
        for factor in self.store.query_nodes(node_type=NodeType.FACTOR):
            edges = self.store.get_edges(factor.id, EdgeType.BELONGS_TO)
            for edge in edges:
                if edge.target_id in concept_ids:
                    factors.append(factor)
                    break
        
        return factors
    
    def find_factors_for_regime(
        self,
        regime_id: str,
        success_only: bool = True,
    ) -> List[Tuple[FactorNode, float]]:
        """
        查找在某市场状态下表现好的因子
        
        Returns:
            [(factor, score), ...]
        """
        edge_type = EdgeType.SUCCEEDED_IN if success_only else None
        
        result = []
        for factor in self.store.query_nodes(node_type=NodeType.FACTOR):
            edges = self.store.get_edges(factor.id, edge_type)
            for edge in edges:
                if edge.target_id == regime_id:
                    result.append((factor, edge.weight))
                    break
        
        # 按分数排序
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
    
    # ============================================================
    # 反思检索
    # ============================================================
    
    def get_factor_reflections(
        self,
        factor_id: str,
        reflection_type: str = None,  # success/failure/neutral
    ) -> List[ReflectionNode]:
        """获取因子的反思记录"""
        reflections = []
        
        edges = self.store.get_edges(factor_id, EdgeType.HAS_REFLECTION)
        for edge in edges:
            node = self.store.get_node(edge.target_id)
            if isinstance(node, ReflectionNode):
                if reflection_type is None or node.reflection_type == reflection_type:
                    reflections.append(node)
        
        # 按时间排序（新的在前）
        reflections.sort(key=lambda x: x.created_at, reverse=True)
        
        return reflections
    
    def find_similar_failures(
        self,
        factor_id: str,
        limit: int = 5,
    ) -> List[Tuple[FactorNode, ReflectionNode]]:
        """
        查找有类似失败模式的因子
        
        用于：避免重复犯错
        """
        # 获取当前因子的失败反思
        my_failures = self.get_factor_reflections(factor_id, reflection_type="failure")
        if not my_failures:
            return []
        
        my_keywords = set()
        for ref in my_failures:
            # 简单提取关键词
            for word in ref.diagnosis.split():
                if len(word) > 2:
                    my_keywords.add(word.lower())
        
        # 查找其他因子的失败反思
        similar = []
        for factor in self.store.query_nodes(node_type=NodeType.FACTOR):
            if factor.id == factor_id:
                continue
            
            failures = self.get_factor_reflections(factor.id, reflection_type="failure")
            for ref in failures:
                ref_keywords = set(w.lower() for w in ref.diagnosis.split() if len(w) > 2)
                overlap = len(my_keywords & ref_keywords)
                if overlap > 2:
                    similar.append((factor, ref, overlap))
        
        # 按相似度排序
        similar.sort(key=lambda x: x[2], reverse=True)
        
        return [(f, r) for f, r, _ in similar[:limit]]
    
    # ============================================================
    # 多跳推理
    # ============================================================
    
    def multi_hop_query(
        self,
        start_id: str,
        edge_types: List[EdgeType],
        max_depth: int = 3,
    ) -> RetrievalResult:
        """
        多跳查询
        
        例: Factor -[FAILED_IN]-> Regime -[SUCCEEDED_IN]-> OtherFactor
        找在同一市场状态下成功的其他因子
        """
        visited = {start_id}
        current_level = [start_id]
        all_nodes = [self.store.get_node(start_id)]
        all_edges = []
        paths = [[start_id]]
        
        for depth, edge_type in enumerate(edge_types[:max_depth]):
            next_level = []
            new_paths = []
            
            for node_id in current_level:
                edges = self.store.get_edges(node_id, edge_type)
                
                for edge in edges:
                    # 确定下一个节点
                    next_id = edge.target_id if edge.source_id == node_id else edge.source_id
                    
                    if next_id not in visited:
                        visited.add(next_id)
                        next_level.append(next_id)
                        
                        node = self.store.get_node(next_id)
                        if node:
                            all_nodes.append(node)
                        all_edges.append(edge)
                        
                        # 更新路径
                        for path in paths:
                            if path[-1] == node_id:
                                new_paths.append(path + [next_id])
            
            current_level = next_level
            if new_paths:
                paths = new_paths
        
        return RetrievalResult(
            nodes=all_nodes,
            edges=all_edges,
            paths=paths,
        )
    
    def find_improvement_chain(
        self,
        factor_id: str,
        max_depth: int = 5,
    ) -> List[FactorNode]:
        """
        追溯因子改进链
        
        Factor -> IMPROVES -> BetterFactor -> IMPROVES -> ...
        """
        chain = []
        current_id = factor_id
        visited = set()
        
        while len(chain) < max_depth:
            if current_id in visited:
                break
            visited.add(current_id)
            
            node = self.store.get_node(current_id)
            if not isinstance(node, FactorNode):
                break
            
            chain.append(node)
            
            # 找改进自它的因子
            edges = self.store.get_edges(current_id, EdgeType.IMPROVES)
            improved = None
            best_fitness = node.fitness
            
            for edge in edges:
                if edge.target_id == current_id:  # 有因子改进自它
                    candidate = self.store.get_node(edge.source_id)
                    if isinstance(candidate, FactorNode) and candidate.fitness > best_fitness:
                        improved = candidate
                        best_fitness = candidate.fitness
            
            if improved:
                current_id = improved.id
            else:
                break
        
        return chain
    
    # ============================================================
    # 子图提取
    # ============================================================
    
    def extract_factor_subgraph(
        self,
        factor_id: str,
        include_reflections: bool = True,
        include_regimes: bool = True,
        include_concepts: bool = True,
        depth: int = 1,
    ) -> RetrievalResult:
        """
        提取因子的相关子图
        
        用于：为LLM提供完整上下文
        """
        nodes = []
        edges = []
        visited = set()
        
        def collect(node_id: str, current_depth: int):
            if node_id in visited or current_depth > depth:
                return
            visited.add(node_id)
            
            node = self.store.get_node(node_id)
            if not node:
                return
            nodes.append(node)
            
            # 获取相关边
            for edge in self.store.get_edges(node_id):
                # 过滤边类型
                if not include_reflections and edge.edge_type == EdgeType.HAS_REFLECTION:
                    continue
                if not include_regimes and edge.edge_type in (EdgeType.FAILED_IN, EdgeType.SUCCEEDED_IN):
                    continue
                if not include_concepts and edge.edge_type == EdgeType.BELONGS_TO:
                    continue
                
                edges.append(edge)
                
                # 递归
                next_id = edge.target_id if edge.source_id == node_id else edge.source_id
                collect(next_id, current_depth + 1)
        
        collect(factor_id, 0)
        
        return RetrievalResult(
            nodes=nodes,
            edges=edges,
            paths=[],
        )
    
    # ============================================================
    # 生成LLM Prompt上下文
    # ============================================================
    
    def generate_context_for_llm(
        self,
        query: str,
        factor_ids: List[str] = None,
        regime_id: str = None,
        max_factors: int = 5,
        max_reflections: int = 3,
    ) -> str:
        """
        为LLM生成结构化上下文
        
        Args:
            query: 用户查询
            factor_ids: 相关因子ID
            regime_id: 当前市场状态
            max_factors: 最多返回因子数
            max_reflections: 每个因子最多反思数
        """
        context_parts = []
        
        # 1. 相关因子
        if factor_ids:
            context_parts.append("## 相关因子")
            for fid in factor_ids[:max_factors]:
                factor = self.store.get_node(fid)
                if isinstance(factor, FactorNode):
                    context_parts.append(f"""
### {factor.name} ({factor.id})
- 类别: {factor.category}
- IC: {factor.ic:.4f}, ICIR: {factor.icir:.4f}
- 换手率: {factor.turnover:.1%}
- 来源: {factor.source} / {factor.reference}
- 代码:
```python
{factor.code[:500]}
```
""")
                    
                    # 反思
                    reflections = self.get_factor_reflections(fid)
                    if reflections:
                        context_parts.append("#### 历史反思")
                        for ref in reflections[:max_reflections]:
                            context_parts.append(f"- [{ref.reflection_type}] {ref.summary}")
        
        # 2. 市场状态
        if regime_id:
            regime = self.store.get_node(regime_id)
            if isinstance(regime, RegimeNode):
                context_parts.append(f"""
## 当前市场状态
- 名称: {regime.name}
- 趋势: {regime.trend}
- 波动: {regime.volatility}
- 流动性: {regime.liquidity}
""")
                
                # 该状态下成功的因子
                success_factors = self.find_factors_for_regime(regime_id, success_only=True)
                if success_factors:
                    context_parts.append("### 该状态下表现好的因子")
                    for f, score in success_factors[:3]:
                        context_parts.append(f"- {f.name} (score={score:.2f})")
        
        # 3. 概念关系
        concepts = self.store.query_nodes(node_type=NodeType.CONCEPT)
        if concepts:
            context_parts.append("\n## 策略概念")
            for c in concepts[:5]:
                if isinstance(c, ConceptNode):
                    context_parts.append(f"- **{c.name}**: {c.description}")
        
        return "\n".join(context_parts)
    
    # ============================================================
    # 统计
    # ============================================================
    
    def get_factor_stats(self, factor_id: str) -> Dict:
        """获取因子统计"""
        factor = self.store.get_node(factor_id)
        if not isinstance(factor, FactorNode):
            return {}
        
        edges = self.store.get_edges(factor_id)
        
        stats = {
            'id': factor_id,
            'name': factor.name,
            'ic': factor.ic,
            'fitness': factor.fitness,
            'total_edges': len(edges),
            'correlations': sum(1 for e in edges if e.edge_type == EdgeType.CORRELATES_WITH),
            'reflections': sum(1 for e in edges if e.edge_type == EdgeType.HAS_REFLECTION),
            'regime_relations': sum(1 for e in edges if e.edge_type in (EdgeType.SUCCEEDED_IN, EdgeType.FAILED_IN)),
        }
        
        return stats
