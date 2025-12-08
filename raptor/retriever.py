"""
RAPTOR Retriever - 层次化检索器

检索策略:
1. Top-Down: 从L3→L2→L1→L0, 逐层筛选
2. Traversal: 找到匹配节点后遍历相关路径
3. Hybrid: 结合两种策略
"""

import logging
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from .tree import RaptorTree, TreeNode

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """检索配置"""
    strategy: str = "hybrid"      # top_down, traversal, hybrid
    top_k: int = 10               # 每层返回数量
    similarity_threshold: float = 0.5
    include_ancestors: bool = True
    include_siblings: bool = True
    max_depth: int = 3


@dataclass
class RetrievalResult:
    """检索结果"""
    nodes: List[TreeNode]
    scores: List[float]
    paths: List[List[TreeNode]]
    context: str                  # 生成的上下文文本
    
    def to_prompt_context(self) -> str:
        """转换为LLM Prompt上下文"""
        return self.context


class RaptorRetriever:
    """RAPTOR层次化检索器"""
    
    def __init__(
        self,
        tree: RaptorTree,
        embedder: Callable = None,
        config: RetrievalConfig = None,
    ):
        self.tree = tree
        self.embedder = embedder or self._default_embedder
        self.config = config or RetrievalConfig()
    
    def _default_embedder(self, text: str) -> List[float]:
        """默认嵌入器"""
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        return np.random.randn(768).tolist()
    
    # ============================================================
    # 主检索接口
    # ============================================================
    
    def retrieve(
        self,
        query: str,
        strategy: str = None,
        top_k: int = None,
    ) -> RetrievalResult:
        """
        检索相关内容
        
        Args:
            query: 查询文本
            strategy: 检索策略 (top_down/traversal/hybrid)
            top_k: 返回数量
        
        Returns:
            RetrievalResult
        """
        strategy = strategy or self.config.strategy
        top_k = top_k or self.config.top_k
        
        if strategy == "top_down":
            return self._top_down_retrieve(query, top_k)
        elif strategy == "traversal":
            return self._traversal_retrieve(query, top_k)
        else:  # hybrid
            return self._hybrid_retrieve(query, top_k)
    
    # ============================================================
    # Top-Down检索
    # ============================================================
    
    def _top_down_retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """
        自顶向下检索
        
        L3 → L2 → L1 → L0
        每层选择最相关的节点，然后展开其子节点
        """
        query_emb = self.embedder(query)
        
        all_nodes = []
        all_scores = []
        paths = []
        
        # 从L3开始
        current_nodes = self.tree.get_level_nodes(3)
        if not current_nodes:
            current_nodes = self.tree.get_level_nodes(2)
        if not current_nodes:
            current_nodes = self.tree.get_level_nodes(1)
        if not current_nodes:
            current_nodes = self.tree.get_level_nodes(0)
        
        # 逐层向下
        for level in range(3, -1, -1):
            if not current_nodes:
                break
            
            # 计算相似度
            scored = []
            for node in current_nodes:
                score = self._compute_similarity(query_emb, node)
                scored.append((node, score))
            
            # 排序取top
            scored.sort(key=lambda x: x[1], reverse=True)
            top_nodes = scored[:max(1, top_k // (4 - level))]
            
            for node, score in top_nodes:
                if score >= self.config.similarity_threshold:
                    all_nodes.append(node)
                    all_scores.append(score)
            
            # 获取子节点作为下一层候选
            next_nodes = []
            for node, _ in top_nodes:
                next_nodes.extend(self.tree.get_children(node.id))
            current_nodes = next_nodes
        
        # 构建上下文
        context = self._build_context(all_nodes, all_scores, query)
        
        return RetrievalResult(
            nodes=all_nodes[:top_k],
            scores=all_scores[:top_k],
            paths=paths,
            context=context,
        )
    
    # ============================================================
    # Traversal检索
    # ============================================================
    
    def _traversal_retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """
        遍历检索
        
        1. 在所有层找最相关的节点
        2. 从这些节点遍历祖先和后代
        """
        query_emb = self.embedder(query)
        
        # 对所有节点计算相似度
        all_scored = []
        for node in self.tree.nodes.values():
            score = self._compute_similarity(query_emb, node)
            all_scored.append((node, score))
        
        # 排序
        all_scored.sort(key=lambda x: x[1], reverse=True)
        
        # 取top seed nodes
        seed_nodes = [n for n, s in all_scored[:top_k // 2] if s >= self.config.similarity_threshold]
        
        # 遍历相关路径
        all_nodes = []
        all_scores = []
        paths = []
        visited = set()
        
        for seed in seed_nodes:
            if seed.id in visited:
                continue
            visited.add(seed.id)
            all_nodes.append(seed)
            all_scores.append(self._compute_similarity(query_emb, seed))
            
            # 添加祖先
            if self.config.include_ancestors:
                for ancestor in self.tree.get_ancestors(seed.id):
                    if ancestor.id not in visited:
                        visited.add(ancestor.id)
                        all_nodes.append(ancestor)
                        all_scores.append(self._compute_similarity(query_emb, ancestor))
            
            # 添加兄弟
            if self.config.include_siblings:
                for sibling in self.tree.get_siblings(seed.id):
                    if sibling.id not in visited:
                        visited.add(sibling.id)
                        all_nodes.append(sibling)
                        all_scores.append(self._compute_similarity(query_emb, sibling))
            
            # 记录路径
            path = self.tree.get_path_to_root(seed.id)
            if path:
                paths.append(path)
        
        # 重新排序
        combined = list(zip(all_nodes, all_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        all_nodes = [n for n, _ in combined[:top_k]]
        all_scores = [s for _, s in combined[:top_k]]
        
        context = self._build_context(all_nodes, all_scores, query)
        
        return RetrievalResult(
            nodes=all_nodes,
            scores=all_scores,
            paths=paths,
            context=context,
        )
    
    # ============================================================
    # Hybrid检索
    # ============================================================
    
    def _hybrid_retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """
        混合检索
        
        结合Top-Down和Traversal
        """
        # 先用top-down获取结构化路径
        td_result = self._top_down_retrieve(query, top_k // 2)
        
        # 再用traversal补充
        tr_result = self._traversal_retrieve(query, top_k // 2)
        
        # 合并去重
        seen = set()
        all_nodes = []
        all_scores = []
        
        for node, score in zip(td_result.nodes, td_result.scores):
            if node.id not in seen:
                seen.add(node.id)
                all_nodes.append(node)
                all_scores.append(score)
        
        for node, score in zip(tr_result.nodes, tr_result.scores):
            if node.id not in seen:
                seen.add(node.id)
                all_nodes.append(node)
                all_scores.append(score)
        
        # 重新排序
        combined = list(zip(all_nodes, all_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        all_nodes = [n for n, _ in combined[:top_k]]
        all_scores = [s for _, s in combined[:top_k]]
        
        context = self._build_context(all_nodes, all_scores, query)
        
        return RetrievalResult(
            nodes=all_nodes,
            scores=all_scores,
            paths=td_result.paths + tr_result.paths,
            context=context,
        )
    
    # ============================================================
    # 专用检索
    # ============================================================
    
    def retrieve_by_category(self, category: str, top_k: int = 10) -> List[TreeNode]:
        """按类别检索因子"""
        results = []
        
        for node in self.tree.get_level_nodes(0):  # L0 = 因子
            if node.metadata.get('category') == category:
                results.append(node)
        
        return results[:top_k]
    
    def retrieve_by_tags(self, tags: List[str], top_k: int = 10) -> List[TreeNode]:
        """按标签检索因子"""
        results = []
        
        for node in self.tree.get_level_nodes(0):
            node_tags = node.metadata.get('tags', [])
            if any(t in node_tags for t in tags):
                results.append(node)
        
        return results[:top_k]
    
    def retrieve_cluster(self, factor_id: str) -> List[TreeNode]:
        """获取因子所在的簇"""
        node = self.tree.get_node(factor_id)
        if not node:
            return []
        
        # 找到L1父节点
        parent = self.tree.get_parent(factor_id)
        if not parent or parent.level != 1:
            return [node]
        
        # 获取同簇因子
        siblings = self.tree.get_children(parent.id)
        return siblings
    
    def retrieve_strategy_factors(self, strategy_name: str) -> List[TreeNode]:
        """获取某策略类型下的所有因子"""
        # 找到L2策略节点
        for node in self.tree.get_level_nodes(2):
            if strategy_name.lower() in node.text.lower():
                return self.tree.get_leaf_factors(node.id)
        
        return []
    
    # ============================================================
    # 辅助方法
    # ============================================================
    
    def _compute_similarity(self, query_emb: List[float], node: TreeNode) -> float:
        """计算相似度"""
        if not node.embedding:
            # 没有embedding，用文本匹配
            return 0.0
        
        return self._cosine_similarity(query_emb, node.embedding)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """余弦相似度"""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def _build_context(
        self, 
        nodes: List[TreeNode], 
        scores: List[float],
        query: str
    ) -> str:
        """构建LLM上下文"""
        parts = [f"## 查询: {query}\n"]
        
        # 按层级组织
        levels = {0: [], 1: [], 2: [], 3: []}
        for node, score in zip(nodes, scores):
            levels[node.level].append((node, score))
        
        # L3: 全局洞察
        if levels[3]:
            parts.append("### 全局洞察")
            for node, score in levels[3]:
                parts.append(f"- {node.summary[:200]}")
        
        # L2: 策略类型
        if levels[2]:
            parts.append("\n### 相关策略类型")
            for node, score in levels[2]:
                parts.append(f"- **{node.text}** (相关度: {score:.2f})")
                if node.summary:
                    parts.append(f"  {node.summary[:150]}")
        
        # L1: 因子簇
        if levels[1]:
            parts.append("\n### 相关因子簇")
            for node, score in levels[1]:
                parts.append(f"- **{node.text}** (相关度: {score:.2f})")
        
        # L0: 具体因子
        if levels[0]:
            parts.append("\n### 相关因子")
            for node, score in levels[0][:5]:  # 最多显示5个
                parts.append(f"""
#### {node.factor_name} (相关度: {score:.2f})
- 类别: {node.metadata.get('category', 'unknown')}
- IC: {node.metadata.get('ic', 0):.4f}
```python
{node.factor_code[:300]}
```
""")
        
        return "\n".join(parts)
    
    # ============================================================
    # 统计
    # ============================================================
    
    def get_retrieval_stats(self) -> Dict:
        """检索器统计"""
        return {
            'tree_stats': self.tree.stats(),
            'config': {
                'strategy': self.config.strategy,
                'top_k': self.config.top_k,
                'threshold': self.config.similarity_threshold,
            },
        }
