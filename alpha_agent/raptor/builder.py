"""
RAPTOR Builder - 树构建器

构建流程:
1. L0: 从因子库导入叶子节点
2. L1: 聚类形成因子簇
3. L2: 聚类形成策略类型
4. L3: 汇总生成全局摘要
"""

import logging
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np

from .tree import RaptorTree, TreeNode

logger = logging.getLogger(__name__)


@dataclass
class BuildConfig:
    """构建配置"""
    # 聚类参数
    l1_clusters: int = 10       # L1 聚类数
    l2_clusters: int = 5        # L2 聚类数
    min_cluster_size: int = 3   # 最小簇大小
    
    # 摘要参数
    max_summary_length: int = 200
    
    # 嵌入
    embedding_dim: int = 768
    use_embeddings: bool = True


class RaptorBuilder:
    """RAPTOR树构建器"""
    
    def __init__(
        self,
        config: BuildConfig = None,
        embedder: Callable = None,      # 文本 -> 向量
        summarizer: Callable = None,    # 文本列表 -> 摘要
    ):
        self.config = config or BuildConfig()
        self.embedder = embedder or self._default_embedder
        self.summarizer = summarizer or self._default_summarizer
        self.tree = RaptorTree()
    
    def _default_embedder(self, text: str) -> List[float]:
        """默认嵌入器 (简单哈希，生产应用LLM embedding)"""
        # 简单的TF-IDF风格伪向量
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        return np.random.randn(self.config.embedding_dim).tolist()
    
    def _default_summarizer(self, texts: List[str]) -> str:
        """默认摘要器 (简单拼接，生产应用LLM)"""
        combined = " | ".join(texts[:5])
        return combined[:self.config.max_summary_length]
    
    # ============================================================
    # L0: 导入因子
    # ============================================================
    
    def import_factors(self, factors: List[Dict]) -> int:
        """
        导入因子作为叶子节点
        
        Args:
            factors: [{'id': ..., 'name': ..., 'code': ..., 'description': ...}, ...]
        
        Returns:
            导入数量
        """
        count = 0
        
        for factor in factors:
            node = TreeNode(
                level=0,
                factor_id=factor.get('id', ''),
                factor_name=factor.get('name', ''),
                factor_code=factor.get('code', '')[:500],
                text=f"{factor.get('name', '')} - {factor.get('description', '')}",
                metadata={
                    'category': factor.get('category', ''),
                    'tags': factor.get('tags', []),
                    'ic': factor.get('ic', 0),
                    'source': factor.get('source', ''),
                },
            )
            
            # 计算embedding
            if self.config.use_embeddings:
                node.embedding = self.embedder(node.text + " " + node.factor_code)
            
            self.tree.add_node(node)
            count += 1
        
        logger.info(f"导入 {count} 个因子作为L0叶子节点")
        return count
    
    def import_from_factor_library(self, library) -> int:
        """从FactorLibrary导入"""
        factors = []
        for record in library.factors.values():
            factors.append({
                'id': record.id,
                'name': record.name,
                'code': record.code,
                'description': record.description,
                'category': record.category,
                'tags': record.tags,
                'ic': record.ic,
                'source': record.source,
            })
        
        return self.import_factors(factors)
    
    # ============================================================
    # L1: 因子簇聚类
    # ============================================================
    
    def build_level1(self) -> int:
        """构建L1层 - 因子簇"""
        l0_nodes = self.tree.get_level_nodes(0)
        
        if len(l0_nodes) < self.config.min_cluster_size:
            logger.warning(f"L0节点数 ({len(l0_nodes)}) 太少，跳过L1构建")
            return 0
        
        # 聚类
        clusters = self._cluster_nodes(l0_nodes, self.config.l1_clusters)
        
        count = 0
        for cluster_id, cluster_nodes in clusters.items():
            if len(cluster_nodes) < self.config.min_cluster_size:
                continue
            
            # 生成簇摘要
            texts = [n.text for n in cluster_nodes]
            summary = self.summarizer(texts)
            
            # 确定簇名称 (取最常见的category)
            categories = [n.metadata.get('category', 'unknown') for n in cluster_nodes]
            main_category = max(set(categories), key=categories.count) if categories else 'mixed'
            
            # 创建L1节点
            l1_node = TreeNode(
                level=1,
                text=f"{main_category}因子簇 ({len(cluster_nodes)}个)",
                summary=summary,
                cluster_id=cluster_id,
                metadata={
                    'main_category': main_category,
                    'factor_count': len(cluster_nodes),
                    'avg_ic': np.mean([n.metadata.get('ic', 0) for n in cluster_nodes]),
                },
            )
            
            if self.config.use_embeddings:
                l1_node.embedding = self.embedder(summary)
            
            self.tree.add_node(l1_node)
            
            # 建立父子关系
            for child_node in cluster_nodes:
                self.tree.link_parent_child(l1_node.id, child_node.id)
                child_node.cluster_id = cluster_id
            
            count += 1
        
        logger.info(f"构建 {count} 个L1因子簇")
        return count
    
    # ============================================================
    # L2: 策略类型聚类
    # ============================================================
    
    def build_level2(self) -> int:
        """构建L2层 - 策略类型"""
        l1_nodes = self.tree.get_level_nodes(1)
        
        if len(l1_nodes) < 2:
            logger.warning(f"L1节点数 ({len(l1_nodes)}) 太少，跳过L2构建")
            return 0
        
        # 聚类
        clusters = self._cluster_nodes(l1_nodes, min(self.config.l2_clusters, len(l1_nodes)))
        
        # 预定义策略类型
        strategy_names = {
            0: "动量策略",
            1: "价值策略",
            2: "质量策略",
            3: "波动率策略",
            4: "量价策略",
        }
        
        count = 0
        for cluster_id, cluster_nodes in clusters.items():
            # 生成策略摘要
            texts = [n.summary or n.text for n in cluster_nodes]
            summary = self.summarizer(texts)
            
            # 策略名称
            strategy_name = strategy_names.get(cluster_id, f"策略类型{cluster_id}")
            
            # 创建L2节点
            l2_node = TreeNode(
                level=2,
                text=strategy_name,
                summary=summary,
                cluster_id=cluster_id,
                metadata={
                    'cluster_count': len(cluster_nodes),
                    'total_factors': sum(n.metadata.get('factor_count', 0) for n in cluster_nodes),
                },
            )
            
            if self.config.use_embeddings:
                l2_node.embedding = self.embedder(summary)
            
            self.tree.add_node(l2_node)
            
            # 建立父子关系
            for child_node in cluster_nodes:
                self.tree.link_parent_child(l2_node.id, child_node.id)
            
            count += 1
        
        logger.info(f"构建 {count} 个L2策略类型")
        return count
    
    # ============================================================
    # L3: 全局洞察
    # ============================================================
    
    def build_level3(self) -> int:
        """构建L3层 - 全局洞察"""
        l2_nodes = self.tree.get_level_nodes(2)
        
        if not l2_nodes:
            logger.warning("无L2节点，跳过L3构建")
            return 0
        
        # 汇总所有策略
        texts = [f"{n.text}: {n.summary}" for n in l2_nodes]
        global_summary = self.summarizer(texts)
        
        # 统计
        total_factors = len(self.tree.get_level_nodes(0))
        total_clusters = len(self.tree.get_level_nodes(1))
        total_strategies = len(l2_nodes)
        
        # 创建全局根节点
        root_node = TreeNode(
            level=3,
            text="因子库全局洞察",
            summary=f"""
因子库概览:
- 总因子数: {total_factors}
- 因子簇数: {total_clusters}  
- 策略类型: {total_strategies}

策略摘要:
{global_summary}
""",
            metadata={
                'total_factors': total_factors,
                'total_clusters': total_clusters,
                'total_strategies': total_strategies,
            },
        )
        
        if self.config.use_embeddings:
            root_node.embedding = self.embedder(root_node.summary)
        
        self.tree.add_node(root_node)
        
        # 连接L2节点
        for l2_node in l2_nodes:
            self.tree.link_parent_child(root_node.id, l2_node.id)
        
        logger.info("构建L3全局洞察节点")
        return 1
    
    # ============================================================
    # 完整构建
    # ============================================================
    
    def build_full_tree(self, factors: List[Dict] = None, library = None) -> RaptorTree:
        """
        完整构建RAPTOR树
        
        Args:
            factors: 因子列表
            library: FactorLibrary实例
        
        Returns:
            构建好的RaptorTree
        """
        logger.info("开始构建RAPTOR树...")
        
        # L0: 导入因子
        if library:
            self.import_from_factor_library(library)
        elif factors:
            self.import_factors(factors)
        else:
            raise ValueError("需要提供factors或library")
        
        # L1: 因子簇
        self.build_level1()
        
        # L2: 策略类型
        self.build_level2()
        
        # L3: 全局洞察
        self.build_level3()
        
        logger.info(f"RAPTOR树构建完成: {self.tree.stats()}")
        
        return self.tree
    
    # ============================================================
    # 聚类算法
    # ============================================================
    
    def _cluster_nodes(
        self, 
        nodes: List[TreeNode], 
        n_clusters: int
    ) -> Dict[int, List[TreeNode]]:
        """
        对节点进行聚类
        
        使用简单的K-Means (生产可用更复杂的聚类)
        """
        if not nodes:
            return {}
        
        # 提取embedding
        embeddings = []
        valid_nodes = []
        
        for node in nodes:
            if node.embedding:
                embeddings.append(node.embedding)
                valid_nodes.append(node)
            else:
                # 没有embedding的话用简单hash
                emb = self._default_embedder(node.text)
                embeddings.append(emb)
                valid_nodes.append(node)
        
        if not embeddings:
            return {}
        
        X = np.array(embeddings)
        
        # 简单K-Means
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(n_clusters, len(valid_nodes)), random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
        except ImportError:
            # 没有sklearn，用简单随机分配
            logger.warning("sklearn未安装，使用随机聚类")
            labels = np.random.randint(0, n_clusters, size=len(valid_nodes))
        
        # 按cluster_id分组
        clusters = {}
        for node, label in zip(valid_nodes, labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)
        
        return clusters
    
    # ============================================================
    # 增量更新
    # ============================================================
    
    def add_factor_incremental(self, factor: Dict) -> str:
        """
        增量添加因子
        
        1. 添加L0节点
        2. 找最近的L1簇并加入
        3. 更新L1/L2/L3摘要
        """
        # 创建L0节点
        node = TreeNode(
            level=0,
            factor_id=factor.get('id', ''),
            factor_name=factor.get('name', ''),
            factor_code=factor.get('code', '')[:500],
            text=f"{factor.get('name', '')} - {factor.get('description', '')}",
            metadata={
                'category': factor.get('category', ''),
                'tags': factor.get('tags', []),
                'ic': factor.get('ic', 0),
            },
        )
        
        if self.config.use_embeddings:
            node.embedding = self.embedder(node.text)
        
        self.tree.add_node(node)
        
        # 找最近的L1簇
        l1_nodes = self.tree.get_level_nodes(1)
        if l1_nodes and node.embedding:
            best_l1 = None
            best_sim = -1
            
            for l1 in l1_nodes:
                if l1.embedding:
                    sim = self._cosine_similarity(node.embedding, l1.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_l1 = l1
            
            if best_l1:
                self.tree.link_parent_child(best_l1.id, node.id)
                logger.info(f"因子 {node.factor_name} 加入簇 {best_l1.text}")
        
        return node.id
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """余弦相似度"""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
