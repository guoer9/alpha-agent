"""
RAPTOR Tree - 层次化树结构

四层金字塔:
- L0: 原始因子 (叶子节点)
- L1: 因子簇 (相似因子聚合，如"短期动量因子群")
- L2: 策略类型 (如"动量策略"、"价值策略")
- L3: 全局洞察 (跨策略的高级知识)
"""

import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """RAPTOR树节点"""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # 层级 (0=叶子/因子, 1=簇, 2=策略, 3=全局)
    level: int = 0
    
    # 内容
    text: str = ""                  # 节点文本/摘要
    summary: str = ""               # LLM生成的摘要
    
    # 关联
    parent_id: str = ""             # 父节点ID
    children_ids: List[str] = field(default_factory=list)  # 子节点ID
    
    # 叶子节点特有 (level=0)
    factor_id: str = ""             # 关联的因子ID
    factor_name: str = ""
    factor_code: str = ""
    
    # 聚类信息
    cluster_id: int = -1            # 所属聚类
    centroid_distance: float = 0.0  # 到聚类中心的距离
    
    # 向量
    embedding: List[float] = field(default_factory=list)
    
    # 元数据
    metadata: Dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        # 不保存embedding到JSON (太大)
        d.pop('embedding', None)
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TreeNode':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def __repr__(self) -> str:
        return f"TreeNode(L{self.level}, id={self.id}, text={self.text[:30]}...)"


class RaptorTree:
    """RAPTOR层次化树"""
    
    def __init__(self, name: str = "factor_tree"):
        self.name = name
        self.nodes: Dict[str, TreeNode] = {}  # id -> node
        self.root_ids: List[str] = []          # L3 根节点
        
        # 层级索引
        self.levels: Dict[int, List[str]] = {
            0: [],  # 因子
            1: [],  # 簇
            2: [],  # 策略
            3: [],  # 全局
        }
    
    def add_node(self, node: TreeNode) -> str:
        """添加节点"""
        self.nodes[node.id] = node
        
        if node.level in self.levels:
            self.levels[node.level].append(node.id)
        
        if node.level == 3:
            self.root_ids.append(node.id)
        
        return node.id
    
    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[TreeNode]:
        """获取子节点"""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]
    
    def get_parent(self, node_id: str) -> Optional[TreeNode]:
        """获取父节点"""
        node = self.get_node(node_id)
        if not node or not node.parent_id:
            return None
        return self.nodes.get(node.parent_id)
    
    def get_level_nodes(self, level: int) -> List[TreeNode]:
        """获取某层所有节点"""
        return [self.nodes[nid] for nid in self.levels.get(level, []) if nid in self.nodes]
    
    def get_ancestors(self, node_id: str) -> List[TreeNode]:
        """获取所有祖先节点 (向上追溯)"""
        ancestors = []
        current = self.get_node(node_id)
        
        while current and current.parent_id:
            parent = self.get_parent(current.id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors
    
    def get_descendants(self, node_id: str) -> List[TreeNode]:
        """获取所有后代节点 (向下展开)"""
        descendants = []
        queue = [node_id]
        
        while queue:
            current_id = queue.pop(0)
            node = self.get_node(current_id)
            if not node:
                continue
            
            for child_id in node.children_ids:
                child = self.get_node(child_id)
                if child:
                    descendants.append(child)
                    queue.append(child_id)
        
        return descendants
    
    def get_leaf_factors(self, node_id: str = None) -> List[TreeNode]:
        """获取叶子因子节点"""
        if node_id:
            # 获取某节点下的所有叶子
            descendants = self.get_descendants(node_id)
            return [d for d in descendants if d.level == 0]
        else:
            # 获取所有叶子
            return self.get_level_nodes(0)
    
    def link_parent_child(self, parent_id: str, child_id: str):
        """建立父子关系"""
        parent = self.get_node(parent_id)
        child = self.get_node(child_id)
        
        if parent and child:
            if child_id not in parent.children_ids:
                parent.children_ids.append(child_id)
            child.parent_id = parent_id
    
    def update_node(self, node_id: str, updates: Dict):
        """更新节点"""
        node = self.get_node(node_id)
        if not node:
            return
        
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)
        
        node.updated_at = datetime.now().isoformat()
    
    def remove_node(self, node_id: str):
        """删除节点"""
        node = self.get_node(node_id)
        if not node:
            return
        
        # 从父节点移除
        if node.parent_id:
            parent = self.get_node(node.parent_id)
            if parent and node_id in parent.children_ids:
                parent.children_ids.remove(node_id)
        
        # 从层级索引移除
        if node.level in self.levels and node_id in self.levels[node.level]:
            self.levels[node.level].remove(node_id)
        
        # 从根移除
        if node_id in self.root_ids:
            self.root_ids.remove(node_id)
        
        del self.nodes[node_id]
    
    # ============================================================
    # 统计
    # ============================================================
    
    def stats(self) -> Dict:
        """树统计"""
        return {
            'name': self.name,
            'total_nodes': len(self.nodes),
            'level_counts': {
                f'L{level}': len(ids) 
                for level, ids in self.levels.items()
            },
            'root_count': len(self.root_ids),
            'avg_children': self._avg_children(),
        }
    
    def _avg_children(self) -> float:
        """平均子节点数"""
        non_leaf = [n for n in self.nodes.values() if n.level > 0]
        if not non_leaf:
            return 0.0
        return sum(len(n.children_ids) for n in non_leaf) / len(non_leaf)
    
    def print_tree(self, node_id: str = None, indent: int = 0):
        """打印树结构"""
        if node_id is None:
            # 从根开始
            for rid in self.root_ids:
                self.print_tree(rid, indent)
            return
        
        node = self.get_node(node_id)
        if not node:
            return
        
        prefix = "  " * indent
        level_name = {0: "因子", 1: "簇", 2: "策略", 3: "全局"}.get(node.level, "?")
        print(f"{prefix}[L{node.level}/{level_name}] {node.text[:50]}...")
        
        for child_id in node.children_ids:
            self.print_tree(child_id, indent + 1)
    
    # ============================================================
    # 持久化
    # ============================================================
    
    def save(self, path: str):
        """保存到文件"""
        data = {
            'name': self.name,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'root_ids': self.root_ids,
            'levels': self.levels,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"RAPTOR树保存到 {path}")
    
    def load(self, path: str):
        """从文件加载"""
        if not Path(path).exists():
            logger.warning(f"文件不存在: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.name = data.get('name', self.name)
        self.root_ids = data.get('root_ids', [])
        self.levels = {int(k): v for k, v in data.get('levels', {}).items()}
        
        self.nodes.clear()
        for nid, node_data in data.get('nodes', {}).items():
            self.nodes[nid] = TreeNode.from_dict(node_data)
        
        logger.info(f"从 {path} 加载 {len(self.nodes)} 个节点")
    
    # ============================================================
    # 查询接口
    # ============================================================
    
    def search_by_text(self, query: str, level: int = None) -> List[TreeNode]:
        """简单文本搜索"""
        query_lower = query.lower()
        results = []
        
        for node in self.nodes.values():
            if level is not None and node.level != level:
                continue
            
            if query_lower in node.text.lower() or query_lower in node.summary.lower():
                results.append(node)
        
        return results
    
    def get_path_to_root(self, node_id: str) -> List[TreeNode]:
        """获取到根的路径"""
        path = []
        current = self.get_node(node_id)
        
        while current:
            path.append(current)
            if not current.parent_id:
                break
            current = self.get_node(current.parent_id)
        
        return path
    
    def get_siblings(self, node_id: str) -> List[TreeNode]:
        """获取兄弟节点"""
        node = self.get_node(node_id)
        if not node or not node.parent_id:
            return []
        
        parent = self.get_node(node.parent_id)
        if not parent:
            return []
        
        return [
            self.nodes[cid] 
            for cid in parent.children_ids 
            if cid != node_id and cid in self.nodes
        ]
