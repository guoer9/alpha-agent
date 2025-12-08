"""
GraphStore - 图存储层

支持两种后端:
1. 内存存储 (开发/测试)
2. Neo4j存储 (生产)
"""

import json
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import asdict
from pathlib import Path
from abc import ABC, abstractmethod

from .schema import (
    NodeType, EdgeType,
    BaseNode, FactorNode, ReflectionNode, RegimeNode, ConceptNode, DataFieldNode,
    GraphEdge,
    PREDEFINED_CONCEPTS, PREDEFINED_REGIMES,
)

logger = logging.getLogger(__name__)


class BaseGraphStore(ABC):
    """图存储基类"""
    
    @abstractmethod
    def add_node(self, node: BaseNode) -> str:
        """添加节点"""
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """获取节点"""
        pass
    
    @abstractmethod
    def update_node(self, node_id: str, updates: Dict) -> bool:
        """更新节点"""
        pass
    
    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """删除节点"""
        pass
    
    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> str:
        """添加边"""
        pass
    
    @abstractmethod
    def get_edges(self, node_id: str, edge_type: EdgeType = None) -> List[GraphEdge]:
        """获取节点的边"""
        pass
    
    @abstractmethod
    def query_nodes(self, node_type: NodeType = None, filters: Dict = None) -> List[BaseNode]:
        """查询节点"""
        pass


class InMemoryGraphStore(BaseGraphStore):
    """内存图存储 - 用于开发和测试"""
    
    def __init__(self, storage_path: str = None):
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.storage_path = storage_path
        
        # 初始化预定义节点
        self._init_predefined()
    
    def _init_predefined(self):
        """初始化预定义节点"""
        for concept in PREDEFINED_CONCEPTS:
            self.nodes[concept.id] = concept
        
        for regime in PREDEFINED_REGIMES:
            self.nodes[regime.id] = regime
    
    def add_node(self, node: BaseNode) -> str:
        """添加节点"""
        if node.id in self.nodes:
            logger.warning(f"节点 {node.id} 已存在，将覆盖")
        self.nodes[node.id] = node
        return node.id
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def update_node(self, node_id: str, updates: Dict) -> bool:
        """更新节点"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)
        
        return True
    
    def delete_node(self, node_id: str) -> bool:
        """删除节点"""
        if node_id not in self.nodes:
            return False
        
        del self.nodes[node_id]
        
        # 删除相关边
        edges_to_delete = [
            eid for eid, edge in self.edges.items()
            if edge.source_id == node_id or edge.target_id == node_id
        ]
        for eid in edges_to_delete:
            del self.edges[eid]
        
        return True
    
    def add_edge(self, edge: GraphEdge) -> str:
        """添加边"""
        self.edges[edge.id] = edge
        return edge.id
    
    def get_edges(self, node_id: str, edge_type: EdgeType = None) -> List[GraphEdge]:
        """获取节点的边"""
        result = []
        for edge in self.edges.values():
            if edge.source_id == node_id or edge.target_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    result.append(edge)
        return result
    
    def query_nodes(
        self, 
        node_type: NodeType = None, 
        filters: Dict = None
    ) -> List[BaseNode]:
        """查询节点"""
        result = []
        
        for node in self.nodes.values():
            # 类型过滤
            if node_type and node.node_type != node_type:
                continue
            
            # 属性过滤
            if filters:
                match = True
                for key, value in filters.items():
                    if not hasattr(node, key):
                        match = False
                        break
                    node_value = getattr(node, key)
                    if isinstance(value, list):
                        # 列表包含检查
                        if node_value not in value:
                            match = False
                            break
                    else:
                        if node_value != value:
                            match = False
                            break
                if not match:
                    continue
            
            result.append(node)
        
        return result
    
    def get_neighbors(
        self, 
        node_id: str, 
        edge_type: EdgeType = None,
        direction: str = "both"  # "out", "in", "both"
    ) -> List[BaseNode]:
        """获取邻居节点"""
        neighbor_ids = set()
        
        for edge in self.edges.values():
            if direction in ("out", "both") and edge.source_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    neighbor_ids.add(edge.target_id)
            
            if direction in ("in", "both") and edge.target_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    neighbor_ids.add(edge.source_id)
        
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]
    
    def get_path(
        self, 
        start_id: str, 
        end_id: str, 
        max_depth: int = 3
    ) -> List[List[str]]:
        """查找两节点间的路径 (BFS)"""
        if start_id == end_id:
            return [[start_id]]
        
        paths = []
        queue = [(start_id, [start_id])]
        visited = {start_id}
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for neighbor in self.get_neighbors(current):
                if neighbor.id == end_id:
                    paths.append(path + [end_id])
                elif neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [neighbor.id]))
        
        return paths
    
    def save(self, path: str = None):
        """保存到文件"""
        save_path = path or self.storage_path
        if not save_path:
            raise ValueError("未指定存储路径")
        
        data = {
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': {eid: edge.to_dict() for eid, edge in self.edges.items()},
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"图保存到 {save_path}")
    
    def load(self, path: str = None):
        """从文件加载"""
        load_path = path or self.storage_path
        if not load_path or not Path(load_path).exists():
            logger.warning(f"文件不存在: {load_path}")
            return
        
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建节点
        self.nodes.clear()
        for nid, node_data in data.get('nodes', {}).items():
            node_type = NodeType(node_data.pop('node_type', 'Factor'))
            node_class = {
                NodeType.FACTOR: FactorNode,
                NodeType.REFLECTION: ReflectionNode,
                NodeType.REGIME: RegimeNode,
                NodeType.CONCEPT: ConceptNode,
                NodeType.DATA_FIELD: DataFieldNode,
            }.get(node_type, FactorNode)
            
            # 过滤无效字段
            valid_fields = {f.name for f in node_class.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in node_data.items() if k in valid_fields}
            
            self.nodes[nid] = node_class(**filtered_data)
        
        # 重建边
        self.edges.clear()
        for eid, edge_data in data.get('edges', {}).items():
            edge_data['edge_type'] = EdgeType(edge_data.get('edge_type', 'CORRELATES_WITH'))
            self.edges[eid] = GraphEdge(**edge_data)
        
        logger.info(f"从 {load_path} 加载 {len(self.nodes)} 节点, {len(self.edges)} 边")
    
    def stats(self) -> Dict:
        """统计信息"""
        node_counts = {}
        for node in self.nodes.values():
            t = node.node_type.value
            node_counts[t] = node_counts.get(t, 0) + 1
        
        edge_counts = {}
        for edge in self.edges.values():
            t = edge.edge_type.value
            edge_counts[t] = edge_counts.get(t, 0) + 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_counts': node_counts,
            'edge_counts': edge_counts,
        }


class Neo4jGraphStore(BaseGraphStore):
    """Neo4j图存储 - 生产环境"""
    
    def __init__(
        self, 
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
        self._connect()
    
    def _connect(self):
        """连接Neo4j"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"已连接到 Neo4j: {self.uri}")
        except ImportError:
            logger.warning("neo4j-driver未安装: pip install neo4j")
            self.driver = None
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            self.driver = None
    
    def _run_query(self, query: str, params: Dict = None) -> List[Dict]:
        """执行Cypher查询"""
        if not self.driver:
            raise RuntimeError("Neo4j未连接")
        
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    
    def add_node(self, node: BaseNode) -> str:
        """添加节点"""
        label = node.node_type.value
        props = node.to_dict()
        props.pop('embedding', None)  # 不存embedding
        
        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET n += $props
        RETURN n.id as id
        """
        
        result = self._run_query(query, {'id': node.id, 'props': props})
        return result[0]['id'] if result else node.id
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """获取节点"""
        query = """
        MATCH (n {id: $id})
        RETURN n, labels(n) as labels
        """
        
        result = self._run_query(query, {'id': node_id})
        if not result:
            return None
        
        data = dict(result[0]['n'])
        labels = result[0]['labels']
        
        node_type = NodeType(labels[0]) if labels else NodeType.FACTOR
        node_class = {
            NodeType.FACTOR: FactorNode,
            NodeType.REFLECTION: ReflectionNode,
            NodeType.REGIME: RegimeNode,
            NodeType.CONCEPT: ConceptNode,
        }.get(node_type, FactorNode)
        
        return node_class(**data)
    
    def update_node(self, node_id: str, updates: Dict) -> bool:
        """更新节点"""
        query = """
        MATCH (n {id: $id})
        SET n += $updates
        RETURN n.id as id
        """
        
        result = self._run_query(query, {'id': node_id, 'updates': updates})
        return len(result) > 0
    
    def delete_node(self, node_id: str) -> bool:
        """删除节点"""
        query = """
        MATCH (n {id: $id})
        DETACH DELETE n
        RETURN count(*) as deleted
        """
        
        result = self._run_query(query, {'id': node_id})
        return result[0]['deleted'] > 0 if result else False
    
    def add_edge(self, edge: GraphEdge) -> str:
        """添加边"""
        rel_type = edge.edge_type.value
        
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.id = $edge_id, r.weight = $weight, r.properties = $properties
        RETURN r.id as id
        """
        
        params = {
            'source_id': edge.source_id,
            'target_id': edge.target_id,
            'edge_id': edge.id,
            'weight': edge.weight,
            'properties': json.dumps(edge.properties),
        }
        
        result = self._run_query(query, params)
        return result[0]['id'] if result else edge.id
    
    def get_edges(self, node_id: str, edge_type: EdgeType = None) -> List[GraphEdge]:
        """获取节点的边"""
        if edge_type:
            query = f"""
            MATCH (n {{id: $id}})-[r:{edge_type.value}]-(m)
            RETURN r, n.id as source, m.id as target, type(r) as rel_type
            """
        else:
            query = """
            MATCH (n {id: $id})-[r]-(m)
            RETURN r, n.id as source, m.id as target, type(r) as rel_type
            """
        
        result = self._run_query(query, {'id': node_id})
        
        edges = []
        for record in result:
            edges.append(GraphEdge(
                id=record['r'].get('id', ''),
                source_id=record['source'],
                target_id=record['target'],
                edge_type=EdgeType(record['rel_type']),
                weight=record['r'].get('weight', 1.0),
            ))
        
        return edges
    
    def query_nodes(
        self, 
        node_type: NodeType = None, 
        filters: Dict = None
    ) -> List[BaseNode]:
        """查询节点"""
        if node_type:
            query = f"MATCH (n:{node_type.value})"
        else:
            query = "MATCH (n)"
        
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"n.{key} = ${key}")
            query += " WHERE " + " AND ".join(conditions)
        
        query += " RETURN n, labels(n) as labels"
        
        result = self._run_query(query, filters or {})
        
        nodes = []
        for record in result:
            data = dict(record['n'])
            labels = record['labels']
            nt = NodeType(labels[0]) if labels else NodeType.FACTOR
            
            node_class = {
                NodeType.FACTOR: FactorNode,
                NodeType.REFLECTION: ReflectionNode,
                NodeType.REGIME: RegimeNode,
                NodeType.CONCEPT: ConceptNode,
            }.get(nt, FactorNode)
            
            nodes.append(node_class(**data))
        
        return nodes
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()


# ============================================================
# 工厂函数
# ============================================================

def GraphStore(
    backend: str = "memory",
    **kwargs
) -> BaseGraphStore:
    """
    创建图存储实例
    
    Args:
        backend: "memory" 或 "neo4j"
        **kwargs: 后端特定参数
    """
    if backend == "memory":
        return InMemoryGraphStore(**kwargs)
    elif backend == "neo4j":
        return Neo4jGraphStore(**kwargs)
    else:
        raise ValueError(f"未知后端: {backend}")
