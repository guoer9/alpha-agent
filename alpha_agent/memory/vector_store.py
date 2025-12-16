"""
向量数据库 - Milvus存储因子嵌入

用于:
1. 存储成功因子的代码嵌入
2. 检索相似因子避免重复
3. RAG增强因子生成
"""

import json
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Milvus
try:
    from pymilvus import (
        connections, Collection, FieldSchema, CollectionSchema, DataType,
        utility
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("pymilvus未安装: pip install pymilvus")

# OpenAI Embeddings
try:
    from langchain_openai import OpenAIEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from ..config import vector_db_config


@dataclass
class FactorRecord:
    """因子记录"""
    factor_id: str
    name: str
    code: str
    description: str
    ic: float
    icir: float
    status: str
    tags: List[str]
    embedding: List[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'factor_id': self.factor_id,
            'name': self.name,
            'code': self.code,
            'description': self.description,
            'ic': self.ic,
            'icir': self.icir,
            'status': self.status,
            'tags': self.tags,
        }


class MilvusStore:
    """Milvus向量存储"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
        embedding_dim: int = None,
    ):
        if not MILVUS_AVAILABLE:
            raise ImportError("请安装pymilvus: pip install pymilvus")
        
        self.host = host or vector_db_config.host
        self.port = port or vector_db_config.port
        self.collection_name = collection_name or vector_db_config.collection_name
        self.embedding_dim = embedding_dim or vector_db_config.embedding_dim
        
        self.collection: Optional[Collection] = None
        self._connected = False
    
    def connect(self) -> bool:
        """连接Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
            self._connected = True
            logger.info(f"Milvus连接成功: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self._connected:
            connections.disconnect("default")
            self._connected = False
    
    def create_collection(self) -> bool:
        """创建集合"""
        if not self._connected:
            self.connect()
        
        # 检查是否存在
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"集合已存在: {self.collection_name}")
            return True
        
        # 定义Schema
        fields = [
            FieldSchema(name="factor_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="ic", dtype=DataType.FLOAT),
            FieldSchema(name="icir", dtype=DataType.FLOAT),
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]
        
        schema = CollectionSchema(fields=fields, description="Alpha因子库")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # 创建索引
        index_params = {
            "metric_type": "IP",  # Inner Product
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        
        logger.info(f"集合创建成功: {self.collection_name}")
        return True
    
    def insert(self, records: List[FactorRecord]) -> int:
        """插入记录"""
        if self.collection is None:
            self.create_collection()
        
        data = [
            [r.factor_id for r in records],
            [r.name for r in records],
            [r.code for r in records],
            [r.description for r in records],
            [r.ic for r in records],
            [r.icir for r in records],
            [r.status for r in records],
            [json.dumps(r.tags) for r in records],
            [r.embedding for r in records],
        ]
        
        result = self.collection.insert(data)
        self.collection.flush()
        
        return result.insert_count
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.7,
    ) -> List[Dict]:
        """搜索相似因子"""
        if self.collection is None:
            return []
        
        self.collection.load()
        
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["factor_id", "name", "code", "ic", "icir", "status"]
        )
        
        matches = []
        for hits in results:
            for hit in hits:
                if hit.score >= min_score:
                    matches.append({
                        'factor_id': hit.entity.get('factor_id'),
                        'name': hit.entity.get('name'),
                        'code': hit.entity.get('code'),
                        'ic': hit.entity.get('ic'),
                        'icir': hit.entity.get('icir'),
                        'score': hit.score,
                    })
        
        return matches
    
    def get_all_factors(
        self,
        limit: int = 10000,
        min_ic: float = None,
    ) -> List[Dict]:
        """
        获取所有因子
        
        Args:
            limit: 最大返回数量
            min_ic: 最小IC阈值过滤
        
        Returns:
            因子列表
        """
        if self.collection is None:
            self.create_collection()
        
        self.collection.load()
        
        # 构建查询表达式
        expr = ""
        if min_ic is not None:
            expr = f"ic >= {min_ic}"
        
        try:
            # 指定输出字段 (不含embedding)
            output_fields = [
                "id", "factor_id", "name", "name_en", 
                "category", "source", "code", "description", 
                "ic", "icir"
            ]
            
            results = self.collection.query(
                expr=expr if expr else "id >= 0",
                output_fields=output_fields,
                limit=limit,
            )
            
            factors = []
            for r in results:
                factor = {
                    'id': str(r.get('id', '')),
                    'factor_id': r.get('factor_id', ''),
                    'name': r.get('name', ''),
                    'name_en': r.get('name_en', ''),
                    'category': r.get('category', ''),
                    'source': r.get('source', ''),
                    'code': r.get('code', ''),
                    'description': r.get('description', ''),
                    'ic': r.get('ic', 0) or 0,
                    'icir': r.get('icir', 0) or 0,
                }
                factors.append(factor)
            
            logger.info(f"从Milvus加载 {len(factors)} 个因子")
            return factors
            
        except Exception as e:
            logger.error(f"获取因子失败: {e}")
            return []
    
    def count(self) -> int:
        """获取因子数量"""
        if self.collection is None:
            return 0
        try:
            return self.collection.num_entities
        except Exception:
            return 0
    
    def delete(self, factor_ids: List[str]) -> int:
        """删除记录"""
        if self.collection is None:
            return 0
        
        expr = f"factor_id in {factor_ids}"
        result = self.collection.delete(expr)
        return result.delete_count


class FactorMemory:
    """因子记忆管理器"""
    
    def __init__(self, store: MilvusStore = None):
        self.store = store or MilvusStore()
        self.embeddings = None
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embeddings = OpenAIEmbeddings()
            except Exception as e:
                logger.warning(f"Embeddings初始化失败: {e}")
    
    def _generate_id(self, code: str) -> str:
        """生成因子ID"""
        return hashlib.md5(code.encode()).hexdigest()[:16]
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本嵌入"""
        if self.embeddings is None:
            return None
        
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"获取嵌入失败: {e}")
            return None
    
    def save_factor(
        self,
        name: str,
        code: str,
        description: str = "",
        ic: float = 0.0,
        icir: float = 0.0,
        status: str = "valid",
        tags: List[str] = None,
    ) -> bool:
        """保存因子到记忆库"""
        factor_id = self._generate_id(code)
        
        # 生成嵌入 (基于代码和描述)
        embed_text = f"{description}\n{code}"
        embedding = self._get_embedding(embed_text)
        
        if embedding is None:
            logger.warning("无法生成嵌入，跳过保存")
            return False
        
        record = FactorRecord(
            factor_id=factor_id,
            name=name,
            code=code,
            description=description,
            ic=ic,
            icir=icir,
            status=status,
            tags=tags or [],
            embedding=embedding,
        )
        
        try:
            self.store.insert([record])
            logger.info(f"因子已保存: {name} (IC={ic:.4f})")
            return True
        except Exception as e:
            logger.error(f"保存失败: {e}")
            return False
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """搜索相似因子"""
        embedding = self._get_embedding(query)
        if embedding is None:
            return []
        
        return self.store.search(embedding, top_k=top_k)
    
    def check_duplicate(self, code: str, threshold: float = 0.95) -> Optional[Dict]:
        """检查是否存在重复因子"""
        embedding = self._get_embedding(code)
        if embedding is None:
            return None
        
        matches = self.store.search(embedding, top_k=1, min_score=threshold)
        return matches[0] if matches else None
