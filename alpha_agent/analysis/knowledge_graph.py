"""
风险知识图谱 - Neo4j实现

用于:
1. 存储因子-风险-行业关系
2. 风险暴露分析
3. 因子共线性检测
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j未安装: pip install neo4j")


@dataclass
class RiskNode:
    """风险节点"""
    node_type: str  # factor, risk, industry, stock
    name: str
    properties: Dict = None


@dataclass
class RiskRelation:
    """风险关系"""
    source: str
    target: str
    relation_type: str  # EXPOSED_TO, BELONGS_TO, CORRELATED_WITH
    weight: float = 1.0


class RiskKnowledgeGraph:
    """风险知识图谱 (Neo4j)"""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("请安装neo4j: pip install neo4j")
        
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
    
    def connect(self) -> bool:
        """连接Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # 验证连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Neo4j连接成功: {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
    
    def init_schema(self):
        """初始化Schema"""
        with self.driver.session() as session:
            # 创建约束
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Factor) REQUIRE f.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Risk) REQUIRE r.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE",
            ]
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass
            
            # 创建索引
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (f:Factor) ON (f.ic)",
                "CREATE INDEX IF NOT EXISTS FOR (f:Factor) ON (f.category)",
            ]
            for index in indexes:
                try:
                    session.run(index)
                except Exception:
                    pass
        
        logger.info("Schema初始化完成")
    
    # ==================== 节点操作 ====================
    
    def add_factor(
        self,
        name: str,
        ic: float = 0.0,
        category: str = "unknown",
        description: str = "",
        **properties,
    ) -> bool:
        """添加因子节点"""
        query = """
        MERGE (f:Factor {name: $name})
        SET f.ic = $ic, f.category = $category, f.description = $description
        """
        for key, value in properties.items():
            query += f", f.{key} = ${key}"
        
        try:
            with self.driver.session() as session:
                session.run(query, name=name, ic=ic, category=category, 
                           description=description, **properties)
            return True
        except Exception as e:
            logger.error(f"添加因子失败: {e}")
            return False
    
    def add_risk(self, name: str, risk_type: str = "market", **properties) -> bool:
        """添加风险节点"""
        query = """
        MERGE (r:Risk {name: $name})
        SET r.type = $risk_type
        """
        try:
            with self.driver.session() as session:
                session.run(query, name=name, risk_type=risk_type, **properties)
            return True
        except Exception as e:
            logger.error(f"添加风险失败: {e}")
            return False
    
    def add_industry(self, name: str, sector: str = "", **properties) -> bool:
        """添加行业节点"""
        query = """
        MERGE (i:Industry {name: $name})
        SET i.sector = $sector
        """
        try:
            with self.driver.session() as session:
                session.run(query, name=name, sector=sector, **properties)
            return True
        except Exception as e:
            logger.error(f"添加行业失败: {e}")
            return False
    
    # ==================== 关系操作 ====================
    
    def add_exposure(
        self,
        factor_name: str,
        risk_name: str,
        weight: float = 1.0,
    ) -> bool:
        """添加因子-风险暴露关系"""
        query = """
        MATCH (f:Factor {name: $factor_name})
        MATCH (r:Risk {name: $risk_name})
        MERGE (f)-[rel:EXPOSED_TO]->(r)
        SET rel.weight = $weight
        """
        try:
            with self.driver.session() as session:
                session.run(query, factor_name=factor_name, 
                           risk_name=risk_name, weight=weight)
            return True
        except Exception as e:
            logger.error(f"添加暴露关系失败: {e}")
            return False
    
    def add_correlation(
        self,
        factor1: str,
        factor2: str,
        correlation: float,
    ) -> bool:
        """添加因子相关性关系"""
        query = """
        MATCH (f1:Factor {name: $factor1})
        MATCH (f2:Factor {name: $factor2})
        MERGE (f1)-[rel:CORRELATED_WITH]->(f2)
        SET rel.correlation = $correlation
        """
        try:
            with self.driver.session() as session:
                session.run(query, factor1=factor1, factor2=factor2, 
                           correlation=correlation)
            return True
        except Exception as e:
            logger.error(f"添加相关性失败: {e}")
            return False
    
    # ==================== 查询操作 ====================
    
    def get_factor_risks(self, factor_name: str) -> List[Dict]:
        """获取因子的风险暴露"""
        query = """
        MATCH (f:Factor {name: $factor_name})-[rel:EXPOSED_TO]->(r:Risk)
        RETURN r.name as risk, r.type as risk_type, rel.weight as weight
        ORDER BY rel.weight DESC
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, factor_name=factor_name)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []
    
    def get_correlated_factors(
        self,
        factor_name: str,
        min_correlation: float = 0.7,
    ) -> List[Dict]:
        """获取高相关因子"""
        query = """
        MATCH (f1:Factor {name: $factor_name})-[rel:CORRELATED_WITH]-(f2:Factor)
        WHERE abs(rel.correlation) >= $min_correlation
        RETURN f2.name as factor, f2.ic as ic, rel.correlation as correlation
        ORDER BY abs(rel.correlation) DESC
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, factor_name=factor_name,
                                    min_correlation=min_correlation)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []
    
    def find_risk_clusters(self, risk_name: str) -> List[Dict]:
        """查找共同暴露于某风险的因子群"""
        query = """
        MATCH (f:Factor)-[:EXPOSED_TO]->(r:Risk {name: $risk_name})
        RETURN f.name as factor, f.ic as ic, f.category as category
        ORDER BY abs(f.ic) DESC
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, risk_name=risk_name)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []
    
    def get_factor_graph(self, factor_name: str, depth: int = 2) -> Dict:
        """获取因子的完整关系图"""
        query = """
        MATCH path = (f:Factor {name: $factor_name})-[*1..$depth]-(related)
        RETURN path
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, factor_name=factor_name, depth=depth)
                paths = []
                for record in result:
                    paths.append(str(record['path']))
                return {'factor': factor_name, 'paths': paths}
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {}
    
    # ==================== 批量操作 ====================
    
    def init_risk_taxonomy(self):
        """初始化风险分类体系"""
        risks = [
            ('market', 'market'),
            ('size', 'style'),
            ('value', 'style'),
            ('momentum', 'style'),
            ('volatility', 'style'),
            ('quality', 'style'),
            ('liquidity', 'style'),
            ('beta', 'style'),
            ('industry', 'sector'),
            ('country', 'macro'),
            ('interest_rate', 'macro'),
            ('inflation', 'macro'),
        ]
        
        for risk_name, risk_type in risks:
            self.add_risk(risk_name, risk_type)
        
        logger.info(f"已初始化 {len(risks)} 个风险节点")
