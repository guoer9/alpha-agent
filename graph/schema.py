"""
GraphRAG Schema - 图节点和边的定义

节点类型:
- Factor: 因子节点
- Reflection: 反思/诊断节点
- Regime: 市场状态节点
- Concept: 概念/策略类型节点
- DataField: 数据字段节点

边类型:
- CORRELATES_WITH: 因子相关性
- DERIVED_FROM: 衍生关系
- HAS_REFLECTION: 因子-反思关系
- FAILED_IN: 在某市场状态失败
- SUCCEEDED_IN: 在某市场状态成功
- BELONGS_TO: 属于某概念
- USES_FIELD: 使用某数据字段
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


class NodeType(Enum):
    """节点类型"""
    FACTOR = "Factor"
    REFLECTION = "Reflection"
    REGIME = "Regime"
    CONCEPT = "Concept"
    DATA_FIELD = "DataField"


class EdgeType(Enum):
    """边类型"""
    # 因子关系
    CORRELATES_WITH = "CORRELATES_WITH"      # 相关性 (带权重)
    DERIVED_FROM = "DERIVED_FROM"            # 衍生自
    SIMILAR_TO = "SIMILAR_TO"                # 相似于
    IMPROVES = "IMPROVES"                    # 改进自
    
    # 反思关系
    HAS_REFLECTION = "HAS_REFLECTION"        # 有反思
    LEARNED_FROM = "LEARNED_FROM"            # 从中学到
    
    # 市场状态关系
    FAILED_IN = "FAILED_IN"                  # 在某状态失败
    SUCCEEDED_IN = "SUCCEEDED_IN"            # 在某状态成功
    SENSITIVE_TO = "SENSITIVE_TO"            # 对某状态敏感
    
    # 概念关系
    BELONGS_TO = "BELONGS_TO"                # 属于某概念
    SUBCATEGORY_OF = "SUBCATEGORY_OF"        # 子类别
    
    # 数据关系
    USES_FIELD = "USES_FIELD"                # 使用某字段


# ============================================================
# 节点定义
# ============================================================

@dataclass
class BaseNode:
    """节点基类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_type: NodeType = NodeType.FACTOR
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""
    
    # 用于向量检索
    embedding: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['node_type'] = self.node_type.value
        return d


@dataclass
class FactorNode(BaseNode):
    """因子节点"""
    node_type: NodeType = NodeType.FACTOR
    
    # 基础信息
    name: str = ""
    name_en: str = ""
    code: str = ""
    description: str = ""
    
    # 来源信息
    source: str = ""               # classic/llm/gp
    reference: str = ""            # 文献出处
    author: str = ""               # 作者
    year: int = 0
    
    # 评估指标
    ic: float = 0.0
    icir: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    fitness: float = 0.0
    
    # 分类
    category: str = ""             # barra_style/technical/fundamental/volume_price
    tags: List[str] = field(default_factory=list)
    
    # 状态
    status: str = "active"         # active/deprecated/experimental
    version: int = 1
    
    def to_cypher_properties(self) -> str:
        """生成Cypher属性字符串"""
        props = {
            'id': self.id,
            'name': self.name,
            'code': self.code[:500],  # 截断
            'ic': self.ic,
            'icir': self.icir,
            'sharpe': self.sharpe,
            'turnover': self.turnover,
            'fitness': self.fitness,
            'category': self.category,
            'source': self.source,
            'status': self.status,
        }
        return str(props)


@dataclass
class ReflectionNode(BaseNode):
    """反思/诊断节点"""
    node_type: NodeType = NodeType.REFLECTION
    
    # 关联因子
    factor_id: str = ""
    
    # 反思内容
    summary: str = ""              # 简要摘要
    diagnosis: str = ""            # 详细诊断
    suggestions: List[str] = field(default_factory=list)  # 改进建议
    
    # 评估时的环境
    eval_period: str = ""          # 评估时间段
    eval_market: str = ""          # 评估市场
    
    # 分类
    reflection_type: str = ""      # success/failure/neutral
    severity: str = ""             # high/medium/low
    
    # 关键指标
    metrics_snapshot: Dict = field(default_factory=dict)


@dataclass
class RegimeNode(BaseNode):
    """市场状态节点"""
    node_type: NodeType = NodeType.REGIME
    
    # 状态标识
    name: str = ""                 # e.g., "牛市上涨", "震荡下跌"
    description: str = ""
    
    # 特征
    volatility: str = ""           # high/medium/low
    trend: str = ""                # up/down/sideways
    liquidity: str = ""            # high/medium/low
    
    # 时间范围
    typical_periods: List[str] = field(default_factory=list)  # ["2020-03", "2022-10"]
    
    # 统计
    avg_return: float = 0.0
    avg_volatility: float = 0.0


@dataclass 
class ConceptNode(BaseNode):
    """概念/策略类型节点"""
    node_type: NodeType = NodeType.CONCEPT
    
    # 概念信息
    name: str = ""                 # e.g., "动量策略", "价值投资"
    description: str = ""
    
    # 层级
    parent_concept: str = ""       # 父概念ID
    level: int = 0                 # 0=根, 1=一级, 2=二级
    
    # 统计
    factor_count: int = 0
    avg_ic: float = 0.0
    
    # 元信息
    aliases: List[str] = field(default_factory=list)  # 别名


@dataclass
class DataFieldNode(BaseNode):
    """数据字段节点"""
    node_type: NodeType = NodeType.DATA_FIELD
    
    name: str = ""                 # e.g., "close", "volume"
    description: str = ""
    data_type: str = ""            # price/volume/fundamental
    
    # 使用统计
    usage_count: int = 0           # 被多少因子使用


# ============================================================
# 边定义
# ============================================================

@dataclass
class GraphEdge:
    """图边"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # 连接
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.CORRELATES_WITH
    
    # 属性
    weight: float = 1.0            # 边权重
    properties: Dict = field(default_factory=dict)
    
    # 时间
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['edge_type'] = self.edge_type.value
        return d


# ============================================================
# 预定义概念
# ============================================================

PREDEFINED_CONCEPTS = [
    ConceptNode(id="concept_momentum", name="动量策略", level=0,
                description="基于价格趋势延续的策略"),
    ConceptNode(id="concept_reversal", name="反转策略", level=0,
                description="基于价格均值回归的策略"),
    ConceptNode(id="concept_value", name="价值投资", level=0,
                description="基于估值低估的策略"),
    ConceptNode(id="concept_quality", name="质量因子", level=0,
                description="基于公司质量的策略"),
    ConceptNode(id="concept_volatility", name="波动率", level=0,
                description="基于波动率的策略"),
    ConceptNode(id="concept_liquidity", name="流动性", level=0,
                description="基于流动性的策略"),
    ConceptNode(id="concept_volume_price", name="量价关系", level=0,
                description="基于成交量和价格关系的策略"),
]

PREDEFINED_REGIMES = [
    RegimeNode(id="regime_bull_low_vol", name="低波动牛市",
               trend="up", volatility="low", liquidity="high"),
    RegimeNode(id="regime_bull_high_vol", name="高波动牛市",
               trend="up", volatility="high", liquidity="high"),
    RegimeNode(id="regime_bear_low_vol", name="低波动熊市",
               trend="down", volatility="low", liquidity="low"),
    RegimeNode(id="regime_bear_high_vol", name="高波动熊市",
               trend="down", volatility="high", liquidity="medium"),
    RegimeNode(id="regime_sideways", name="震荡市",
               trend="sideways", volatility="medium", liquidity="medium"),
]
