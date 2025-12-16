"""
分析模块

技术栈:
- Neo4j: 风险知识图谱
- Brinson归因
- 市场状态识别
"""

from .knowledge_graph import RiskKnowledgeGraph
from .risk_analysis import RiskAnalyzer, RiskReport
from .attribution import brinson_attribution, factor_attribution, BrinsonResult
from .market_regime import MarketRegimeDetector, MarketState, detect_style_rotation, detect_sector_rotation

__all__ = [
    # 知识图谱
    'RiskKnowledgeGraph',
    # 风险分析
    'RiskAnalyzer', 'RiskReport',
    # 收益归因
    'brinson_attribution', 'factor_attribution', 'BrinsonResult',
    # 市场状态
    'MarketRegimeDetector', 'MarketState', 
    'detect_style_rotation', 'detect_sector_rotation',
]
