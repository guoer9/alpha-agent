"""
因子评估模块

基于Qlib框架的完整回测指标体系

支持:
1. IC/ICIR指标计算 + 等级评定
2. 收益风险指标 (夏普/索提诺/卡玛/信息比率)
3. 验证与筛选
4. 报告生成

注意: ML模型回测请使用 modeling/qlib_model_zoo.py 中的 QlibBenchmark
"""

from .metrics import (
    BacktestMetrics,
    ICMetrics,
    ReturnMetrics,
    RiskMetrics,
    compute_all_metrics,
    compute_ic_metrics,
    compute_return_metrics,
    compute_risk_metrics,
)
from .evaluator import FactorEvaluator, EvaluatorConfig

__all__ = [
    # 指标数据类
    'BacktestMetrics',
    'ICMetrics',
    'ReturnMetrics', 
    'RiskMetrics',
    
    # 计算函数
    'compute_all_metrics',
    'compute_ic_metrics',
    'compute_return_metrics',
    'compute_risk_metrics',
    
    # 评估器
    'FactorEvaluator',
    'EvaluatorConfig',
]
