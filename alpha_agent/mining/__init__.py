"""
因子挖掘模块

- GP搜索引擎: 遗传规划因子搜索
- 回测模块: 简单回测 + Qlib完整回测
"""

from .gp_engine import GPEngine
from .backtest import (
    run_backtest, 
    BacktestResult, 
    format_backtest_report,
    run_qlib_backtest,
    run_qlib_weight_backtest,
    run_qlib_factor_analysis,
    plot_backtest_result,
    compute_simple_backtest,
    QLIB_AVAILABLE,
)

__all__ = [
    # GP
    'GPEngine', 
    # 回测
    'run_backtest', 
    'BacktestResult', 
    'format_backtest_report',
    'run_qlib_backtest',
    'run_qlib_weight_backtest',
    'run_qlib_factor_analysis',
    'plot_backtest_result',
    'compute_simple_backtest',
    'QLIB_AVAILABLE',
]
