"""
回测模块 - Qlib完整回测集成

支持:
1. 简单分组回测 (无需Qlib)
2. Qlib TopK策略回测
3. Qlib权重策略回测
4. 完整风险分析和报告

参考Qlib文档: https://qlib.readthedocs.io/en/latest/component/backtest.html
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Qlib回测
try:
    import qlib
    from qlib.data import D
    from qlib.backtest import backtest as qlib_backtest, executor
    from qlib.backtest.decision import OrderDir
    from qlib.contrib.strategy import TopkDropoutStrategy, WeightStrategyBase
    from qlib.contrib.evaluate import risk_analysis, backtest_daily
    from qlib.contrib.report import analysis_position, analysis_model
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("Qlib未安装，部分回测功能不可用")


@dataclass
class BacktestResult:
    """回测结果"""
    # 收益
    total_return: float = 0.0
    annual_return: float = 0.0
    excess_return: float = 0.0
    
    # 风险
    volatility: float = 0.0
    max_drawdown: float = 0.0
    
    # 风险调整
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # 交易统计
    win_rate: float = 0.0
    turnover: float = 0.0
    avg_holding_period: float = 0.0
    total_trades: int = 0
    
    # 时间序列
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    benchmark_curve: Optional[pd.Series] = None
    
    # Qlib分析结果
    risk_analysis_report: Optional[pd.DataFrame] = None
    positions: Optional[pd.DataFrame] = None


def compute_simple_backtest(
    factor: pd.Series,
    returns: pd.Series,
    n_groups: int = 5,
    long_short: bool = True,
) -> BacktestResult:
    """
    简单分组回测
    
    参数:
        factor: 因子值
        returns: 未来收益
        n_groups: 分组数
        long_short: 是否多空
    """
    result = BacktestResult()
    
    # 对齐
    valid_mask = factor.notna() & returns.notna()
    factor = factor[valid_mask]
    returns = returns[valid_mask]
    
    if len(factor) < 100:
        return result
    
    # 分组
    try:
        groups = pd.qcut(factor, n_groups, labels=False, duplicates='drop')
    except ValueError:
        return result
    
    # 计算每组收益
    group_returns = returns.groupby(groups).mean()
    
    # 多空收益
    if long_short:
        top_return = group_returns.iloc[-1]  # 最高组做多
        bottom_return = group_returns.iloc[0]  # 最低组做空
        strategy_return = (top_return - bottom_return) / 2
    else:
        strategy_return = group_returns.iloc[-1]  # 只做多
    
    # 计算累积收益
    if hasattr(returns.index, 'get_level_values'):
        # 多级索引 (datetime, instrument)
        dates = returns.index.get_level_values(0).unique()
    else:
        dates = returns.index.unique()
    
    daily_returns = []
    for date in dates:
        mask = returns.index.get_level_values(0) == date if hasattr(returns.index, 'get_level_values') else returns.index == date
        day_factor = factor[mask]
        day_returns = returns[mask]
        
        if len(day_factor) < n_groups:
            continue
        
        try:
            day_groups = pd.qcut(day_factor, n_groups, labels=False, duplicates='drop')
            day_group_returns = day_returns.groupby(day_groups).mean()
            
            if long_short:
                day_ret = (day_group_returns.iloc[-1] - day_group_returns.iloc[0]) / 2
            else:
                day_ret = day_group_returns.iloc[-1]
            
            daily_returns.append({'date': date, 'return': day_ret})
        except Exception:
            continue
    
    if not daily_returns:
        return result
    
    daily_df = pd.DataFrame(daily_returns).set_index('date')
    equity = (1 + daily_df['return']).cumprod()
    
    # 计算指标
    result.equity_curve = equity
    result.total_return = equity.iloc[-1] - 1
    result.annual_return = result.total_return * (252 / len(equity))
    result.volatility = daily_df['return'].std() * np.sqrt(252)
    
    # 最大回撤
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max
    result.max_drawdown = drawdown.max()
    
    # Sharpe
    if result.volatility > 0:
        result.sharpe_ratio = result.annual_return / result.volatility
    
    # Calmar
    if result.max_drawdown > 0:
        result.calmar_ratio = result.annual_return / result.max_drawdown
    
    # 胜率
    result.win_rate = (daily_df['return'] > 0).mean()
    
    return result


def run_backtest(
    factor: pd.Series,
    returns: pd.Series,
    method: str = 'simple',
    **kwargs,
) -> BacktestResult:
    """
    运行回测
    
    参数:
        factor: 因子值
        returns: 未来收益
        method: 回测方法 ('simple', 'qlib', 'qlib_topk')
    """
    if method == 'simple':
        return compute_simple_backtest(factor, returns, **kwargs)
    
    if method in ['qlib', 'qlib_topk']:
        if not QLIB_AVAILABLE:
            raise ImportError("请安装qlib: pip install pyqlib")
        return run_qlib_backtest(factor, **kwargs)
    
    if method == 'qlib_weight':
        if not QLIB_AVAILABLE:
            raise ImportError("请安装qlib: pip install pyqlib")
        return run_qlib_weight_backtest(factor, **kwargs)
    
    raise ValueError(f"未知方法: {method}")


# ==================== Qlib完整回测 ====================

class FactorStrategy(WeightStrategyBase if QLIB_AVAILABLE else object):
    """
    因子权重策略 - 根据因子值分配权重
    
    参考: https://qlib.readthedocs.io/en/latest/component/strategy.html
    """
    
    def __init__(
        self,
        signal: pd.DataFrame,
        topk: int = 50,
        n_drop: int = 5,
        only_tradable: bool = True,
        **kwargs,
    ):
        if QLIB_AVAILABLE:
            super().__init__(signal=signal, **kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.only_tradable = only_tradable
        self.signal = signal
    
    def generate_trade_decision(self, execute_result=None):
        """生成交易决策"""
        # 这里简化处理，实际应该继承WeightStrategyBase
        pass


def prepare_qlib_prediction(
    factor: pd.Series,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    将因子转换为Qlib预测格式
    
    Qlib预测格式: DataFrame with MultiIndex (datetime, instrument)
    """
    # 确保是MultiIndex
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("因子必须是MultiIndex格式: (datetime, instrument)")
    
    # 转换为DataFrame
    pred_df = factor.to_frame(name='score')
    
    # 过滤日期范围
    if start_date or end_date:
        dates = pred_df.index.get_level_values(0)
        mask = pd.Series(True, index=pred_df.index)
        if start_date:
            mask &= dates >= pd.Timestamp(start_date)
        if end_date:
            mask &= dates <= pd.Timestamp(end_date)
        pred_df = pred_df[mask]
    
    return pred_df


def run_qlib_backtest(
    factor: pd.Series,
    start_date: str = None,
    end_date: str = None,
    topk: int = 50,
    n_drop: int = 5,
    benchmark: str = "SH000300",
    account: float = 1e8,
    exchange_kwargs: Dict = None,
    verbose: bool = True,
) -> BacktestResult:
    """
    Qlib TopK策略回测
    
    参数:
        factor: 因子值 (MultiIndex: datetime, instrument)
        start_date: 开始日期
        end_date: 结束日期
        topk: 持仓股票数
        n_drop: 每次调仓剔除数量
        benchmark: 基准
        account: 初始资金
        exchange_kwargs: 交易所参数
        verbose: 是否打印详情
    
    返回:
        BacktestResult
    """
    if not QLIB_AVAILABLE:
        raise ImportError("请安装qlib")
    
    result = BacktestResult()
    
    # 准备预测数据
    try:
        pred_df = prepare_qlib_prediction(factor, start_date, end_date)
    except ValueError as e:
        logger.warning(f"因子格式错误: {e}，使用简单回测")
        return compute_simple_backtest(factor, pd.Series(), n_groups=5)
    
    if len(pred_df) == 0:
        logger.warning("预测数据为空")
        return result
    
    # 获取日期范围
    dates = pred_df.index.get_level_values(0)
    start = start_date or str(dates.min().date())
    end = end_date or str(dates.max().date())
    
    if verbose:
        print(f"\n{'='*60}")
        print("Qlib TopK策略回测")
        print(f"{'='*60}")
        print(f"日期范围: {start} ~ {end}")
        print(f"TopK: {topk}, N_Drop: {n_drop}")
        print(f"初始资金: {account:,.0f}")
    
    # 创建策略
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": pred_df,
            "topk": topk,
            "n_drop": n_drop,
        },
    }
    
    # 执行器配置
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }
    
    # 交易所配置
    default_exchange_kwargs = {
        "freq": "day",
        "limit_threshold": 0.095,  # 涨跌停限制
        "deal_price": "close",
        "open_cost": 0.0005,  # 买入费用
        "close_cost": 0.0015,  # 卖出费用（含印花税）
        "min_cost": 5,  # 最小费用
    }
    if exchange_kwargs:
        default_exchange_kwargs.update(exchange_kwargs)
    
    try:
        # 运行回测
        portfolio_metric_dict, indicator_dict = qlib_backtest(
            pred=pred_df,
            strategy=strategy_config,
            executor=executor_config,
            backtest_config={
                "start_time": start,
                "end_time": end,
                "account": account,
                "benchmark": benchmark,
                "exchange_kwargs": default_exchange_kwargs,
            },
        )
        
        # 提取结果
        result = _parse_qlib_backtest_result(
            portfolio_metric_dict, 
            indicator_dict,
            benchmark=benchmark,
        )
        
        if verbose:
            print(format_backtest_report(result))
        
    except Exception as e:
        logger.error(f"Qlib回测失败: {e}")
        raise
    
    return result


def run_qlib_weight_backtest(
    factor: pd.Series,
    start_date: str = None,
    end_date: str = None,
    method: str = 'rank',  # rank, zscore, softmax
    long_only: bool = True,
    benchmark: str = "SH000300",
    account: float = 1e8,
    **kwargs,
) -> BacktestResult:
    """
    Qlib权重策略回测
    
    根据因子值计算权重：
    - rank: 排名权重
    - zscore: Z-score标准化
    - softmax: Softmax权重
    """
    if not QLIB_AVAILABLE:
        raise ImportError("请安装qlib")
    
    # 计算权重
    if method == 'rank':
        weights = factor.groupby(level=0).apply(lambda x: x.rank(pct=True))
    elif method == 'zscore':
        weights = factor.groupby(level=0).apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    elif method == 'softmax':
        weights = factor.groupby(level=0).apply(
            lambda x: np.exp(x) / np.exp(x).sum()
        )
    else:
        raise ValueError(f"未知权重方法: {method}")
    
    # 只做多：过滤负权重
    if long_only:
        weights = weights.clip(lower=0)
        weights = weights.groupby(level=0).apply(lambda x: x / x.sum())
    
    # 转换为Qlib格式并回测
    return run_qlib_backtest(
        factor=weights,
        start_date=start_date,
        end_date=end_date,
        topk=len(weights.groupby(level=0).first()),  # 使用所有股票
        benchmark=benchmark,
        account=account,
        **kwargs,
    )


def _parse_qlib_backtest_result(
    portfolio_metric_dict: Dict,
    indicator_dict: Dict,
    benchmark: str = "SH000300",
) -> BacktestResult:
    """解析Qlib回测结果"""
    result = BacktestResult()
    
    # 获取portfolio数据
    for freq, (report_df, positions) in portfolio_metric_dict.items():
        # 解析报告
        if report_df is not None and len(report_df) > 0:
            # 收益曲线
            if 'return' in report_df.columns:
                result.daily_returns = report_df['return']
                result.equity_curve = (1 + result.daily_returns).cumprod()
            
            # 基准曲线
            if 'bench' in report_df.columns:
                result.benchmark_curve = (1 + report_df['bench']).cumprod()
            
            # 计算指标
            if result.equity_curve is not None:
                result.total_return = result.equity_curve.iloc[-1] - 1
                n_days = len(result.equity_curve)
                result.annual_return = (1 + result.total_return) ** (252 / n_days) - 1
                
                # 波动率
                result.volatility = result.daily_returns.std() * np.sqrt(252)
                
                # 最大回撤
                running_max = result.equity_curve.cummax()
                drawdown = (running_max - result.equity_curve) / running_max
                result.max_drawdown = drawdown.max()
                
                # Sharpe
                if result.volatility > 0:
                    result.sharpe_ratio = result.annual_return / result.volatility
                
                # Calmar
                if result.max_drawdown > 0:
                    result.calmar_ratio = result.annual_return / result.max_drawdown
                
                # 胜率
                result.win_rate = (result.daily_returns > 0).mean()
                
                # 超额收益
                if result.benchmark_curve is not None:
                    bench_return = result.benchmark_curve.iloc[-1] - 1
                    result.excess_return = result.total_return - bench_return
            
            # 持仓信息
            if positions is not None:
                result.positions = positions
                result.total_trades = len(positions)
                
                # 换手率
                if 'turnover' in report_df.columns:
                    result.turnover = report_df['turnover'].mean()
        
        # 只取第一个freq的结果
        break
    
    # 风险分析报告
    try:
        if result.daily_returns is not None:
            risk_report = risk_analysis(result.daily_returns)
            result.risk_analysis_report = risk_report
    except Exception as e:
        logger.warning(f"风险分析失败: {e}")
    
    return result


def run_qlib_factor_analysis(
    factor: pd.Series,
    price_field: str = "$close",
    start_date: str = None,
    end_date: str = None,
) -> Dict:
    """
    Qlib因子分析
    
    计算:
    - IC/ICIR
    - 分组收益
    - 换手率分析
    """
    if not QLIB_AVAILABLE:
        raise ImportError("请安装qlib")
    
    result = {}
    
    # 确保MultiIndex
    if not isinstance(factor.index, pd.MultiIndex):
        return result
    
    dates = factor.index.get_level_values(0).unique()
    instruments = factor.index.get_level_values(1).unique()
    
    # 获取收益数据
    try:
        returns = D.features(
            instruments=list(instruments),
            fields=[f"Ref({price_field}, -1) / {price_field} - 1"],
            start_time=str(dates.min()),
            end_time=str(dates.max()),
        )
        returns = returns.iloc[:, 0]
        returns.name = 'return'
    except Exception as e:
        logger.error(f"获取收益数据失败: {e}")
        return result
    
    # 对齐数据
    aligned = pd.concat([factor, returns], axis=1).dropna()
    factor_aligned = aligned.iloc[:, 0]
    returns_aligned = aligned.iloc[:, 1]
    
    # 计算每日IC
    daily_ic = factor_aligned.groupby(level=0).apply(
        lambda x: x.corr(returns_aligned.loc[x.index], method='spearman')
    )
    
    result['ic_mean'] = daily_ic.mean()
    result['ic_std'] = daily_ic.std()
    result['icir'] = result['ic_mean'] / (result['ic_std'] + 1e-8)
    result['ic_series'] = daily_ic
    
    # 分组分析
    n_groups = 5
    group_returns = []
    for date in dates:
        try:
            day_factor = factor_aligned.loc[date]
            day_returns = returns_aligned.loc[date]
            groups = pd.qcut(day_factor, n_groups, labels=False, duplicates='drop')
            group_ret = day_returns.groupby(groups).mean()
            group_returns.append(group_ret)
        except Exception:
            continue
    
    if group_returns:
        group_df = pd.DataFrame(group_returns)
        result['group_returns'] = group_df.mean()
        result['long_short'] = result['group_returns'].iloc[-1] - result['group_returns'].iloc[0]
    
    return result


def format_backtest_report(result: BacktestResult) -> str:
    """格式化回测报告"""
    report = f"""
{'='*60}
回测报告
{'='*60}

【收益指标】
  总收益:      {result.total_return:+.2%}
  年化收益:    {result.annual_return:+.2%}
  超额收益:    {result.excess_return:+.2%}

【风险指标】  
  年化波动:    {result.volatility:.2%}
  最大回撤:    {result.max_drawdown:.2%}

【风险调整】
  Sharpe:      {result.sharpe_ratio:.2f}
  Calmar:      {result.calmar_ratio:.2f}
  Sortino:     {result.sortino_ratio:.2f}
  IR:          {result.information_ratio:.2f}

【交易统计】
  胜率:        {result.win_rate:.1%}
  换手率:      {result.turnover:.2%}
  交易次数:    {result.total_trades}

{'='*60}
"""
    return report


def plot_backtest_result(
    result: BacktestResult,
    title: str = "策略回测",
    save_path: str = None,
):
    """
    绘制回测结果图表
    
    包含:
    1. 累积收益曲线
    2. 回撤曲线
    3. 月度收益热力图
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib未安装，无法绘图")
        return
    
    if result.equity_curve is None:
        logger.warning("无收益曲线数据")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    
    # 1. 累积收益曲线
    ax1 = axes[0]
    ax1.plot(result.equity_curve.index, result.equity_curve, label='策略', linewidth=1.5)
    if result.benchmark_curve is not None:
        ax1.plot(result.benchmark_curve.index, result.benchmark_curve, 
                 label='基准', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('累积收益')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 2. 回撤曲线
    ax2 = axes[1]
    running_max = result.equity_curve.cummax()
    drawdown = (result.equity_curve - running_max) / running_max
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    ax2.plot(drawdown.index, drawdown, color='red', linewidth=1)
    ax2.set_ylabel('回撤')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 3. 日收益分布
    ax3 = axes[2]
    if result.daily_returns is not None:
        ax3.hist(result.daily_returns, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax3.axvline(x=result.daily_returns.mean(), color='green', linestyle='--', 
                    linewidth=1, label=f'均值: {result.daily_returns.mean():.4f}')
    ax3.set_xlabel('日收益率')
    ax3.set_ylabel('频数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")
    
    plt.show()


# ==================== 导出 ====================

__all__ = [
    'BacktestResult',
    'run_backtest',
    'compute_simple_backtest',
    'run_qlib_backtest',
    'run_qlib_weight_backtest',
    'run_qlib_factor_analysis',
    'format_backtest_report',
    'plot_backtest_result',
    'QLIB_AVAILABLE',
]
