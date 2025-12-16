"""
回测指标计算模块

参考Qlib框架的完整回测指标体系:
https://qlib.readthedocs.io/en/latest/component/report.html

指标分类:
1. IC指标 - 预测能力
2. 收益指标 - 收益能力
3. 风险指标 - 风险控制
4. 综合指标 - 综合评估
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import numpy as np
from scipy import stats


# ============================================================
# 指标数据类
# ============================================================

@dataclass
class ICMetrics:
    """IC相关指标 - 衡量因子预测能力"""
    
    # Pearson IC
    ic_mean: float = 0.0           # IC均值
    ic_std: float = 0.0            # IC标准差
    icir: float = 0.0              # IC信息比 = IC_mean / IC_std
    
    # Rank IC (Spearman)
    rank_ic_mean: float = 0.0      # Rank IC均值
    rank_ic_std: float = 0.0       # Rank IC标准差
    rank_icir: float = 0.0         # Rank ICIR = Rank_IC_mean / Rank_IC_std
    
    # IC统计
    ic_positive_ratio: float = 0.0 # IC>0的比例
    ic_abs_gt_2pct: float = 0.0    # |IC|>2%的比例
    
    # 等级评定
    ic_grade: str = ""             # IC等级 A/B/C/D/F
    icir_grade: str = ""           # ICIR等级 A/B/C/D/F
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReturnMetrics:
    """收益相关指标"""
    
    # 收益率
    total_return: float = 0.0      # 累计收益
    annual_return: float = 0.0     # 年化收益
    excess_return: float = 0.0     # 超额收益 (相对基准)
    
    # 收益分布
    daily_return_mean: float = 0.0 # 日均收益
    daily_return_std: float = 0.0  # 日收益标准差
    
    # 胜率
    win_rate: float = 0.0          # 胜率 (正收益天数/总天数)
    profit_loss_ratio: float = 0.0 # 盈亏比 (平均盈利/平均亏损)
    
    # 换手率
    turnover: float = 0.0          # 平均换手率
    turnover_std: float = 0.0      # 换手率标准差
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RiskMetrics:
    """风险相关指标"""
    
    # 波动率
    volatility: float = 0.0        # 年化波动率
    downside_volatility: float = 0.0  # 下行波动率
    
    # 回撤
    max_drawdown: float = 0.0      # 最大回撤
    max_drawdown_duration: int = 0 # 最大回撤持续天数
    avg_drawdown: float = 0.0      # 平均回撤
    
    # 风险调整收益
    sharpe_ratio: float = 0.0      # 夏普比率 = (年化收益-无风险利率) / 年化波动率
    sortino_ratio: float = 0.0     # 索提诺比率 = (年化收益-无风险利率) / 下行波动率
    calmar_ratio: float = 0.0      # 卡玛比率 = 年化收益 / 最大回撤
    information_ratio: float = 0.0 # 信息比率 = 超额收益 / 跟踪误差
    
    # 尾部风险
    var_95: float = 0.0            # 95% VaR (Value at Risk)
    cvar_95: float = 0.0           # 95% CVaR (Conditional VaR)
    
    # 等级评定
    sharpe_grade: str = ""         # 夏普比率等级
    drawdown_grade: str = ""       # 回撤等级
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BacktestMetrics:
    """完整回测指标汇总"""
    
    # 子指标
    ic: ICMetrics = field(default_factory=ICMetrics)
    returns: ReturnMetrics = field(default_factory=ReturnMetrics)
    risk: RiskMetrics = field(default_factory=RiskMetrics)
    
    # 元信息
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0
    
    # 综合评分
    overall_score: float = 0.0     # 综合得分 (0-100)
    overall_grade: str = ""        # 综合等级 A/B/C/D/F
    
    # 分年度统计
    yearly_stats: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'ic': self.ic.to_dict(),
            'returns': self.returns.to_dict(),
            'risk': self.risk.to_dict(),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'trading_days': self.trading_days,
            'overall_score': self.overall_score,
            'overall_grade': self.overall_grade,
            'yearly_stats': self.yearly_stats,
        }
    
    def to_flat_dict(self) -> Dict:
        """扁平化为单层字典，便于存储"""
        return {
            # IC指标
            'ic': self.ic.ic_mean,
            'icir': self.ic.icir,
            'rank_ic': self.ic.rank_ic_mean,
            'rank_icir': self.ic.rank_icir,
            'ic_grade': self.ic.ic_grade,
            'icir_grade': self.ic.icir_grade,
            
            # 收益指标
            'ann_return': self.returns.annual_return,
            'total_return': self.returns.total_return,
            'excess_return': self.returns.excess_return,
            'turnover': self.returns.turnover,
            'win_rate': self.returns.win_rate,
            
            # 风险指标
            'sharpe': self.risk.sharpe_ratio,
            'sortino': self.risk.sortino_ratio,
            'calmar': self.risk.calmar_ratio,
            'information_ratio': self.risk.information_ratio,
            'max_drawdown': self.risk.max_drawdown,
            'volatility': self.risk.volatility,
            
            # 综合
            'overall_score': self.overall_score,
            'overall_grade': self.overall_grade,
        }


# ============================================================
# 指标计算函数
# ============================================================

def compute_ic_metrics(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
) -> ICMetrics:
    """
    计算IC相关指标
    
    Args:
        factor_values: 因子值序列 (T x N)
        forward_returns: 未来收益率序列 (T x N)
    
    Returns:
        ICMetrics
    """
    if factor_values.size == 0 or forward_returns.size == 0:
        return ICMetrics()
    
    # 确保维度匹配
    if factor_values.ndim == 1:
        factor_values = factor_values.reshape(-1, 1)
    if forward_returns.ndim == 1:
        forward_returns = forward_returns.reshape(-1, 1)
    
    T = factor_values.shape[0]
    
    # 计算每期IC
    ic_series = []
    rank_ic_series = []
    
    for t in range(T):
        f = factor_values[t]
        r = forward_returns[t]
        
        # 去除NaN
        valid = ~(np.isnan(f) | np.isnan(r))
        if valid.sum() < 10:
            continue
        
        f_valid = f[valid]
        r_valid = r[valid]
        
        # Pearson IC
        ic, _ = stats.pearsonr(f_valid, r_valid)
        ic_series.append(ic)
        
        # Spearman Rank IC
        rank_ic, _ = stats.spearmanr(f_valid, r_valid)
        rank_ic_series.append(rank_ic)
    
    if not ic_series:
        return ICMetrics()
    
    ic_arr = np.array(ic_series)
    rank_ic_arr = np.array(rank_ic_series)
    
    # 计算统计量
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0
    
    rank_ic_mean = float(np.mean(rank_ic_arr))
    rank_ic_std = float(np.std(rank_ic_arr))
    rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0
    
    return ICMetrics(
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        rank_ic_mean=rank_ic_mean,
        rank_ic_std=rank_ic_std,
        rank_icir=rank_icir,
        ic_positive_ratio=float((ic_arr > 0).mean()),
        ic_abs_gt_2pct=float((np.abs(ic_arr) > 0.02).mean()),
        ic_grade=_compute_ic_grade(rank_ic_mean),
        icir_grade=_compute_icir_grade(rank_icir),
    )


def compute_return_metrics(
    returns: np.ndarray,
    benchmark_returns: np.ndarray = None,
    turnover: np.ndarray = None,
    annual_trading_days: int = 252,
) -> ReturnMetrics:
    """
    计算收益相关指标
    
    Args:
        returns: 日收益率序列
        benchmark_returns: 基准收益率序列
        turnover: 换手率序列
        annual_trading_days: 年化交易日
    """
    if returns.size == 0:
        return ReturnMetrics()
    
    returns = np.asarray(returns).flatten()
    returns = returns[~np.isnan(returns)]
    
    T = len(returns)
    if T == 0:
        return ReturnMetrics()
    
    # 累计收益
    total_return = float(np.prod(1 + returns) - 1)
    
    # 年化收益
    annual_return = float((1 + total_return) ** (annual_trading_days / T) - 1)
    
    # 超额收益
    excess_return = 0.0
    if benchmark_returns is not None:
        bench = np.asarray(benchmark_returns).flatten()
        bench = bench[~np.isnan(bench)]
        if len(bench) == len(returns):
            bench_total = float(np.prod(1 + bench) - 1)
            excess_return = total_return - bench_total
    
    # 日收益统计
    daily_mean = float(np.mean(returns))
    daily_std = float(np.std(returns))
    
    # 胜率
    win_rate = float((returns > 0).mean())
    
    # 盈亏比
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    if len(gains) > 0 and len(losses) > 0:
        profit_loss_ratio = float(np.abs(np.mean(gains) / np.mean(losses)))
    else:
        profit_loss_ratio = 0.0
    
    # 换手率
    turnover_mean = 0.0
    turnover_std = 0.0
    if turnover is not None:
        turnover = np.asarray(turnover).flatten()
        turnover = turnover[~np.isnan(turnover)]
        if len(turnover) > 0:
            turnover_mean = float(np.mean(turnover))
            turnover_std = float(np.std(turnover))
    
    return ReturnMetrics(
        total_return=total_return,
        annual_return=annual_return,
        excess_return=excess_return,
        daily_return_mean=daily_mean,
        daily_return_std=daily_std,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        turnover=turnover_mean,
        turnover_std=turnover_std,
    )


def compute_risk_metrics(
    returns: np.ndarray,
    benchmark_returns: np.ndarray = None,
    risk_free_rate: float = 0.03,
    annual_trading_days: int = 252,
) -> RiskMetrics:
    """
    计算风险相关指标
    
    Args:
        returns: 日收益率序列
        benchmark_returns: 基准收益率序列
        risk_free_rate: 年化无风险利率
        annual_trading_days: 年化交易日
    """
    if returns.size == 0:
        return RiskMetrics()
    
    returns = np.asarray(returns).flatten()
    returns = returns[~np.isnan(returns)]
    
    T = len(returns)
    if T == 0:
        return RiskMetrics()
    
    # 日无风险利率
    daily_rf = risk_free_rate / annual_trading_days
    
    # 波动率
    daily_vol = float(np.std(returns))
    volatility = daily_vol * np.sqrt(annual_trading_days)
    
    # 下行波动率 (只考虑负收益)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_vol = float(np.std(negative_returns)) * np.sqrt(annual_trading_days)
    else:
        downside_vol = 0.0
    
    # 最大回撤
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = float(np.abs(np.min(drawdowns)))
    avg_drawdown = float(np.abs(np.mean(drawdowns[drawdowns < 0]))) if (drawdowns < 0).any() else 0.0
    
    # 最大回撤持续时间
    max_dd_duration = _compute_max_drawdown_duration(drawdowns)
    
    # 年化收益
    total_return = float(np.prod(1 + returns) - 1)
    annual_return = float((1 + total_return) ** (annual_trading_days / T) - 1)
    
    # 夏普比率
    excess_daily = np.mean(returns) - daily_rf
    sharpe = float(excess_daily / daily_vol * np.sqrt(annual_trading_days)) if daily_vol > 0 else 0.0
    
    # 索提诺比率
    if downside_vol > 0:
        sortino = float((annual_return - risk_free_rate) / downside_vol)
    else:
        sortino = 0.0
    
    # 卡玛比率
    calmar = float(annual_return / max_drawdown) if max_drawdown > 0 else 0.0
    
    # 信息比率
    information_ratio = 0.0
    if benchmark_returns is not None:
        bench = np.asarray(benchmark_returns).flatten()
        bench = bench[~np.isnan(bench)]
        if len(bench) == len(returns):
            excess = returns - bench
            tracking_error = float(np.std(excess)) * np.sqrt(annual_trading_days)
            if tracking_error > 0:
                excess_annual = float(np.mean(excess)) * annual_trading_days
                information_ratio = excess_annual / tracking_error
    
    # VaR和CVaR
    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(np.mean(returns[returns <= var_95]))
    
    return RiskMetrics(
        volatility=volatility,
        downside_volatility=downside_vol,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        avg_drawdown=avg_drawdown,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        information_ratio=information_ratio,
        var_95=var_95,
        cvar_95=cvar_95,
        sharpe_grade=_compute_sharpe_grade(sharpe),
        drawdown_grade=_compute_drawdown_grade(max_drawdown),
    )


def compute_all_metrics(
    factor_values: np.ndarray = None,
    forward_returns: np.ndarray = None,
    strategy_returns: np.ndarray = None,
    benchmark_returns: np.ndarray = None,
    turnover: np.ndarray = None,
    risk_free_rate: float = 0.03,
    start_date: str = "",
    end_date: str = "",
) -> BacktestMetrics:
    """
    计算全部回测指标
    
    Args:
        factor_values: 因子值 (用于IC计算)
        forward_returns: 未来收益率 (用于IC计算)
        strategy_returns: 策略日收益率
        benchmark_returns: 基准收益率
        turnover: 换手率序列
        risk_free_rate: 无风险利率
    """
    metrics = BacktestMetrics(
        start_date=start_date,
        end_date=end_date,
    )
    
    # IC指标
    if factor_values is not None and forward_returns is not None:
        metrics.ic = compute_ic_metrics(factor_values, forward_returns)
    
    # 收益指标
    if strategy_returns is not None:
        metrics.returns = compute_return_metrics(
            strategy_returns, benchmark_returns, turnover
        )
        metrics.trading_days = len(strategy_returns)
    
    # 风险指标
    if strategy_returns is not None:
        metrics.risk = compute_risk_metrics(
            strategy_returns, benchmark_returns, risk_free_rate
        )
    
    # 综合评分
    metrics.overall_score = _compute_overall_score(metrics)
    metrics.overall_grade = _compute_overall_grade(metrics.overall_score)
    
    return metrics


# ============================================================
# 等级评定函数
# ============================================================

def _compute_ic_grade(rank_ic: float) -> str:
    """
    Rank IC等级评定
    
    A: >5%  (优秀)
    B: 3-5% (良好)
    C: 2-3% (合格)
    D: 1-2% (较弱)
    F: <1%  (无效)
    """
    if rank_ic >= 0.05:
        return 'A'
    elif rank_ic >= 0.03:
        return 'B'
    elif rank_ic >= 0.02:
        return 'C'
    elif rank_ic >= 0.01:
        return 'D'
    else:
        return 'F'


def _compute_icir_grade(rank_icir: float) -> str:
    """
    Rank ICIR等级评定
    
    A: >1.5 (优秀)
    B: 1.0-1.5 (良好)
    C: 0.5-1.0 (合格)
    D: 0.3-0.5 (较弱)
    F: <0.3 (无效)
    """
    if rank_icir >= 1.5:
        return 'A'
    elif rank_icir >= 1.0:
        return 'B'
    elif rank_icir >= 0.5:
        return 'C'
    elif rank_icir >= 0.3:
        return 'D'
    else:
        return 'F'


def _compute_sharpe_grade(sharpe: float) -> str:
    """
    夏普比率等级评定
    
    A: >2.0 (优秀)
    B: 1.5-2.0 (良好)
    C: 1.0-1.5 (合格)
    D: 0.5-1.0 (较弱)
    F: <0.5 (无效)
    """
    if sharpe >= 2.0:
        return 'A'
    elif sharpe >= 1.5:
        return 'B'
    elif sharpe >= 1.0:
        return 'C'
    elif sharpe >= 0.5:
        return 'D'
    else:
        return 'F'


def _compute_drawdown_grade(max_drawdown: float) -> str:
    """
    最大回撤等级评定
    
    A: <10% (优秀)
    B: 10-20% (良好)
    C: 20-30% (合格)
    D: 30-40% (较弱)
    F: >40% (无效)
    """
    if max_drawdown < 0.10:
        return 'A'
    elif max_drawdown < 0.20:
        return 'B'
    elif max_drawdown < 0.30:
        return 'C'
    elif max_drawdown < 0.40:
        return 'D'
    else:
        return 'F'


def _compute_max_drawdown_duration(drawdowns: np.ndarray) -> int:
    """计算最大回撤持续天数"""
    in_drawdown = drawdowns < 0
    max_duration = 0
    current_duration = 0
    
    for i in range(len(in_drawdown)):
        if in_drawdown[i]:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return int(max_duration)


def _compute_overall_score(metrics: BacktestMetrics) -> float:
    """
    计算综合评分 (0-100)
    
    权重:
    - IC指标: 30%
    - 夏普比率: 25%
    - 回撤控制: 20%
    - 年化收益: 15%
    - 换手率: 10%
    """
    score = 0.0
    
    # IC评分 (30分)
    ic_score = min(100, max(0, metrics.ic.rank_ic_mean * 1000))  # 5%->50分
    score += ic_score * 0.3
    
    # 夏普评分 (25分)
    sharpe_score = min(100, max(0, metrics.risk.sharpe_ratio * 40))  # 2.5->100分
    score += sharpe_score * 0.25
    
    # 回撤评分 (20分)
    dd_score = max(0, 100 - metrics.risk.max_drawdown * 250)  # 40%->0分
    score += dd_score * 0.20
    
    # 收益评分 (15分)
    ret_score = min(100, max(0, metrics.returns.annual_return * 200))  # 50%->100分
    score += ret_score * 0.15
    
    # 换手率评分 (10分) - 低换手得高分
    to_score = max(0, 100 - metrics.returns.turnover * 100)  # 100%->0分
    score += to_score * 0.10
    
    return round(score, 2)


def _compute_overall_grade(score: float) -> str:
    """
    综合等级评定
    
    A: >80
    B: 60-80
    C: 40-60
    D: 20-40
    F: <20
    """
    if score >= 80:
        return 'A'
    elif score >= 60:
        return 'B'
    elif score >= 40:
        return 'C'
    elif score >= 20:
        return 'D'
    else:
        return 'F'
