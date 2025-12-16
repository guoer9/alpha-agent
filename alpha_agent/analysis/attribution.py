"""
收益归因分析 - Brinson模型

Brinson归因将组合收益分解为:
1. 配置效应 (Allocation Effect): 行业配置贡献
2. 选择效应 (Selection Effect): 个股选择贡献
3. 交互效应 (Interaction Effect): 配置与选择的交互

参考: Brinson, Hood, Beebower (1986)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BrinsonResult:
    """Brinson归因结果"""
    # 总效应
    total_return: float = 0.0
    benchmark_return: float = 0.0
    active_return: float = 0.0
    
    # 分解效应
    allocation_effect: float = 0.0
    selection_effect: float = 0.0
    interaction_effect: float = 0.0
    
    # 按行业分解
    sector_attribution: Optional[pd.DataFrame] = None
    
    # 时间序列
    daily_attribution: Optional[pd.DataFrame] = None


def brinson_attribution(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    sector_map: Dict[str, str] = None,
) -> BrinsonResult:
    """
    Brinson归因分析
    
    参数:
        portfolio_weights: 组合权重 (index=date, columns=stocks)
        benchmark_weights: 基准权重
        portfolio_returns: 组合个股收益
        benchmark_returns: 基准个股收益
        sector_map: 股票到行业的映射
    
    返回:
        BrinsonResult
    """
    result = BrinsonResult()
    
    # 对齐数据
    dates = portfolio_weights.index.intersection(benchmark_weights.index)
    stocks = portfolio_weights.columns.intersection(benchmark_weights.columns)
    
    if len(dates) == 0 or len(stocks) == 0:
        logger.warning("数据无法对齐")
        return result
    
    pw = portfolio_weights.loc[dates, stocks]
    bw = benchmark_weights.loc[dates, stocks]
    pr = portfolio_returns.loc[dates, stocks]
    br = benchmark_returns.loc[dates, stocks]
    
    # 计算总收益
    portfolio_ret = (pw * pr).sum(axis=1)
    benchmark_ret = (bw * br).sum(axis=1)
    
    result.total_return = (1 + portfolio_ret).prod() - 1
    result.benchmark_return = (1 + benchmark_ret).prod() - 1
    result.active_return = result.total_return - result.benchmark_return
    
    # 如果没有行业映射，使用个股级别
    if sector_map is None:
        sector_map = {s: s for s in stocks}
    
    # 按行业聚合
    sectors = list(set(sector_map.values()))
    
    daily_allocation = []
    daily_selection = []
    daily_interaction = []
    
    for date in dates:
        pw_day = pw.loc[date]
        bw_day = bw.loc[date]
        pr_day = pr.loc[date]
        br_day = br.loc[date]
        
        alloc = 0.0
        selec = 0.0
        inter = 0.0
        
        for sector in sectors:
            sector_stocks = [s for s in stocks if sector_map.get(s) == sector]
            if not sector_stocks:
                continue
            
            # 行业权重
            wp_s = pw_day[sector_stocks].sum()
            wb_s = bw_day[sector_stocks].sum()
            
            # 行业收益
            if wb_s > 0:
                rb_s = (bw_day[sector_stocks] * br_day[sector_stocks]).sum() / wb_s
            else:
                rb_s = br_day[sector_stocks].mean()
            
            if wp_s > 0:
                rp_s = (pw_day[sector_stocks] * pr_day[sector_stocks]).sum() / wp_s
            else:
                rp_s = pr_day[sector_stocks].mean()
            
            rb_total = (bw_day * br_day).sum()
            
            # Brinson分解
            alloc += (wp_s - wb_s) * (rb_s - rb_total)
            selec += wb_s * (rp_s - rb_s)
            inter += (wp_s - wb_s) * (rp_s - rb_s)
        
        daily_allocation.append(alloc)
        daily_selection.append(selec)
        daily_interaction.append(inter)
    
    # 汇总
    result.allocation_effect = np.sum(daily_allocation)
    result.selection_effect = np.sum(daily_selection)
    result.interaction_effect = np.sum(daily_interaction)
    
    # 每日归因
    result.daily_attribution = pd.DataFrame({
        'allocation': daily_allocation,
        'selection': daily_selection,
        'interaction': daily_interaction,
    }, index=dates)
    
    return result


def factor_attribution(
    factor_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    factor_names: List[str] = None,
) -> Dict:
    """
    因子归因 - 基于多因子回归
    
    参数:
        factor_returns: 因子收益 (index=date, columns=factors)
        portfolio_returns: 组合收益
        factor_names: 因子名称
    
    返回:
        归因结果字典
    """
    from sklearn.linear_model import LinearRegression
    
    # 对齐
    aligned = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
    if len(aligned) < 30:
        return {}
    
    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]
    
    # 回归
    model = LinearRegression()
    model.fit(X, y)
    
    # 归因
    factor_names = factor_names or list(X.columns)
    exposures = dict(zip(factor_names, model.coef_))
    
    # 因子贡献
    contributions = {}
    for i, name in enumerate(factor_names):
        contributions[name] = model.coef_[i] * X.iloc[:, i].mean()
    
    # Alpha (残差)
    y_pred = model.predict(X)
    alpha = (y - y_pred).mean() * 252
    
    return {
        'exposures': exposures,
        'contributions': contributions,
        'alpha': alpha,
        'r2': model.score(X, y),
        'residual': y - y_pred,
    }


def performance_attribution_report(result: BrinsonResult) -> str:
    """生成归因报告"""
    report = f"""
{'='*60}
Brinson 收益归因分析
{'='*60}

【总体表现】
  组合收益:    {result.total_return:+.2%}
  基准收益:    {result.benchmark_return:+.2%}
  超额收益:    {result.active_return:+.2%}

【归因分解】
  配置效应:    {result.allocation_effect:+.4%}
  选择效应:    {result.selection_effect:+.4%}
  交互效应:    {result.interaction_effect:+.4%}
  ─────────────────────────
  合计:        {result.allocation_effect + result.selection_effect + result.interaction_effect:+.4%}

【解读】
  配置效应 > 0: 行业配置优于基准
  选择效应 > 0: 个股选择优于基准
  交互效应: 配置与选择的协同效果

{'='*60}
"""
    return report
