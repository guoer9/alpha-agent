"""
因子评估器 - 计算IC、ICIR等指标
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .base import FactorResult, FactorStatus
from ..config import factor_config


@dataclass
class EvaluationResult:
    """评估结果"""
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    rank_icir: float = 0.0
    
    # 分组收益
    top_return: float = 0.0
    bottom_return: float = 0.0
    long_short_return: float = 0.0
    
    # 稳定性
    ic_std: float = 0.0
    positive_ic_ratio: float = 0.0
    
    # 状态
    status: FactorStatus = FactorStatus.PENDING
    recommendation: str = ""


class FactorEvaluator:
    """因子评估器"""
    
    def __init__(
        self,
        ic_excellent: float = None,
        ic_good: float = None,
        ic_minimum: float = None,
        rolling_window: int = 20,
    ):
        self.ic_excellent = ic_excellent or factor_config.ic_excellent
        self.ic_good = ic_good or factor_config.ic_good
        self.ic_minimum = ic_minimum or factor_config.ic_minimum
        self.rolling_window = rolling_window
    
    def evaluate(
        self,
        factor: pd.Series,
        target: pd.Series,
        n_groups: int = 5,
    ) -> EvaluationResult:
        """
        评估因子
        
        参数:
            factor: 因子值
            target: 目标收益
            n_groups: 分组数
        
        返回:
            EvaluationResult
        """
        result = EvaluationResult()
        
        # 对齐数据
        valid_mask = factor.notna() & target.notna()
        factor_valid = factor[valid_mask]
        target_valid = target[valid_mask]
        
        if len(factor_valid) < 100:
            result.status = FactorStatus.FAILED
            result.recommendation = "有效样本不足"
            return result
        
        # 1. 计算IC (Spearman相关)
        result.ic = factor_valid.corr(target_valid, method='spearman')
        result.rank_ic = factor_valid.rank().corr(target_valid.rank())
        
        # 2. 计算滚动IC和ICIR
        rolling_ic = self._compute_rolling_ic(factor_valid, target_valid)
        if len(rolling_ic) > 0:
            result.icir = rolling_ic.mean() / (rolling_ic.std() + 1e-8)
            result.ic_std = rolling_ic.std()
            result.positive_ic_ratio = (rolling_ic > 0).mean()
            result.rank_icir = result.icir  # 简化处理
        
        # 3. 分组收益分析
        group_returns = self._compute_group_returns(factor_valid, target_valid, n_groups)
        if group_returns is not None:
            result.top_return = group_returns.iloc[-1]
            result.bottom_return = group_returns.iloc[0]
            result.long_short_return = result.top_return - result.bottom_return
        
        # 4. 判定状态
        result.status, result.recommendation = self._determine_status(result)
        
        return result
    
    def _compute_rolling_ic(
        self,
        factor: pd.Series,
        target: pd.Series,
    ) -> pd.Series:
        """计算滚动IC"""
        rolling_ic = []
        
        for i in range(self.rolling_window, len(factor)):
            window_factor = factor.iloc[i-self.rolling_window:i]
            window_target = target.iloc[i-self.rolling_window:i]
            ic = window_factor.corr(window_target, method='spearman')
            if not np.isnan(ic):
                rolling_ic.append(ic)
        
        return pd.Series(rolling_ic)
    
    def _compute_group_returns(
        self,
        factor: pd.Series,
        target: pd.Series,
        n_groups: int,
    ) -> Optional[pd.Series]:
        """计算分组收益"""
        try:
            # 分组
            factor_groups = pd.qcut(factor, n_groups, labels=False, duplicates='drop')
            
            # 计算每组平均收益
            group_returns = target.groupby(factor_groups).mean()
            return group_returns.sort_index()
        except Exception:
            return None
    
    def _determine_status(
        self,
        result: EvaluationResult,
    ) -> Tuple[FactorStatus, str]:
        """判定因子状态"""
        abs_ic = abs(result.ic)
        
        if abs_ic >= self.ic_excellent:
            return FactorStatus.EXCELLENT, f"优秀因子 (|IC|={abs_ic:.4f} >= {self.ic_excellent})"
        
        if abs_ic >= self.ic_good:
            return FactorStatus.GOOD, f"良好因子 (|IC|={abs_ic:.4f} >= {self.ic_good})"
        
        if abs_ic >= self.ic_minimum:
            return FactorStatus.MARGINAL, f"边缘因子 (|IC|={abs_ic:.4f} >= {self.ic_minimum})"
        
        return FactorStatus.POOR, f"无效因子 (|IC|={abs_ic:.4f} < {self.ic_minimum})"


def evaluate_factor(
    factor: pd.Series,
    target: pd.Series,
    name: str = "factor",
) -> FactorResult:
    """
    便捷函数: 评估因子
    
    返回:
        FactorResult
    """
    evaluator = FactorEvaluator()
    eval_result = evaluator.evaluate(factor, target)
    
    return FactorResult(
        name=name,
        code="",
        ic=eval_result.ic,
        icir=eval_result.icir,
        rank_ic=eval_result.rank_ic,
        status=eval_result.status,
        values=factor,
    )


def format_evaluation_report(result: EvaluationResult) -> str:
    """格式化评估报告"""
    report = f"""
{'='*50}
因子评估报告
{'='*50}
IC:         {result.ic:+.4f}
ICIR:       {result.icir:+.4f}
Rank IC:    {result.rank_ic:+.4f}
IC Std:     {result.ic_std:.4f}
正IC比例:   {result.positive_ic_ratio:.1%}

分组收益:
  Top组:    {result.top_return:+.4f}
  Bottom组: {result.bottom_return:+.4f}
  多空:     {result.long_short_return:+.4f}

状态: {result.status.value}
建议: {result.recommendation}
{'='*50}
"""
    return report
