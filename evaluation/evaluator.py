"""
因子评估器 - 对接Qlib框架

参考Qlib的回测评估流程:
https://qlib.readthedocs.io/en/latest/component/backtest.html
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import logging
import numpy as np

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

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """评估器配置"""
    
    # 回测参数
    start_date: str = "2018-01-01"
    end_date: str = "2023-12-31"
    benchmark: str = "SH000300"    # 基准: 沪深300
    
    # IC计算参数
    ic_lag: int = 1               # IC计算的收益滞后期
    ic_method: str = "rank"       # IC计算方法: "pearson" / "rank"
    
    # 收益计算参数
    annual_trading_days: int = 252
    risk_free_rate: float = 0.03  # 年化无风险利率
    
    # 筛选阈值
    min_ic: float = 0.02          # 最小IC
    min_icir: float = 0.5         # 最小ICIR
    min_rank_ic: float = 0.025    # 最小Rank IC
    min_sharpe: float = 0.5       # 最小夏普比率
    max_drawdown: float = 0.30    # 最大回撤
    max_turnover: float = 0.50    # 最大换手率
    
    # 综合评分权重
    weight_ic: float = 0.30
    weight_sharpe: float = 0.25
    weight_drawdown: float = 0.20
    weight_return: float = 0.15
    weight_turnover: float = 0.10


class FactorEvaluator:
    """
    因子评估器
    
    支持两种模式:
    1. 快速评估 (IC-based): 只计算IC指标，用于快速筛选
    2. 完整回测: 计算所有指标，用于最终验证
    
    Usage:
        evaluator = FactorEvaluator(config)
        
        # 快速评估
        quick_result = evaluator.quick_evaluate(factor_code)
        
        # 完整回测
        full_result = evaluator.full_evaluate(factor_code)
    """
    
    def __init__(
        self,
        config: EvaluatorConfig = None,
        data_handler = None,      # Qlib DataHandler
        backtest_handler = None,  # 回测处理器
    ):
        self.config = config or EvaluatorConfig()
        self.data_handler = data_handler
        self.backtest_handler = backtest_handler
    
    def quick_evaluate(self, factor_code: str) -> Dict:
        """
        快速评估 - 只计算IC指标
        
        用于Phase 1: LLM探索阶段的快速筛选
        
        Returns:
            {
                'ic': float,
                'icir': float,
                'rank_ic': float,
                'rank_icir': float,
                'passed': bool,
                'reason': str,
            }
        """
        try:
            # 计算因子值
            factor_values = self._compute_factor(factor_code)
            if factor_values is None:
                return self._failed_result("因子计算失败")
            
            # 获取未来收益
            forward_returns = self._get_forward_returns()
            if forward_returns is None:
                return self._failed_result("获取收益失败")
            
            # 计算IC
            ic_metrics = compute_ic_metrics(factor_values, forward_returns)
            
            # 判断是否通过
            passed, reason = self._check_quick_pass(ic_metrics)
            
            return {
                'ic': ic_metrics.ic_mean,
                'icir': ic_metrics.icir,
                'rank_ic': ic_metrics.rank_ic_mean,
                'rank_icir': ic_metrics.rank_icir,
                'ic_grade': ic_metrics.ic_grade,
                'icir_grade': ic_metrics.icir_grade,
                'passed': passed,
                'reason': reason,
            }
        
        except Exception as e:
            logger.error(f"快速评估失败: {e}")
            return self._failed_result(str(e))
    
    def full_evaluate(self, factor_code: str) -> BacktestMetrics:
        """
        完整回测评估
        
        用于Phase 2: GP优胜者验证
        
        Returns:
            BacktestMetrics 完整指标
        """
        try:
            # 计算因子值
            factor_values = self._compute_factor(factor_code)
            if factor_values is None:
                return BacktestMetrics()
            
            # 获取未来收益 (IC用)
            forward_returns = self._get_forward_returns()
            
            # 模拟回测获取策略收益
            backtest_result = self._run_backtest(factor_values)
            strategy_returns = backtest_result.get('returns')
            benchmark_returns = backtest_result.get('benchmark_returns')
            turnover = backtest_result.get('turnover')
            
            # 计算所有指标
            metrics = compute_all_metrics(
                factor_values=factor_values,
                forward_returns=forward_returns,
                strategy_returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                turnover=turnover,
                risk_free_rate=self.config.risk_free_rate,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            
            # 添加分年度统计
            metrics.yearly_stats = self._compute_yearly_stats(
                strategy_returns, factor_values, forward_returns
            )
            
            return metrics
        
        except Exception as e:
            logger.error(f"完整回测失败: {e}")
            return BacktestMetrics()
    
    def evaluate(self, factor_code: str, full_backtest: bool = False) -> Dict:
        """
        统一评估接口
        
        Args:
            factor_code: 因子代码
            full_backtest: 是否完整回测
        
        Returns:
            Dict 指标字典
        """
        if full_backtest:
            metrics = self.full_evaluate(factor_code)
            return metrics.to_flat_dict()
        else:
            return self.quick_evaluate(factor_code)
    
    def validate(self, metrics: BacktestMetrics) -> tuple[bool, List[str]]:
        """
        验证因子是否通过筛选条件
        
        Returns:
            (passed, reasons)
        """
        reasons = []
        
        # IC验证
        if metrics.ic.ic_mean < self.config.min_ic:
            reasons.append(f"IC={metrics.ic.ic_mean:.4f}<{self.config.min_ic}")
        
        # ICIR验证
        if metrics.ic.icir < self.config.min_icir:
            reasons.append(f"ICIR={metrics.ic.icir:.2f}<{self.config.min_icir}")
        
        # Rank IC验证
        if metrics.ic.rank_ic_mean < self.config.min_rank_ic:
            reasons.append(f"RankIC={metrics.ic.rank_ic_mean:.4f}<{self.config.min_rank_ic}")
        
        # 夏普比率验证
        if metrics.risk.sharpe_ratio < self.config.min_sharpe:
            reasons.append(f"Sharpe={metrics.risk.sharpe_ratio:.2f}<{self.config.min_sharpe}")
        
        # 最大回撤验证
        if metrics.risk.max_drawdown > self.config.max_drawdown:
            reasons.append(f"MaxDD={metrics.risk.max_drawdown:.1%}>{self.config.max_drawdown:.1%}")
        
        # 换手率验证
        if metrics.returns.turnover > self.config.max_turnover:
            reasons.append(f"Turnover={metrics.returns.turnover:.1%}>{self.config.max_turnover:.1%}")
        
        passed = len(reasons) == 0
        return passed, reasons
    
    # ============================================================
    # 内部方法
    # ============================================================
    
    def _compute_factor(self, factor_code: str) -> Optional[np.ndarray]:
        """计算因子值"""
        if self.data_handler:
            # 使用Qlib DataHandler
            try:
                # 这里应该执行因子代码并获取因子值
                # factor_values = exec_factor(factor_code, self.data_handler)
                pass
            except Exception as e:
                logger.error(f"因子计算失败: {e}")
                return None
        
        # Mock实现 (测试用)
        T, N = 500, 100  # 500天, 100只股票
        return np.random.randn(T, N)
    
    def _get_forward_returns(self) -> Optional[np.ndarray]:
        """获取未来收益率"""
        if self.data_handler:
            # 从Qlib获取收益数据
            pass
        
        # Mock实现
        T, N = 500, 100
        return np.random.randn(T, N) * 0.02
    
    def _run_backtest(self, factor_values: np.ndarray) -> Dict:
        """运行回测"""
        if self.backtest_handler:
            # 使用Qlib回测框架
            pass
        
        # Mock实现
        T = factor_values.shape[0]
        returns = np.random.randn(T) * 0.01 + 0.0003  # 日均0.03%
        benchmark = np.random.randn(T) * 0.008
        turnover = np.random.uniform(0.1, 0.5, T)
        
        return {
            'returns': returns,
            'benchmark_returns': benchmark,
            'turnover': turnover,
        }
    
    def _compute_yearly_stats(
        self,
        strategy_returns: np.ndarray,
        factor_values: np.ndarray,
        forward_returns: np.ndarray,
    ) -> Dict:
        """计算分年度统计"""
        # Mock实现
        years = ['2019', '2020', '2021', '2022', '2023']
        yearly_stats = {}
        
        for year in years:
            yearly_stats[year] = {
                'ic': np.random.uniform(0.02, 0.05),
                'rank_ic': np.random.uniform(0.025, 0.06),
                'return': np.random.uniform(-0.1, 0.3),
                'sharpe': np.random.uniform(0.5, 2.0),
                'max_drawdown': np.random.uniform(0.05, 0.25),
            }
        
        return yearly_stats
    
    def _check_quick_pass(self, ic_metrics: ICMetrics) -> tuple[bool, str]:
        """检查快速筛选是否通过"""
        if ic_metrics.rank_ic_mean < self.config.min_ic * 0.75:  # 快速筛选阈值略低
            return False, f"RankIC={ic_metrics.rank_ic_mean:.4f}过低"
        
        if ic_metrics.rank_icir < self.config.min_icir * 0.6:
            return False, f"RankICIR={ic_metrics.rank_icir:.2f}过低"
        
        return True, "通过快速筛选"
    
    def _failed_result(self, reason: str) -> Dict:
        """返回失败结果"""
        return {
            'ic': 0,
            'icir': 0,
            'rank_ic': 0,
            'rank_icir': 0,
            'ic_grade': 'F',
            'icir_grade': 'F',
            'passed': False,
            'reason': reason,
        }
    
    # ============================================================
    # 报告生成
    # ============================================================
    
    def generate_report(self, metrics: BacktestMetrics) -> str:
        """生成评估报告"""
        passed, reasons = self.validate(metrics)
        status = "✓ 通过" if passed else "✗ 未通过"
        
        report = f"""
================================================================================
                           因子评估报告
================================================================================
评估时间: {metrics.start_date} ~ {metrics.end_date}
交易天数: {metrics.trading_days}
综合评分: {metrics.overall_score:.1f} ({metrics.overall_grade})
验证结果: {status}

--------------------------------------------------------------------------------
                           IC指标 (预测能力)
--------------------------------------------------------------------------------
  IC均值:       {metrics.ic.ic_mean:.4f}      (Pearson相关)
  ICIR:         {metrics.ic.icir:.2f}        (IC信息比)
  Rank IC均值:  {metrics.ic.rank_ic_mean:.4f}  (Spearman相关) [{metrics.ic.ic_grade}]
  Rank ICIR:    {metrics.ic.rank_icir:.2f}    (Rank IC信息比) [{metrics.ic.icir_grade}]
  IC正向率:     {metrics.ic.ic_positive_ratio:.1%}

--------------------------------------------------------------------------------
                           收益指标
--------------------------------------------------------------------------------
  累计收益:     {metrics.returns.total_return:.1%}
  年化收益:     {metrics.returns.annual_return:.1%}
  超额收益:     {metrics.returns.excess_return:.1%}
  胜率:         {metrics.returns.win_rate:.1%}
  盈亏比:       {metrics.returns.profit_loss_ratio:.2f}
  换手率:       {metrics.returns.turnover:.1%}

--------------------------------------------------------------------------------
                           风险指标
--------------------------------------------------------------------------------
  年化波动率:   {metrics.risk.volatility:.1%}
  最大回撤:     {metrics.risk.max_drawdown:.1%}  [{metrics.risk.drawdown_grade}]
  回撤天数:     {metrics.risk.max_drawdown_duration}天
  
  夏普比率:     {metrics.risk.sharpe_ratio:.2f}  [{metrics.risk.sharpe_grade}]
  索提诺比率:   {metrics.risk.sortino_ratio:.2f}
  卡玛比率:     {metrics.risk.calmar_ratio:.2f}
  信息比率:     {metrics.risk.information_ratio:.2f}
  
  95% VaR:      {metrics.risk.var_95:.2%}
  95% CVaR:     {metrics.risk.cvar_95:.2%}

--------------------------------------------------------------------------------
                           分年度统计
--------------------------------------------------------------------------------"""
        
        for year, stats in metrics.yearly_stats.items():
            report += f"\n  {year}: IC={stats.get('ic', 0):.3f}, Return={stats.get('return', 0):.1%}, Sharpe={stats.get('sharpe', 0):.2f}"
        
        if not passed:
            report += f"\n\n未通过原因: {', '.join(reasons)}"
        
        report += "\n================================================================================"
        
        return report
