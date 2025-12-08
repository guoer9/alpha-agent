"""
风险分析模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskReport:
    """风险报告"""
    factor_name: str
    # 风险暴露
    market_beta: float = 0.0
    size_exposure: float = 0.0
    value_exposure: float = 0.0
    momentum_exposure: float = 0.0
    volatility_exposure: float = 0.0
    # 风险指标
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    # 建议
    risk_level: str = "medium"
    recommendations: List[str] = None


class RiskAnalyzer:
    """风险分析器"""
    
    def __init__(self):
        self.risk_factors = {}
    
    def compute_factor_exposure(
        self,
        factor_returns: pd.Series,
        risk_factors: pd.DataFrame,
    ) -> Dict[str, float]:
        """计算因子风险暴露"""
        # 对齐数据
        aligned = pd.concat([factor_returns, risk_factors], axis=1).dropna()
        if len(aligned) < 30:
            return {}
        
        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]
        
        # 回归计算暴露
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            exposures = dict(zip(X.columns, model.coef_))
            exposures['r2'] = model.score(X, y)
            return exposures
        except Exception as e:
            logger.error(f"计算暴露失败: {e}")
            return {}
    
    def compute_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """计算VaR"""
        return np.percentile(returns.dropna(), (1 - confidence) * 100)
    
    def compute_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """计算CVaR (Expected Shortfall)"""
        var = self.compute_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def compute_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def analyze(
        self,
        factor_returns: pd.Series,
        risk_factors: pd.DataFrame = None,
    ) -> RiskReport:
        """完整风险分析"""
        report = RiskReport(factor_name=factor_returns.name or "factor")
        
        # 基础风险指标
        report.var_95 = self.compute_var(factor_returns, 0.95)
        report.cvar_95 = self.compute_cvar(factor_returns, 0.95)
        report.max_drawdown = self.compute_max_drawdown(factor_returns)
        
        # 风险暴露
        if risk_factors is not None:
            exposures = self.compute_factor_exposure(factor_returns, risk_factors)
            report.market_beta = exposures.get('market', 0)
            report.size_exposure = exposures.get('size', 0)
            report.value_exposure = exposures.get('value', 0)
            report.momentum_exposure = exposures.get('momentum', 0)
            report.volatility_exposure = exposures.get('volatility', 0)
        
        # 风险等级
        if abs(report.max_drawdown) > 0.2:
            report.risk_level = "high"
        elif abs(report.max_drawdown) > 0.1:
            report.risk_level = "medium"
        else:
            report.risk_level = "low"
        
        # 建议
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: RiskReport) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        if abs(report.market_beta) > 0.5:
            recommendations.append("市场Beta暴露较高，考虑对冲市场风险")
        
        if abs(report.max_drawdown) > 0.15:
            recommendations.append("最大回撤较大，建议控制仓位或设置止损")
        
        if abs(report.volatility_exposure) > 0.3:
            recommendations.append("波动率暴露较高，在高波动市场可能表现不稳定")
        
        if not recommendations:
            recommendations.append("风险水平可接受")
        
        return recommendations
