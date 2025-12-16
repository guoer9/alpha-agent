"""
市场状态识别 - 判断当前市场环境

识别市场状态:
1. 牛市/熊市/震荡
2. 高/低波动
3. 风格轮动 (价值/成长)
4. 行业轮动

方法:
- 隐马尔可夫模型 (HMM)
- 规则判断
- 技术指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn未安装: pip install hmmlearn")


class MarketState(Enum):
    """市场状态"""
    BULL = "bull"           # 牛市
    BEAR = "bear"           # 熊市
    SIDEWAYS = "sideways"   # 震荡
    HIGH_VOL = "high_vol"   # 高波动
    LOW_VOL = "low_vol"     # 低波动


@dataclass
class RegimeResult:
    """市场状态识别结果"""
    current_state: MarketState
    state_probability: Dict[str, float]
    state_history: Optional[pd.Series] = None
    indicators: Optional[Dict] = None


class MarketRegimeDetector:
    """市场状态识别器"""
    
    def __init__(
        self,
        method: str = 'rule',  # rule, hmm
        n_states: int = 3,
        lookback: int = 60,
    ):
        self.method = method
        self.n_states = n_states
        self.lookback = lookback
        self.hmm_model = None
    
    def fit(self, returns: pd.Series):
        """训练模型 (仅HMM需要)"""
        if self.method != 'hmm':
            return
        
        if not HMM_AVAILABLE:
            raise ImportError("请安装hmmlearn: pip install hmmlearn")
        
        # 准备特征
        features = self._prepare_features(returns)
        
        # 训练HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
        )
        self.hmm_model.fit(features)
    
    def _prepare_features(self, returns: pd.Series) -> np.ndarray:
        """准备HMM特征"""
        features = pd.DataFrame()
        features['return'] = returns
        features['volatility'] = returns.rolling(20).std()
        features['momentum'] = returns.rolling(20).mean()
        features = features.dropna()
        return features.values
    
    def detect(self, returns: pd.Series) -> RegimeResult:
        """识别当前市场状态"""
        if self.method == 'hmm':
            return self._detect_hmm(returns)
        return self._detect_rule(returns)
    
    def _detect_rule(self, returns: pd.Series) -> RegimeResult:
        """基于规则的状态识别"""
        if len(returns) < self.lookback:
            return RegimeResult(
                current_state=MarketState.SIDEWAYS,
                state_probability={'sideways': 1.0},
            )
        
        recent = returns.tail(self.lookback)
        
        # 计算指标
        cum_return = (1 + recent).prod() - 1
        volatility = recent.std() * np.sqrt(252)
        sharpe = recent.mean() / recent.std() * np.sqrt(252) if recent.std() > 0 else 0
        
        # 趋势判断
        ma_20 = returns.rolling(20).mean().iloc[-1]
        ma_60 = returns.rolling(60).mean().iloc[-1]
        
        # 状态判断
        indicators = {
            'cum_return': cum_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'ma_20': ma_20,
            'ma_60': ma_60,
        }
        
        probabilities = {}
        
        # 牛市: 正收益 + 上升趋势
        if cum_return > 0.1 and ma_20 > ma_60:
            current_state = MarketState.BULL
            probabilities = {'bull': 0.7, 'sideways': 0.2, 'bear': 0.1}
        # 熊市: 负收益 + 下降趋势
        elif cum_return < -0.1 and ma_20 < ma_60:
            current_state = MarketState.BEAR
            probabilities = {'bear': 0.7, 'sideways': 0.2, 'bull': 0.1}
        # 震荡: 其他情况
        else:
            current_state = MarketState.SIDEWAYS
            probabilities = {'sideways': 0.6, 'bull': 0.2, 'bear': 0.2}
        
        # 波动率状态
        if volatility > 0.25:
            indicators['vol_state'] = 'high'
        elif volatility < 0.15:
            indicators['vol_state'] = 'low'
        else:
            indicators['vol_state'] = 'normal'
        
        return RegimeResult(
            current_state=current_state,
            state_probability=probabilities,
            indicators=indicators,
        )
    
    def _detect_hmm(self, returns: pd.Series) -> RegimeResult:
        """基于HMM的状态识别"""
        if self.hmm_model is None:
            self.fit(returns)
        
        features = self._prepare_features(returns)
        
        # 预测状态
        states = self.hmm_model.predict(features)
        probs = self.hmm_model.predict_proba(features)
        
        # 当前状态
        current_idx = states[-1]
        current_probs = probs[-1]
        
        # 映射到市场状态
        state_mapping = {
            0: MarketState.BEAR,
            1: MarketState.SIDEWAYS,
            2: MarketState.BULL,
        }
        
        current_state = state_mapping.get(current_idx, MarketState.SIDEWAYS)
        
        state_probability = {
            'bear': float(current_probs[0]),
            'sideways': float(current_probs[1]),
            'bull': float(current_probs[2]) if len(current_probs) > 2 else 0,
        }
        
        # 历史状态
        state_history = pd.Series(
            [state_mapping.get(s, MarketState.SIDEWAYS).value for s in states],
            index=returns.index[-len(states):],
        )
        
        return RegimeResult(
            current_state=current_state,
            state_probability=state_probability,
            state_history=state_history,
        )


def detect_style_rotation(
    value_returns: pd.Series,
    growth_returns: pd.Series,
    window: int = 60,
) -> Dict:
    """
    检测价值/成长风格轮动
    
    返回:
        当前风格偏好和历史
    """
    # 相对强度
    relative = value_returns.rolling(window).sum() - growth_returns.rolling(window).sum()
    
    current = relative.iloc[-1]
    
    if current > 0.05:
        style = 'value'
    elif current < -0.05:
        style = 'growth'
    else:
        style = 'neutral'
    
    return {
        'current_style': style,
        'relative_strength': current,
        'history': relative,
    }


def detect_sector_rotation(
    sector_returns: pd.DataFrame,
    window: int = 20,
) -> Dict:
    """
    检测行业轮动
    
    参数:
        sector_returns: 行业收益 (columns=sector names)
        window: 观察窗口
    
    返回:
        热门/冷门行业
    """
    # 区间收益
    period_returns = sector_returns.tail(window).sum()
    
    # 排序
    ranked = period_returns.sort_values(ascending=False)
    
    return {
        'hot_sectors': list(ranked.head(3).index),
        'cold_sectors': list(ranked.tail(3).index),
        'sector_ranking': ranked,
        'dispersion': period_returns.std(),  # 行业分化程度
    }


def format_regime_report(result: RegimeResult) -> str:
    """格式化状态报告"""
    report = f"""
{'='*60}
市场状态识别报告
{'='*60}

【当前状态】
  主状态: {result.current_state.value.upper()}

【状态概率】
"""
    for state, prob in result.state_probability.items():
        bar = '█' * int(prob * 20)
        report += f"  {state:10s}: {bar} {prob:.1%}\n"
    
    if result.indicators:
        report += f"""
【市场指标】
  累积收益: {result.indicators.get('cum_return', 0):+.2%}
  年化波动: {result.indicators.get('volatility', 0):.2%}
  夏普比率: {result.indicators.get('sharpe', 0):.2f}
  波动状态: {result.indicators.get('vol_state', 'N/A')}
"""
    
    report += f"{'='*60}\n"
    return report
