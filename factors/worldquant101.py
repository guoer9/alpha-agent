"""
WorldQuant 101 Formulaic Alphas

来源: Kakushadze (2016) "101 Formulaic Alphas"
论文: https://arxiv.org/abs/1601.00991
包含: 101 个公式化因子
"""

from dataclasses import dataclass, field
from typing import List
from .classic_factors import ClassicFactor, FactorCategory


# ============================================================
# WorldQuant 101 Alphas
# 选取最有代表性的因子
# ============================================================

WORLDQUANT_101_FACTORS = [
    # ==================== Alpha#1 ====================
    ClassicFactor(
        id="wq101_alpha001",
        name="WQ#001 排名反转",
        name_en="WQ101_Alpha001",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#001: (rank(Ts_ArgMax(SignedPower(returns, 2), 5)) - 0.5)
    短期反转信号
    """
    returns = df['close'].pct_change()
    signed_power = returns.abs() ** 2 * returns.apply(lambda x: 1 if x >= 0 else -1)
    argmax = signed_power.rolling(5).apply(lambda x: x.argmax())
    return argmax.rank(pct=True) - 0.5
''',
        description="基于收益率平方的排名反转",
        logic="极端收益后的反转效应",
        reference="Kakushadze (2016) 101 Formulaic Alphas",
        author="Kakushadze",
        year=2016,
        historical_ic=0.020,
        tags=["worldquant101", "reversal", "short_term"],
    ),
    
    # ==================== Alpha#2 ====================
    ClassicFactor(
        id="wq101_alpha002",
        name="WQ#002 量价背离",
        name_en="WQ101_Alpha002",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#002: -1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)
    量价背离信号
    """
    import numpy as np
    delta_log_vol = np.log(df['volume'] + 1).diff(2)
    price_change = (df['close'] - df['open']) / df['open']
    corr = delta_log_vol.rolling(6).corr(price_change)
    return -corr
''',
        description="成交量变化与价格变化的负相关",
        logic="量价背离预示趋势可能反转",
        reference="Kakushadze (2016)",
        historical_ic=0.018,
        tags=["worldquant101", "volume_price", "divergence"],
    ),
    
    # ==================== Alpha#3 ====================
    ClassicFactor(
        id="wq101_alpha003",
        name="WQ#003 开盘量价",
        name_en="WQ101_Alpha003",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#003: -1 * correlation(rank(open), rank(volume), 10)
    开盘价与成交量的负相关
    """
    corr = df['open'].rank(pct=True).rolling(10).corr(df['volume'].rank(pct=True))
    return -corr
''',
        description="开盘价排名与成交量排名的负相关",
        logic="高开低成交量可能回落",
        reference="Kakushadze (2016)",
        historical_ic=0.015,
        tags=["worldquant101", "volume_price", "open"],
    ),
    
    # ==================== Alpha#4 ====================
    ClassicFactor(
        id="wq101_alpha004",
        name="WQ#004 低位排名",
        name_en="WQ101_Alpha004",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#004: -1 * Ts_Rank(rank(low), 9)
    最低价时序排名的负值
    """
    low_rank = df['low'].rank(pct=True)
    ts_rank = low_rank.rolling(9).apply(lambda x: x.rank().iloc[-1] / len(x))
    return -ts_rank
''',
        description="最低价在时序中的排名位置",
        logic="近期创新低后的反弹",
        reference="Kakushadze (2016)",
        historical_ic=0.022,
        tags=["worldquant101", "reversal", "low"],
    ),
    
    # ==================== Alpha#5 ====================
    ClassicFactor(
        id="wq101_alpha005",
        name="WQ#005 VWAP动量",
        name_en="WQ101_Alpha005",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#005: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    VWAP相关动量
    """
    vwap = df['amount'] / (df['volume'] + 1e-8)
    vwap_ma = vwap.rolling(10).mean()
    term1 = (df['open'] - vwap_ma).rank(pct=True)
    term2 = (df['close'] - vwap).abs().rank(pct=True)
    return term1 * (-term2)
''',
        description="基于VWAP的复合动量",
        logic="开盘相对VWAP均线的位置与收盘偏离的组合",
        reference="Kakushadze (2016)",
        historical_ic=0.025,
        tags=["worldquant101", "vwap", "momentum"],
    ),
    
    # ==================== Alpha#6 ====================
    ClassicFactor(
        id="wq101_alpha006",
        name="WQ#006 开盘量相关",
        name_en="WQ101_Alpha006",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#006: -1 * correlation(open, volume, 10)
    开盘价与成交量10日相关性
    """
    return -df['open'].rolling(10).corr(df['volume'])
''',
        description="开盘价与成交量的负相关",
        logic="开盘高但成交量低不健康",
        reference="Kakushadze (2016)",
        historical_ic=0.015,
        tags=["worldquant101", "correlation", "volume"],
    ),
    
    # ==================== Alpha#7 ====================
    ClassicFactor(
        id="wq101_alpha007",
        name="WQ#007 成交额动量",
        name_en="WQ101_Alpha007",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#007: 成交额变化与价格变化
    """
    adv20 = df['amount'].rolling(20).mean()
    delta_close = df['close'].diff(7)
    condition = df['amount'] < adv20
    result = -delta_close.abs().rank(pct=True) * delta_close.apply(lambda x: 1 if x >= 0 else -1)
    return result.where(condition, -1)
''',
        description="基于成交额的条件动量",
        logic="低成交额时的价格变化信号",
        reference="Kakushadze (2016)",
        historical_ic=0.020,
        tags=["worldquant101", "amount", "conditional"],
    ),
    
    # ==================== Alpha#8 ====================
    ClassicFactor(
        id="wq101_alpha008",
        name="WQ#008 开高低综合",
        name_en="WQ101_Alpha008",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#008: -1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))
    """
    returns = df['close'].pct_change()
    term = df['open'].rolling(5).sum() * returns.rolling(5).sum()
    delta = term - term.shift(10)
    return -delta.rank(pct=True)
''',
        description="开盘价与收益率组合的变化",
        logic="开盘动量的变化",
        reference="Kakushadze (2016)",
        historical_ic=0.018,
        tags=["worldquant101", "open", "momentum"],
    ),
    
    # ==================== Alpha#9 ====================
    ClassicFactor(
        id="wq101_alpha009",
        name="WQ#009 价格延迟",
        name_en="WQ101_Alpha009",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#009: 基于收盘价变化的延迟信号
    """
    delta_close = df['close'].diff(1)
    ts_min = delta_close.rolling(5).min()
    ts_max = delta_close.rolling(5).max()
    
    condition = ts_min > 0
    result = delta_close.copy()
    result = result.where(condition, 
                          delta_close.where(ts_max < 0, -delta_close))
    return result
''',
        description="基于价格变化趋势的条件信号",
        logic="趋势延续或反转的判断",
        reference="Kakushadze (2016)",
        historical_ic=0.022,
        tags=["worldquant101", "trend", "conditional"],
    ),
    
    # ==================== Alpha#10 ====================
    ClassicFactor(
        id="wq101_alpha010",
        name="WQ#010 趋势排名",
        name_en="WQ101_Alpha010",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#010: rank(delta_close > 0 ? delta_close : delta_close)
    """
    delta_close = df['close'].diff(1)
    ts_min = delta_close.rolling(4).min()
    ts_max = delta_close.rolling(4).max()
    
    result = delta_close.where(ts_min > 0, 
                               delta_close.where(ts_max < 0, -delta_close))
    return result.rank(pct=True)
''',
        description="基于价格变化方向的排名",
        logic="趋势强度排名",
        reference="Kakushadze (2016)",
        historical_ic=0.020,
        tags=["worldquant101", "trend", "rank"],
    ),
    
    # ==================== Alpha#12 ====================
    ClassicFactor(
        id="wq101_alpha012",
        name="WQ#012 量价动量",
        name_en="WQ101_Alpha012",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#012: sign(delta(volume, 1)) * (-1 * delta(close, 1))
    成交量变化方向与价格变化的组合
    """
    import numpy as np
    delta_vol = df['volume'].diff(1)
    delta_close = df['close'].diff(1)
    return np.sign(delta_vol) * (-delta_close)
''',
        description="量增价跌或量缩价涨信号",
        logic="量价背离的简单捕捉",
        reference="Kakushadze (2016)",
        historical_ic=0.025,
        tags=["worldquant101", "volume_price", "divergence"],
    ),
    
    # ==================== Alpha#13 ====================
    ClassicFactor(
        id="wq101_alpha013",
        name="WQ#013 协方差排名",
        name_en="WQ101_Alpha013",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#013: -1 * rank(covariance(rank(close), rank(volume), 5))
    收盘价排名与成交量排名的协方差
    """
    close_rank = df['close'].rank(pct=True)
    vol_rank = df['volume'].rank(pct=True)
    cov = close_rank.rolling(5).cov(vol_rank)
    return -cov.rank(pct=True)
''',
        description="量价排名协方差的负值",
        logic="量价同向变化的程度",
        reference="Kakushadze (2016)",
        historical_ic=0.018,
        tags=["worldquant101", "covariance", "rank"],
    ),
    
    # ==================== Alpha#14 ====================
    ClassicFactor(
        id="wq101_alpha014",
        name="WQ#014 收益相关",
        name_en="WQ101_Alpha014",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#014: -1 * rank(delta(returns, 3)) * correlation(open, volume, 10)
    """
    returns = df['close'].pct_change()
    delta_ret = returns.diff(3)
    corr = df['open'].rolling(10).corr(df['volume'])
    return -delta_ret.rank(pct=True) * corr
''',
        description="收益率变化与开盘量相关的组合",
        logic="动量变化与流动性的交互",
        reference="Kakushadze (2016)",
        historical_ic=0.020,
        tags=["worldquant101", "returns", "correlation"],
    ),
    
    # ==================== Alpha#15 ====================
    ClassicFactor(
        id="wq101_alpha015",
        name="WQ#015 高价相关",
        name_en="WQ101_Alpha015",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#015: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
    """
    high_rank = df['high'].rank(pct=True)
    vol_rank = df['volume'].rank(pct=True)
    corr = high_rank.rolling(3).corr(vol_rank)
    return -corr.rank(pct=True).rolling(3).sum()
''',
        description="最高价与成交量排名相关性的累计",
        logic="高点放量的持续性",
        reference="Kakushadze (2016)",
        historical_ic=0.015,
        tags=["worldquant101", "high", "volume"],
    ),
    
    # ==================== Alpha#16 ====================
    ClassicFactor(
        id="wq101_alpha016",
        name="WQ#016 高价协方差",
        name_en="WQ101_Alpha016",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#016: -1 * rank(covariance(rank(high), rank(volume), 5))
    """
    high_rank = df['high'].rank(pct=True)
    vol_rank = df['volume'].rank(pct=True)
    cov = high_rank.rolling(5).cov(vol_rank)
    return -cov.rank(pct=True)
''',
        description="最高价与成交量排名的协方差",
        logic="高点与成交量的同步程度",
        reference="Kakushadze (2016)",
        historical_ic=0.018,
        tags=["worldquant101", "covariance", "high"],
    ),
    
    # ==================== Alpha#17 ====================
    ClassicFactor(
        id="wq101_alpha017",
        name="WQ#017 收盘动量",
        name_en="WQ101_Alpha017",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#017: 基于收盘价和成交量的复合排名
    """
    ts_rank_vol = df['volume'].rolling(5).apply(lambda x: x.rank().iloc[-1] / len(x))
    delta_close = df['close'].diff(1)
    ts_rank_close = df['close'].rolling(5).apply(lambda x: x.rank().iloc[-1] / len(x))
    return -ts_rank_vol.rank(pct=True) * delta_close.rank(pct=True) * ts_rank_close.rank(pct=True)
''',
        description="成交量、价格变化、价格位置的三重排名",
        logic="多维度动量信号",
        reference="Kakushadze (2016)",
        historical_ic=0.022,
        tags=["worldquant101", "multi_factor", "rank"],
    ),
    
    # ==================== Alpha#18 ====================
    ClassicFactor(
        id="wq101_alpha018",
        name="WQ#018 收盘相关",
        name_en="WQ101_Alpha018",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#018: -1 * rank(stddev(abs(close - open), 5) + (close - open) + correlation(close, open, 10))
    """
    diff = df['close'] - df['open']
    std_term = diff.abs().rolling(5).std()
    corr_term = df['close'].rolling(10).corr(df['open'])
    combined = std_term + diff + corr_term
    return -combined.rank(pct=True)
''',
        description="开收价差的波动与相关性组合",
        logic="日内波动模式",
        reference="Kakushadze (2016)",
        historical_ic=0.020,
        tags=["worldquant101", "intraday", "volatility"],
    ),
    
    # ==================== Alpha#19 ====================
    ClassicFactor(
        id="wq101_alpha019",
        name="WQ#019 收益反转",
        name_en="WQ101_Alpha019",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#019: (-1 * sign((close - delay(close, 7)) + delta(close, 7))) * (1 + rank(1 + sum(returns, 250)))
    """
    import numpy as np
    returns = df['close'].pct_change()
    delay_close = df['close'].shift(7)
    delta_close = df['close'].diff(7)
    sign_term = np.sign((df['close'] - delay_close) + delta_close)
    rank_term = 1 + (1 + returns.rolling(250).sum()).rank(pct=True)
    return -sign_term * rank_term
''',
        description="周动量与年度动量的组合",
        logic="短期与长期动量的交互",
        reference="Kakushadze (2016)",
        historical_ic=0.018,
        tags=["worldquant101", "momentum", "multi_horizon"],
    ),
    
    # ==================== Alpha#20 ====================
    ClassicFactor(
        id="wq101_alpha020",
        name="WQ#020 开高低差",
        name_en="WQ101_Alpha020",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#020: (-1 * rank(open - delay(high, 1))) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))
    """
    term1 = (df['open'] - df['high'].shift(1)).rank(pct=True)
    term2 = (df['open'] - df['close'].shift(1)).rank(pct=True)
    term3 = (df['open'] - df['low'].shift(1)).rank(pct=True)
    return -term1 * term2 * term3
''',
        description="开盘价相对昨日高低收的位置",
        logic="跳空幅度的综合度量",
        reference="Kakushadze (2016)",
        historical_ic=0.022,
        tags=["worldquant101", "gap", "open"],
    ),
    
    # ==================== Alpha#21 ====================
    ClassicFactor(
        id="wq101_alpha021",
        name="WQ#021 均值回归",
        name_en="WQ101_Alpha021",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#021: 基于成交量和收盘价均线的条件信号
    """
    vol_ma8 = df['volume'].rolling(8).mean()
    vol_std8 = df['volume'].rolling(8).std()
    close_ma8 = df['close'].rolling(8).mean()
    
    cond1 = (vol_ma8 / vol_std8) < 1
    cond2 = close_ma8 + df['close'].rolling(8).std() < df['close']
    
    result = -1 * (df['close'].rolling(2).sum() / 2 - df['close'])
    result = result.where(cond1, -1)
    return result
''',
        description="基于波动率条件的均值回归",
        logic="低波动时的均值回归信号",
        reference="Kakushadze (2016)",
        historical_ic=0.020,
        tags=["worldquant101", "mean_reversion", "conditional"],
    ),
    
    # ==================== Alpha#22 ====================
    ClassicFactor(
        id="wq101_alpha022",
        name="WQ#022 高价相关变化",
        name_en="WQ101_Alpha022",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#022: -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))
    """
    corr = df['high'].rolling(5).corr(df['volume'])
    delta_corr = corr.diff(5)
    std_rank = df['close'].rolling(20).std().rank(pct=True)
    return -delta_corr * std_rank
''',
        description="高价成交量相关性变化与波动率的组合",
        logic="相关性变化在高波动中的放大",
        reference="Kakushadze (2016)",
        historical_ic=0.018,
        tags=["worldquant101", "correlation", "volatility"],
    ),
    
    # ==================== Alpha#23 ====================
    ClassicFactor(
        id="wq101_alpha023",
        name="WQ#023 高价延迟",
        name_en="WQ101_Alpha023",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#023: 基于最高价均线的条件信号
    """
    sma_high = df['high'].rolling(20).mean()
    delta_high = df['high'].diff(2)
    
    result = -delta_high.where(sma_high < df['high'], 0)
    return result
''',
        description="突破高价均线后的动量",
        logic="新高后的动量延续",
        reference="Kakushadze (2016)",
        historical_ic=0.020,
        tags=["worldquant101", "breakout", "high"],
    ),
    
    # ==================== Alpha#24 ====================
    ClassicFactor(
        id="wq101_alpha024",
        name="WQ#024 收盘延迟",
        name_en="WQ101_Alpha024",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#024: 基于收盘价变化的条件信号
    """
    sma_close = df['close'].rolling(100).mean()
    delta_sma = sma_close.diff(100) / 100
    delta_close = df['close'].diff(3)
    
    result = -delta_close.where(delta_sma < df['close'] - sma_close, 
                                (df['close'] - sma_close) - delta_sma)
    return result
''',
        description="相对长期均线的动量",
        logic="长期趋势与短期动量的交互",
        reference="Kakushadze (2016)",
        historical_ic=0.022,
        tags=["worldquant101", "trend", "ma"],
    ),
    
    # ==================== Alpha#25 ====================
    ClassicFactor(
        id="wq101_alpha025",
        name="WQ#025 收益加权",
        name_en="WQ101_Alpha025",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#025: rank((-1 * returns) * adv20 * vwap * (high - close))
    """
    returns = df['close'].pct_change()
    adv20 = df['amount'].rolling(20).mean()
    vwap = df['amount'] / (df['volume'] + 1e-8)
    
    combined = (-returns) * adv20 * vwap * (df['high'] - df['close'])
    return combined.rank(pct=True)
''',
        description="收益、流动性、VWAP、上影的综合",
        logic="多因子复合信号",
        reference="Kakushadze (2016)",
        historical_ic=0.025,
        tags=["worldquant101", "multi_factor", "composite"],
    ),
    
    # ==================== Alpha#26 ====================
    ClassicFactor(
        id="wq101_alpha026",
        name="WQ#026 高价相关累计",
        name_en="WQ101_Alpha026",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#026: -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    """
    vol_ts_rank = df['volume'].rolling(5).apply(lambda x: x.rank().iloc[-1] / len(x))
    high_ts_rank = df['high'].rolling(5).apply(lambda x: x.rank().iloc[-1] / len(x))
    corr = vol_ts_rank.rolling(5).corr(high_ts_rank)
    return -corr.rolling(3).max()
''',
        description="成交量与高价时序排名相关性的最大值",
        logic="量价同步的极值",
        reference="Kakushadze (2016)",
        historical_ic=0.018,
        tags=["worldquant101", "ts_rank", "correlation"],
    ),
    
    # ==================== Alpha#27 ====================
    ClassicFactor(
        id="wq101_alpha027",
        name="WQ#027 量相关排名",
        name_en="WQ101_Alpha027",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#027: 基于量价相关性的条件信号
    """
    vol_rank = df['volume'].rank(pct=True)
    vwap = df['amount'] / (df['volume'] + 1e-8)
    vwap_rank = vwap.rank(pct=True)
    corr = vol_rank.rolling(6).corr(vwap_rank)
    
    mean_corr = corr.rolling(2).mean()
    return mean_corr.rank(pct=True).where(mean_corr > 0.5, -1)
''',
        description="量价相关性排名的条件信号",
        logic="强相关时的动量",
        reference="Kakushadze (2016)",
        historical_ic=0.020,
        tags=["worldquant101", "conditional", "correlation"],
    ),
    
    # ==================== Alpha#28 ====================
    ClassicFactor(
        id="wq101_alpha028",
        name="WQ#028 成交额相关",
        name_en="WQ101_Alpha028",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#028: scale(correlation(adv20, low, 5) + (high + low) / 2 - close)
    """
    adv20 = df['amount'].rolling(20).mean()
    corr = adv20.rolling(5).corr(df['low'])
    mid = (df['high'] + df['low']) / 2
    combined = corr + mid - df['close']
    # scale: normalize to sum to 1
    return combined / (combined.abs().sum() + 1e-8)
''',
        description="流动性与低价相关性加中价偏离",
        logic="流动性与价格位置的交互",
        reference="Kakushadze (2016)",
        historical_ic=0.015,
        tags=["worldquant101", "liquidity", "position"],
    ),
    
    # ==================== Alpha#29 ====================
    ClassicFactor(
        id="wq101_alpha029",
        name="WQ#029 收益排名",
        name_en="WQ101_Alpha029",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#029: 多周期收益的时序排名组合
    """
    returns = df['close'].pct_change()
    
    ts_rank5 = returns.rolling(5).apply(lambda x: x.rank().iloc[-1] / len(x))
    ts_rank_ret = ts_rank5.rolling(2).apply(lambda x: x.rank().iloc[-1] / len(x))
    
    delta_ret = returns.rolling(6).sum().diff(5)
    delta_scale = delta_ret / (delta_ret.abs().sum() + 1e-8)
    
    return min(ts_rank_ret.iloc[-1] if len(ts_rank_ret) > 0 else 0, delta_scale.iloc[-1] if len(delta_scale) > 0 else 0)
''',
        description="短期收益排名与中期收益变化的最小值",
        logic="保守的动量估计",
        reference="Kakushadze (2016)",
        historical_ic=0.018,
        tags=["worldquant101", "momentum", "conservative"],
    ),
    
    # ==================== Alpha#30 ====================
    ClassicFactor(
        id="wq101_alpha030",
        name="WQ#030 收盘变化",
        name_en="WQ101_Alpha030",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    WQ Alpha#030: 基于价格变化和成交量的复合信号
    """
    import numpy as np
    delta_close = df['close'].diff(1)
    sign_delta = np.sign(delta_close) + np.sign(delta_close.shift(1)) + np.sign(delta_close.shift(2))
    vol_sum = df['volume'].rolling(5).sum()
    vol_sum3 = df['volume'].rolling(3).sum()
    
    return (1.0 - sign_delta.rank(pct=True)) * vol_sum3 / vol_sum
''',
        description="价格变化方向与成交量的组合",
        logic="趋势强度与成交量的交互",
        reference="Kakushadze (2016)",
        historical_ic=0.020,
        tags=["worldquant101", "trend", "volume"],
    ),
]


def get_worldquant101_factors():
    """获取WorldQuant 101因子"""
    return WORLDQUANT_101_FACTORS


def get_worldquant101_count():
    """获取WorldQuant 101因子数量"""
    return len(WORLDQUANT_101_FACTORS)
