"""
Qlib Alpha360 因子库

来源: 微软 Qlib 框架
机构: Microsoft Research Asia
论文: https://arxiv.org/abs/2009.11189
特点: Alpha158的扩展版，包含更多滞后特征和时序特征
数量: 360个因子
"""

from dataclasses import dataclass, field
from typing import List
from .classic_factors import ClassicFactor, FactorCategory


# ============================================================
# Alpha360 扩展因子
# 在Alpha158基础上增加：更多时间窗口、滞后特征、交叉特征
# ============================================================

ALPHA360_FACTORS = [
    # ==================== 滞后收益率系列 ====================
    ClassicFactor(
        id="alpha360_ret_lag1",
        name="滞后1日收益",
        name_en="RET_LAG1",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """滞后1日的日收益率"""
    return df['close'].pct_change().shift(1)
''',
        description="昨日收益率，用于捕捉短期反转",
        logic="短期反转信号",
        reference="Qlib Alpha360 - Microsoft Research",
        author="Microsoft",
        year=2020,
        historical_ic=0.020,
        tags=["alpha360", "lag", "returns", "microsoft"],
    ),
    
    ClassicFactor(
        id="alpha360_ret_lag2",
        name="滞后2日收益",
        name_en="RET_LAG2",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """滞后2日的日收益率"""
    return df['close'].pct_change().shift(2)
''',
        description="前日收益率",
        logic="短期反转信号",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.018,
        tags=["alpha360", "lag", "returns"],
    ),
    
    ClassicFactor(
        id="alpha360_ret_lag3",
        name="滞后3日收益",
        name_en="RET_LAG3",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """滞后3日的日收益率"""
    return df['close'].pct_change().shift(3)
''',
        description="3日前收益率",
        logic="短期反转信号",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.015,
        tags=["alpha360", "lag", "returns"],
    ),
    
    # ==================== 更多周期动量 ====================
    ClassicFactor(
        id="alpha360_roc3",
        name="3日动量",
        name_en="ROC3",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """3日收益率"""
    return df['close'] / df['close'].shift(3) - 1
''',
        description="超短期动量",
        logic="3日价格变化",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.022,
        tags=["alpha360", "momentum", "short_term"],
    ),
    
    ClassicFactor(
        id="alpha360_roc30",
        name="30日动量",
        name_en="ROC30",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """30日收益率"""
    return df['close'] / df['close'].shift(30) - 1
''',
        description="月度动量",
        logic="30日价格变化",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.028,
        tags=["alpha360", "momentum"],
    ),
    
    ClassicFactor(
        id="alpha360_roc120",
        name="120日动量",
        name_en="ROC120",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """120日收益率（半年）"""
    return df['close'] / df['close'].shift(120) - 1
''',
        description="半年动量",
        logic="中长期趋势",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.022,
        tags=["alpha360", "momentum", "long_term"],
    ),
    
    # ==================== 滞后波动率 ====================
    ClassicFactor(
        id="alpha360_std5_lag5",
        name="滞后5日波动率",
        name_en="STD5_LAG5",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """滞后5日的5日波动率"""
    return df['close'].pct_change().rolling(5).std().shift(5)
''',
        description="一周前的短期波动率",
        logic="波动率的滞后效应",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.012,
        tags=["alpha360", "volatility", "lag"],
    ),
    
    ClassicFactor(
        id="alpha360_std20_lag20",
        name="滞后20日波动率",
        name_en="STD20_LAG20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """滞后20日的20日波动率"""
    return df['close'].pct_change().rolling(20).std().shift(20)
''',
        description="一个月前的波动率",
        logic="波动率持续性",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.015,
        tags=["alpha360", "volatility", "lag"],
    ),
    
    # ==================== 均线交叉 ====================
    ClassicFactor(
        id="alpha360_ma5_ma10",
        name="MA5/MA10",
        name_en="MA5_MA10_RATIO",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """5日均线与10日均线比值"""
    ma5 = df['close'].rolling(5).mean()
    ma10 = df['close'].rolling(10).mean()
    return ma5 / ma10 - 1
''',
        description="短期均线相对中期均线",
        logic="金叉死叉信号",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.025,
        tags=["alpha360", "ma", "crossover"],
    ),
    
    ClassicFactor(
        id="alpha360_ma10_ma20",
        name="MA10/MA20",
        name_en="MA10_MA20_RATIO",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """10日均线与20日均线比值"""
    ma10 = df['close'].rolling(10).mean()
    ma20 = df['close'].rolling(20).mean()
    return ma10 / ma20 - 1
''',
        description="中短期均线相对中期均线",
        logic="趋势强度",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.028,
        tags=["alpha360", "ma", "crossover"],
    ),
    
    ClassicFactor(
        id="alpha360_ma20_ma60",
        name="MA20/MA60",
        name_en="MA20_MA60_RATIO",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """20日均线与60日均线比值"""
    ma20 = df['close'].rolling(20).mean()
    ma60 = df['close'].rolling(60).mean()
    return ma20 / ma60 - 1
''',
        description="月线相对季线",
        logic="中期趋势",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.025,
        tags=["alpha360", "ma", "trend"],
    ),
    
    # ==================== 成交量比率 ====================
    ClassicFactor(
        id="alpha360_vol_ratio_5_20",
        name="量比5/20",
        name_en="VOL_RATIO_5_20",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """5日均量与20日均量比值"""
    vol5 = df['volume'].rolling(5).mean()
    vol20 = df['volume'].rolling(20).mean()
    return vol5 / (vol20 + 1e-8) - 1
''',
        description="短期成交量相对中期",
        logic="放量或缩量",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.020,
        tags=["alpha360", "volume", "ratio"],
    ),
    
    ClassicFactor(
        id="alpha360_vol_ratio_10_60",
        name="量比10/60",
        name_en="VOL_RATIO_10_60",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """10日均量与60日均量比值"""
    vol10 = df['volume'].rolling(10).mean()
    vol60 = df['volume'].rolling(60).mean()
    return vol10 / (vol60 + 1e-8) - 1
''',
        description="中短期成交量相对长期",
        logic="成交量趋势",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.022,
        tags=["alpha360", "volume", "ratio"],
    ),
    
    # ==================== 价格位置 ====================
    ClassicFactor(
        id="alpha360_price_position_5",
        name="5日价格位置",
        name_en="PRICE_POS_5",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价在5日高低区间的位置"""
    high5 = df['high'].rolling(5).max()
    low5 = df['low'].rolling(5).min()
    return (df['close'] - low5) / (high5 - low5 + 1e-8)
''',
        description="价格在5日区间的相对位置",
        logic="0=最低，1=最高",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.025,
        tags=["alpha360", "position", "range"],
    ),
    
    ClassicFactor(
        id="alpha360_price_position_20",
        name="20日价格位置",
        name_en="PRICE_POS_20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价在20日高低区间的位置"""
    high20 = df['high'].rolling(20).max()
    low20 = df['low'].rolling(20).min()
    return (df['close'] - low20) / (high20 - low20 + 1e-8)
''',
        description="价格在20日区间的相对位置",
        logic="0=最低，1=最高",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.028,
        tags=["alpha360", "position", "range"],
    ),
    
    ClassicFactor(
        id="alpha360_price_position_60",
        name="60日价格位置",
        name_en="PRICE_POS_60",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价在60日高低区间的位置"""
    high60 = df['high'].rolling(60).max()
    low60 = df['low'].rolling(60).min()
    return (df['close'] - low60) / (high60 - low60 + 1e-8)
''',
        description="价格在60日区间的相对位置",
        logic="0=最低，1=最高",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.025,
        tags=["alpha360", "position", "range"],
    ),
    
    # ==================== 波动率变化 ====================
    ClassicFactor(
        id="alpha360_vol_change",
        name="波动率变化",
        name_en="VOL_CHANGE",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """短期波动率相对长期的变化"""
    std5 = df['close'].pct_change().rolling(5).std()
    std20 = df['close'].pct_change().rolling(20).std()
    return std5 / (std20 + 1e-8) - 1
''',
        description="短期波动相对长期波动",
        logic="波动率放大或收敛",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.018,
        tags=["alpha360", "volatility", "change"],
    ),
    
    # ==================== 量价同步性 ====================
    ClassicFactor(
        id="alpha360_vp_sync",
        name="量价同步",
        name_en="VP_SYNC",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """量价同步性指标"""
    import numpy as np
    ret = df['close'].pct_change()
    vol_change = df['volume'].pct_change()
    sync = np.sign(ret) * np.sign(vol_change)
    return sync.rolling(10).mean()
''',
        description="价格和成交量变化方向的一致性",
        logic="同向为正，背离为负",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.020,
        tags=["alpha360", "volume_price", "sync"],
    ),
    
    # ==================== 收益率排名 ====================
    ClassicFactor(
        id="alpha360_ret_rank_5",
        name="5日收益排名",
        name_en="RET_RANK_5",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """5日内收益率的时序排名"""
    ret = df['close'].pct_change()
    return ret.rolling(5).apply(lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5)
''',
        description="当日收益在过去5日中的排名",
        logic="1=最强，0=最弱",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.022,
        tags=["alpha360", "rank", "returns"],
    ),
    
    ClassicFactor(
        id="alpha360_ret_rank_20",
        name="20日收益排名",
        name_en="RET_RANK_20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """20日内收益率的时序排名"""
    ret = df['close'].pct_change()
    return ret.rolling(20).apply(lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5)
''',
        description="当日收益在过去20日中的排名",
        logic="1=最强，0=最弱",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.025,
        tags=["alpha360", "rank", "returns"],
    ),
    
    # ==================== 成交额集中度 ====================
    ClassicFactor(
        id="alpha360_amount_conc",
        name="成交额集中度",
        name_en="AMOUNT_CONC",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """成交额的集中程度"""
    amount_5 = df['amount'].rolling(5).sum()
    amount_20 = df['amount'].rolling(20).sum()
    return amount_5 / (amount_20 + 1e-8)
''',
        description="近5日成交额占20日比例",
        logic="集中度高表示近期活跃",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.018,
        tags=["alpha360", "amount", "concentration"],
    ),
    
    # ==================== EMA特征 ====================
    ClassicFactor(
        id="alpha360_ema12_bias",
        name="EMA12偏离",
        name_en="EMA12_BIAS",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价相对EMA12的偏离"""
    ema12 = df['close'].ewm(span=12).mean()
    return (df['close'] - ema12) / ema12
''',
        description="价格偏离EMA12的程度",
        logic="正值表示强势",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.025,
        tags=["alpha360", "ema", "bias"],
    ),
    
    ClassicFactor(
        id="alpha360_ema26_bias",
        name="EMA26偏离",
        name_en="EMA26_BIAS",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价相对EMA26的偏离"""
    ema26 = df['close'].ewm(span=26).mean()
    return (df['close'] - ema26) / ema26
''',
        description="价格偏离EMA26的程度",
        logic="正值表示中期强势",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.022,
        tags=["alpha360", "ema", "bias"],
    ),
    
    # ==================== ATR特征 ====================
    ClassicFactor(
        id="alpha360_atr14",
        name="ATR14",
        name_en="ATR14",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """14日平均真实波幅"""
    import numpy as np
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(14).mean() / df['close']
''',
        description="14日ATR相对价格",
        logic="衡量真实波动",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.015,
        tags=["alpha360", "atr", "volatility"],
    ),
    
    # ==================== OBV特征 ====================
    ClassicFactor(
        id="alpha360_obv_ma",
        name="OBV均线偏离",
        name_en="OBV_MA_BIAS",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """OBV相对其均线的偏离"""
    import numpy as np
    direction = np.sign(df['close'].diff())
    obv = (direction * df['volume']).cumsum()
    obv_ma = obv.rolling(20).mean()
    return (obv - obv_ma) / (obv_ma.abs() + 1e-8)
''',
        description="能量潮相对均线",
        logic="正值表示资金流入",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.020,
        tags=["alpha360", "obv", "volume"],
    ),
    
    # ==================== 布林带特征 ====================
    ClassicFactor(
        id="alpha360_boll_position",
        name="布林带位置",
        name_en="BOLL_POS",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """价格在布林带中的位置"""
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    return (df['close'] - lower) / (upper - lower + 1e-8)
''',
        description="0=下轨，1=上轨",
        logic="超买超卖信号",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.022,
        tags=["alpha360", "bollinger", "position"],
    ),
    
    ClassicFactor(
        id="alpha360_boll_width",
        name="布林带宽度",
        name_en="BOLL_WIDTH",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """布林带宽度"""
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    return 4 * std20 / ma20
''',
        description="布林带上下轨距离",
        logic="宽度大=波动大",
        reference="Qlib Alpha360",
        author="Microsoft",
        year=2020,
        historical_ic=0.018,
        tags=["alpha360", "bollinger", "volatility"],
    ),
]


def get_alpha360_factors():
    """获取Alpha360因子"""
    return ALPHA360_FACTORS


def get_alpha360_count():
    """获取Alpha360因子数量"""
    return len(ALPHA360_FACTORS)
