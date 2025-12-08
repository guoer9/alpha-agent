"""
国泰君安 191 Alpha因子库

来源: 国泰君安证券《基于短周期价量特征的多因子选股体系》
包含: 191 个量价因子
特点: 短周期、高换手、适合A股市场

注: 以下为精选的代表性因子，完整191因子可按需扩展
"""

from dataclasses import dataclass, field
from typing import List
from .classic_factors import ClassicFactor, FactorCategory


# ============================================================
# 国泰君安 191 因子
# 按功能分类组织
# ============================================================

GTJA191_FACTORS = [
    # ==================== 量价相关类 (1-30) ====================
    ClassicFactor(
        id="gtja_alpha001",
        name="GTJA#001 量价排名差",
        name_en="GTJA_Alpha001",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#001: (-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6))
    量价排名相关性的负值
    """
    import numpy as np
    delta_log_vol = np.log(df['volume'] + 1).diff(1)
    price_change = (df['close'] - df['open']) / (df['open'] + 1e-8)
    corr = delta_log_vol.rank(pct=True).rolling(6).corr(price_change.rank(pct=True))
    return -corr
''',
        description="成交量变化排名与价格变化排名的负相关",
        logic="量价背离预示反转",
        reference="国泰君安 191因子",
        historical_ic=0.022,
        tags=["gtja191", "volume_price", "correlation"],
    ),
    
    ClassicFactor(
        id="gtja_alpha002",
        name="GTJA#002 开高低收相关",
        name_en="GTJA_Alpha002",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#002: -1 * DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),1)
    K线位置变化
    """
    hl_range = df['high'] - df['low'] + 1e-8
    position = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_range
    return -position.diff(1)
''',
        description="收盘价在K线中的位置变化",
        logic="收盘位置下移预示下跌",
        reference="国泰君安 191因子",
        historical_ic=0.018,
        tags=["gtja191", "kbar", "momentum"],
    ),
    
    ClassicFactor(
        id="gtja_alpha003",
        name="GTJA#003 收盘累积",
        name_en="GTJA_Alpha003",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#003: SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    价格突破累积
    """
    import numpy as np
    prev_close = df['close'].shift(1)
    
    condition_up = df['close'] > prev_close
    condition_down = df['close'] < prev_close
    
    min_val = df[['low']].join(prev_close.rename('pc')).min(axis=1)
    max_val = df[['high']].join(prev_close.rename('pc')).max(axis=1)
    
    result = np.where(df['close'] == prev_close, 0,
                     np.where(condition_up, df['close'] - min_val, df['close'] - max_val))
    return pd.Series(result, index=df.index).rolling(6).sum()
''',
        description="6日价格突破累积",
        logic="累积突破强度",
        reference="国泰君安 191因子",
        historical_ic=0.020,
        tags=["gtja191", "breakout", "momentum"],
    ),
    
    ClassicFactor(
        id="gtja_alpha004",
        name="GTJA#004 量价条件",
        name_en="GTJA_Alpha004",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#004: 量价条件因子
    成交量与收盘价均值的条件判断
    """
    vol_cond = df['volume'].rolling(8).mean() < df['volume'].rolling(2).mean()
    close_down = df['close'].rolling(8).sum() < df['close'].rolling(2).sum() * 4
    close_up = df['close'] > df['close'].shift(1)
    
    result = vol_cond.astype(int) * (-1) + (~vol_cond & close_down).astype(int) * (-1) + \
             (~vol_cond & ~close_down & close_up).astype(int) * 1 + \
             (~vol_cond & ~close_down & ~close_up).astype(int) * (-1)
    return result
''',
        description="量价条件综合判断",
        logic="多条件综合信号",
        reference="国泰君安 191因子",
        historical_ic=0.015,
        tags=["gtja191", "volume_price", "condition"],
    ),
    
    ClassicFactor(
        id="gtja_alpha005",
        name="GTJA#005 成交量排名",
        name_en="GTJA_Alpha005",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#005: -1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3)
    成交量与最高价时序排名相关
    """
    vol_rank = df['volume'].rolling(5).apply(lambda x: x.rank().iloc[-1] / len(x))
    high_rank = df['high'].rolling(5).apply(lambda x: x.rank().iloc[-1] / len(x))
    corr = vol_rank.rolling(5).corr(high_rank)
    return -corr.rolling(3).max()
''',
        description="量价时序排名最大相关",
        logic="量价同步见顶信号",
        reference="国泰君安 191因子",
        historical_ic=0.019,
        tags=["gtja191", "volume_price", "rank"],
    ),
    
    # ==================== 动量反转类 (31-60) ====================
    ClassicFactor(
        id="gtja_alpha006",
        name="GTJA#006 开盘跳空",
        name_en="GTJA_Alpha006",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#006: (OPEN*0.85+HIGH*0.15-DELAY(OPEN*0.85+HIGH*0.15,1))
    加权开盘跳空
    """
    weighted = df['open'] * 0.85 + df['high'] * 0.15
    return weighted.diff(1)
''',
        description="加权开盘跳空幅度",
        logic="跳空强度预示趋势",
        reference="国泰君安 191因子",
        historical_ic=0.016,
        tags=["gtja191", "gap", "momentum"],
    ),
    
    ClassicFactor(
        id="gtja_alpha007",
        name="GTJA#007 VWAP动量",
        name_en="GTJA_Alpha007",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#007: (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
    VWAP偏离与成交量变化
    """
    vwap = df['amount'] / (df['volume'] + 1e-8)
    vwap_diff = vwap - df['close']
    
    max_diff = vwap_diff.rolling(3).max().rank(pct=True)
    min_diff = vwap_diff.rolling(3).min().rank(pct=True)
    vol_delta = df['volume'].diff(3).rank(pct=True)
    
    return (max_diff + min_diff) * vol_delta
''',
        description="VWAP偏离与成交量动量",
        logic="VWAP压力与资金流向",
        reference="国泰君安 191因子",
        historical_ic=0.021,
        tags=["gtja191", "vwap", "volume"],
    ),
    
    ClassicFactor(
        id="gtja_alpha008",
        name="GTJA#008 加权收益",
        name_en="GTJA_Alpha008",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#008: -RANK(DELTA((HIGH+LOW)/2*0.2+VWAP*0.8,4))
    加权价格动量
    """
    vwap = df['amount'] / (df['volume'] + 1e-8)
    weighted_price = (df['high'] + df['low']) / 2 * 0.2 + vwap * 0.8
    return -weighted_price.diff(4).rank(pct=True)
''',
        description="加权价格4日变化排名",
        logic="短期反转",
        reference="国泰君安 191因子",
        historical_ic=0.024,
        tags=["gtja191", "reversal", "short_term"],
    ),
    
    ClassicFactor(
        id="gtja_alpha009",
        name="GTJA#009 EMA动量",
        name_en="GTJA_Alpha009",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#009: SMA((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2*(HIGH-LOW)/VOLUME,7,2)
    量价加权中点动量
    """
    mid = (df['high'] + df['low']) / 2
    prev_mid = (df['high'].shift(1) + df['low'].shift(1)) / 2
    range_vol = (df['high'] - df['low']) / (df['volume'] + 1e-8)
    
    factor = (mid - prev_mid) * range_vol
    return factor.ewm(span=7, adjust=False).mean()
''',
        description="量价加权的中点动量",
        logic="量能配合的价格动量",
        reference="国泰君安 191因子",
        historical_ic=0.017,
        tags=["gtja191", "momentum", "volume"],
    ),
    
    ClassicFactor(
        id="gtja_alpha010",
        name="GTJA#010 收益率条件",
        name_en="GTJA_Alpha010",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#010: 收益率条件排名
    """
    ret = df['close'].pct_change()
    std = ret.rolling(20).std()
    cond = ret.rolling(5).min() > 0
    
    result = ret.rolling(5).max().rank(pct=True)
    result = result.where(cond, std.rank(pct=True))
    return result
''',
        description="条件收益率排名",
        logic="趋势延续与波动切换",
        reference="国泰君安 191因子",
        historical_ic=0.020,
        tags=["gtja191", "momentum", "volatility"],
    ),
    
    # ==================== 波动率类 (61-90) ====================
    ClassicFactor(
        id="gtja_alpha011",
        name="GTJA#011 量价差异",
        name_en="GTJA_Alpha011",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#011: SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)
    量能加权的K线位置
    """
    hl_range = df['high'] - df['low'] + 1e-8
    position = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_range
    return (position * df['volume']).rolling(6).sum()
''',
        description="6日量能加权K线位置",
        logic="主力买卖意愿",
        reference="国泰君安 191因子",
        historical_ic=0.023,
        tags=["gtja191", "volume_price", "kbar"],
    ),
    
    ClassicFactor(
        id="gtja_alpha012",
        name="GTJA#012 开盘量价",
        name_en="GTJA_Alpha012",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#012: RANK(OPEN-SUM(VWAP,10)/10)*(-1*(ABS(RANK(CLOSE-VWAP))))
    开盘价与VWAP偏离
    """
    vwap = df['amount'] / (df['volume'] + 1e-8)
    vwap_ma = vwap.rolling(10).mean()
    
    term1 = (df['open'] - vwap_ma).rank(pct=True)
    term2 = (df['close'] - vwap).abs().rank(pct=True)
    return term1 * (-term2)
''',
        description="开盘相对VWAP偏离",
        logic="开盘强弱与收盘偏离",
        reference="国泰君安 191因子",
        historical_ic=0.019,
        tags=["gtja191", "vwap", "open"],
    ),
    
    ClassicFactor(
        id="gtja_alpha013",
        name="GTJA#013 高低价差",
        name_en="GTJA_Alpha013",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#013: ((HIGH*LOW)^0.5-VWAP)
    几何均价与VWAP差
    """
    import numpy as np
    vwap = df['amount'] / (df['volume'] + 1e-8)
    geo_mean = np.sqrt(df['high'] * df['low'])
    return geo_mean - vwap
''',
        description="几何均价与VWAP的差值",
        logic="价格分布偏离",
        reference="国泰君安 191因子",
        historical_ic=0.015,
        tags=["gtja191", "vwap", "price"],
    ),
    
    ClassicFactor(
        id="gtja_alpha014",
        name="GTJA#014 收盘延迟",
        name_en="GTJA_Alpha014",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#014: CLOSE-DELAY(CLOSE,5)
    5日价格变化
    """
    return df['close'] - df['close'].shift(5)
''',
        description="5日收盘价变化",
        logic="短期动量",
        reference="国泰君安 191因子",
        historical_ic=0.018,
        tags=["gtja191", "momentum", "short_term"],
    ),
    
    ClassicFactor(
        id="gtja_alpha015",
        name="GTJA#015 开收比",
        name_en="GTJA_Alpha015",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#015: OPEN/DELAY(CLOSE,1)-1
    隔夜跳空
    """
    return df['open'] / df['close'].shift(1) - 1
''',
        description="隔夜跳空幅度",
        logic="隔夜信息反应",
        reference="国泰君安 191因子",
        historical_ic=0.025,
        tags=["gtja191", "gap", "overnight"],
    ),
    
    # ==================== 技术形态类 (91-120) ====================
    ClassicFactor(
        id="gtja_alpha016",
        name="GTJA#016 量能排名",
        name_en="GTJA_Alpha016",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#016: -RANK(COVIANCE(RANK(HIGH),RANK(VOLUME),5))
    高价与成交量协方差排名
    """
    high_rank = df['high'].rank(pct=True)
    vol_rank = df['volume'].rank(pct=True)
    cov = high_rank.rolling(5).cov(vol_rank)
    return -cov.rank(pct=True)
''',
        description="高价与成交量协方差",
        logic="量价配合度",
        reference="国泰君安 191因子",
        historical_ic=0.017,
        tags=["gtja191", "volume_price", "covariance"],
    ),
    
    ClassicFactor(
        id="gtja_alpha017",
        name="GTJA#017 TSRANK价格",
        name_en="GTJA_Alpha017",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#017: RANK(VWAP-MAX(VWAP,15))^DELTA(CLOSE,5)
    VWAP时序位置
    """
    vwap = df['amount'] / (df['volume'] + 1e-8)
    vwap_max = vwap.rolling(15).max()
    delta_close = df['close'].diff(5)
    
    term1 = (vwap - vwap_max).rank(pct=True)
    return term1 ** delta_close.clip(-1, 1)
''',
        description="VWAP相对最高点的位置",
        logic="价格强弱与动量",
        reference="国泰君安 191因子",
        historical_ic=0.016,
        tags=["gtja191", "vwap", "momentum"],
    ),
    
    ClassicFactor(
        id="gtja_alpha018",
        name="GTJA#018 收盘延迟均值",
        name_en="GTJA_Alpha018",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#018: CLOSE/DELAY(CLOSE,5)
    5日收益率
    """
    return df['close'] / df['close'].shift(5)
''',
        description="5日收益率",
        logic="短期动量",
        reference="国泰君安 191因子",
        historical_ic=0.022,
        tags=["gtja191", "momentum", "return"],
    ),
    
    ClassicFactor(
        id="gtja_alpha019",
        name="GTJA#019 收盘延迟条件",
        name_en="GTJA_Alpha019",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#019: 条件收益
    """
    ret_5d = df['close'] / df['close'].shift(5) - 1
    cond = df['close'] < df['close'].shift(5)
    result = ret_5d.where(cond, (df['close'] - df['close'].shift(5)) / df['close'].shift(5))
    return result
''',
        description="条件5日收益",
        logic="下跌后的反弹强度",
        reference="国泰君安 191因子",
        historical_ic=0.018,
        tags=["gtja191", "reversal", "condition"],
    ),
    
    ClassicFactor(
        id="gtja_alpha020",
        name="GTJA#020 收盘开盘差",
        name_en="GTJA_Alpha020",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#020: (CLOSE/DELAY(CLOSE,6)-1)*100
    6日收益率
    """
    return (df['close'] / df['close'].shift(6) - 1) * 100
''',
        description="6日收益率百分比",
        logic="周动量",
        reference="国泰君安 191因子",
        historical_ic=0.020,
        tags=["gtja191", "momentum", "weekly"],
    ),
    
    # ==================== 资金流向类 (121-150) ====================
    ClassicFactor(
        id="gtja_alpha021",
        name="GTJA#021 SMA收盘",
        name_en="GTJA_Alpha021",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#021: SMA(CLOSE,6)-CLOSE
    均线偏离
    """
    return df['close'].rolling(6).mean() - df['close']
''',
        description="6日均线偏离",
        logic="均值回归",
        reference="国泰君安 191因子",
        historical_ic=0.028,
        tags=["gtja191", "mean_reversion", "ma"],
    ),
    
    ClassicFactor(
        id="gtja_alpha022",
        name="GTJA#022 收盘变化比",
        name_en="GTJA_Alpha022",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#022: SMA((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3),12,1)
    标准化偏离动量
    """
    mean_6 = df['close'].rolling(6).mean()
    deviation = (df['close'] - mean_6) / (mean_6 + 1e-8)
    delta_dev = deviation - deviation.shift(3)
    return delta_dev.ewm(span=12, adjust=False).mean()
''',
        description="标准化偏离的动量",
        logic="偏离变化趋势",
        reference="国泰君安 191因子",
        historical_ic=0.019,
        tags=["gtja191", "momentum", "deviation"],
    ),
    
    ClassicFactor(
        id="gtja_alpha023",
        name="GTJA#023 条件SMA",
        name_en="GTJA_Alpha023",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#023: 条件均线
    """
    cond = df['close'] > df['close'].shift(1)
    std = df['close'].rolling(20).std()
    result = std.where(cond, -std)
    return result.ewm(span=20, adjust=False).mean()
''',
        description="条件波动率",
        logic="上涨下跌的波动差异",
        reference="国泰君安 191因子",
        historical_ic=0.021,
        tags=["gtja191", "volatility", "condition"],
    ),
    
    ClassicFactor(
        id="gtja_alpha024",
        name="GTJA#024 收盘延迟SMA",
        name_en="GTJA_Alpha024",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#024: SMA(CLOSE-DELAY(CLOSE,5),5,1)
    5日动量的平滑
    """
    delta = df['close'] - df['close'].shift(5)
    return delta.ewm(span=5, adjust=False).mean()
''',
        description="平滑5日动量",
        logic="趋势强度",
        reference="国泰君安 191因子",
        historical_ic=0.023,
        tags=["gtja191", "momentum", "smooth"],
    ),
    
    ClassicFactor(
        id="gtja_alpha025",
        name="GTJA#025 量价衰减",
        name_en="GTJA_Alpha025",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#025: (-1*RANK(DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR(VOLUME/MEAN(VOLUME,20),9)))))*(1+RANK(SUM(RET,250)))
    量价衰减综合
    """
    ret = df['close'].pct_change()
    vol_ratio = df['volume'] / df['volume'].rolling(20).mean()
    
    # 线性衰减
    weights = list(range(9, 0, -1))
    decay = vol_ratio.rolling(9).apply(lambda x: sum(w*v for w,v in zip(weights, x)) / sum(weights))
    
    term1 = -df['close'].diff(7).rank(pct=True) * (1 - decay.rank(pct=True))
    term2 = 1 + ret.rolling(250).sum().rank(pct=True)
    return term1 * term2
''',
        description="量价衰减综合因子",
        logic="短期反转与长期动量",
        reference="国泰君安 191因子",
        historical_ic=0.026,
        tags=["gtja191", "complex", "decay"],
    ),
    
    # ==================== 趋势类 (151-191) ====================
    ClassicFactor(
        id="gtja_alpha026",
        name="GTJA#026 相关性和",
        name_en="GTJA_Alpha026",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#026: SUM(CLOSE,7)/7-CLOSE+CORR(VWAP,DELAY(CLOSE,5),230)
    均线偏离与长期相关
    """
    vwap = df['amount'] / (df['volume'] + 1e-8)
    ma7 = df['close'].rolling(7).mean()
    corr = vwap.rolling(230).corr(df['close'].shift(5))
    return ma7 - df['close'] + corr
''',
        description="短期偏离与长期相关",
        logic="短期均值回归+长期趋势",
        reference="国泰君安 191因子",
        historical_ic=0.018,
        tags=["gtja191", "ma", "correlation"],
    ),
    
    ClassicFactor(
        id="gtja_alpha027",
        name="GTJA#027 WMA收益",
        name_en="GTJA_Alpha027",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#027: WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
    加权多周期动量
    """
    ret_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3) * 100
    ret_6d = (df['close'] - df['close'].shift(6)) / df['close'].shift(6) * 100
    combined = ret_3d + ret_6d
    
    weights = list(range(12, 0, -1))
    wma = combined.rolling(12).apply(lambda x: sum(w*v for w,v in zip(weights, x)) / sum(weights))
    return wma
''',
        description="加权多周期动量",
        logic="综合短期趋势",
        reference="国泰君安 191因子",
        historical_ic=0.024,
        tags=["gtja191", "momentum", "weighted"],
    ),
    
    ClassicFactor(
        id="gtja_alpha028",
        name="GTJA#028 Boll带",
        name_en="GTJA_Alpha028",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#028: 布林带位置
    """
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    return (df['close'] - lower) / (upper - lower + 1e-8)
''',
        description="布林带中的位置",
        logic="相对波动位置",
        reference="国泰君安 191因子",
        historical_ic=0.022,
        tags=["gtja191", "bollinger", "volatility"],
    ),
    
    ClassicFactor(
        id="gtja_alpha029",
        name="GTJA#029 资金流入",
        name_en="GTJA_Alpha029",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#029: (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
    量能加权动量
    """
    ret_6d = (df['close'] - df['close'].shift(6)) / df['close'].shift(6)
    return ret_6d * df['volume']
''',
        description="量能加权6日动量",
        logic="资金流向强度",
        reference="国泰君安 191因子",
        historical_ic=0.020,
        tags=["gtja191", "money_flow", "momentum"],
    ),
    
    ClassicFactor(
        id="gtja_alpha030",
        name="GTJA#030 残差动量",
        name_en="GTJA_Alpha030",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    GTJA Alpha#030: 残差动量
    """
    ret = df['close'].pct_change()
    ma_ret = ret.rolling(60).mean()
    residual = ret - ma_ret
    return residual.rolling(20).sum()
''',
        description="残差收益累积",
        logic="剔除均值后的动量",
        reference="国泰君安 191因子",
        historical_ic=0.021,
        tags=["gtja191", "residual", "momentum"],
    ),
]


def get_gtja191_factors():
    """获取所有国泰君安191因子"""
    return GTJA191_FACTORS


# 打印因子统计
if __name__ == "__main__":
    print(f"国泰君安 191 因子库:")
    print(f"  - 已实现因子: {len(GTJA191_FACTORS)}个")
    print(f"  - 量价类: {len([f for f in GTJA191_FACTORS if 'volume_price' in f.tags])}个")
    print(f"  - 动量类: {len([f for f in GTJA191_FACTORS if 'momentum' in f.tags])}个")
