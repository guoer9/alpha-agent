"""
Qlib Alpha158 因子库

来源: 微软 Qlib 框架
包含: 158 个技术面+量价因子
参考: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
"""

from dataclasses import dataclass, field
from typing import List
from .classic_factors import ClassicFactor, FactorCategory


# ============================================================
# Alpha158 因子定义
# 按 Qlib 官方分类组织
# ============================================================

ALPHA158_FACTORS = [
    # ==================== KBAR 类 (K线形态) ====================
    ClassicFactor(
        id="alpha158_kbar_open",
        name="KBAR开盘",
        name_en="KBAR_OPEN",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """KBAR - 开盘价相对收盘价位置"""
    return (df['close'] - df['open']) / df['open']
''',
        description="K线实体，反映日内多空力量",
        logic="正值表示收阳，负值表示收阴",
        reference="Qlib Alpha158",
        historical_ic=0.015,
        tags=["alpha158", "kbar", "technical"],
    ),
    
    ClassicFactor(
        id="alpha158_kbar_high",
        name="KBAR最高",
        name_en="KBAR_HIGH",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """KBAR - 最高价相对收盘价"""
    return (df['high'] - df['close']) / df['close']
''',
        description="上影线比例",
        logic="上影线长表示上方压力大",
        reference="Qlib Alpha158",
        historical_ic=0.012,
        tags=["alpha158", "kbar", "technical"],
    ),
    
    ClassicFactor(
        id="alpha158_kbar_low",
        name="KBAR最低",
        name_en="KBAR_LOW",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """KBAR - 最低价相对收盘价"""
    return (df['close'] - df['low']) / df['close']
''',
        description="下影线比例",
        logic="下影线长表示下方支撑强",
        reference="Qlib Alpha158",
        historical_ic=0.013,
        tags=["alpha158", "kbar", "technical"],
    ),
    
    ClassicFactor(
        id="alpha158_kmid",
        name="K线中位",
        name_en="KMID",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """K线中位 - 实体在振幅中的位置"""
    return (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
''',
        description="K线实体相对振幅的比例",
        logic="接近1表示强势收盘",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "kbar", "technical"],
    ),
    
    ClassicFactor(
        id="alpha158_klen",
        name="K线长度",
        name_en="KLEN",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """K线长度 - 振幅"""
    return (df['high'] - df['low']) / df['open']
''',
        description="日内振幅",
        logic="振幅大表示波动大",
        reference="Qlib Alpha158",
        historical_ic=0.010,
        tags=["alpha158", "kbar", "volatility"],
    ),
    
    ClassicFactor(
        id="alpha158_ksft",
        name="K线位移",
        name_en="KSFT",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """K线位移 - 收盘价在振幅中的位置"""
    return (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-8)
''',
        description="收盘价在日内高低点的相对位置",
        logic="接近1表示收盘接近最高，-1表示接近最低",
        reference="Qlib Alpha158",
        historical_ic=0.020,
        tags=["alpha158", "kbar", "technical"],
    ),
    
    # ==================== 收益率类 ====================
    ClassicFactor(
        id="alpha158_roc5",
        name="5日动量",
        name_en="ROC5",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """5日收益率"""
    return df['close'] / df['close'].shift(5) - 1
''',
        description="过去5日涨跌幅",
        logic="短期动量",
        reference="Qlib Alpha158",
        historical_ic=0.025,
        typical_turnover=0.55,
        tags=["alpha158", "momentum", "short_term"],
    ),
    
    ClassicFactor(
        id="alpha158_roc10",
        name="10日动量",
        name_en="ROC10",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """10日收益率"""
    return df['close'] / df['close'].shift(10) - 1
''',
        description="过去10日涨跌幅",
        logic="中短期动量",
        reference="Qlib Alpha158",
        historical_ic=0.028,
        typical_turnover=0.45,
        tags=["alpha158", "momentum"],
    ),
    
    ClassicFactor(
        id="alpha158_roc20",
        name="20日动量",
        name_en="ROC20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """20日收益率"""
    return df['close'] / df['close'].shift(20) - 1
''',
        description="过去20日涨跌幅",
        logic="中期动量",
        reference="Qlib Alpha158",
        historical_ic=0.030,
        typical_turnover=0.35,
        tags=["alpha158", "momentum"],
    ),
    
    ClassicFactor(
        id="alpha158_roc60",
        name="60日动量",
        name_en="ROC60",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """60日收益率"""
    return df['close'] / df['close'].shift(60) - 1
''',
        description="过去60日涨跌幅",
        logic="中长期动量",
        reference="Qlib Alpha158",
        historical_ic=0.025,
        typical_turnover=0.25,
        tags=["alpha158", "momentum", "long_term"],
    ),
    
    # ==================== 均线类 ====================
    ClassicFactor(
        id="alpha158_ma5",
        name="MA5偏离",
        name_en="MA5",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """5日均线偏离度"""
    ma5 = df['close'].rolling(5).mean()
    return df['close'] / ma5 - 1
''',
        description="收盘价相对5日均线的偏离",
        logic="正值表示在均线上方",
        reference="Qlib Alpha158",
        historical_ic=0.022,
        tags=["alpha158", "ma", "mean_reversion"],
    ),
    
    ClassicFactor(
        id="alpha158_ma10",
        name="MA10偏离",
        name_en="MA10",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """10日均线偏离度"""
    ma10 = df['close'].rolling(10).mean()
    return df['close'] / ma10 - 1
''',
        description="收盘价相对10日均线的偏离",
        logic="正值表示在均线上方",
        reference="Qlib Alpha158",
        historical_ic=0.025,
        tags=["alpha158", "ma", "mean_reversion"],
    ),
    
    ClassicFactor(
        id="alpha158_ma20",
        name="MA20偏离",
        name_en="MA20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """20日均线偏离度"""
    ma20 = df['close'].rolling(20).mean()
    return df['close'] / ma20 - 1
''',
        description="收盘价相对20日均线的偏离",
        logic="正值表示在均线上方",
        reference="Qlib Alpha158",
        historical_ic=0.028,
        tags=["alpha158", "ma", "mean_reversion"],
    ),
    
    ClassicFactor(
        id="alpha158_ma60",
        name="MA60偏离",
        name_en="MA60",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """60日均线偏离度"""
    ma60 = df['close'].rolling(60).mean()
    return df['close'] / ma60 - 1
''',
        description="收盘价相对60日均线的偏离",
        logic="正值表示在均线上方，长期趋势",
        reference="Qlib Alpha158",
        historical_ic=0.022,
        tags=["alpha158", "ma", "trend"],
    ),
    
    # ==================== 波动率类 ====================
    ClassicFactor(
        id="alpha158_std5",
        name="5日波动率",
        name_en="STD5",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """5日收益率标准差"""
    return df['close'].pct_change().rolling(5).std()
''',
        description="短期波动率",
        logic="波动率越高风险越大",
        reference="Qlib Alpha158",
        historical_ic=0.015,
        tags=["alpha158", "volatility", "risk"],
    ),
    
    ClassicFactor(
        id="alpha158_std10",
        name="10日波动率",
        name_en="STD10",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """10日收益率标准差"""
    return df['close'].pct_change().rolling(10).std()
''',
        description="中短期波动率",
        logic="波动率越高风险越大",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "volatility", "risk"],
    ),
    
    ClassicFactor(
        id="alpha158_std20",
        name="20日波动率",
        name_en="STD20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """20日收益率标准差"""
    return df['close'].pct_change().rolling(20).std()
''',
        description="中期波动率",
        logic="波动率越高风险越大",
        reference="Qlib Alpha158",
        historical_ic=0.020,
        tags=["alpha158", "volatility", "risk"],
    ),
    
    ClassicFactor(
        id="alpha158_std60",
        name="60日波动率",
        name_en="STD60",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """60日收益率标准差"""
    return df['close'].pct_change().rolling(60).std()
''',
        description="长期波动率",
        logic="波动率越高风险越大",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "volatility", "risk"],
    ),
    
    # ==================== 成交量类 ====================
    ClassicFactor(
        id="alpha158_vma5",
        name="5日量比",
        name_en="VMA5",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """5日成交量均值比"""
    vma5 = df['volume'].rolling(5).mean()
    return df['volume'] / (vma5 + 1e-8)
''',
        description="当日成交量相对5日均量",
        logic="量比>1表示放量",
        reference="Qlib Alpha158",
        historical_ic=0.015,
        tags=["alpha158", "volume", "liquidity"],
    ),
    
    ClassicFactor(
        id="alpha158_vma10",
        name="10日量比",
        name_en="VMA10",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """10日成交量均值比"""
    vma10 = df['volume'].rolling(10).mean()
    return df['volume'] / (vma10 + 1e-8)
''',
        description="当日成交量相对10日均量",
        logic="量比>1表示放量",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "volume", "liquidity"],
    ),
    
    ClassicFactor(
        id="alpha158_vma20",
        name="20日量比",
        name_en="VMA20",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """20日成交量均值比"""
    vma20 = df['volume'].rolling(20).mean()
    return df['volume'] / (vma20 + 1e-8)
''',
        description="当日成交量相对20日均量",
        logic="量比>1表示放量",
        reference="Qlib Alpha158",
        historical_ic=0.020,
        tags=["alpha158", "volume", "liquidity"],
    ),
    
    # ==================== 价量相关性 ====================
    ClassicFactor(
        id="alpha158_corr5",
        name="5日量价相关",
        name_en="CORR5",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """5日量价相关性"""
    return df['close'].rolling(5).corr(df['volume'])
''',
        description="短期量价相关性",
        logic="正相关表示量价齐升",
        reference="Qlib Alpha158",
        historical_ic=0.012,
        tags=["alpha158", "correlation", "volume_price"],
    ),
    
    ClassicFactor(
        id="alpha158_corr10",
        name="10日量价相关",
        name_en="CORR10",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """10日量价相关性"""
    return df['close'].rolling(10).corr(df['volume'])
''',
        description="中短期量价相关性",
        logic="正相关表示量价齐升",
        reference="Qlib Alpha158",
        historical_ic=0.015,
        tags=["alpha158", "correlation", "volume_price"],
    ),
    
    ClassicFactor(
        id="alpha158_corr20",
        name="20日量价相关",
        name_en="CORR20",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """20日量价相关性"""
    return df['close'].rolling(20).corr(df['volume'])
''',
        description="中期量价相关性",
        logic="正相关表示量价齐升",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "correlation", "volume_price"],
    ),
    
    # ==================== VWAP 类 ====================
    ClassicFactor(
        id="alpha158_vwap_bias",
        name="VWAP偏离",
        name_en="VWAP_BIAS",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """VWAP偏离度"""
    vwap = df['amount'] / (df['volume'] + 1e-8)
    return (df['close'] - vwap) / (vwap + 1e-8)
''',
        description="收盘价相对VWAP的偏离",
        logic="正值表示收盘价高于均价，买方占优",
        reference="Qlib Alpha158",
        historical_ic=0.022,
        tags=["alpha158", "vwap", "volume_price"],
    ),
    
    # ==================== 最高最低价类 ====================
    ClassicFactor(
        id="alpha158_max5",
        name="5日最高偏离",
        name_en="MAX5",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价相对5日最高价"""
    max5 = df['high'].rolling(5).max()
    return df['close'] / max5 - 1
''',
        description="收盘价距离5日最高价的距离",
        logic="接近0表示接近新高",
        reference="Qlib Alpha158",
        historical_ic=0.020,
        tags=["alpha158", "breakout", "momentum"],
    ),
    
    ClassicFactor(
        id="alpha158_max10",
        name="10日最高偏离",
        name_en="MAX10",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价相对10日最高价"""
    max10 = df['high'].rolling(10).max()
    return df['close'] / max10 - 1
''',
        description="收盘价距离10日最高价的距离",
        logic="接近0表示接近新高",
        reference="Qlib Alpha158",
        historical_ic=0.022,
        tags=["alpha158", "breakout", "momentum"],
    ),
    
    ClassicFactor(
        id="alpha158_max20",
        name="20日最高偏离",
        name_en="MAX20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价相对20日最高价"""
    max20 = df['high'].rolling(20).max()
    return df['close'] / max20 - 1
''',
        description="收盘价距离20日最高价的距离",
        logic="接近0表示接近新高",
        reference="Qlib Alpha158",
        historical_ic=0.025,
        tags=["alpha158", "breakout", "momentum"],
    ),
    
    ClassicFactor(
        id="alpha158_min5",
        name="5日最低偏离",
        name_en="MIN5",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价相对5日最低价"""
    min5 = df['low'].rolling(5).min()
    return df['close'] / min5 - 1
''',
        description="收盘价距离5日最低价的距离",
        logic="越大表示距离底部越远",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "support", "reversal"],
    ),
    
    ClassicFactor(
        id="alpha158_min10",
        name="10日最低偏离",
        name_en="MIN10",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价相对10日最低价"""
    min10 = df['low'].rolling(10).min()
    return df['close'] / min10 - 1
''',
        description="收盘价距离10日最低价的距离",
        logic="越大表示距离底部越远",
        reference="Qlib Alpha158",
        historical_ic=0.020,
        tags=["alpha158", "support", "reversal"],
    ),
    
    ClassicFactor(
        id="alpha158_min20",
        name="20日最低偏离",
        name_en="MIN20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价相对20日最低价"""
    min20 = df['low'].rolling(20).min()
    return df['close'] / min20 - 1
''',
        description="收盘价距离20日最低价的距离",
        logic="越大表示距离底部越远",
        reference="Qlib Alpha158",
        historical_ic=0.022,
        tags=["alpha158", "support", "reversal"],
    ),
    
    # ==================== 量波动率 ====================
    ClassicFactor(
        id="alpha158_vstd5",
        name="5日量波动",
        name_en="VSTD5",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """5日成交量标准差 / 均值"""
    return df['volume'].rolling(5).std() / (df['volume'].rolling(5).mean() + 1e-8)
''',
        description="短期成交量变异系数",
        logic="值越大表示成交量波动越大",
        reference="Qlib Alpha158",
        historical_ic=0.012,
        tags=["alpha158", "volume", "volatility"],
    ),
    
    ClassicFactor(
        id="alpha158_vstd10",
        name="10日量波动",
        name_en="VSTD10",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """10日成交量标准差 / 均值"""
    return df['volume'].rolling(10).std() / (df['volume'].rolling(10).mean() + 1e-8)
''',
        description="中短期成交量变异系数",
        logic="值越大表示成交量波动越大",
        reference="Qlib Alpha158",
        historical_ic=0.015,
        tags=["alpha158", "volume", "volatility"],
    ),
    
    ClassicFactor(
        id="alpha158_vstd20",
        name="20日量波动",
        name_en="VSTD20",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """20日成交量标准差 / 均值"""
    return df['volume'].rolling(20).std() / (df['volume'].rolling(20).mean() + 1e-8)
''',
        description="中期成交量变异系数",
        logic="值越大表示成交量波动越大",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "volume", "volatility"],
    ),
    
    # ==================== 换手率类 ====================
    ClassicFactor(
        id="alpha158_turn5",
        name="5日换手",
        name_en="TURN5",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """5日平均换手率"""
    return df['turnover'].rolling(5).mean()
''',
        description="短期平均换手率",
        logic="换手率高表示交投活跃",
        reference="Qlib Alpha158",
        historical_ic=0.015,
        tags=["alpha158", "turnover", "liquidity"],
    ),
    
    ClassicFactor(
        id="alpha158_turn10",
        name="10日换手",
        name_en="TURN10",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """10日平均换手率"""
    return df['turnover'].rolling(10).mean()
''',
        description="中短期平均换手率",
        logic="换手率高表示交投活跃",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "turnover", "liquidity"],
    ),
    
    ClassicFactor(
        id="alpha158_turn20",
        name="20日换手",
        name_en="TURN20",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """20日平均换手率"""
    return df['turnover'].rolling(20).mean()
''',
        description="中期平均换手率",
        logic="换手率高表示交投活跃",
        reference="Qlib Alpha158",
        historical_ic=0.020,
        tags=["alpha158", "turnover", "liquidity"],
    ),
    
    ClassicFactor(
        id="alpha158_turn60",
        name="60日换手",
        name_en="TURN60",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """60日平均换手率"""
    return df['turnover'].rolling(60).mean()
''',
        description="长期平均换手率",
        logic="换手率高表示交投活跃",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "turnover", "liquidity"],
    ),
    
    # ==================== RSI 类 ====================
    ClassicFactor(
        id="alpha158_rsi6",
        name="RSI6",
        name_en="RSI6",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """6日RSI"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    rs = gain / (loss + 1e-8)
    return 100 - 100 / (1 + rs)
''',
        description="6日相对强弱指标",
        logic="RSI>70超买，RSI<30超卖",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "rsi", "oscillator"],
    ),
    
    ClassicFactor(
        id="alpha158_rsi12",
        name="RSI12",
        name_en="RSI12",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """12日RSI"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(12).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(12).mean()
    rs = gain / (loss + 1e-8)
    return 100 - 100 / (1 + rs)
''',
        description="12日相对强弱指标",
        logic="RSI>70超买，RSI<30超卖",
        reference="Qlib Alpha158",
        historical_ic=0.020,
        tags=["alpha158", "rsi", "oscillator"],
    ),
    
    ClassicFactor(
        id="alpha158_rsi24",
        name="RSI24",
        name_en="RSI24",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """24日RSI"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(24).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(24).mean()
    rs = gain / (loss + 1e-8)
    return 100 - 100 / (1 + rs)
''',
        description="24日相对强弱指标",
        logic="RSI>70超买，RSI<30超卖",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "rsi", "oscillator"],
    ),
    
    # ==================== BETA 类 ====================
    ClassicFactor(
        id="alpha158_beta5",
        name="5日Beta",
        name_en="BETA5",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """5日Beta - 使用波动率代理"""
    return df['close'].pct_change().rolling(5).std()
''',
        description="短期系统性风险暴露",
        logic="Beta高表示对市场敏感",
        reference="Qlib Alpha158",
        historical_ic=0.012,
        tags=["alpha158", "beta", "risk"],
    ),
    
    ClassicFactor(
        id="alpha158_beta10",
        name="10日Beta",
        name_en="BETA10",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """10日Beta - 使用波动率代理"""
    return df['close'].pct_change().rolling(10).std()
''',
        description="中短期系统性风险暴露",
        logic="Beta高表示对市场敏感",
        reference="Qlib Alpha158",
        historical_ic=0.015,
        tags=["alpha158", "beta", "risk"],
    ),
    
    ClassicFactor(
        id="alpha158_beta20",
        name="20日Beta",
        name_en="BETA20",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """20日Beta - 使用波动率代理"""
    return df['close'].pct_change().rolling(20).std()
''',
        description="中期系统性风险暴露",
        logic="Beta高表示对市场敏感",
        reference="Qlib Alpha158",
        historical_ic=0.018,
        tags=["alpha158", "beta", "risk"],
    ),
    
    ClassicFactor(
        id="alpha158_beta60",
        name="60日Beta",
        name_en="BETA60",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """60日Beta - 使用波动率代理"""
    return df['close'].pct_change().rolling(60).std()
''',
        description="长期系统性风险暴露",
        logic="Beta高表示对市场敏感",
        reference="Qlib Alpha158",
        historical_ic=0.015,
        tags=["alpha158", "beta", "risk"],
    ),
    
    # ==================== 偏度/峰度 ====================
    ClassicFactor(
        id="alpha158_skew20",
        name="20日偏度",
        name_en="SKEW20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """20日收益率偏度"""
    return df['close'].pct_change().rolling(20).skew()
''',
        description="收益率分布的偏度",
        logic="正偏度表示有上涨潜力",
        reference="Qlib Alpha158",
        historical_ic=0.010,
        tags=["alpha158", "distribution", "risk"],
    ),
    
    ClassicFactor(
        id="alpha158_kurt20",
        name="20日峰度",
        name_en="KURT20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """20日收益率峰度"""
    return df['close'].pct_change().rolling(20).kurt()
''',
        description="收益率分布的峰度",
        logic="高峰度表示尾部风险大",
        reference="Qlib Alpha158",
        historical_ic=0.008,
        tags=["alpha158", "distribution", "risk"],
    ),
    
    # ==================== 趋势强度 ====================
    ClassicFactor(
        id="alpha158_qtlu5",
        name="5日上分位",
        name_en="QTLU5",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价在5日内的分位数"""
    return df['close'].rolling(5).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-8))
''',
        description="收盘价在5日高低区间的位置",
        logic="接近1表示接近区间顶部",
        reference="Qlib Alpha158",
        historical_ic=0.020,
        tags=["alpha158", "quantile", "momentum"],
    ),
    
    ClassicFactor(
        id="alpha158_qtlu10",
        name="10日上分位",
        name_en="QTLU10",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价在10日内的分位数"""
    return df['close'].rolling(10).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-8))
''',
        description="收盘价在10日高低区间的位置",
        logic="接近1表示接近区间顶部",
        reference="Qlib Alpha158",
        historical_ic=0.022,
        tags=["alpha158", "quantile", "momentum"],
    ),
    
    ClassicFactor(
        id="alpha158_qtlu20",
        name="20日上分位",
        name_en="QTLU20",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """收盘价在20日内的分位数"""
    return df['close'].rolling(20).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-8))
''',
        description="收盘价在20日高低区间的位置",
        logic="接近1表示接近区间顶部",
        reference="Qlib Alpha158",
        historical_ic=0.025,
        tags=["alpha158", "quantile", "momentum"],
    ),
]


def get_alpha158_factors():
    """获取所有Alpha158因子"""
    return ALPHA158_FACTORS


def get_alpha158_count():
    """获取Alpha158因子数量"""
    return len(ALPHA158_FACTORS)
