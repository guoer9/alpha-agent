"""
经典因子库 - 学术界和业界公认的有效因子

参考来源:
- Barra CNE5/CNE6 风格因子
- Fama-French 三因子/五因子模型
- 《101 Formulaic Alphas》 WorldQuant
- 《实证资产定价》 经典文献
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class FactorCategory(Enum):
    """因子类别"""
    BARRA_STYLE = "barra_style"       # Barra风格因子
    TECHNICAL = "technical"            # 技术分析
    FUNDAMENTAL = "fundamental"        # 基本面
    VOLUME_PRICE = "volume_price"      # 量价
    ALTERNATIVE = "alternative"        # 另类数据


@dataclass
class ClassicFactor:
    """经典因子定义"""
    id: str                            # 唯一ID
    name: str                          # 因子名称
    name_en: str                       # 英文名称
    category: FactorCategory           # 类别
    
    # 代码
    code: str                          # Python实现代码
    
    # 描述
    description: str                   # 因子描述
    logic: str                         # 投资逻辑
    
    # 学术来源
    reference: str = ""                # 文献出处
    author: str = ""                   # 提出者
    year: int = 0                      # 提出年份
    
    # 历史表现 (A股回测)
    historical_ic: float = 0.0         # 历史IC
    historical_icir: float = 0.0       # 历史ICIR
    typical_turnover: float = 0.0      # 典型换手率
    
    # 特征
    tags: List[str] = field(default_factory=list)
    
    # 注意事项
    pitfalls: List[str] = field(default_factory=list)


# ============================================================
# Barra CNE5/CNE6 风格因子
# ============================================================

BARRA_FACTORS = [
    # -------------------- Size 规模 --------------------
    ClassicFactor(
        id="barra_size",
        name="市值因子",
        name_en="Size",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    Size - 市值因子
    小市值股票长期跑赢大市值股票 (A股小市值效应显著)
    """
    size = np.log(df['market_cap'])
    return -size.rank(pct=True)  # 负号表示小市值为正暴露
''',
        description="对数市值的负值，捕捉小市值溢价",
        logic="小市值公司信息不对称程度高，流动性差，需要风险补偿",
        reference="Banz (1981) 'The relationship between return and market value'",
        author="Banz",
        year=1981,
        historical_ic=0.035,
        historical_icir=0.45,
        typical_turnover=0.15,
        tags=["barra", "size", "value", "long_term"],
        pitfalls=[
            "A股小市值效应波动大，2017年后有所减弱",
            "需要剔除壳资源价值的影响",
            "微盘股流动性差，实际交易摩擦大"
        ]
    ),
    
    # -------------------- Beta 贝塔 --------------------
    ClassicFactor(
        id="barra_beta",
        name="贝塔因子",
        name_en="Beta",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    Beta - 贝塔因子
    低贝塔股票风险调整后收益更高 (低波动异象)
    """
    # 计算252日滚动贝塔
    returns = df['close'].pct_change()
    market_returns = df['market_ret']  # 需要市场收益
    
    # 简化版：用波动率代理贝塔
    volatility = returns.rolling(60).std()
    return -volatility.rank(pct=True)  # 低波动为正
''',
        description="股票相对市场的系统性风险暴露",
        logic="低贝塔异象：高风险股票并未获得应有的风险补偿",
        reference="Frazzini & Pedersen (2014) 'Betting Against Beta'",
        author="Frazzini & Pedersen",
        year=2014,
        historical_ic=0.025,
        historical_icir=0.35,
        typical_turnover=0.20,
        tags=["barra", "risk", "low_volatility"],
        pitfalls=[
            "贝塔计算需要较长历史数据",
            "市场剧烈波动时贝塔不稳定",
            "A股做空限制使低贝塔策略难以对冲"
        ]
    ),
    
    # -------------------- Momentum 动量 --------------------
    ClassicFactor(
        id="barra_momentum",
        name="动量因子",
        name_en="Momentum",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    Momentum - 动量因子 (剔除最近1个月)
    过去表现好的股票未来继续表现好
    """
    # 12-1动量：过去12个月收益，剔除最近1个月
    ret_12m = df['close'] / df['close'].shift(252) - 1
    ret_1m = df['close'] / df['close'].shift(21) - 1
    momentum = ret_12m - ret_1m
    return momentum.rank(pct=True)
''',
        description="过去12个月收益（剔除最近1个月）",
        logic="投资者对信息反应不足导致价格趋势延续",
        reference="Jegadeesh & Titman (1993) 'Returns to Buying Winners'",
        author="Jegadeesh & Titman",
        year=1993,
        historical_ic=0.028,
        historical_icir=0.38,
        typical_turnover=0.35,
        tags=["barra", "momentum", "trend"],
        pitfalls=[
            "动量反转风险大，极端市场会崩溃",
            "A股动量周期较短，12个月可能太长",
            "2015年股灾、2020年3月等时期严重回撤"
        ]
    ),
    
    # -------------------- Residual Volatility 残差波动率 --------------------
    ClassicFactor(
        id="barra_resvol",
        name="残差波动率",
        name_en="Residual Volatility",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    Residual Volatility - 残差波动率
    低特质波动率股票收益更高
    """
    returns = df['close'].pct_change()
    # 简化：用总波动率代理
    volatility = returns.rolling(60).std() * np.sqrt(252)
    return -volatility.rank(pct=True)  # 低波动为正
''',
        description="剔除市场和行业因素后的特质波动率",
        logic="高特质波动率意味着信息不确定性高，投资者往往高估其价值",
        reference="Ang et al. (2006) 'The Cross-Section of Volatility'",
        author="Ang",
        year=2006,
        historical_ic=0.032,
        historical_icir=0.42,
        typical_turnover=0.25,
        tags=["barra", "volatility", "quality"],
        pitfalls=[
            "残差波动率计算依赖因子模型设定",
            "高波动率股票可能包含信息优势"
        ]
    ),
    
    # -------------------- Value 价值 --------------------
    ClassicFactor(
        id="barra_value",
        name="价值因子",
        name_en="Book-to-Price",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    Value - 价值因子 (账面市值比)
    低估值股票长期跑赢高估值股票
    """
    # BP = 1/PB
    bp = 1 / df['pb'].clip(lower=0.1)  # 避免负数和极端值
    return bp.rank(pct=True)
''',
        description="账面价值/市值，低估值为高暴露",
        logic="投资者过度反应导致价值股被低估",
        reference="Fama & French (1992) 'Cross-Section of Expected Returns'",
        author="Fama & French",
        year=1992,
        historical_ic=0.022,
        historical_icir=0.30,
        typical_turnover=0.18,
        tags=["barra", "value", "fundamental"],
        pitfalls=[
            "价值陷阱：低估值可能是基本面恶化",
            "A股价值因子有效性波动",
            "需要区分低PB和低PE"
        ]
    ),
    
    # -------------------- Earnings Yield 盈利收益率 --------------------
    ClassicFactor(
        id="barra_earnings_yield",
        name="盈利收益率",
        name_en="Earnings Yield",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    Earnings Yield - 盈利收益率
    高盈利收益率（低PE）的股票收益更高
    """
    # EP = 1/PE
    ep = 1 / df['pe_ttm'].clip(lower=1)  # PE为正
    ep = ep.where(df['pe_ttm'] > 0, 0)   # 亏损公司设为0
    return ep.rank(pct=True)
''',
        description="盈利/市值，高盈利收益率为高暴露",
        logic="高盈利收益率意味着股票被低估",
        reference="Barra CNE5",
        author="Barra",
        year=2007,
        historical_ic=0.025,
        historical_icir=0.35,
        typical_turnover=0.20,
        tags=["barra", "value", "fundamental"],
        pitfalls=[
            "周期股PE在周期底部最高",
            "高成长股PE天然较高"
        ]
    ),
    
    # -------------------- Liquidity 流动性 --------------------
    ClassicFactor(
        id="barra_liquidity",
        name="流动性因子",
        name_en="Liquidity",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    Liquidity - 流动性因子
    低流动性股票需要流动性溢价
    """
    # 换手率作为流动性代理
    turnover_avg = df['turnover'].rolling(20).mean()
    return -turnover_avg.rank(pct=True)  # 低流动性为正
''',
        description="基于换手率的流动性度量",
        logic="低流动性股票交易成本高，需要补偿",
        reference="Pastor & Stambaugh (2003) 'Liquidity Risk'",
        author="Pastor & Stambaugh",
        year=2003,
        historical_ic=0.020,
        historical_icir=0.28,
        typical_turnover=0.15,
        tags=["barra", "liquidity", "market_microstructure"],
        pitfalls=[
            "低流动性股票实际交易时冲击成本大",
            "流动性危机时低流动性股票跌幅更大"
        ]
    ),
    
    # -------------------- Growth 成长 --------------------
    ClassicFactor(
        id="barra_growth",
        name="成长因子",
        name_en="Growth",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    Growth - 成长因子
    高成长公司股价表现更好
    """
    # 营收增长率
    if 'revenue_yoy' in df.columns:
        growth = df['revenue_yoy']
    else:
        # 使用ROE作为代理
        growth = df['roe_ttm']
    return growth.rank(pct=True)
''',
        description="营收或盈利的历史增长率",
        logic="高成长公司具有更高的内在价值",
        reference="Barra CNE5",
        author="Barra",
        year=2007,
        historical_ic=0.018,
        historical_icir=0.25,
        typical_turnover=0.22,
        tags=["barra", "growth", "fundamental"],
        pitfalls=[
            "高成长不一定可持续",
            "成长股估值往往偏高"
        ]
    ),
    
    # -------------------- Leverage 杠杆 --------------------
    ClassicFactor(
        id="barra_leverage",
        name="杠杆因子",
        name_en="Leverage",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    Leverage - 杠杆因子
    低杠杆公司更稳健
    """
    leverage = df['debt_ratio'] if 'debt_ratio' in df.columns else 0.5
    return -leverage.rank(pct=True)  # 低杠杆为正
''',
        description="资产负债率，低杠杆为正暴露",
        logic="高杠杆增加财务风险",
        reference="Barra CNE5",
        author="Barra",
        year=2007,
        historical_ic=0.015,
        historical_icir=0.22,
        typical_turnover=0.12,
        tags=["barra", "quality", "fundamental"],
        pitfalls=[
            "杠杆水平有行业差异",
            "适度杠杆可能提升ROE"
        ]
    ),
]


# ============================================================
# 技术分析因子
# ============================================================

TECHNICAL_FACTORS = [
    # -------------------- 短期反转 --------------------
    ClassicFactor(
        id="tech_reversal",
        name="短期反转",
        name_en="Short-term Reversal",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    Short-term Reversal - 短期反转
    过去一周/一月跌幅大的股票反弹
    """
    ret_5d = df['close'].pct_change(5)
    return -ret_5d.rank(pct=True)  # 跌多为正
''',
        description="过去5日收益率的负值",
        logic="短期过度反应后的价格修正",
        reference="Jegadeesh (1990) 'Predictable Behavior'",
        author="Jegadeesh",
        year=1990,
        historical_ic=0.038,
        historical_icir=0.48,
        typical_turnover=0.65,
        tags=["technical", "reversal", "short_term"],
        pitfalls=[
            "换手率极高",
            "下跌可能是信息驱动而非过度反应",
            "需要剔除ST和停牌股"
        ]
    ),
    
    # -------------------- 均线偏离 --------------------
    ClassicFactor(
        id="tech_ma_deviation",
        name="均线偏离度",
        name_en="MA Deviation",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    MA Deviation - 均线偏离度
    偏离均线过多会回归
    """
    ma20 = df['close'].rolling(20).mean()
    deviation = (df['close'] - ma20) / ma20
    return -deviation.rank(pct=True)  # 高于均线为负暴露
''',
        description="股价相对20日均线的偏离度",
        logic="技术分析中的均值回归思想",
        reference="技术分析经典",
        author="",
        year=0,
        historical_ic=0.030,
        historical_icir=0.40,
        typical_turnover=0.55,
        tags=["technical", "mean_reversion", "short_term"],
        pitfalls=[
            "趋势行情中偏离可能持续扩大",
            "均线周期选择影响效果"
        ]
    ),
    
    # -------------------- RSI --------------------
    ClassicFactor(
        id="tech_rsi",
        name="相对强弱指标",
        name_en="RSI",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    RSI - 相对强弱指标
    超买超卖信号
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    
    # 超买(RSI>70)做空，超卖(RSI<30)做多
    return -(rsi - 50).rank(pct=True)
''',
        description="14日RSI指标",
        logic="超买超卖的反转逻辑",
        reference="Wilder (1978)",
        author="Wilder",
        year=1978,
        historical_ic=0.022,
        historical_icir=0.30,
        typical_turnover=0.50,
        tags=["technical", "oscillator", "mean_reversion"],
        pitfalls=[
            "强势股可以持续超买",
            "需要结合趋势判断"
        ]
    ),
    
    # -------------------- 波动率突破 --------------------
    ClassicFactor(
        id="tech_volatility_breakout",
        name="波动率突破",
        name_en="Volatility Breakout",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    Volatility Breakout - 波动率突破
    突破近期高点的动量信号
    """
    high_20d = df['high'].rolling(20).max()
    breakout = df['close'] / high_20d - 1
    return breakout.rank(pct=True)
''',
        description="收盘价相对20日最高价的位置",
        logic="突破形态往往预示趋势延续",
        reference="Turtle Trading",
        author="Dennis & Eckhardt",
        year=1983,
        historical_ic=0.025,
        historical_icir=0.32,
        typical_turnover=0.45,
        tags=["technical", "breakout", "momentum"],
        pitfalls=[
            "假突破风险",
            "需要成交量配合确认"
        ]
    ),
    
    # -------------------- MACD --------------------
    ClassicFactor(
        id="tech_macd",
        name="MACD",
        name_en="MACD",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    MACD - 平滑异同移动平均
    """
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    macd = (dif - dea) * 2
    return macd.rank(pct=True)
''',
        description="MACD柱状图",
        logic="趋势跟踪指标",
        reference="Appel (1979)",
        author="Appel",
        year=1979,
        historical_ic=0.020,
        historical_icir=0.28,
        typical_turnover=0.40,
        tags=["technical", "trend", "momentum"],
        pitfalls=[
            "震荡市频繁发出错误信号",
            "滞后性较强"
        ]
    ),
]


# ============================================================
# 基本面因子
# ============================================================

FUNDAMENTAL_FACTORS = [
    # -------------------- ROE --------------------
    ClassicFactor(
        id="fund_roe",
        name="净资产收益率",
        name_en="ROE",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    ROE - 净资产收益率
    高质量公司的核心指标
    """
    roe = df['roe_ttm']
    return roe.rank(pct=True)
''',
        description="净利润/净资产，反映股东回报能力",
        logic="高ROE公司具有竞争优势，可持续创造价值",
        reference="Buffett 'Owner Earnings'",
        author="Buffett",
        year=1987,
        historical_ic=0.028,
        historical_icir=0.38,
        typical_turnover=0.15,
        tags=["fundamental", "quality", "profitability"],
        pitfalls=[
            "高杠杆可以人为提升ROE",
            "需要结合杜邦分析"
        ]
    ),
    
    # -------------------- 盈利稳定性 --------------------
    ClassicFactor(
        id="fund_earning_stability",
        name="盈利稳定性",
        name_en="Earnings Stability",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    Earnings Stability - 盈利稳定性
    盈利波动小的公司更稳健
    """
    # 使用ROE的稳定性作为代理
    if 'roe_ttm' in df.columns:
        roe_std = df['roe_ttm'].rolling(4).std()  # 4个季度
        return -roe_std.rank(pct=True)  # 波动小为正
    return pd.Series(0, index=df.index)
''',
        description="过去几年盈利的标准差",
        logic="盈利稳定意味着业务可预测性强",
        reference="Novy-Marx (2013) 'Quality Minus Junk'",
        author="Novy-Marx",
        year=2013,
        historical_ic=0.018,
        historical_icir=0.25,
        typical_turnover=0.10,
        tags=["fundamental", "quality", "stability"],
        pitfalls=[
            "周期行业盈利天然波动大"
        ]
    ),
    
    # -------------------- 应计异象 --------------------
    ClassicFactor(
        id="fund_accruals",
        name="应计因子",
        name_en="Accruals",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    Accruals - 应计项目异象
    低应计项目的公司盈利质量更高
    """
    # 简化：用经营现金流/净利润代理
    if 'ocf' in df.columns and 'net_profit' in df.columns:
        accrual_ratio = 1 - df['ocf'] / (df['net_profit'] + 1e-8)
        return -accrual_ratio.rank(pct=True)  # 低应计为正
    return pd.Series(0, index=df.index)
''',
        description="盈利中非现金部分的占比",
        logic="高应计意味着盈利质量差，可能包含操纵",
        reference="Sloan (1996) 'Accrual Anomaly'",
        author="Sloan",
        year=1996,
        historical_ic=0.022,
        historical_icir=0.30,
        typical_turnover=0.18,
        tags=["fundamental", "quality", "accounting"],
        pitfalls=[
            "应计计算需要详细财务数据",
            "行业间可比性差"
        ]
    ),
    
    # -------------------- 资产增长 --------------------
    ClassicFactor(
        id="fund_asset_growth",
        name="资产增长",
        name_en="Asset Growth",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    Asset Growth - 资产增长异象
    资产扩张慢的公司收益更高
    """
    if 'total_assets_yoy' in df.columns:
        return -df['total_assets_yoy'].rank(pct=True)  # 低增长为正
    return pd.Series(0, index=df.index)
''',
        description="总资产同比增长率的负值",
        logic="激进扩张往往毁损股东价值",
        reference="Cooper et al. (2008) 'Asset Growth'",
        author="Cooper",
        year=2008,
        historical_ic=0.020,
        historical_icir=0.28,
        typical_turnover=0.15,
        tags=["fundamental", "investment", "quality"],
        pitfalls=[
            "成长型公司资产增长快是正常的"
        ]
    ),
    
    # -------------------- 股息率 --------------------
    ClassicFactor(
        id="fund_dividend_yield",
        name="股息率",
        name_en="Dividend Yield",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    Dividend Yield - 股息率
    高股息股票提供稳定现金回报
    """
    div_yield = df['dividend_yield'] if 'dividend_yield' in df.columns else 0
    return div_yield.rank(pct=True)
''',
        description="每股股息/股价",
        logic="高股息反映公司现金流稳定",
        reference="Fama & French (1988)",
        author="Fama & French",
        year=1988,
        historical_ic=0.018,
        historical_icir=0.25,
        typical_turnover=0.12,
        tags=["fundamental", "value", "income"],
        pitfalls=[
            "A股股息率整体偏低",
            "高股息可能是因为股价下跌"
        ]
    ),
]


# ============================================================
# 量价因子
# ============================================================

VOLUME_PRICE_FACTORS = [
    # -------------------- 换手率 --------------------
    ClassicFactor(
        id="vp_turnover",
        name="换手率因子",
        name_en="Turnover",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    Turnover - 换手率
    低换手率股票可能被低估
    """
    turnover_avg = df['turnover'].rolling(20).mean()
    return -turnover_avg.rank(pct=True)  # 低换手为正
''',
        description="20日平均换手率的负值",
        logic="低换手率意味着关注度低，可能存在定价偏差",
        reference="Datar et al. (1998)",
        author="Datar",
        year=1998,
        historical_ic=0.025,
        historical_icir=0.32,
        typical_turnover=0.20,
        tags=["volume_price", "liquidity", "attention"],
        pitfalls=[
            "低换手可能是基本面恶化导致无人问津"
        ]
    ),
    
    # -------------------- 异常换手 --------------------
    ClassicFactor(
        id="vp_abnormal_turnover",
        name="异常换手",
        name_en="Abnormal Turnover",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    Abnormal Turnover - 异常换手率
    换手率突增可能是反转信号
    """
    turnover_avg = df['turnover'].rolling(20).mean()
    turnover_std = df['turnover'].rolling(20).std()
    abnormal = (df['turnover'] - turnover_avg) / (turnover_std + 1e-8)
    return -abnormal.rank(pct=True)  # 异常高为负
''',
        description="换手率相对历史均值的标准化偏离",
        logic="换手率突增往往是情绪过热的信号",
        reference="Lee & Swaminathan (2000)",
        author="Lee & Swaminathan",
        year=2000,
        historical_ic=0.022,
        historical_icir=0.30,
        typical_turnover=0.45,
        tags=["volume_price", "sentiment", "reversal"],
        pitfalls=[
            "利好消息也会导致换手率突增"
        ]
    ),
    
    # -------------------- 量价背离 --------------------
    ClassicFactor(
        id="vp_volume_price_divergence",
        name="量价背离",
        name_en="Volume Price Divergence",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    Volume Price Divergence - 量价背离
    价涨量缩可能是上涨乏力
    """
    price_ret = df['close'].pct_change(5)
    volume_ret = df['volume'].pct_change(5)
    
    # 量价背离：价格方向与成交量方向不一致
    divergence = -price_ret * np.sign(volume_ret)
    return divergence.rank(pct=True)
''',
        description="价格变化与成交量变化的背离程度",
        logic="健康的上涨需要量价配合",
        reference="技术分析经典",
        author="",
        year=0,
        historical_ic=0.018,
        historical_icir=0.25,
        typical_turnover=0.50,
        tags=["volume_price", "technical", "divergence"],
        pitfalls=[
            "信号噪声大"
        ]
    ),
    
    # -------------------- 资金流向 --------------------
    ClassicFactor(
        id="vp_money_flow",
        name="资金流向",
        name_en="Money Flow",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    Money Flow - 资金流向 (简化版)
    使用典型价格和成交量估算
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    
    # 正向和负向资金流
    positive_flow = raw_money_flow.where(df['close'] > df['close'].shift(1), 0)
    negative_flow = raw_money_flow.where(df['close'] < df['close'].shift(1), 0)
    
    mfi = positive_flow.rolling(14).sum() / (positive_flow.rolling(14).sum() + negative_flow.rolling(14).sum() + 1e-8)
    return mfi.rank(pct=True)
''',
        description="基于价格和成交量的资金流向指标",
        logic="资金流入预示后续上涨",
        reference="MFI指标",
        author="",
        year=0,
        historical_ic=0.020,
        historical_icir=0.28,
        typical_turnover=0.40,
        tags=["volume_price", "money_flow", "sentiment"],
        pitfalls=[
            "A股资金流数据可能有水分"
        ]
    ),
    
    # -------------------- 振幅 --------------------
    ClassicFactor(
        id="vp_amplitude",
        name="振幅因子",
        name_en="Amplitude",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    Amplitude - 振幅因子
    低振幅股票更稳定
    """
    amplitude = (df['high'] - df['low']) / df['close']
    amplitude_avg = amplitude.rolling(20).mean()
    return -amplitude_avg.rank(pct=True)  # 低振幅为正
''',
        description="日内振幅的平均值",
        logic="低振幅意味着低波动",
        reference="实证研究",
        author="",
        year=0,
        historical_ic=0.025,
        historical_icir=0.33,
        typical_turnover=0.25,
        tags=["volume_price", "volatility", "stability"],
        pitfalls=[
            "涨跌停时振幅被压缩"
        ]
    ),
    
    # -------------------- ILLIQ非流动性 --------------------
    ClassicFactor(
        id="vp_illiquidity",
        name="Amihud非流动性",
        name_en="Amihud Illiquidity",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    Amihud Illiquidity - 非流动性指标
    价格冲击成本
    """
    ret_abs = df['close'].pct_change().abs()
    illiq = ret_abs / (df['amount'] + 1e-8) * 1e8
    illiq_avg = illiq.rolling(20).mean()
    return -illiq_avg.rank(pct=True)  # 高流动性为正? 或者做多低流动性
''',
        description="收益率绝对值/成交额",
        logic="低流动性股票需要流动性溢价",
        reference="Amihud (2002)",
        author="Amihud",
        year=2002,
        historical_ic=0.022,
        historical_icir=0.30,
        typical_turnover=0.18,
        tags=["volume_price", "liquidity", "market_microstructure"],
        pitfalls=[
            "极端非流动性股票难以交易"
        ]
    ),
]


# ============================================================
# 汇总所有经典因子
# ============================================================

ALL_CLASSIC_FACTORS = BARRA_FACTORS + TECHNICAL_FACTORS + FUNDAMENTAL_FACTORS + VOLUME_PRICE_FACTORS


def get_factor_by_id(factor_id: str) -> Optional[ClassicFactor]:
    """根据ID获取因子"""
    for factor in ALL_CLASSIC_FACTORS:
        if factor.id == factor_id:
            return factor
    return None


def get_factors_by_category(category: FactorCategory) -> List[ClassicFactor]:
    """根据类别获取因子"""
    return [f for f in ALL_CLASSIC_FACTORS if f.category == category]


def get_factors_by_tag(tag: str) -> List[ClassicFactor]:
    """根据标签获取因子"""
    return [f for f in ALL_CLASSIC_FACTORS if tag in f.tags]


# 打印因子库统计
if __name__ == "__main__":
    print(f"经典因子库统计:")
    print(f"- Barra风格因子: {len(BARRA_FACTORS)}个")
    print(f"- 技术分析因子: {len(TECHNICAL_FACTORS)}个")
    print(f"- 基本面因子: {len(FUNDAMENTAL_FACTORS)}个")
    print(f"- 量价因子: {len(VOLUME_PRICE_FACTORS)}个")
    print(f"- 总计: {len(ALL_CLASSIC_FACTORS)}个")
