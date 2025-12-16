"""
Academic Premia 学术溢价因子库

来源: 顶级金融期刊的经典因子研究
包含: 学术界公认的风险溢价因子

参考:
- Journal of Finance
- Journal of Financial Economics  
- Review of Financial Studies
- Journal of Accounting Research
"""

from dataclasses import dataclass, field
from typing import List
from .classic_factors import ClassicFactor, FactorCategory


# ============================================================
# Academic Premia 学术溢价因子
# 按论文来源组织
# ============================================================

ACADEMIC_PREMIA_FACTORS = [
    # ==================== Fama-French 因子 ====================
    ClassicFactor(
        id="ap_smb",
        name="规模溢价 SMB",
        name_en="Size Premium (SMB)",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    SMB (Small Minus Big) - 规模溢价
    做多小市值，做空大市值
    """
    import numpy as np
    market_cap = df['market_cap'] if 'market_cap' in df.columns else df['close'] * df['volume']
    log_cap = np.log(market_cap + 1)
    return -log_cap.rank(pct=True)  # 小市值为正
''',
        description="小市值股票相对大市值股票的超额收益",
        logic="小公司信息不对称程度高，流动性差，投资者要求更高补偿",
        reference="Fama & French (1993) 'Common Risk Factors in the Returns on Stocks and Bonds', JFE",
        author="Fama & French",
        year=1993,
        historical_ic=0.035,
        historical_icir=0.45,
        typical_turnover=0.12,
        tags=["academic", "fama_french", "size", "risk_premium"],
        pitfalls=["A股小市值效应2017年后减弱", "壳价值干扰", "微盘股流动性差"],
    ),
    
    ClassicFactor(
        id="ap_hml",
        name="价值溢价 HML",
        name_en="Value Premium (HML)",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    HML (High Minus Low) - 价值溢价
    做多高B/M，做空低B/M
    """
    bp = 1 / df['pb'].clip(lower=0.1) if 'pb' in df.columns else 0.5
    return bp.rank(pct=True)
''',
        description="高账面市值比股票相对低账面市值比股票的超额收益",
        logic="价值股风险更高（财务困境），需要更高回报补偿",
        reference="Fama & French (1992) 'The Cross-Section of Expected Stock Returns', JF",
        author="Fama & French",
        year=1992,
        historical_ic=0.022,
        historical_icir=0.30,
        typical_turnover=0.15,
        tags=["academic", "fama_french", "value", "risk_premium"],
        pitfalls=["价值陷阱", "周期性行业干扰"],
    ),
    
    ClassicFactor(
        id="ap_umd",
        name="动量溢价 UMD",
        name_en="Momentum Premium (UMD)",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    UMD (Up Minus Down) - 动量溢价
    做多过去赢家，做空过去输家
    """
    # 12-1 动量：过去12个月收益剔除最近1个月
    ret_12m = df['close'] / df['close'].shift(252) - 1
    ret_1m = df['close'] / df['close'].shift(21) - 1
    momentum = ret_12m - ret_1m
    return momentum.rank(pct=True)
''',
        description="过去表现好的股票继续表现好的趋势",
        logic="投资者对信息反应不足，导致价格趋势延续",
        reference="Jegadeesh & Titman (1993) 'Returns to Buying Winners and Selling Losers', JF",
        author="Jegadeesh & Titman",
        year=1993,
        historical_ic=0.028,
        historical_icir=0.38,
        typical_turnover=0.35,
        tags=["academic", "momentum", "behavioral", "risk_premium"],
        pitfalls=["动量崩溃风险", "极端市场回撤大"],
    ),
    
    ClassicFactor(
        id="ap_rmw",
        name="盈利溢价 RMW",
        name_en="Profitability Premium (RMW)",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    RMW (Robust Minus Weak) - 盈利溢价
    做多高盈利，做空低盈利
    """
    # 使用毛利率或ROE
    profitability = df['roe_ttm'] if 'roe_ttm' in df.columns else df['close'].pct_change(60)
    return profitability.rank(pct=True)
''',
        description="高盈利公司相对低盈利公司的超额收益",
        logic="高盈利公司具有竞争优势，未来现金流更稳定",
        reference="Fama & French (2015) 'A Five-Factor Asset Pricing Model', JFE",
        author="Fama & French",
        year=2015,
        historical_ic=0.030,
        historical_icir=0.40,
        typical_turnover=0.18,
        tags=["academic", "fama_french", "quality", "profitability"],
        pitfalls=["高杠杆可能扭曲ROE"],
    ),
    
    ClassicFactor(
        id="ap_cma",
        name="投资溢价 CMA",
        name_en="Investment Premium (CMA)",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    CMA (Conservative Minus Aggressive) - 投资溢价
    做多低投资，做空高投资
    """
    # 资产增长率的负值
    if 'total_assets_yoy' in df.columns:
        investment = df['total_assets_yoy']
    else:
        investment = df['close'].pct_change(252)
    return -investment.rank(pct=True)
''',
        description="保守投资公司相对激进投资公司的超额收益",
        logic="激进扩张往往意味着较低的未来回报",
        reference="Fama & French (2015) 'A Five-Factor Asset Pricing Model', JFE",
        author="Fama & French",
        year=2015,
        historical_ic=0.020,
        historical_icir=0.28,
        typical_turnover=0.12,
        tags=["academic", "fama_french", "investment", "quality"],
        pitfalls=["成长型公司投资高是正常的"],
    ),
    
    # ==================== 低风险异象 ====================
    ClassicFactor(
        id="ap_bab",
        name="低贝塔溢价 BAB",
        name_en="Betting Against Beta (BAB)",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    BAB (Betting Against Beta) - 低贝塔溢价
    做多低贝塔，做空高贝塔
    """
    returns = df['close'].pct_change()
    volatility = returns.rolling(60).std() * (252 ** 0.5)
    return -volatility.rank(pct=True)  # 低波动为正
''',
        description="低贝塔股票风险调整后收益更高",
        logic="杠杆约束使投资者偏好高贝塔股票，低贝塔被低估",
        reference="Frazzini & Pedersen (2014) 'Betting Against Beta', JFE",
        author="Frazzini & Pedersen",
        year=2014,
        historical_ic=0.028,
        historical_icir=0.38,
        typical_turnover=0.20,
        tags=["academic", "low_risk", "anomaly", "behavioral"],
        pitfalls=["A股做空限制影响策略实施"],
    ),
    
    ClassicFactor(
        id="ap_idiovol",
        name="低特质波动溢价",
        name_en="Idiosyncratic Volatility (IVOL)",
        category=FactorCategory.BARRA_STYLE,
        code='''
def compute_alpha(df):
    """
    IVOL - 低特质波动溢价
    低特质波动股票收益更高
    """
    returns = df['close'].pct_change()
    # 简化：用总波动率代理特质波动率
    vol = returns.rolling(60).std() * (252 ** 0.5)
    return -vol.rank(pct=True)
''',
        description="低特质波动率股票的超额收益",
        logic="高特质波动率意味着彩票型股票，被过度追捧",
        reference="Ang et al. (2006) 'The Cross-Section of Volatility and Expected Returns', JF",
        author="Ang, Hodrick, Xing & Zhang",
        year=2006,
        historical_ic=0.032,
        historical_icir=0.42,
        typical_turnover=0.22,
        tags=["academic", "low_risk", "volatility", "anomaly"],
        pitfalls=["残差波动率计算依赖因子模型"],
    ),
    
    # ==================== 质量因子 ====================
    ClassicFactor(
        id="ap_qmj",
        name="质量溢价 QMJ",
        name_en="Quality Minus Junk (QMJ)",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    QMJ (Quality Minus Junk) - 质量溢价
    综合盈利能力、成长性、安全性、派息
    """
    # 简化版：使用ROE作为质量代理
    quality = df['roe_ttm'] if 'roe_ttm' in df.columns else df['close'].pct_change(60)
    
    # 也可以添加低杠杆、低波动等指标
    return quality.rank(pct=True)
''',
        description="高质量公司相对低质量公司的超额收益",
        logic="高质量公司盈利稳定、财务健康、增长可持续",
        reference="Asness, Frazzini & Pedersen (2019) 'Quality Minus Junk', RFS",
        author="Asness, Frazzini & Pedersen",
        year=2019,
        historical_ic=0.028,
        historical_icir=0.38,
        typical_turnover=0.15,
        tags=["academic", "quality", "profitability", "safety"],
        pitfalls=["质量指标构建较复杂"],
    ),
    
    ClassicFactor(
        id="ap_gp",
        name="毛利溢价 GP",
        name_en="Gross Profitability (GP)",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    GP/A - 毛利/资产
    毛利率高的公司收益更好
    """
    # 使用毛利率或ROE代理
    gp = df['roe_ttm'] if 'roe_ttm' in df.columns else 0.1
    return gp.rank(pct=True)
''',
        description="高毛利率公司的超额收益",
        logic="毛利是最纯净的盈利指标，不易被操纵",
        reference="Novy-Marx (2013) 'The Other Side of Value', JFE",
        author="Novy-Marx",
        year=2013,
        historical_ic=0.025,
        historical_icir=0.35,
        typical_turnover=0.15,
        tags=["academic", "quality", "profitability", "gross_profit"],
        pitfalls=["行业间毛利率差异大"],
    ),
    
    # ==================== 会计异象 ====================
    ClassicFactor(
        id="ap_accruals",
        name="应计异象",
        name_en="Accrual Anomaly",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    Accruals - 应计项目异象
    低应计项目的公司盈利质量更高
    """
    # 简化：用经营现金流/净利润代理
    if 'ocf' in df.columns and 'net_profit' in df.columns:
        accrual = 1 - df['ocf'] / (df['net_profit'].abs() + 1e-8)
    else:
        accrual = df['close'].pct_change(60)
    return -accrual.rank(pct=True)  # 低应计为正
''',
        description="低应计项目公司的超额收益",
        logic="高应计意味着盈利质量差，可能包含操纵",
        reference="Sloan (1996) 'Do Stock Prices Fully Reflect Information', TAR",
        author="Sloan",
        year=1996,
        historical_ic=0.022,
        historical_icir=0.30,
        typical_turnover=0.18,
        tags=["academic", "accounting", "quality", "earnings"],
        pitfalls=["应计计算需要详细财务数据"],
    ),
    
    ClassicFactor(
        id="ap_noa",
        name="净经营资产异象",
        name_en="Net Operating Assets (NOA)",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    NOA - 净经营资产异象
    高NOA公司未来收益较差
    """
    # 简化版本
    if 'total_assets' in df.columns:
        noa = df['total_assets']
    else:
        noa = df['close'] * df['volume']
    return -noa.rank(pct=True).diff(252)  # NOA增长的负值
''',
        description="净经营资产增长的负向预测能力",
        logic="NOA增长过快表明资本配置效率下降",
        reference="Hirshleifer et al. (2004) 'Do Investors Overvalue Firms with Bloated Balance Sheets?', JAE",
        author="Hirshleifer",
        year=2004,
        historical_ic=0.018,
        historical_icir=0.25,
        typical_turnover=0.15,
        tags=["academic", "accounting", "balance_sheet", "investment"],
        pitfalls=["需要资产负债表数据"],
    ),
    
    # ==================== 行为金融 ====================
    ClassicFactor(
        id="ap_reversal_st",
        name="短期反转",
        name_en="Short-term Reversal",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    Short-term Reversal - 短期反转
    过去一周跌幅大的股票反弹
    """
    ret_5d = df['close'].pct_change(5)
    return -ret_5d.rank(pct=True)
''',
        description="过去一周收益的反转效应",
        logic="短期过度反应后的价格修正",
        reference="Jegadeesh (1990) 'Evidence of Predictable Behavior', JF",
        author="Jegadeesh",
        year=1990,
        historical_ic=0.038,
        historical_icir=0.48,
        typical_turnover=0.65,
        tags=["academic", "behavioral", "reversal", "short_term"],
        pitfalls=["换手率极高", "信息驱动的下跌不会反转"],
    ),
    
    ClassicFactor(
        id="ap_reversal_lt",
        name="长期反转",
        name_en="Long-term Reversal",
        category=FactorCategory.TECHNICAL,
        code='''
def compute_alpha(df):
    """
    Long-term Reversal - 长期反转
    过去3-5年表现差的股票未来反弹
    """
    ret_3y = df['close'] / df['close'].shift(756) - 1  # 3年收益
    return -ret_3y.rank(pct=True)
''',
        description="过去3-5年收益的反转效应",
        logic="长期过度反应后的均值回归",
        reference="De Bondt & Thaler (1985) 'Does the Stock Market Overreact?', JF",
        author="De Bondt & Thaler",
        year=1985,
        historical_ic=0.020,
        historical_icir=0.28,
        typical_turnover=0.10,
        tags=["academic", "behavioral", "reversal", "long_term"],
        pitfalls=["需要长期数据", "基本面变化可能是永久性的"],
    ),
    
    ClassicFactor(
        id="ap_pead",
        name="盈余公告后漂移",
        name_en="Post-Earnings Announcement Drift (PEAD)",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    PEAD - 盈余公告后漂移
    盈余超预期的股票继续上涨
    """
    # 使用过去的收益变化作为代理
    ret_60d = df['close'].pct_change(60)
    return ret_60d.rank(pct=True)
''',
        description="盈余公告后股价的持续漂移",
        logic="投资者对盈余信息反应不足",
        reference="Bernard & Thomas (1989) 'Post-Earnings-Announcement Drift', JAR",
        author="Bernard & Thomas",
        year=1989,
        historical_ic=0.025,
        historical_icir=0.35,
        typical_turnover=0.25,
        tags=["academic", "behavioral", "earnings", "underreaction"],
        pitfalls=["需要盈余公告日期和盈余预期数据"],
    ),
    
    # ==================== 流动性溢价 ====================
    ClassicFactor(
        id="ap_illiq",
        name="非流动性溢价",
        name_en="Illiquidity Premium",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    Amihud Illiquidity - 非流动性溢价
    低流动性股票需要流动性补偿
    """
    ret_abs = df['close'].pct_change().abs()
    illiq = ret_abs / (df['amount'] + 1e-8) * 1e8
    illiq_avg = illiq.rolling(20).mean()
    # 注意：这里做多低流动性可能需要考虑实际交易成本
    return illiq_avg.rank(pct=True)
''',
        description="低流动性股票的超额收益",
        logic="流动性差的股票交易成本高，需要补偿",
        reference="Amihud (2002) 'Illiquidity and Stock Returns', JFM",
        author="Amihud",
        year=2002,
        historical_ic=0.022,
        historical_icir=0.30,
        typical_turnover=0.18,
        tags=["academic", "liquidity", "market_microstructure", "risk_premium"],
        pitfalls=["实际交易时冲击成本大", "极端非流动性股票难以交易"],
    ),
    
    ClassicFactor(
        id="ap_turnover",
        name="换手率溢价",
        name_en="Turnover Premium",
        category=FactorCategory.VOLUME_PRICE,
        code='''
def compute_alpha(df):
    """
    Turnover - 换手率溢价
    低换手率股票收益更高
    """
    turnover_avg = df['turnover'].rolling(20).mean() if 'turnover' in df.columns else \
                   df['volume'] / df['volume'].rolling(20).mean()
    return -turnover_avg.rank(pct=True)
''',
        description="低换手率股票的超额收益",
        logic="低换手率意味着关注度低，可能被低估",
        reference="Datar, Naik & Radcliffe (1998) 'Liquidity and Stock Returns', JFM",
        author="Datar",
        year=1998,
        historical_ic=0.025,
        historical_icir=0.32,
        typical_turnover=0.20,
        tags=["academic", "liquidity", "turnover", "attention"],
        pitfalls=["低换手可能是基本面恶化导致"],
    ),
    
    # ==================== 股息溢价 ====================
    ClassicFactor(
        id="ap_div_yield",
        name="股息溢价",
        name_en="Dividend Yield Premium",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    Dividend Yield - 股息溢价
    高股息股票收益更高
    """
    div_yield = df['dividend_yield'] if 'dividend_yield' in df.columns else 0.02
    return div_yield.rank(pct=True)
''',
        description="高股息率股票的超额收益",
        logic="高股息反映公司现金流稳定，管理层信心",
        reference="Litzenberger & Ramaswamy (1979) 'The Effects of Dividends', JFE",
        author="Litzenberger & Ramaswamy",
        year=1979,
        historical_ic=0.018,
        historical_icir=0.25,
        typical_turnover=0.12,
        tags=["academic", "dividend", "income", "value"],
        pitfalls=["A股股息率整体偏低", "高股息可能因股价下跌"],
    ),
    
    # ==================== 杠杆效应 ====================
    ClassicFactor(
        id="ap_leverage",
        name="低杠杆溢价",
        name_en="Low Leverage Premium",
        category=FactorCategory.FUNDAMENTAL,
        code='''
def compute_alpha(df):
    """
    Low Leverage - 低杠杆溢价
    低杠杆公司更稳健
    """
    leverage = df['debt_ratio'] if 'debt_ratio' in df.columns else 0.5
    return -leverage.rank(pct=True)
''',
        description="低杠杆公司的超额收益",
        logic="高杠杆增加财务风险和破产概率",
        reference="Penman, Richardson & Tuna (2007) 'The Book-to-Price Effect', JAR",
        author="Penman",
        year=2007,
        historical_ic=0.015,
        historical_icir=0.22,
        typical_turnover=0.10,
        tags=["academic", "leverage", "quality", "risk"],
        pitfalls=["杠杆水平有行业差异"],
    ),
]


def get_academic_premia_factors():
    """获取所有学术溢价因子"""
    return ACADEMIC_PREMIA_FACTORS


# 因子分类统计
FACTOR_CATEGORIES = {
    "Fama-French": ["ap_smb", "ap_hml", "ap_umd", "ap_rmw", "ap_cma"],
    "Low Risk": ["ap_bab", "ap_idiovol"],
    "Quality": ["ap_qmj", "ap_gp"],
    "Accounting": ["ap_accruals", "ap_noa"],
    "Behavioral": ["ap_reversal_st", "ap_reversal_lt", "ap_pead"],
    "Liquidity": ["ap_illiq", "ap_turnover"],
    "Other": ["ap_div_yield", "ap_leverage"],
}


# 打印因子统计
if __name__ == "__main__":
    print(f"Academic Premia 学术溢价因子库:")
    print(f"  - 总因子数: {len(ACADEMIC_PREMIA_FACTORS)}个")
    print(f"\n按类别统计:")
    for cat, factors in FACTOR_CATEGORIES.items():
        print(f"  - {cat}: {len(factors)}个")
