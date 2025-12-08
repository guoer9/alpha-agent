"""
A股数据字典 - 沪深A股日频数据完整定义

基于Qlib数据格式和常见量化数据源
"""

from .data_schema import (
    DataSchema, FieldSchema, DataValidator,
    DataFrequency, DataType,
)


# ============================================================
# A股日频数据字典
# ============================================================

CN_STOCK_DAILY_SCHEMA = DataSchema(
    name="A股日频数据",
    version="2.0",
    description="""
沪深A股日频行情与基本面数据，适用于因子挖掘和量化策略研究。

数据来源: Qlib / Wind / 聚宽 / Tushare
更新频率: 每日收盘后
历史范围: 2010年至今
""",
    primary_key=['date', 'instrument'],
    time_column='date',
    entity_column='instrument',
    start_date="2010-01-01",
    end_date="2024-12-01",
    universe_size=5000,
    fields={
        # ==================== 价格类 ====================
        'open': FieldSchema(
            name='open',
            dtype='float64',
            description='开盘价（前复权），集合竞价确定',
            data_type=DataType.PRICE,
            missing_rate=0.001,
            update_frequency=DataFrequency.DAILY,
            min_value=0.01,
            max_value=10000,
            typical_range=(5, 200),
            is_adjusted=True,
            usage_examples=[
                "gap = (open - close.shift(1)) / close.shift(1)  # 跳空缺口",
                "open_ret = open / open.shift(1) - 1  # 开盘收益",
            ],
            common_pitfalls=[
                "涨跌停时开盘价可能不准确（一字板）",
                "集合竞价波动大，慎用于高频策略",
                "必须使用复权价格计算收益率",
            ],
            related_fields=['high', 'low', 'close', 'vwap'],
        ),
        
        'high': FieldSchema(
            name='high',
            dtype='float64',
            description='最高价（前复权），日内最高成交价',
            data_type=DataType.PRICE,
            missing_rate=0.001,
            min_value=0.01,
            max_value=10000,
            typical_range=(5, 200),
            is_adjusted=True,
            usage_examples=[
                "upper_shadow = (high - np.maximum(open, close)) / close  # 上影线比例",
                "range_pct = (high - low) / close  # 日内振幅",
                "n_day_high = high.rolling(20).max()  # 20日最高",
                "breakout = close / high.rolling(20).max()  # 突破信号",
            ],
            common_pitfalls=[
                "涨停时 high = close，无法计算上影线",
                "除权除息日价格跳变，需用复权价",
            ],
            related_fields=['open', 'low', 'close'],
        ),
        
        'low': FieldSchema(
            name='low',
            dtype='float64',
            description='最低价（前复权），日内最低成交价',
            data_type=DataType.PRICE,
            missing_rate=0.001,
            min_value=0.01,
            max_value=10000,
            typical_range=(5, 200),
            is_adjusted=True,
            usage_examples=[
                "lower_shadow = (np.minimum(open, close) - low) / close  # 下影线",
                "support = low.rolling(20).min()  # 支撑位",
                "drawdown = close / close.cummax() - 1  # 回撤",
            ],
            common_pitfalls=[
                "跌停时 low = close，无法计算下影线",
            ],
            related_fields=['open', 'high', 'close'],
        ),
        
        'close': FieldSchema(
            name='close',
            dtype='float64',
            description='收盘价（前复权）- 最核心的价格字段',
            data_type=DataType.PRICE,
            missing_rate=0.0,
            min_value=0.01,
            max_value=10000,
            typical_range=(5, 200),
            is_adjusted=True,
            usage_examples=[
                "ret = close.pct_change()  # 日收益率",
                "ma20 = close.rolling(20).mean()  # 20日均线",
                "momentum = close / close.shift(5) - 1  # 5日动量",
                "volatility = close.pct_change().rolling(20).std()  # 20日波动率",
                "ret_rank = close.pct_change().rank(pct=True)  # 收益率截面排名",
            ],
            common_pitfalls=[
                "⚠️ 必须使用复权价格！否则除权日会出现假跳空",
                "ST股票涨跌停限制为5%，需单独处理",
                "新股上市首日/前5日涨跌停限制不同",
                "科创板/创业板(注册制)涨跌停±20%",
            ],
            related_fields=['open', 'high', 'low', 'volume', 'amount'],
        ),
        
        'vwap': FieldSchema(
            name='vwap',
            dtype='float64',
            description='成交量加权平均价 VWAP = amount / volume',
            data_type=DataType.PRICE,
            missing_rate=0.01,
            typical_range=(5, 200),
            usage_examples=[
                "vwap_bias = (close - vwap) / vwap  # VWAP偏离度",
                "# 收盘价>VWAP表示当日买方占优",
            ],
            common_pitfalls=[
                "停牌日VWAP为空",
                "成交量极小时VWAP波动大",
            ],
            related_fields=['close', 'volume', 'amount'],
        ),
        
        # ==================== 成交量类 ====================
        'volume': FieldSchema(
            name='volume',
            dtype='float64',
            description='成交量（股），当日总成交股数',
            data_type=DataType.VOLUME,
            missing_rate=0.001,
            min_value=0,
            max_value=1e12,
            typical_range=(1e6, 1e9),
            usage_examples=[
                "vol_ma = volume.rolling(20).mean()  # 成交量均线",
                "vol_ratio = volume / volume.rolling(20).mean()  # 量比",
                "vol_std = volume.rolling(20).std()  # 成交量波动",
            ],
            common_pitfalls=[
                "停牌日成交量为0，rolling计算需注意",
                "不同股票市值差异大，成交量绝对值不可比",
                "建议使用换手率或量比进行标准化",
                "科创板引入盘后固定价格交易，成交量构成有变化",
            ],
            related_fields=['amount', 'turnover', 'close'],
        ),
        
        'amount': FieldSchema(
            name='amount',
            dtype='float64',
            description='成交额（元），当日总成交金额',
            data_type=DataType.VOLUME,
            missing_rate=0.001,
            min_value=0,
            max_value=1e14,
            typical_range=(1e7, 1e10),
            usage_examples=[
                "avg_price = amount / volume  # 均价（约等于VWAP）",
                "amount_ratio = amount / amount.rolling(20).mean()  # 金额比",
                "illiq = close.pct_change().abs() / amount * 1e8  # Amihud非流动性",
            ],
            common_pitfalls=[
                "成交额受价格影响，高价股成交额天然更大",
            ],
            related_fields=['volume', 'vwap'],
        ),
        
        'turnover': FieldSchema(
            name='turnover',
            dtype='float64',
            description='换手率 = 成交量 / 流通股本，最常用的标准化成交量指标',
            data_type=DataType.VOLUME,
            missing_rate=0.01,
            min_value=0,
            max_value=1.0,
            typical_range=(0.005, 0.10),
            usage_examples=[
                "turnover_ma = turnover.rolling(20).mean()  # 平均换手率",
                "turnover_std = turnover.rolling(20).std()  # 换手率波动",
                "high_turnover = turnover > turnover.rolling(60).quantile(0.9)  # 异常高换手",
                "illiquidity = 1 / (turnover + 1e-8)  # 非流动性",
            ],
            common_pitfalls=[
                "新股上市初期换手率极高（可达50%+），需剔除",
                "限售股解禁会改变流通股本，导致换手率计算基数变化",
                "部分数据源的换手率单位不同（有的是百分比形式）",
            ],
            related_fields=['volume', 'float_share'],
        ),
        
        # ==================== 收益类（衍生）====================
        'returns': FieldSchema(
            name='returns',
            dtype='float64',
            description='日收益率 = close / close.shift(1) - 1',
            data_type=DataType.PRICE,
            missing_rate=0.01,
            min_value=-0.20,
            max_value=0.20,
            typical_range=(-0.05, 0.05),
            lookback_required=1,
            usage_examples=[
                "cum_ret = (1 + returns).cumprod() - 1  # 累计收益",
                "ret_5d = close / close.shift(5) - 1  # 5日收益率",
                "excess_ret = returns - market_ret  # 超额收益",
            ],
            common_pitfalls=[
                "涨跌停限制: 主板±10%，科创板/创业板(注册制)±20%，ST±5%",
                "新股上市首日不设涨跌停（但有临停机制）",
                "复牌股票可能有大幅跳空",
            ],
            related_fields=['close'],
        ),
        
        # ==================== 基本面类 ====================
        'pe_ttm': FieldSchema(
            name='pe_ttm',
            dtype='float64',
            description='市盈率TTM = 市值 / 过去12个月净利润',
            data_type=DataType.FUNDAMENTAL,
            missing_rate=0.05,
            min_value=-1000,
            max_value=10000,
            typical_range=(10, 50),
            lag_days=1,
            usage_examples=[
                "ep = 1 / pe_ttm.clip(lower=1)  # 盈利收益率（PE倒数）",
                "pe_rank = pe_ttm.rank(pct=True)  # PE截面排名",
                "pe_zscore = (pe_ttm - pe_ttm.mean()) / pe_ttm.std()  # 标准化",
            ],
            common_pitfalls=[
                "亏损公司PE为负或无穷大，需特殊处理",
                "周期股PE在底部时最高（盈利最差时）",
                "成长股PE天然较高，不能简单比较",
                "建议使用PE倒数(EP)或分位数处理",
            ],
            related_fields=['pb', 'ps', 'market_cap', 'net_profit'],
        ),
        
        'pb': FieldSchema(
            name='pb',
            dtype='float64',
            description='市净率 = 市值 / 净资产',
            data_type=DataType.FUNDAMENTAL,
            missing_rate=0.03,
            min_value=0,
            max_value=100,
            typical_range=(1, 10),
            lag_days=1,
            usage_examples=[
                "bp = 1 / pb.clip(lower=0.1)  # 账面市值比（价值因子）",
            ],
            common_pitfalls=[
                "净资产为负时PB无意义（资不抵债）",
                "轻资产公司（如互联网）PB天然很高",
                "银行股PB常年低于1，是行业特性",
            ],
            related_fields=['pe_ttm', 'market_cap', 'roe_ttm'],
        ),
        
        'ps_ttm': FieldSchema(
            name='ps_ttm',
            dtype='float64',
            description='市销率TTM = 市值 / 过去12个月营收',
            data_type=DataType.FUNDAMENTAL,
            missing_rate=0.05,
            min_value=0,
            max_value=1000,
            typical_range=(1, 20),
            lag_days=1,
            usage_examples=[
                "sp = 1 / ps_ttm.clip(lower=0.1)  # 营收市值比",
            ],
            common_pitfalls=[
                "亏损但有收入的公司可以用PS估值",
                "不同行业利润率差异大，PS可比性有限",
            ],
            related_fields=['pe_ttm', 'market_cap'],
        ),
        
        'market_cap': FieldSchema(
            name='market_cap',
            dtype='float64',
            description='总市值（元）= 股价 × 总股本',
            data_type=DataType.FUNDAMENTAL,
            missing_rate=0.0,
            min_value=1e8,
            max_value=5e13,
            typical_range=(1e9, 1e11),
            usage_examples=[
                "size = np.log(market_cap)  # 对数市值（规模因子）",
                "is_small = market_cap < market_cap.quantile(0.3)  # 小市值",
                "cap_weight = market_cap / market_cap.sum()  # 市值权重",
            ],
            common_pitfalls=[
                "市值因子（小市值效应）在A股长期有效但波动大",
                "建议使用对数市值减少偏度",
                "2017年后小市值效应有所减弱",
            ],
            related_fields=['float_cap', 'pe_ttm', 'pb'],
        ),
        
        'float_cap': FieldSchema(
            name='float_cap',
            dtype='float64',
            description='流通市值（元）= 股价 × 流通股本',
            data_type=DataType.FUNDAMENTAL,
            missing_rate=0.01,
            min_value=1e7,
            max_value=5e13,
            typical_range=(5e8, 5e10),
            usage_examples=[
                "float_ratio = float_cap / market_cap  # 流通比例",
            ],
            common_pitfalls=[
                "限售股解禁会增加流通市值",
            ],
            related_fields=['market_cap', 'turnover'],
        ),
        
        'roe_ttm': FieldSchema(
            name='roe_ttm',
            dtype='float64',
            description='净资产收益率TTM = 净利润 / 净资产',
            data_type=DataType.FUNDAMENTAL,
            missing_rate=0.05,
            min_value=-1.0,
            max_value=1.0,
            typical_range=(0.05, 0.20),
            lag_days=1,
            usage_examples=[
                "quality = roe_ttm  # 质量因子",
                "roe_stable = roe_ttm.rolling(4).std()  # ROE稳定性（季度）",
            ],
            common_pitfalls=[
                "高ROE可能源于高杠杆而非真正的高质量",
                "需要结合资产负债率一起分析（杜邦分析）",
            ],
            related_fields=['pe_ttm', 'pb', 'debt_ratio'],
        ),
        
        'debt_ratio': FieldSchema(
            name='debt_ratio',
            dtype='float64',
            description='资产负债率 = 总负债 / 总资产',
            data_type=DataType.FUNDAMENTAL,
            missing_rate=0.05,
            min_value=0,
            max_value=1.0,
            typical_range=(0.3, 0.7),
            lag_days=1,
            usage_examples=[
                "low_leverage = debt_ratio < 0.5  # 低杠杆公司",
            ],
            common_pitfalls=[
                "杠杆水平有显著行业差异（银行vs科技）",
                "适度杠杆可能提升ROE",
            ],
            related_fields=['roe_ttm'],
        ),
        
        'dividend_yield': FieldSchema(
            name='dividend_yield',
            dtype='float64',
            description='股息率 = 每股股息 / 股价',
            data_type=DataType.FUNDAMENTAL,
            missing_rate=0.10,
            min_value=0,
            max_value=0.20,
            typical_range=(0, 0.05),
            lag_days=1,
            usage_examples=[
                "high_div = dividend_yield > 0.03  # 高股息股票",
            ],
            common_pitfalls=[
                "A股整体股息率偏低",
                "高股息可能是因为股价下跌而非分红增加",
            ],
            related_fields=['pe_ttm'],
        ),
        
        # ==================== 行业分类 ====================
        'industry': FieldSchema(
            name='industry',
            dtype='category',
            description='申万一级行业分类（共31个行业）',
            data_type=DataType.FUNDAMENTAL,
            missing_rate=0.0,
            usage_examples=[
                "# 行业中性化",
                "factor_neutral = factor - factor.groupby('industry').transform('mean')",
                "# 行业哑变量",
                "industry_dummies = pd.get_dummies(industry, prefix='ind')",
            ],
            common_pitfalls=[
                "行业分类会调整，需使用Point-in-Time数据",
                "部分公司跨多个行业，分类可能不够精确",
                "申万行业分类2021年有较大调整",
            ],
            related_fields=[],
        ),
    }
)


# ============================================================
# 便捷函数
# ============================================================

def get_cn_stock_schema() -> DataSchema:
    """获取A股数据字典"""
    return CN_STOCK_DAILY_SCHEMA


def get_cn_stock_validator() -> DataValidator:
    """获取A股数据验证器"""
    return DataValidator(CN_STOCK_DAILY_SCHEMA)


def generate_llm_data_context(df=None) -> str:
    """
    生成给LLM的数据上下文
    
    Args:
        df: 可选，传入实际数据会添加统计信息
    """
    schema = CN_STOCK_DAILY_SCHEMA
    
    if df is not None:
        validator = DataValidator(schema)
        return validator.generate_llm_context(df)
    
    return schema.to_llm_prompt()


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    schema = get_cn_stock_schema()
    print(f"A股数据字典: {schema.name} v{schema.version}")
    print(f"字段数量: {len(schema.fields)}")
    print(f"时间范围: {schema.start_date} ~ {schema.end_date}")
    print("\n字段列表:")
    for name, field in schema.fields.items():
        print(f"  - {name}: {field.description[:30]}...")
    
    print("\n" + "="*50)
    print("LLM Prompt 预览:")
    print("="*50)
    print(schema.to_llm_prompt()[:2000] + "...")
