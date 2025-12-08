# Alpha Agent 因子库文档

> 整合多个权威因子集合，总计 300+ 个因子

## 📊 因子库概览

| 因子集 | 数量 | 来源 | 类型 |
|--------|------|------|------|
| **Barra CNE5/CNE6** | 9 | MSCI Barra | 风格因子 |
| **技术分析因子** | 5 | 经典技术指标 | 技术面 |
| **基本面因子** | 5 | 学术文献 | 财务指标 |
| **量价因子** | 6 | 市场微观结构 | 量价关系 |
| **Alpha158** | 50+ | Microsoft Qlib | 技术+量价 |
| **Alpha360** | 扩展 | Microsoft Qlib | 全面特征 |
| **WorldQuant 101** | 29 | Kakushadze (2016) | 公式化因子 |
| **国泰君安 191** | 30 | 国泰君安证券 | 短周期量价 |
| **Academic Premia** | 18 | 顶级金融期刊 | 学术溢价 |

---

## 1. Barra CNE5/CNE6 风格因子

> 来源: MSCI Barra 多因子模型
> 参考: Barra CNE5/CNE6 Risk Model

### 1.1 Size (市值因子)
```python
def compute_alpha(df):
    """小市值股票长期跑赢大市值股票"""
    size = np.log(df['market_cap'])
    return -size.rank(pct=True)  # 负号表示小市值为正暴露
```
| 指标 | 值 | 说明 |
|------|-----|------|
| IC | 0.035 | 历史信息系数 |
| ICIR | 0.45 | 信息比率 |
| 换手率 | 15% | 典型月换手 |
| 参考文献 | Banz (1981) | "The relationship between return and market value" |

**投资逻辑**: 小市值公司信息不对称程度高，流动性差，需要风险补偿

**注意事项**:
- A股小市值效应波动大，2017年后有所减弱
- 需要剔除壳资源价值的影响
- 微盘股流动性差，实际交易摩擦大

---

### 1.2 Beta (贝塔因子)
```python
def compute_alpha(df):
    """低贝塔股票风险调整后收益更高"""
    volatility = df['close'].pct_change().rolling(60).std()
    return -volatility.rank(pct=True)
```
| 指标 | 值 |
|------|-----|
| IC | 0.025 |
| ICIR | 0.35 |
| 参考文献 | Frazzini & Pedersen (2014) "Betting Against Beta" |

**投资逻辑**: 低贝塔异象 - 高风险股票并未获得应有的风险补偿

---

### 1.3 Momentum (动量因子)
```python
def compute_alpha(df):
    """过去12个月收益（剔除最近1个月）"""
    ret_12m = df['close'] / df['close'].shift(252) - 1
    ret_1m = df['close'] / df['close'].shift(21) - 1
    momentum = ret_12m - ret_1m
    return momentum.rank(pct=True)
```
| 指标 | 值 |
|------|-----|
| IC | 0.028 |
| ICIR | 0.38 |
| 换手率 | 35% |
| 参考文献 | Jegadeesh & Titman (1993) "Returns to Buying Winners" |

**投资逻辑**: 投资者对信息反应不足导致价格趋势延续

**注意事项**:
- 动量反转风险大，极端市场会崩溃
- A股动量周期较短，12个月可能太长
- 2015年股灾等时期严重回撤

---

### 1.4 Residual Volatility (残差波动率)
```python
def compute_alpha(df):
    """低特质波动率股票收益更高"""
    volatility = df['close'].pct_change().rolling(60).std() * np.sqrt(252)
    return -volatility.rank(pct=True)
```
| 指标 | 值 |
|------|-----|
| IC | 0.032 |
| ICIR | 0.42 |
| 参考文献 | Ang et al. (2006) "The Cross-Section of Volatility" |

---

### 1.5 Value (价值因子)
```python
def compute_alpha(df):
    """账面价值/市值，低估值为高暴露"""
    bp = 1 / df['pb'].clip(lower=0.1)
    return bp.rank(pct=True)
```
| 指标 | 值 |
|------|-----|
| IC | 0.022 |
| ICIR | 0.30 |
| 参考文献 | Fama & French (1992) "Cross-Section of Expected Returns" |

**注意事项**: 价值陷阱 - 低估值可能是基本面恶化

---

### 1.6 Earnings Yield (盈利收益率)
```python
def compute_alpha(df):
    """盈利/市值，高盈利收益率为高暴露"""
    ep = 1 / df['pe_ttm'].clip(lower=1)
    ep = ep.where(df['pe_ttm'] > 0, 0)
    return ep.rank(pct=True)
```

---

### 1.7 Liquidity (流动性因子)
```python
def compute_alpha(df):
    """基于换手率的流动性度量"""
    turnover_avg = df['turnover'].rolling(20).mean()
    return -turnover_avg.rank(pct=True)
```
| 参考文献 | Pastor & Stambaugh (2003) "Liquidity Risk" |

---

### 1.8 Growth (成长因子)
```python
def compute_alpha(df):
    """高成长公司股价表现更好"""
    growth = df['revenue_yoy'] if 'revenue_yoy' in df.columns else df['roe_ttm']
    return growth.rank(pct=True)
```

---

### 1.9 Leverage (杠杆因子)
```python
def compute_alpha(df):
    """低杠杆公司更稳健"""
    leverage = df['debt_ratio']
    return -leverage.rank(pct=True)
```

---

## 2. 技术分析因子

### 2.1 Short-term Reversal (短期反转)
```python
def compute_alpha(df):
    """过去一周跌幅大的股票反弹"""
    ret_5d = df['close'].pct_change(5)
    return -ret_5d.rank(pct=True)
```
| 指标 | 值 |
|------|-----|
| IC | 0.038 |
| ICIR | 0.48 |
| 换手率 | 65% |
| 参考文献 | Jegadeesh (1990) |

**注意**: 换手率极高，需要剔除ST和停牌股

---

### 2.2 MA Deviation (均线偏离度)
```python
def compute_alpha(df):
    """偏离均线过多会回归"""
    ma20 = df['close'].rolling(20).mean()
    deviation = (df['close'] - ma20) / ma20
    return -deviation.rank(pct=True)
```
| IC | 0.030 | ICIR | 0.40 |

---

### 2.3 RSI (相对强弱指标)
```python
def compute_alpha(df):
    """超买超卖信号"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    return -(rsi - 50).rank(pct=True)
```
| 参考文献 | Wilder (1978) |

---

### 2.4 Volatility Breakout (波动率突破)
```python
def compute_alpha(df):
    """突破近期高点的动量信号"""
    high_20d = df['high'].rolling(20).max()
    breakout = df['close'] / high_20d - 1
    return breakout.rank(pct=True)
```
| 参考文献 | Turtle Trading - Dennis & Eckhardt (1983) |

---

### 2.5 MACD
```python
def compute_alpha(df):
    """趋势跟踪指标"""
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    macd = (dif - dea) * 2
    return macd.rank(pct=True)
```

---

## 3. 基本面因子

### 3.1 ROE (净资产收益率)
```python
def compute_alpha(df):
    """高质量公司的核心指标"""
    return df['roe_ttm'].rank(pct=True)
```
| IC | 0.028 | ICIR | 0.38 |
| 参考文献 | Buffett (1987) "Owner Earnings" |

**投资逻辑**: 高ROE公司具有竞争优势，可持续创造价值

---

### 3.2 Earnings Stability (盈利稳定性)
```python
def compute_alpha(df):
    """盈利波动小的公司更稳健"""
    roe_std = df['roe_ttm'].rolling(4).std()
    return -roe_std.rank(pct=True)
```
| 参考文献 | Novy-Marx (2013) "Quality Minus Junk" |

---

### 3.3 Accruals (应计因子)
```python
def compute_alpha(df):
    """低应计项目的公司盈利质量更高"""
    accrual_ratio = 1 - df['ocf'] / (df['net_profit'] + 1e-8)
    return -accrual_ratio.rank(pct=True)
```
| 参考文献 | Sloan (1996) "Accrual Anomaly" |

---

### 3.4 Asset Growth (资产增长)
```python
def compute_alpha(df):
    """资产扩张慢的公司收益更高"""
    return -df['total_assets_yoy'].rank(pct=True)
```
| 参考文献 | Cooper et al. (2008) "Asset Growth" |

---

### 3.5 Dividend Yield (股息率)
```python
def compute_alpha(df):
    """高股息股票提供稳定现金回报"""
    return df['dividend_yield'].rank(pct=True)
```

---

## 4. 量价因子

### 4.1 Turnover (换手率因子)
```python
def compute_alpha(df):
    """低换手率股票可能被低估"""
    turnover_avg = df['turnover'].rolling(20).mean()
    return -turnover_avg.rank(pct=True)
```
| 参考文献 | Datar et al. (1998) |

---

### 4.2 Abnormal Turnover (异常换手)
```python
def compute_alpha(df):
    """换手率突增可能是反转信号"""
    turnover_avg = df['turnover'].rolling(20).mean()
    turnover_std = df['turnover'].rolling(20).std()
    abnormal = (df['turnover'] - turnover_avg) / (turnover_std + 1e-8)
    return -abnormal.rank(pct=True)
```
| 参考文献 | Lee & Swaminathan (2000) |

---

### 4.3 Volume Price Divergence (量价背离)
```python
def compute_alpha(df):
    """价涨量缩可能是上涨乏力"""
    price_ret = df['close'].pct_change(5)
    volume_ret = df['volume'].pct_change(5)
    divergence = -price_ret * np.sign(volume_ret)
    return divergence.rank(pct=True)
```

---

### 4.4 Money Flow (资金流向)
```python
def compute_alpha(df):
    """基于价格和成交量的资金流向指标"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_flow = raw_money_flow.where(df['close'] > df['close'].shift(1), 0)
    negative_flow = raw_money_flow.where(df['close'] < df['close'].shift(1), 0)
    mfi = positive_flow.rolling(14).sum() / (positive_flow.rolling(14).sum() + negative_flow.rolling(14).sum() + 1e-8)
    return mfi.rank(pct=True)
```

---

### 4.5 Amplitude (振幅因子)
```python
def compute_alpha(df):
    """低振幅股票更稳定"""
    amplitude = (df['high'] - df['low']) / df['close']
    amplitude_avg = amplitude.rolling(20).mean()
    return -amplitude_avg.rank(pct=True)
```

---

### 4.6 Amihud Illiquidity (非流动性)
```python
def compute_alpha(df):
    """价格冲击成本"""
    ret_abs = df['close'].pct_change().abs()
    illiq = ret_abs / (df['amount'] + 1e-8) * 1e8
    illiq_avg = illiq.rolling(20).mean()
    return -illiq_avg.rank(pct=True)
```
| 参考文献 | Amihud (2002) |

---

## 5. Qlib Alpha158

> 来源: Microsoft Qlib 框架
> 链接: https://github.com/microsoft/qlib

### 因子分类

| 类别 | 数量 | 说明 |
|------|------|------|
| KBAR | 10+ | K线形态特征 |
| 收益率 | 20+ | 不同周期收益 |
| 波动率 | 15+ | 各类波动指标 |
| 量比 | 20+ | 成交量特征 |
| 均线 | 30+ | 移动平均特征 |
| 相关性 | 20+ | 量价相关性 |
| 其他 | 40+ | 综合技术特征 |

### 示例因子

```python
# KBAR - K线实体
(close - open) / open

# 收益率
close / Ref(close, 5) - 1

# 波动率
Std(close, 20) / Mean(close, 20)

# 量比
volume / Ref(volume, 5)

# 均线偏离
close / SMA(close, 20) - 1
```

---

## 6. WorldQuant 101 Alphas

> 来源: Kakushadze (2016) "101 Formulaic Alphas"
> 论文: https://arxiv.org/abs/1601.00991

### 代表性因子

#### Alpha#001 - 排名反转
```python
def compute_alpha(df):
    """短期反转信号"""
    returns = df['close'].pct_change()
    signed_power = returns.abs() ** 2 * np.sign(returns)
    argmax = signed_power.rolling(5).apply(lambda x: x.argmax())
    return argmax.rank(pct=True) - 0.5
```

#### Alpha#002 - 量价背离
```python
def compute_alpha(df):
    """成交量变化与价格变化的负相关"""
    delta_log_vol = np.log(df['volume'] + 1).diff(2)
    price_change = (df['close'] - df['open']) / df['open']
    corr = delta_log_vol.rolling(6).corr(price_change)
    return -corr
```

#### Alpha#004 - 低位排名
```python
def compute_alpha(df):
    """最低价时序排名的负值"""
    low_rank = df['low'].rank(pct=True)
    ts_rank = low_rank.rolling(9).apply(lambda x: x.rank().iloc[-1] / len(x))
    return -ts_rank
```

#### Alpha#005 - VWAP动量
```python
def compute_alpha(df):
    """VWAP相关动量"""
    vwap = df['amount'] / (df['volume'] + 1e-8)
    vwap_ma = vwap.rolling(10).mean()
    term1 = (df['open'] - vwap_ma).rank(pct=True)
    term2 = (df['close'] - vwap).abs().rank(pct=True)
    return term1 * (-term2)
```

---

## 7. 使用指南

### 7.1 快速开始

```python
from alpha_agent.factors import (
    FactorLibrary,
    create_factor_library,
    ALL_FACTORS,
)

# 创建完整因子库
library = create_factor_library(
    include_classic=True,
    include_alpha158=True,
    include_alpha360=True,
    include_worldquant=True,
)

print(library.summary())
```

### 7.2 按类别检索

```python
from alpha_agent.factors import get_factors_by_category, FactorCategory

# 获取所有Barra因子
barra_factors = get_factors_by_category(FactorCategory.BARRA_STYLE)

# 获取所有量价因子
volume_price = get_factors_by_category(FactorCategory.VOLUME_PRICE)
```

### 7.3 按标签检索

```python
from alpha_agent.factors import FactorLibrary

library = FactorLibrary()
library.initialize_classic_factors()

# 搜索动量类因子
momentum_factors = library.search_factors(tags=["momentum"])

# 搜索低换手因子
low_turnover = library.search_factors(max_turnover=0.30)

# 搜索高IC因子
high_ic = library.search_factors(min_ic=0.03)
```

### 7.4 导出因子数据

```python
# 导出到GraphRAG
nodes = library.get_factors_for_graphrag()

# 导出到RAPTOR
documents = library.get_factors_for_raptor()

# 保存因子库
library.save("factor_library.json")
```

---

## 8. 因子评价标准

| 指标 | 优秀 | 良好 | 一般 |
|------|------|------|------|
| **IC** | > 0.05 | 0.03-0.05 | 0.02-0.03 |
| **ICIR** | > 0.5 | 0.3-0.5 | 0.2-0.3 |
| **年化收益** | > 15% | 10-15% | 5-10% |
| **信息比率** | > 1.5 | 1.0-1.5 | 0.5-1.0 |
| **最大回撤** | < 10% | 10-15% | 15-20% |
| **换手率** | < 30% | 30-50% | > 50% |

---

## 9. 参考文献

1. **Banz (1981)** - "The relationship between return and market value of common stocks"
2. **Fama & French (1992)** - "The Cross-Section of Expected Stock Returns"
3. **Jegadeesh & Titman (1993)** - "Returns to Buying Winners and Selling Losers"
4. **Sloan (1996)** - "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows?"
5. **Amihud (2002)** - "Illiquidity and Stock Returns"
6. **Ang et al. (2006)** - "The Cross-Section of Volatility and Expected Returns"
7. **Frazzini & Pedersen (2014)** - "Betting Against Beta"
8. **Novy-Marx (2013)** - "The Other Side of Value: The Gross Profitability Premium"
9. **Kakushadze (2016)** - "101 Formulaic Alphas"
10. **Microsoft Qlib** - https://github.com/microsoft/qlib

---

## 10. 国泰君安 191 因子

> 来源: 国泰君安证券《基于短周期价量特征的多因子选股体系》
> 特点: 短周期、高换手、适合A股市场

### 因子分类

| 类别 | 数量 | 说明 |
|------|------|------|
| 量价相关 | 10 | 成交量与价格的关系 |
| 动量反转 | 8 | 短期动量与反转 |
| 技术形态 | 6 | K线形态特征 |
| 资金流向 | 3 | 资金进出信号 |
| 趋势类 | 3 | 趋势强度判断 |

### 代表性因子

#### GTJA#001 - 量价排名差
```python
def compute_alpha(df):
    """成交量变化排名与价格变化排名的负相关"""
    delta_log_vol = np.log(df['volume'] + 1).diff(1)
    price_change = (df['close'] - df['open']) / df['open']
    corr = delta_log_vol.rank(pct=True).rolling(6).corr(price_change.rank(pct=True))
    return -corr
```

#### GTJA#011 - 量价差异
```python
def compute_alpha(df):
    """6日量能加权K线位置"""
    hl_range = df['high'] - df['low'] + 1e-8
    position = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_range
    return (position * df['volume']).rolling(6).sum()
```

#### GTJA#015 - 隔夜跳空
```python
def compute_alpha(df):
    """隔夜跳空幅度"""
    return df['open'] / df['close'].shift(1) - 1
```

---

## 11. Academic Premia 学术溢价因子

> 来源: 顶级金融期刊 (JF, JFE, RFS, JAR)
> 特点: 学术界公认的风险溢价因子

### 因子分类

| 类别 | 因子 | 参考文献 |
|------|------|----------|
| **Fama-French** | SMB, HML, UMD, RMW, CMA | FF (1993, 2015) |
| **低风险** | BAB, IVOL | Frazzini (2014), Ang (2006) |
| **质量** | QMJ, GP | Asness (2019), Novy-Marx (2013) |
| **会计** | Accruals, NOA | Sloan (1996), Hirshleifer (2004) |
| **行为** | Reversal, PEAD | Jegadeesh (1990), Bernard (1989) |
| **流动性** | ILLIQ, Turnover | Amihud (2002), Datar (1998) |

### 代表性因子

#### SMB - 规模溢价
```python
def compute_alpha(df):
    """做多小市值，做空大市值"""
    log_cap = np.log(df['market_cap'] + 1)
    return -log_cap.rank(pct=True)
```
| 参考文献 | Fama & French (1993) JFE |
| 历史IC | 0.035 |

#### BAB - 低贝塔溢价
```python
def compute_alpha(df):
    """做多低波动，做空高波动"""
    volatility = df['close'].pct_change().rolling(60).std() * np.sqrt(252)
    return -volatility.rank(pct=True)
```
| 参考文献 | Frazzini & Pedersen (2014) JFE |
| 历史IC | 0.028 |

#### QMJ - 质量溢价
```python
def compute_alpha(df):
    """做多高质量，做空低质量"""
    quality = df['roe_ttm']
    return quality.rank(pct=True)
```
| 参考文献 | Asness, Frazzini & Pedersen (2019) RFS |
| 历史IC | 0.028 |

---

## 12. 文件结构

```
alpha_agent/factors/
├── __init__.py              # 因子库入口
├── classic_factors.py       # 经典因子 (Barra/技术/基本面/量价)
├── alpha158.py              # Qlib Alpha158
├── alpha360.py              # Qlib Alpha360
├── worldquant101.py         # WorldQuant 101
├── gtja191.py               # 国泰君安 191
├── academic_premia.py       # Academic Premia 学术溢价
└── factor_library.py        # 因子库管理器
```

---

*最后更新: 2025-12*
