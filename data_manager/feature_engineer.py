"""
特征工程模块
深度挖掘数据价值，计算高级因子

因子分类:
1. 价格因子 - 动量、反转、波动率
2. 量价因子 - 量价背离、资金流向
3. 技术因子 - MACD、RSI、布林带等
4. 统计因子 - 偏度、峰度、相关性
5. 时序因子 - 趋势、周期、季节性
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FactorInfo:
    """因子信息"""
    name: str
    category: str
    description: str
    lookback: int  # 回看周期


class FeatureEngineer:
    """特征工程师"""
    
    # 因子注册表
    FACTOR_REGISTRY: Dict[str, FactorInfo] = {}
    
    def __init__(self):
        """初始化"""
        self._register_factors()
    
    def _register_factors(self):
        """注册所有因子"""
        # 价格因子
        self.FACTOR_REGISTRY.update({
            "mom_5": FactorInfo("mom_5", "momentum", "5日动量", 5),
            "mom_10": FactorInfo("mom_10", "momentum", "10日动量", 10),
            "mom_20": FactorInfo("mom_20", "momentum", "20日动量", 20),
            "mom_60": FactorInfo("mom_60", "momentum", "60日动量", 60),
            "rev_5": FactorInfo("rev_5", "reversal", "5日反转", 5),
            "rev_20": FactorInfo("rev_20", "reversal", "20日反转", 20),
            "vol_5": FactorInfo("vol_5", "volatility", "5日波动率", 5),
            "vol_20": FactorInfo("vol_20", "volatility", "20日波动率", 20),
            "vol_60": FactorInfo("vol_60", "volatility", "60日波动率", 60),
        })
        
        # 量价因子
        self.FACTOR_REGISTRY.update({
            "vp_corr_5": FactorInfo("vp_corr_5", "volume_price", "5日量价相关", 5),
            "vp_corr_20": FactorInfo("vp_corr_20", "volume_price", "20日量价相关", 20),
            "amount_ratio": FactorInfo("amount_ratio", "volume_price", "成交额比率", 20),
            "volume_ma_ratio": FactorInfo("volume_ma_ratio", "volume_price", "量比", 5),
        })
        
        # 技术因子
        self.FACTOR_REGISTRY.update({
            "rsi_6": FactorInfo("rsi_6", "technical", "6日RSI", 6),
            "rsi_14": FactorInfo("rsi_14", "technical", "14日RSI", 14),
            "macd": FactorInfo("macd", "technical", "MACD", 26),
            "macd_signal": FactorInfo("macd_signal", "technical", "MACD信号线", 26),
            "macd_hist": FactorInfo("macd_hist", "technical", "MACD柱", 26),
            "bb_position": FactorInfo("bb_position", "technical", "布林带位置", 20),
            "atr_14": FactorInfo("atr_14", "technical", "14日ATR", 14),
        })
    
    # ============ 计算所有因子 ============
    
    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子
        
        Args:
            df: 原始数据，需包含 date, open, high, low, close, volume, amount
            
        Returns:
            包含所有因子的 DataFrame
        """
        df = df.copy()
        
        # 确保基础列存在
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")
        
        # 计算收益率
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # 1. 价格因子
        df = self._compute_price_factors(df)
        
        # 2. 量价因子
        df = self._compute_volume_price_factors(df)
        
        # 3. 技术因子
        df = self._compute_technical_factors(df)
        
        # 4. 统计因子
        df = self._compute_statistical_factors(df)
        
        # 5. 时序因子
        df = self._compute_time_series_factors(df)
        
        # 6. 高级因子
        df = self._compute_advanced_factors(df)
        
        return df
    
    # ============ 价格因子 ============
    
    def _compute_price_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格因子"""
        
        # 动量因子
        for period in [5, 10, 20, 60]:
            df[f"mom_{period}"] = df["close"].pct_change(period)
        
        # 反转因子（短期动量的负值）
        for period in [5, 10, 20]:
            df[f"rev_{period}"] = -df["close"].pct_change(period)
        
        # 波动率因子
        for period in [5, 10, 20, 60]:
            df[f"vol_{period}"] = df["returns"].rolling(period).std() * np.sqrt(252)
        
        # 波动率变化
        df["vol_change"] = df["vol_5"] / (df["vol_20"] + 1e-10) - 1
        
        # 价格位置
        for period in [5, 10, 20, 60]:
            high = df["high"].rolling(period).max()
            low = df["low"].rolling(period).min()
            df[f"price_pos_{period}"] = (df["close"] - low) / (high - low + 1e-10)
        
        # 均线偏离
        for period in [5, 10, 20, 60]:
            ma = df["close"].rolling(period).mean()
            df[f"ma_bias_{period}"] = df["close"] / ma - 1
        
        # 均线斜率
        for period in [5, 10, 20]:
            ma = df["close"].rolling(period).mean()
            df[f"ma_slope_{period}"] = ma.pct_change(5)
        
        # 价格加速度
        df["price_acc"] = df["returns"].diff()
        
        return df
    
    # ============ 量价因子 ============
    
    def _compute_volume_price_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算量价因子"""
        
        # 量价相关性
        for period in [5, 10, 20]:
            df[f"vp_corr_{period}"] = df["returns"].rolling(period).corr(
                df["volume"].pct_change()
            )
        
        # 量比
        df["volume_ma_5"] = df["volume"].rolling(5).mean()
        df["volume_ma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma_5"] + 1e-10)
        df["volume_ma_ratio"] = df["volume_ma_5"] / (df["volume_ma_20"] + 1e-10)
        
        # 成交额因子
        if "amount" in df.columns:
            df["amount_ma_5"] = df["amount"].rolling(5).mean()
            df["amount_ma_20"] = df["amount"].rolling(20).mean()
            df["amount_ratio"] = df["amount_ma_5"] / (df["amount_ma_20"] + 1e-10)
        
        # 量价背离
        # 价格上涨但成交量下降
        price_up = df["returns"] > 0
        volume_down = df["volume"].pct_change() < 0
        df["vp_divergence"] = (price_up & volume_down).astype(int) - \
                              (~price_up & ~volume_down).astype(int)
        
        # 资金流向（简化版）
        df["money_flow"] = df["close"] * df["volume"]
        df["money_flow_ma"] = df["money_flow"].rolling(20).mean()
        df["money_flow_ratio"] = df["money_flow"] / (df["money_flow_ma"] + 1e-10)
        
        # OBV (On-Balance Volume)
        df["obv"] = (np.sign(df["returns"]) * df["volume"]).cumsum()
        df["obv_ma"] = df["obv"].rolling(20).mean()
        df["obv_signal"] = df["obv"] - df["obv_ma"]
        
        return df
    
    # ============ 技术因子 ============
    
    def _compute_technical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        
        # RSI
        for period in [6, 14, 24]:
            df[f"rsi_{period}"] = self._rsi(df["close"], period)
        
        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = self._macd(df["close"])
        
        # 布林带
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = self._bollinger(df["close"])
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / \
                            (df["bb_upper"] - df["bb_lower"] + 1e-10)
        
        # ATR
        df["atr_14"] = self._atr(df["high"], df["low"], df["close"], 14)
        df["atr_ratio"] = df["atr_14"] / (df["close"] + 1e-10)
        
        # KDJ
        df["k"], df["d"], df["j"] = self._kdj(df["high"], df["low"], df["close"])
        
        # CCI
        df["cci_14"] = self._cci(df["high"], df["low"], df["close"], 14)
        
        # Williams %R
        df["willr_14"] = self._williams_r(df["high"], df["low"], df["close"], 14)
        
        # ADX
        df["adx_14"] = self._adx(df["high"], df["low"], df["close"], 14)
        
        return df
    
    # ============ 统计因子 ============
    
    def _compute_statistical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算统计因子"""
        
        # 偏度
        for period in [20, 60]:
            df[f"skew_{period}"] = df["returns"].rolling(period).skew()
        
        # 峰度
        for period in [20, 60]:
            df[f"kurt_{period}"] = df["returns"].rolling(period).kurt()
        
        # 最大回撤
        for period in [20, 60]:
            df[f"max_dd_{period}"] = self._rolling_max_drawdown(df["close"], period)
        
        # 夏普比率（滚动）
        for period in [20, 60]:
            returns = df["returns"].rolling(period)
            df[f"sharpe_{period}"] = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
        
        # 信息比率（相对于均线）
        df["ir_20"] = (df["close"] - df["close"].rolling(20).mean()) / \
                      (df["close"].rolling(20).std() + 1e-10)
        
        # 自相关性
        for lag in [1, 5, 10]:
            df[f"autocorr_{lag}"] = df["returns"].rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        return df
    
    # ============ 时序因子 ============
    
    def _compute_time_series_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算时序因子"""
        
        # 趋势强度
        for period in [10, 20, 60]:
            df[f"trend_{period}"] = self._trend_strength(df["close"], period)
        
        # 趋势方向
        df["trend_direction"] = np.sign(df["close"].rolling(20).mean().diff())
        
        # 周期性（简化）
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek if "date" in df.columns else 0
        df["month"] = pd.to_datetime(df["date"]).dt.month if "date" in df.columns else 0
        
        # 季节性收益
        if "date" in df.columns:
            df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)
            df["is_month_start"] = pd.to_datetime(df["date"]).dt.is_month_start.astype(int)
        
        # 连涨/连跌天数
        df["up_days"] = self._consecutive_days(df["returns"] > 0)
        df["down_days"] = self._consecutive_days(df["returns"] < 0)
        
        return df
    
    # ============ 高级因子 ============
    
    def _compute_advanced_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算高级因子"""
        
        # 波动率调整动量
        df["risk_adj_mom_20"] = df["mom_20"] / (df["vol_20"] + 1e-10)
        
        # 信息离散度
        df["idio_vol"] = df["returns"].rolling(20).std() - \
                         df["returns"].rolling(60).std()
        
        # 流动性调整收益
        if "turnover" in df.columns:
            df["liq_adj_ret"] = df["returns"] / (df["turnover"].rolling(20).mean() + 1e-10)
        
        # 价格效率
        abs_ret = df["returns"].abs().rolling(20).sum()
        net_ret = df["returns"].rolling(20).sum().abs()
        df["price_efficiency"] = net_ret / (abs_ret + 1e-10)
        
        # 异常成交量
        vol_mean = df["volume"].rolling(60).mean()
        vol_std = df["volume"].rolling(60).std()
        df["abnormal_volume"] = (df["volume"] - vol_mean) / (vol_std + 1e-10)
        
        # 隐含波动率代理（使用ATR）
        df["implied_vol_proxy"] = df["atr_14"] / df["close"] * np.sqrt(252)
        
        # 动量质量
        df["mom_quality"] = df["mom_20"].rolling(20).mean() / \
                           (df["mom_20"].rolling(20).std() + 1e-10)
        
        return df
    
    # ============ 辅助函数 ============
    
    def _rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _macd(self, prices: pd.Series, 
              fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _bollinger(self, prices: pd.Series, period: int = 20, std: int = 2) -> Tuple:
        """布林带"""
        middle = prices.rolling(period).mean()
        rolling_std = prices.rolling(period).std()
        upper = middle + std * rolling_std
        lower = middle - std * rolling_std
        return upper, middle, lower
    
    def _atr(self, high: pd.Series, low: pd.Series, 
             close: pd.Series, period: int = 14) -> pd.Series:
        """ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _kdj(self, high: pd.Series, low: pd.Series, 
             close: pd.Series, period: int = 9) -> Tuple:
        """KDJ"""
        low_min = low.rolling(period).min()
        high_max = high.rolling(period).max()
        rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j
    
    def _cci(self, high: pd.Series, low: pd.Series, 
             close: pd.Series, period: int = 14) -> pd.Series:
        """CCI"""
        tp = (high + low + close) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - ma) / (0.015 * md + 1e-10)
    
    def _williams_r(self, high: pd.Series, low: pd.Series, 
                    close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        high_max = high.rolling(period).max()
        low_min = low.rolling(period).min()
        return -100 * (high_max - close) / (high_max - low_min + 1e-10)
    
    def _adx(self, high: pd.Series, low: pd.Series, 
             close: pd.Series, period: int = 14) -> pd.Series:
        """ADX"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = self._atr(high, low, close, 1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx.rolling(period).mean()
    
    def _rolling_max_drawdown(self, prices: pd.Series, period: int) -> pd.Series:
        """滚动最大回撤"""
        def max_dd(x):
            cummax = x.cummax()
            drawdown = (x - cummax) / cummax
            return drawdown.min()
        return prices.rolling(period).apply(max_dd)
    
    def _trend_strength(self, prices: pd.Series, period: int) -> pd.Series:
        """趋势强度（R²）"""
        def r_squared(y):
            x = np.arange(len(y))
            if len(y) < 2:
                return np.nan
            corr = np.corrcoef(x, y)[0, 1]
            return corr ** 2 if not np.isnan(corr) else np.nan
        return prices.rolling(period).apply(r_squared)
    
    def _consecutive_days(self, condition: pd.Series) -> pd.Series:
        """计算连续天数"""
        groups = (~condition).cumsum()
        return condition.groupby(groups).cumsum()
    
    # ============ 因子分析 ============
    
    def analyze_factors(self, df: pd.DataFrame, target: str = "returns") -> pd.DataFrame:
        """
        分析因子有效性
        
        Args:
            df: 包含因子的 DataFrame
            target: 目标变量（通常是未来收益）
            
        Returns:
            因子分析结果
        """
        results = []
        
        # 获取所有因子列（排除基础列）
        base_cols = ["date", "open", "high", "low", "close", "volume", "amount", 
                    "returns", "log_returns", target]
        factor_cols = [c for c in df.columns if c not in base_cols]
        
        for factor in factor_cols:
            if df[factor].isna().all():
                continue
            
            # 计算 IC
            ic = df[factor].corr(df[target].shift(-1))
            
            # 计算 Rank IC
            rank_ic = df[factor].rank().corr(df[target].shift(-1).rank())
            
            # 计算 IC 的 IR
            rolling_ic = df[factor].rolling(20).corr(df[target].shift(-1))
            ic_ir = rolling_ic.mean() / (rolling_ic.std() + 1e-10)
            
            results.append({
                "factor": factor,
                "IC": ic,
                "Rank_IC": rank_ic,
                "ICIR": ic_ir,
                "coverage": 1 - df[factor].isna().mean(),
            })
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("IC", key=abs, ascending=False)
        
        return result_df
    
    def get_top_factors(self, df: pd.DataFrame, 
                        target: str = "returns",
                        top_n: int = 20) -> List[str]:
        """获取最有效的因子"""
        analysis = self.analyze_factors(df, target)
        return analysis.head(top_n)["factor"].tolist()


if __name__ == "__main__":
    # 测试
    print("特征工程测试")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n),
        "open": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "high": 100 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n)),
        "low": 100 + np.cumsum(np.random.randn(n) * 0.5) - np.abs(np.random.randn(n)),
        "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "volume": np.random.randint(1000000, 10000000, n),
        "amount": np.random.randint(10000000, 100000000, n),
    })
    
    # 计算因子
    engineer = FeatureEngineer()
    df_factors = engineer.compute_all_factors(df)
    
    print(f"原始列数: {len(df.columns)}")
    print(f"因子列数: {len(df_factors.columns)}")
    print(f"\n新增因子: {len(df_factors.columns) - len(df.columns)}")
    
    # 因子分析
    print("\n因子有效性分析 (Top 10):")
    analysis = engineer.analyze_factors(df_factors)
    print(analysis.head(10).to_string())
