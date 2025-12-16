"""
工业级数据处理模块
- 并行处理：多进程加速
- 增量更新：只处理变化的数据
- 数据校验：完整性检查
- 高级特征：技术指标、因子计算
"""

import os
import struct
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class DataProcessor:
    """工业级数据处理器"""
    
    def __init__(self, 
                 qlib_data_dir: str = "~/.qlib/qlib_data/cn_data",
                 max_workers: int = 8):
        """
        初始化
        
        Args:
            qlib_data_dir: Qlib 数据目录
            max_workers: 最大并行数
        """
        self.qlib_data_dir = os.path.expanduser(qlib_data_dir)
        self.features_dir = os.path.join(self.qlib_data_dir, "features")
        self.instruments_dir = os.path.join(self.qlib_data_dir, "instruments")
        self.calendars_dir = os.path.join(self.qlib_data_dir, "calendars")
        self.max_workers = max_workers
        
        # 元数据目录
        self.meta_dir = os.path.join(self.qlib_data_dir, ".meta")
        
        for d in [self.features_dir, self.instruments_dir, self.calendars_dir, self.meta_dir]:
            os.makedirs(d, exist_ok=True)
    
    # ============ 并行处理 ============
    
    def process_stocks_parallel(self,
                                data: Dict[str, pd.DataFrame],
                                show_progress: bool = True) -> Dict:
        """
        并行处理多只股票数据
        
        Args:
            data: {symbol: DataFrame} 字典
            show_progress: 显示进度
            
        Returns:
            处理结果统计
        """
        import time
        start_time = time.time()
        
        results = {
            "total": len(data),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }
        
        completed = 0
        
        def process_one(item):
            symbol, df = item
            try:
                # 判断市场
                if symbol.startswith("6"):
                    market = "sh"
                elif symbol.startswith(("0", "3")):
                    market = "sz"
                elif symbol.startswith(("8", "4")):
                    market = "bj"
                else:
                    market = "sz"
                
                # 检查是否需要更新
                if self._need_update(symbol, market, df):
                    success = self._process_single_stock(df, symbol, market)
                    return symbol, "success" if success else "failed"
                else:
                    return symbol, "skipped"
                    
            except Exception as e:
                return symbol, f"error: {e}"
        
        # 使用线程池（I/O密集型）
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_one, item): item[0] 
                      for item in data.items()}
            
            for future in as_completed(futures):
                symbol, status = future.result()
                completed += 1
                
                if status == "success":
                    results["success"] += 1
                elif status == "skipped":
                    results["skipped"] += 1
                elif status == "failed":
                    results["failed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"{symbol}: {status}")
                
                if show_progress:
                    elapsed = time.time() - start_time
                    speed = completed / elapsed if elapsed > 0 else 0
                    print(f"\r处理进度: {completed}/{results['total']} | "
                          f"成功: {results['success']} | "
                          f"跳过: {results['skipped']} | "
                          f"失败: {results['failed']} | "
                          f"速度: {speed:.1f}/s", end="")
        
        if show_progress:
            print()
            elapsed = time.time() - start_time
            print(f"处理完成! 耗时: {elapsed:.1f}s")
        
        return results
    
    def _need_update(self, symbol: str, market: str, df: pd.DataFrame) -> bool:
        """检查是否需要更新"""
        qlib_symbol = f"{market}{symbol}".lower()
        meta_file = os.path.join(self.meta_dir, f"{qlib_symbol}.json")
        
        if not os.path.exists(meta_file):
            return True
        
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            
            # 检查数据哈希
            current_hash = self._compute_data_hash(df)
            if meta.get("data_hash") != current_hash:
                return True
            
            # 检查最新日期
            if "date" in df.columns:
                latest_date = df["date"].max()
                if isinstance(latest_date, pd.Timestamp):
                    latest_date = latest_date.strftime("%Y-%m-%d")
                if meta.get("latest_date") != latest_date:
                    return True
            
            return False
            
        except:
            return True
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """计算数据哈希"""
        if df.empty:
            return ""
        
        # 只用关键列计算哈希
        key_cols = ["date", "close", "volume"]
        key_cols = [c for c in key_cols if c in df.columns]
        
        data_str = df[key_cols].to_string()
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def _save_meta(self, symbol: str, market: str, df: pd.DataFrame):
        """保存元数据"""
        qlib_symbol = f"{market}{symbol}".lower()
        meta_file = os.path.join(self.meta_dir, f"{qlib_symbol}.json")
        
        meta = {
            "symbol": qlib_symbol,
            "data_hash": self._compute_data_hash(df),
            "latest_date": df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns else None,
            "rows": len(df),
            "updated_at": datetime.now().isoformat(),
        }
        
        with open(meta_file, "w") as f:
            json.dump(meta, f)
    
    # ============ 单只股票处理 ============
    
    def _process_single_stock(self, 
                              df: pd.DataFrame,
                              symbol: str,
                              market: str) -> bool:
        """处理单只股票数据"""
        if df.empty:
            return False
        
        df = df.copy()
        
        # 确保日期列
        if "date" not in df.columns:
            return False
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")
        
        # 构建 Qlib 股票代码
        qlib_symbol = f"{market}{symbol}".lower()
        stock_dir = os.path.join(self.features_dir, qlib_symbol)
        os.makedirs(stock_dir, exist_ok=True)
        
        # 计算额外特征
        df = self._compute_features(df)
        
        # 保存各个特征
        features = ["open", "high", "low", "close", "volume", "amount", 
                   "vwap", "adjclose", "factor", "change",
                   "returns", "log_returns", "volatility", "turnover"]
        
        for feature in features:
            if feature in df.columns:
                self._save_feature_bin(
                    df[feature].values,
                    os.path.join(stock_dir, f"{feature}.day.bin")
                )
        
        # 保存元数据
        self._save_meta(symbol, market, df)
        
        return True
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算衍生特征"""
        # 基础特征
        if "vwap" not in df.columns and "amount" in df.columns and "volume" in df.columns:
            df["vwap"] = df["amount"] / (df["volume"] * 100 + 1e-10)
        
        if "adjclose" not in df.columns:
            df["adjclose"] = df["close"]
        
        if "factor" not in df.columns:
            df["factor"] = 1.0
        
        if "change" not in df.columns and "close" in df.columns:
            df["change"] = df["close"].diff()
        
        # 收益率
        if "close" in df.columns:
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # 波动率（20日滚动）
        if "returns" in df.columns:
            df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(252)
        
        # 换手率（如果有）
        if "turnover" not in df.columns:
            df["turnover"] = 0.0
        
        return df
    
    def _save_feature_bin(self, values: np.ndarray, filepath: str):
        """保存特征为 Qlib 二进制格式"""
        values = np.array(values, dtype=np.float32)
        values = np.nan_to_num(values, nan=np.nan)
        
        with open(filepath, "wb") as f:
            values.tofile(f)
    
    # ============ 高级特征计算 ============
    
    def compute_alpha_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 因子
        
        包含:
        - 动量因子
        - 波动率因子
        - 流动性因子
        - 技术指标
        """
        df = df.copy()
        
        # ===== 动量因子 =====
        # 短期动量 (5日)
        df["mom_5"] = df["close"].pct_change(5)
        # 中期动量 (20日)
        df["mom_20"] = df["close"].pct_change(20)
        # 长期动量 (60日)
        df["mom_60"] = df["close"].pct_change(60)
        
        # ===== 波动率因子 =====
        # 已实现波动率
        df["vol_5"] = df["returns"].rolling(5).std() * np.sqrt(252)
        df["vol_20"] = df["returns"].rolling(20).std() * np.sqrt(252)
        
        # 波动率变化
        df["vol_change"] = df["vol_5"] / df["vol_20"] - 1
        
        # ===== 流动性因子 =====
        # 换手率均值
        if "turnover" in df.columns:
            df["turnover_5"] = df["turnover"].rolling(5).mean()
            df["turnover_20"] = df["turnover"].rolling(20).mean()
        
        # 成交额均值
        if "amount" in df.columns:
            df["amount_5"] = df["amount"].rolling(5).mean()
            df["amount_20"] = df["amount"].rolling(20).mean()
            df["amount_ratio"] = df["amount_5"] / df["amount_20"]
        
        # ===== 技术指标 =====
        # RSI
        df["rsi_14"] = self._compute_rsi(df["close"], 14)
        
        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = self._compute_macd(df["close"])
        
        # 布林带
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = self._compute_bollinger(df["close"])
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
        
        # ATR
        df["atr_14"] = self._compute_atr(df["high"], df["low"], df["close"], 14)
        
        # ===== 价格位置 =====
        # 相对高低点位置
        df["high_20"] = df["high"].rolling(20).max()
        df["low_20"] = df["low"].rolling(20).min()
        df["price_position"] = (df["close"] - df["low_20"]) / (df["high_20"] - df["low_20"] + 1e-10)
        
        # 均线偏离
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_60"] = df["close"].rolling(60).mean()
        df["ma_bias_5"] = df["close"] / df["ma_5"] - 1
        df["ma_bias_20"] = df["close"] / df["ma_20"] - 1
        
        return df
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算 RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _compute_macd(self, prices: pd.Series, 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """计算 MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _compute_bollinger(self, prices: pd.Series, period: int = 20, std: int = 2) -> Tuple:
        """计算布林带"""
        middle = prices.rolling(period).mean()
        rolling_std = prices.rolling(period).std()
        upper = middle + std * rolling_std
        lower = middle - std * rolling_std
        return upper, middle, lower
    
    def _compute_atr(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series, period: int = 14) -> pd.Series:
        """计算 ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    # ============ 数据完整性检查 ============
    
    def check_data_integrity(self, symbols: List[str] = None) -> Dict:
        """
        检查数据完整性
        
        Returns:
            {
                "total": int,
                "complete": int,
                "incomplete": List[str],
                "missing_dates": Dict[str, List[str]],
                "issues": List[str]
            }
        """
        result = {
            "total": 0,
            "complete": 0,
            "incomplete": [],
            "missing_dates": {},
            "issues": []
        }
        
        # 获取交易日历
        calendar_file = os.path.join(self.calendars_dir, "day.txt")
        if not os.path.exists(calendar_file):
            result["issues"].append("交易日历不存在")
            return result
        
        with open(calendar_file, "r") as f:
            calendar = set(line.strip() for line in f if line.strip())
        
        # 获取要检查的股票
        if symbols is None:
            symbols = os.listdir(self.features_dir)
        
        result["total"] = len(symbols)
        
        for symbol in symbols:
            stock_dir = os.path.join(self.features_dir, symbol)
            if not os.path.isdir(stock_dir):
                continue
            
            # 检查必要文件
            required = ["close.day.bin", "volume.day.bin"]
            missing_files = [f for f in required 
                           if not os.path.exists(os.path.join(stock_dir, f))]
            
            if missing_files:
                result["incomplete"].append(symbol)
                result["issues"].append(f"{symbol}: 缺少文件 {missing_files}")
                continue
            
            # 检查数据长度
            close_file = os.path.join(stock_dir, "close.day.bin")
            with open(close_file, "rb") as f:
                data = np.fromfile(f, dtype=np.float32)
            
            # 检查空值比例
            null_ratio = np.isnan(data).sum() / len(data)
            if null_ratio > 0.1:
                result["issues"].append(f"{symbol}: 空值比例 {null_ratio:.1%}")
            
            result["complete"] += 1
        
        return result
    
    # ============ 股票池管理 ============
    
    def update_instruments(self, 
                           name: str,
                           stocks: List[str],
                           start_date: str = "2008-01-01",
                           end_date: str = "2030-12-31"):
        """更新股票池文件"""
        filepath = os.path.join(self.instruments_dir, f"{name}.txt")
        
        with open(filepath, "w") as f:
            for stock in stocks:
                if stock.startswith(("6", "5")):
                    qlib_code = f"sh{stock}"
                elif stock.startswith(("0", "3")):
                    qlib_code = f"sz{stock}"
                elif stock.startswith(("8", "4")):
                    qlib_code = f"bj{stock}"
                else:
                    qlib_code = f"sz{stock}"
                
                f.write(f"{qlib_code}\t{start_date}\t{end_date}\n")
        
        logger.info(f"股票池已更新: {filepath} ({len(stocks)} 只)")
    
    def update_calendar(self, trade_dates: List[str]):
        """更新交易日历"""
        filepath = os.path.join(self.calendars_dir, "day.txt")
        
        # 标准化日期格式
        dates = []
        for d in trade_dates:
            if "-" not in d:
                d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            dates.append(d)
        
        dates = sorted(set(dates))
        
        with open(filepath, "w") as f:
            for d in dates:
                f.write(f"{d}\n")
        
        logger.info(f"交易日历已更新: {len(dates)} 天")


if __name__ == "__main__":
    # 测试
    processor = DataProcessor()
    
    # 检查数据完整性
    print("检查数据完整性...")
    result = processor.check_data_integrity()
    print(f"总计: {result['total']}, 完整: {result['complete']}, 不完整: {len(result['incomplete'])}")
    
    if result["issues"]:
        print(f"问题: {result['issues'][:5]}")
