"""
数据展示模块
提供数据可视化和分析功能
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import struct

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib 未安装，图表功能不可用")


class DataViewer:
    """数据查看器"""
    
    def __init__(self, qlib_data_dir: str = "~/.qlib/qlib_data/cn_data"):
        """
        初始化
        
        Args:
            qlib_data_dir: Qlib 数据目录
        """
        self.qlib_data_dir = os.path.expanduser(qlib_data_dir)
        self.features_dir = os.path.join(self.qlib_data_dir, "features")
        self.instruments_dir = os.path.join(self.qlib_data_dir, "instruments")
        self.calendars_dir = os.path.join(self.qlib_data_dir, "calendars")
    
    # ============ 数据读取 ============
    
    def read_stock_data(self, 
                        symbol: str,
                        features: List[str] = None,
                        start_date: str = None,
                        end_date: str = None) -> pd.DataFrame:
        """
        读取股票数据
        
        Args:
            symbol: 股票代码，如 "sh600000" 或 "sz000001"
            features: 特征列表，默认 ["open", "high", "low", "close", "volume"]
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票数据 DataFrame
        """
        if features is None:
            features = ["open", "high", "low", "close", "volume", "amount"]
        
        stock_dir = os.path.join(self.features_dir, symbol.lower())
        
        if not os.path.exists(stock_dir):
            print(f"股票 {symbol} 数据不存在")
            return pd.DataFrame()
        
        # 读取日历
        calendar = self._read_calendar()
        if not calendar:
            print("交易日历不存在")
            return pd.DataFrame()
        
        # 读取各特征
        data = {"date": calendar}
        
        for feature in features:
            feature_file = os.path.join(stock_dir, f"{feature}.day.bin")
            if os.path.exists(feature_file):
                values = self._read_bin_file(feature_file)
                # 对齐长度
                if len(values) < len(calendar):
                    values = np.concatenate([np.full(len(calendar) - len(values), np.nan), values])
                elif len(values) > len(calendar):
                    values = values[-len(calendar):]
                data[feature] = values
            else:
                data[feature] = np.nan
        
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        
        # 过滤日期
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        
        # 去除全为 NaN 的行
        df = df.dropna(subset=["close"])
        
        return df
    
    def _read_bin_file(self, filepath: str) -> np.ndarray:
        """读取 Qlib 二进制文件"""
        with open(filepath, "rb") as f:
            return np.fromfile(f, dtype=np.float32)
    
    def _read_calendar(self) -> List[str]:
        """读取交易日历"""
        filepath = os.path.join(self.calendars_dir, "day.txt")
        if not os.path.exists(filepath):
            return []
        
        with open(filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]
    
    # ============ 数据概览 ============
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要"""
        summary = {
            "data_dir": self.qlib_data_dir,
            "total_stocks": 0,
            "instruments": {},
            "calendar": {},
            "features": [],
        }
        
        # 统计股票数量
        if os.path.exists(self.features_dir):
            stocks = [d for d in os.listdir(self.features_dir)
                     if os.path.isdir(os.path.join(self.features_dir, d))]
            summary["total_stocks"] = len(stocks)
            
            # 按市场分类
            sh_stocks = [s for s in stocks if s.startswith("sh")]
            sz_stocks = [s for s in stocks if s.startswith("sz")]
            bj_stocks = [s for s in stocks if s.startswith("bj")]
            
            summary["by_market"] = {
                "上海": len(sh_stocks),
                "深圳": len(sz_stocks),
                "北京": len(bj_stocks),
            }
        
        # 股票池信息
        if os.path.exists(self.instruments_dir):
            for f in os.listdir(self.instruments_dir):
                if f.endswith(".txt"):
                    name = f.replace(".txt", "")
                    filepath = os.path.join(self.instruments_dir, f)
                    with open(filepath, "r") as file:
                        count = sum(1 for _ in file)
                    summary["instruments"][name] = count
        
        # 日历信息
        calendar = self._read_calendar()
        if calendar:
            summary["calendar"] = {
                "start": calendar[0],
                "end": calendar[-1],
                "total_days": len(calendar),
            }
        
        # 可用特征
        if summary["total_stocks"] > 0:
            sample_stock = stocks[0]
            stock_dir = os.path.join(self.features_dir, sample_stock)
            summary["features"] = [f.replace(".day.bin", "") 
                                   for f in os.listdir(stock_dir)
                                   if f.endswith(".day.bin")]
        
        return summary
    
    def print_summary(self):
        """打印数据摘要"""
        summary = self.get_data_summary()
        
        print("=" * 60)
        print("Qlib 数据概览")
        print("=" * 60)
        print(f"数据目录: {summary['data_dir']}")
        print(f"股票总数: {summary['total_stocks']}")
        
        if "by_market" in summary:
            print("\n按市场分布:")
            for market, count in summary["by_market"].items():
                print(f"  {market}: {count} 只")
        
        if summary["instruments"]:
            print("\n股票池:")
            for name, count in summary["instruments"].items():
                print(f"  {name}: {count} 只")
        
        if summary["calendar"]:
            cal = summary["calendar"]
            print(f"\n交易日历: {cal['start']} ~ {cal['end']} ({cal['total_days']} 天)")
        
        if summary["features"]:
            print(f"\n可用特征: {', '.join(summary['features'])}")
        
        print("=" * 60)
    
    # ============ 股票查询 ============
    
    def search_stocks(self, keyword: str) -> List[str]:
        """搜索股票"""
        if not os.path.exists(self.features_dir):
            return []
        
        stocks = os.listdir(self.features_dir)
        return [s for s in stocks if keyword.lower() in s.lower()]
    
    def get_stock_info(self, symbol: str) -> Dict:
        """获取股票信息"""
        stock_dir = os.path.join(self.features_dir, symbol.lower())
        
        if not os.path.exists(stock_dir):
            return {}
        
        info = {
            "symbol": symbol,
            "features": [],
            "data_points": 0,
            "date_range": None,
        }
        
        # 可用特征
        info["features"] = [f.replace(".day.bin", "") 
                           for f in os.listdir(stock_dir)
                           if f.endswith(".day.bin")]
        
        # 数据点数
        close_file = os.path.join(stock_dir, "close.day.bin")
        if os.path.exists(close_file):
            values = self._read_bin_file(close_file)
            info["data_points"] = len(values)
            
            # 日期范围
            calendar = self._read_calendar()
            if calendar and len(values) <= len(calendar):
                start_idx = len(calendar) - len(values)
                info["date_range"] = (calendar[start_idx], calendar[-1])
        
        return info
    
    # ============ 可视化 ============
    
    def plot_stock(self,
                   symbol: str,
                   start_date: str = None,
                   end_date: str = None,
                   figsize: Tuple[int, int] = (14, 8)):
        """
        绘制股票K线图
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            figsize: 图表大小
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib 未安装，无法绘图")
            return
        
        df = self.read_stock_data(symbol, start_date=start_date, end_date=end_date)
        
        if df.empty:
            print(f"无数据: {symbol}")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, 
                                  gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 价格图
        ax1 = axes[0]
        ax1.plot(df["date"], df["close"], label="收盘价", linewidth=1.5)
        ax1.fill_between(df["date"], df["low"], df["high"], alpha=0.3, label="最高-最低")
        ax1.set_title(f"{symbol} 股票走势", fontsize=14)
        ax1.set_ylabel("价格")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # 成交量
        ax2 = axes[1]
        colors = ['red' if df["close"].iloc[i] >= df["close"].iloc[i-1] else 'green' 
                  for i in range(1, len(df))]
        colors = ['gray'] + colors
        ax2.bar(df["date"], df["volume"], color=colors, alpha=0.7)
        ax2.set_ylabel("成交量")
        ax2.grid(True, alpha=0.3)
        
        # 收益率
        ax3 = axes[2]
        returns = df["close"].pct_change() * 100
        ax3.bar(df["date"], returns, color=['red' if r >= 0 else 'green' for r in returns], alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel("日收益率(%)")
        ax3.set_xlabel("日期")
        ax3.grid(True, alpha=0.3)
        
        # 格式化日期
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_comparison(self,
                        symbols: List[str],
                        start_date: str = None,
                        end_date: str = None,
                        normalize: bool = True,
                        figsize: Tuple[int, int] = (14, 6)):
        """
        多股票对比图
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            normalize: 是否归一化（便于对比）
            figsize: 图表大小
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib 未安装，无法绘图")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for symbol in symbols:
            df = self.read_stock_data(symbol, start_date=start_date, end_date=end_date)
            
            if df.empty:
                continue
            
            if normalize:
                # 归一化到起始值为 1
                values = df["close"] / df["close"].iloc[0]
                ax.plot(df["date"], values, label=symbol, linewidth=1.5)
            else:
                ax.plot(df["date"], df["close"], label=symbol, linewidth=1.5)
        
        ax.set_title("股票走势对比", fontsize=14)
        ax.set_xlabel("日期")
        ax.set_ylabel("归一化价格" if normalize else "价格")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_market_overview(self,
                             index: str = "csi300",
                             start_date: str = None,
                             end_date: str = None,
                             figsize: Tuple[int, int] = (14, 10)):
        """
        市场概览图
        
        Args:
            index: 股票池名称
            start_date: 开始日期
            end_date: 结束日期
            figsize: 图表大小
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib 未安装，无法绘图")
            return
        
        # 获取股票池
        instruments_file = os.path.join(self.instruments_dir, f"{index}.txt")
        if not os.path.exists(instruments_file):
            print(f"股票池 {index} 不存在")
            return
        
        stocks = []
        with open(instruments_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    stocks.append(parts[0])
        
        # 计算市场统计
        all_returns = []
        
        for symbol in stocks[:50]:  # 只取前50只，加快速度
            df = self.read_stock_data(symbol, start_date=start_date, end_date=end_date)
            if not df.empty:
                returns = df["close"].pct_change().dropna()
                all_returns.append(returns)
        
        if not all_returns:
            print("无数据")
            return
        
        # 合并收益率
        returns_df = pd.concat(all_returns, axis=1)
        avg_returns = returns_df.mean(axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 平均收益率走势
        ax1 = axes[0, 0]
        cumulative = (1 + avg_returns).cumprod()
        ax1.plot(cumulative.index, cumulative.values)
        ax1.set_title(f"{index.upper()} 累计收益")
        ax1.set_ylabel("累计收益")
        ax1.grid(True, alpha=0.3)
        
        # 收益率分布
        ax2 = axes[0, 1]
        ax2.hist(avg_returns * 100, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_title("日收益率分布")
        ax2.set_xlabel("收益率(%)")
        ax2.set_ylabel("频次")
        
        # 滚动波动率
        ax3 = axes[1, 0]
        rolling_vol = avg_returns.rolling(20).std() * np.sqrt(252) * 100
        ax3.plot(rolling_vol.index, rolling_vol.values)
        ax3.set_title("20日滚动波动率")
        ax3.set_ylabel("年化波动率(%)")
        ax3.grid(True, alpha=0.3)
        
        # 统计信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        市场统计 ({index.upper()})
        ─────────────────────
        样本股票数: {len(stocks)}
        分析股票数: {len(all_returns)}
        
        平均日收益: {avg_returns.mean()*100:.3f}%
        日收益标准差: {avg_returns.std()*100:.3f}%
        年化收益: {avg_returns.mean()*252*100:.2f}%
        年化波动率: {avg_returns.std()*np.sqrt(252)*100:.2f}%
        夏普比率: {avg_returns.mean()/avg_returns.std()*np.sqrt(252):.2f}
        最大单日涨幅: {avg_returns.max()*100:.2f}%
        最大单日跌幅: {avg_returns.min()*100:.2f}%
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    # ============ 数据统计 ============
    
    def get_stock_stats(self, symbol: str, 
                        start_date: str = None,
                        end_date: str = None) -> Dict:
        """
        获取股票统计信息
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            统计信息字典
        """
        df = self.read_stock_data(symbol, start_date=start_date, end_date=end_date)
        
        if df.empty:
            return {}
        
        returns = df["close"].pct_change().dropna()
        
        stats = {
            "symbol": symbol,
            "start_date": df["date"].iloc[0].strftime("%Y-%m-%d"),
            "end_date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "trading_days": len(df),
            "start_price": df["close"].iloc[0],
            "end_price": df["close"].iloc[-1],
            "total_return": (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100,
            "avg_daily_return": returns.mean() * 100,
            "daily_volatility": returns.std() * 100,
            "annualized_return": returns.mean() * 252 * 100,
            "annualized_volatility": returns.std() * np.sqrt(252) * 100,
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            "max_drawdown": self._calculate_max_drawdown(df["close"]) * 100,
            "avg_volume": df["volume"].mean(),
            "avg_amount": df["amount"].mean() if "amount" in df.columns else 0,
        }
        
        return stats
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """计算最大回撤"""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()
    
    def print_stock_stats(self, symbol: str,
                          start_date: str = None,
                          end_date: str = None):
        """打印股票统计信息"""
        stats = self.get_stock_stats(symbol, start_date, end_date)
        
        if not stats:
            print(f"无数据: {symbol}")
            return
        
        print("=" * 50)
        print(f"股票统计: {stats['symbol']}")
        print("=" * 50)
        print(f"时间范围: {stats['start_date']} ~ {stats['end_date']}")
        print(f"交易天数: {stats['trading_days']}")
        print(f"起始价格: {stats['start_price']:.2f}")
        print(f"结束价格: {stats['end_price']:.2f}")
        print(f"总收益率: {stats['total_return']:.2f}%")
        print(f"年化收益: {stats['annualized_return']:.2f}%")
        print(f"年化波动: {stats['annualized_volatility']:.2f}%")
        print(f"夏普比率: {stats['sharpe_ratio']:.2f}")
        print(f"最大回撤: {stats['max_drawdown']:.2f}%")
        print(f"日均成交量: {stats['avg_volume']:,.0f}")
        print("=" * 50)


if __name__ == "__main__":
    # 测试
    viewer = DataViewer()
    
    # 打印数据摘要
    viewer.print_summary()
    
    # 查看股票信息
    print("\n平安银行信息:")
    info = viewer.get_stock_info("sz000001")
    print(info)
