"""
数据管理模块

功能：
- DataFetcher: 实时行情 + 历史数据获取
- DataProcessor: 数据处理和转换
- DataViewer: 数据展示和可视化
- DailyUpdater: 每日数据更新
- FeatureEngineer: 因子计算

因子说明：
- 使用 Qlib 标准 Alpha158，不自定义因子
- Alpha158 是工业级标准，经过大量验证
"""

from .data_fetcher import DataFetcher
from .data_processor import DataProcessor
from .data_viewer import DataViewer
from .daily_updater import DailyUpdater
from .feature_engineer import FeatureEngineer

__all__ = [
    "DataFetcher",
    "DataProcessor", 
    "DataViewer",
    "DailyUpdater",
    "FeatureEngineer",
]
