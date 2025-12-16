"""
统一特征表 - Feast特征存储

功能:
1. 特征注册和版本管理
2. 点查询和批量查询
3. 特征物化和在线服务
4. 与Qlib数据对接

参考: https://docs.feast.dev/
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Feast可选
try:
    from feast import FeatureStore as FeastStore
    from feast import Entity, Feature, FeatureView, FileSource, ValueType
    from feast.infra.offline_stores.file_source import FileSource
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    logger.info("Feast未安装，使用本地特征存储")


@dataclass
class FeatureDefinition:
    """特征定义"""
    name: str
    dtype: str = "float"  # float, int, string
    description: str = ""
    category: str = "factor"  # factor, fundamental, technical
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class FeatureSet:
    """特征集合"""
    name: str
    features: List[str]
    entity: str = "instrument"
    ttl: int = 86400  # 1天


class LocalFeatureStore:
    """本地特征存储 (无需Feast)"""
    
    def __init__(self, store_path: str = "./feature_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.features_dir = self.store_path / "features"
        self.features_dir.mkdir(exist_ok=True)
        
        self.registry_path = self.store_path / "registry.json"
        self.registry: Dict[str, FeatureDefinition] = {}
        
        self._load_registry()
    
    def _load_registry(self):
        """加载注册表"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                self.registry = {
                    k: FeatureDefinition(**v) for k, v in data.items()
                }
    
    def _save_registry(self):
        """保存注册表"""
        data = {k: v.__dict__ for k, v in self.registry.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_feature(
        self,
        name: str,
        dtype: str = "float",
        description: str = "",
        category: str = "factor",
    ) -> FeatureDefinition:
        """注册特征"""
        feature = FeatureDefinition(
            name=name,
            dtype=dtype,
            description=description,
            category=category,
        )
        self.registry[name] = feature
        self._save_registry()
        logger.info(f"注册特征: {name}")
        return feature
    
    def write_features(
        self,
        df: pd.DataFrame,
        feature_names: List[str] = None,
    ):
        """
        写入特征数据
        
        参数:
            df: 特征数据 (index=MultiIndex(datetime, instrument))
            feature_names: 特征列名
        """
        if feature_names is None:
            feature_names = [c for c in df.columns if c not in ['datetime', 'instrument']]
        
        # 按日期分区存储
        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values(0).unique()
        else:
            dates = df.index.unique()
        
        for date in dates:
            date_str = pd.Timestamp(date).strftime('%Y%m%d')
            date_dir = self.features_dir / date_str
            date_dir.mkdir(exist_ok=True)
            
            if isinstance(df.index, pd.MultiIndex):
                day_data = df.loc[date]
            else:
                day_data = df.loc[[date]]
            
            # 保存parquet
            output_path = date_dir / "features.parquet"
            day_data[feature_names].to_parquet(output_path)
        
        # 注册特征
        for name in feature_names:
            if name not in self.registry:
                self.register_feature(name)
        
        logger.info(f"写入特征: {len(feature_names)}个, {len(dates)}天")
    
    def read_features(
        self,
        feature_names: List[str],
        start_date: str = None,
        end_date: str = None,
        instruments: List[str] = None,
    ) -> pd.DataFrame:
        """
        读取特征数据
        
        参数:
            feature_names: 特征名列表
            start_date: 开始日期
            end_date: 结束日期
            instruments: 股票代码列表
        """
        dfs = []
        
        # 遍历日期目录
        for date_dir in sorted(self.features_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            
            date_str = date_dir.name
            date = pd.Timestamp(date_str)
            
            # 日期过滤
            if start_date and date < pd.Timestamp(start_date):
                continue
            if end_date and date > pd.Timestamp(end_date):
                continue
            
            # 读取parquet
            parquet_path = date_dir / "features.parquet"
            if not parquet_path.exists():
                continue
            
            df = pd.read_parquet(parquet_path)
            
            # 特征过滤
            available = [f for f in feature_names if f in df.columns]
            if not available:
                continue
            
            df = df[available]
            df['datetime'] = date
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs)
        result = result.reset_index()
        result = result.set_index(['datetime', 'instrument'])
        
        # 股票过滤
        if instruments:
            result = result.loc[(slice(None), instruments), :]
        
        return result
    
    def get_latest_features(
        self,
        feature_names: List[str],
        instruments: List[str] = None,
    ) -> pd.DataFrame:
        """获取最新特征 (在线服务)"""
        # 找到最新日期
        date_dirs = sorted(self.features_dir.iterdir(), reverse=True)
        
        for date_dir in date_dirs:
            parquet_path = date_dir / "features.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                available = [f for f in feature_names if f in df.columns]
                if available:
                    if instruments:
                        df = df.loc[df.index.isin(instruments)]
                    return df[available]
        
        return pd.DataFrame()
    
    def list_features(self, category: str = None) -> List[FeatureDefinition]:
        """列出所有特征"""
        features = list(self.registry.values())
        if category:
            features = [f for f in features if f.category == category]
        return features
    
    def delete_feature(self, name: str):
        """删除特征"""
        if name in self.registry:
            del self.registry[name]
            self._save_registry()
            logger.info(f"删除特征: {name}")


class FeatureStore:
    """
    统一特征存储接口
    
    自动选择后端:
    - Feast已安装: 使用Feast
    - 未安装: 使用本地存储
    """
    
    def __init__(
        self,
        store_path: str = "./feature_store",
        use_feast: bool = True,
    ):
        self.store_path = store_path
        
        if use_feast and FEAST_AVAILABLE:
            try:
                self.backend = FeastStore(repo_path=store_path)
                self.use_feast = True
                logger.info("使用Feast特征存储")
            except Exception as e:
                logger.warning(f"Feast初始化失败: {e}, 使用本地存储")
                self.backend = LocalFeatureStore(store_path)
                self.use_feast = False
        else:
            self.backend = LocalFeatureStore(store_path)
            self.use_feast = False
            logger.info("使用本地特征存储")
    
    def register(self, name: str, **kwargs) -> FeatureDefinition:
        """注册特征"""
        if self.use_feast:
            # Feast注册逻辑
            pass
        return self.backend.register_feature(name, **kwargs)
    
    def write(self, df: pd.DataFrame, feature_names: List[str] = None):
        """写入特征"""
        self.backend.write_features(df, feature_names)
    
    def read(
        self,
        feature_names: List[str],
        start_date: str = None,
        end_date: str = None,
        instruments: List[str] = None,
    ) -> pd.DataFrame:
        """读取特征"""
        return self.backend.read_features(
            feature_names, start_date, end_date, instruments
        )
    
    def get_latest(
        self,
        feature_names: List[str],
        instruments: List[str] = None,
    ) -> pd.DataFrame:
        """获取最新特征"""
        return self.backend.get_latest_features(feature_names, instruments)
    
    def list(self, category: str = None) -> List[FeatureDefinition]:
        """列出特征"""
        return self.backend.list_features(category)


# 全局实例
_feature_store: Optional[FeatureStore] = None


def get_feature_store(store_path: str = "./feature_store") -> FeatureStore:
    """获取特征存储单例"""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore(store_path)
    return _feature_store
