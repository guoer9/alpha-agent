"""
数据预处理模块

功能:
1. 添加派生字段（market_cap, market_ret 等）
2. 准备训练/测试数据集
3. 处理缺失值和异常值
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    'add_derived_fields',
    'prepare_train_test_data',
    'split_by_date',
    'handle_missing_values',
    'QLIB_AVAILABLE_FIELDS',
    'DERIVED_FIELDS',
]


# ============================================================
# 字段定义
# ============================================================

# Qlib默认可用的技术指标字段
QLIB_AVAILABLE_FIELDS = {
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'change', 'factor', 'turn', 'adj_factor',
}

# 需要动态计算的派生字段
DERIVED_FIELDS = {
    'market_cap',      # 市值估算 = close * volume * 100
    'market_ret',      # 市场收益 = 所有股票平均收益
    'returns',         # 日收益率 = close.pct_change()
    'amount',          # 成交额 = close * volume
    'amplitude',       # 振幅 = (high - low) / close.shift(1)
    'turnover',        # 换手率别名
    'adv5', 'adv10', 'adv20',  # 平均成交量
}


# ============================================================
# 派生字段计算
# ============================================================

def add_derived_fields(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    为DataFrame添加派生字段，支持更多因子
    
    Args:
        df: 包含基础OHLCV数据的DataFrame
        inplace: 是否原地修改
        
    Returns:
        添加派生字段后的DataFrame
    """
    if not inplace:
        df = df.copy()
    
    # 检查是否有MultiIndex（按股票分组）
    has_grouper = (
        hasattr(df.index, 'get_level_values') and 
        'instrument' in df.index.names
    )
    
    # 日收益率
    if 'returns' not in df.columns and 'close' in df.columns:
        if has_grouper:
            df['returns'] = df.groupby(level='instrument')['close'].pct_change()
        else:
            df['returns'] = df['close'].pct_change()
    
    # 市值估算（使用成交量代理流通股本）
    if 'market_cap' not in df.columns:
        if 'close' in df.columns and 'volume' in df.columns:
            df['market_cap'] = df['close'] * df['volume'] * 100
    
    # 市场收益（所有股票的平均收益）
    if 'market_ret' not in df.columns and 'returns' in df.columns:
        if has_grouper:
            df['market_ret'] = df.groupby(level='datetime')['returns'].transform('mean')
        else:
            df['market_ret'] = df['returns'].mean()
    
    # 成交额
    if 'amount' not in df.columns:
        if 'close' in df.columns and 'volume' in df.columns:
            df['amount'] = df['close'] * df['volume']
    
    # 振幅
    if 'amplitude' not in df.columns:
        if all(c in df.columns for c in ['high', 'low', 'close']):
            if has_grouper:
                df['amplitude'] = (df['high'] - df['low']) / df.groupby(level='instrument')['close'].shift(1)
            else:
                df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # 换手率别名
    if 'turnover' not in df.columns and 'turn' in df.columns:
        df['turnover'] = df['turn']
    
    # ADV (平均成交量)
    if 'adv5' not in df.columns and 'volume' in df.columns:
        if has_grouper:
            df['adv5'] = df.groupby(level='instrument')['volume'].rolling(5).mean().reset_index(0, drop=True)
            df['adv10'] = df.groupby(level='instrument')['volume'].rolling(10).mean().reset_index(0, drop=True)
            df['adv20'] = df.groupby(level='instrument')['volume'].rolling(20).mean().reset_index(0, drop=True)
        else:
            df['adv5'] = df['volume'].rolling(5).mean()
            df['adv10'] = df['volume'].rolling(10).mean()
            df['adv20'] = df['volume'].rolling(20).mean()

    # 基本面字段兜底：避免 Milvus 因子因字段缺失直接 KeyError
    try:
        from .factor_cleaner import FUNDAMENTAL_FIELDS

        missing_fields = [f for f in FUNDAMENTAL_FIELDS if f not in df.columns]
        for field_name in missing_fields:
            df[field_name] = np.nan
    except Exception:
        # 避免因为可选依赖或导入失败影响主流程
        pass
    
    return df


# ============================================================
# 训练/测试数据准备
# ============================================================

def split_by_date(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按日期分割训练集和测试集
    
    Args:
        df: 数据DataFrame
        train_start: 训练开始日期
        train_end: 训练结束日期
        test_start: 测试开始日期
        test_end: 测试结束日期
        
    Returns:
        (训练集DataFrame, 测试集DataFrame)
    """
    # 获取日期索引
    if hasattr(df.index, 'get_level_values') and 'datetime' in df.index.names:
        dates = df.index.get_level_values('datetime')
    else:
        dates = df.index
    
    train_mask = (dates >= train_start) & (dates <= train_end)
    test_mask = (dates >= test_start) & (dates <= test_end)
    
    return df.loc[train_mask], df.loc[test_mask]


def handle_missing_values(
    df: pd.DataFrame,
    target_col: str = 'target',
    feature_cols: Optional[List[str]] = None,
    strategy: str = 'fill_zero',
) -> pd.DataFrame:
    """
    处理缺失值
    
    Args:
        df: 数据DataFrame
        target_col: 目标列名
        feature_cols: 特征列名列表
        strategy: 缺失值处理策略 ('fill_zero', 'fill_mean', 'drop')
        
    Returns:
        处理后的DataFrame
    """
    result = df.copy()
    
    # 目标列：删除缺失值
    if target_col in result.columns:
        result = result.dropna(subset=[target_col])
    
    # 特征列：根据策略处理
    if feature_cols is None:
        feature_cols = [c for c in result.columns if c != target_col]
    
    if strategy == 'fill_zero':
        result[feature_cols] = result[feature_cols].fillna(0)
    elif strategy == 'fill_mean':
        for col in feature_cols:
            if col in result.columns:
                result[col] = result[col].fillna(result[col].mean())
    elif strategy == 'drop':
        result = result.dropna(subset=feature_cols, how='any')
    
    return result


def prepare_train_test_data(
    factor_df: pd.DataFrame,
    target: pd.Series,
    train_start: str = "2022-01-01",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2023-12-31",
    fill_strategy: str = 'fill_zero',
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    准备训练和测试数据
    
    Args:
        factor_df: 因子值DataFrame
        target: 目标值Series
        train_start: 训练开始日期
        train_end: 训练结束日期
        test_start: 测试开始日期
        test_end: 测试结束日期
        fill_strategy: 缺失值填充策略
        
    Returns:
        (X_train, y_train, X_test, y_test)
    """
    # 合并因子和目标
    full_df = factor_df.copy()
    full_df['target'] = target
    
    # 获取特征列
    feature_cols = [c for c in factor_df.columns]
    
    # 处理缺失值
    full_df = handle_missing_values(
        full_df, 
        target_col='target',
        feature_cols=feature_cols,
        strategy=fill_strategy,
    )
    
    if full_df.empty:
        logger.warning("处理缺失值后数据为空")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
    
    # 按日期分割
    train_data, test_data = split_by_date(
        full_df, train_start, train_end, test_start, test_end
    )
    
    # 分离特征和目标
    feature_cols = [c for c in feature_cols if c in train_data.columns]
    
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test = test_data[feature_cols]
    y_test = test_data['target']
    
    logger.info(f"训练集: {len(X_train):,} 样本, {train_start} ~ {train_end}")
    logger.info(f"测试集: {len(X_test):,} 样本, {test_start} ~ {test_end}")
    
    return X_train, y_train, X_test, y_test
