#!/usr/bin/env python
"""
因子筛选与回测完整Pipeline

功能:
1. 从Milvus向量数据库/因子库提取候选因子
2. 执行分层筛选 (快速IC → 去重 → 聚类 → 正交化)
3. 封装因子供模型使用
4. 训练预测模型并回测
5. 输出回测报告

使用方法:
    # 完整Pipeline (提取 → 筛选 → 回测)
    python run_factor_selection.py --mode full
    
    # 从因子库加载因子
    python run_factor_selection.py --mode full --source library --library-sets alpha158
    
    # 仅筛选
    python run_factor_selection.py --mode select --source milvus
    
    # 仅回测 (使用已筛选因子)
    python run_factor_selection.py --mode backtest --input output/selection/selected_factors.json
    
    # 自定义参数
    python run_factor_selection.py --max-factors 30 --corr-threshold 0.7 --model lgb

API使用:
    >>> from alpha_agent.run_factor_selection import quick_start
    >>> result = quick_start(factor_sets=['alpha158'], max_factors=20)
"""
from __future__ import annotations

# 抑制Gym弃用警告（gym直接print到stderr，需要临时重定向）
import sys as _sys
import io as _io
_original_stderr = _sys.stderr
_sys.stderr = _io.StringIO()
try:
    import gym  # 触发警告但被捕获
except ImportError:
    pass
finally:
    _sys.stderr = _original_stderr
del _sys, _io, _original_stderr

__all__ = [
    # 配置类
    'PipelineConfig',
    'PipelineMode',
    'FactorSource',
    'BacktestResult',
    'ComparisonResult',
    # 核心函数
    'run_pipeline',
    'run_full_pipeline',
    'run_selection',
    'run_backtest',
    # 便捷API
    'quick_start',
    'run_from_json',
    'compare_factor_sets',  # 因子集对比
    # 数据加载
    'load_qlib_data',
    'load_factors_from_milvus',
    'load_factors_from_json',
    'load_factors_from_library',
    # 因子清洗
    'clean_factor_code',
    'clean_factors',
    # Qlib集成
    'run_qlib_benchmark',
    'run_qlib_with_custom_factors',
    'generate_qlib_config',
]

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

# 抑制警告
warnings.filterwarnings('ignore')


# ============================================================
# 配置类定义
# ============================================================

class FactorSource(str, Enum):
    """因子来源枚举"""
    MILVUS = "milvus"
    FILE = "file"
    LIBRARY = "library"  # 从alpha_agent.factors加载


class PipelineMode(str, Enum):
    """Pipeline运行模式"""
    FULL = "full"
    SELECT = "select"
    SELECT_MILVUS = "select-milvus"  # Milvus因子筛选模式
    BACKTEST = "backtest"
    COMPARE = "compare"  # 因子集对比模式
    QLIB_BENCHMARK = "qlib-benchmark"
    QLIB_CUSTOM = "qlib-custom"


@dataclass
class PipelineConfig:
    """
    因子筛选与回测Pipeline配置
    
    集中管理所有参数，支持从命令行参数、配置文件或代码构建
    """
    # 运行模式
    mode: PipelineMode = PipelineMode.FULL
    
    # 数据来源
    source: FactorSource = FactorSource.MILVUS
    input_file: Optional[str] = None
    library_sets: List[str] = field(default_factory=lambda: ["alpha158"])  # library模式的因子集
    
    # 筛选参数
    max_factors: int = 30
    quick_ic_threshold: float = 0.005
    corr_threshold: float = 0.7
    enable_cluster: bool = True
    n_clusters: int = 10
    
    # 数据参数
    instruments: str = "csi300"
    return_days: int = 5
    
    # 回测时间段
    train_start: str = "2022-01-01"
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2023-12-31"
    
    # 模型参数
    model_type: str = "lgb"
    qlib_models: List[str] = field(default_factory=lambda: ["lgb", "lgb_light", "xgb", "linear"])
    
    # 输出
    output_dir: str = "output/selection"
    save_intermediate: bool = True
    
    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: Optional[str] = None
    milvus_min_ic: Optional[float] = None
    
    # 对比模式配置
    compare_sets: List[str] = field(default_factory=lambda: ["alpha158", "worldquant101", "gtja191"])
    max_factors_per_set: int = 50
    
    def validate(self) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        
        if self.mode in [PipelineMode.BACKTEST, PipelineMode.QLIB_CUSTOM] and not self.input_file:
            errors.append(f"{self.mode.value}模式需要指定input_file")
        
        if self.source == FactorSource.FILE and not self.input_file:
            errors.append("source=file时需要指定input_file")
        
        if self.max_factors < 1:
            errors.append("max_factors必须大于0")
        
        if not (0 < self.corr_threshold <= 1):
            errors.append("corr_threshold必须在(0, 1]之间")
        
        return errors
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        """从命令行参数构建配置"""
        qlib_models = [m.strip() for m in args.qlib_models.split(',')] if hasattr(args, 'qlib_models') else ["lgb"]
        library_sets = [s.strip() for s in args.library_sets.split(',')] if hasattr(args, 'library_sets') else ["alpha158"]
        compare_sets = [s.strip() for s in args.compare_sets.split(',')] if hasattr(args, 'compare_sets') else ["alpha158", "worldquant101", "gtja191"]
        max_factors_per_set = getattr(args, 'max_factors_per_set', 50)
        
        return cls(
            mode=PipelineMode(args.mode),
            source=FactorSource(args.source),
            input_file=args.input,
            library_sets=library_sets,
            max_factors=args.max_factors,
            quick_ic_threshold=args.quick_ic,
            corr_threshold=args.corr_threshold,
            instruments=args.instruments,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            model_type=args.model,
            qlib_models=qlib_models,
            output_dir=args.output,
            compare_sets=compare_sets,
            max_factors_per_set=max_factors_per_set,
        )

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据加载
# ============================================================

def load_qlib_data(
    instruments: str = "csi300",
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    return_days: int = 5,
) -> tuple:
    """
    加载Qlib数据
    
    参考: https://www.wuzao.com/p/qlib/document/start/getdata.html
    
    Args:
        instruments: 股票池 (csi300, csi500, all)
        start_date: 开始日期
        end_date: 结束日期
        return_days: 收益计算天数
    
    Returns:
        (data, target) - DataFrame和目标收益
    """
    try:
        import qlib
        from qlib.data import D
        from qlib.config import REG_CN
        
        # Qlib数据路径
        provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
        if not os.path.exists(provider_uri):
            logger.warning(f"Qlib数据不存在: {provider_uri}")
            logger.info("请运行: python -m qlib.run.data_collector qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn")
            return None, None
        
        # 初始化Qlib
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        logger.info(f"Qlib初始化成功: {provider_uri}")
        
        # 获取股票池
        instruments_list = D.instruments(instruments)
        logger.info(f"股票池: {instruments}")
        
        # 定义字段 - 使用Qlib表达式语法
        fields = [
            "$close",    # 收盘价
            "$open",     # 开盘价
            "$high",     # 最高价
            "$low",      # 最低价
            "$volume",   # 成交量
            "$vwap",     # 成交均价
            "$turn",     # 换手率
            "$factor",   # 复权因子
        ]
        
        # 加载数据
        logger.info(f"加载数据: {start_date} ~ {end_date}")
        df = D.features(
            instruments_list,
            fields,
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )
        
        # 重命名列
        df.columns = ['close', 'open', 'high', 'low', 'volume', 'vwap', 'turn', 'adj_factor']
        
        # 添加派生指标以支持更多因子
        # 日收益率
        df['returns'] = df.groupby(level='instrument')['close'].pct_change()
        
        # 市值估算 (用价格*成交量*100近似，单位：元)
        df['market_cap'] = df['close'] * df['volume'] * 100
        
        # 市场收益 (所有股票平均收益)
        df['market_ret'] = df.groupby(level='datetime')['returns'].transform('mean')
        
        # 换手率
        if 'turn' in df.columns:
            df['turnover'] = df['turn']
        
        # 成交额
        df['amount'] = df['close'] * df['volume']
        
        # 振幅
        df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)
        
        # 计算未来收益作为预测目标
        # Ref($close, -N) = N天后的收盘价
        target = df['close'].groupby(level='instrument').pct_change(return_days).shift(-return_days)
        
        # 统计
        n_stocks = df.index.get_level_values('instrument').nunique()
        n_days = df.index.get_level_values('datetime').nunique()
        
        logger.info(f"Qlib数据加载完成:")
        logger.info(f"  - 股票数: {n_stocks}")
        logger.info(f"  - 交易日: {n_days}")
        logger.info(f"  - 总记录: {len(df):,}")
        
        return df, target
        
    except ImportError:
        logger.error("Qlib未安装: pip install pyqlib")
        return None, None
    except Exception as e:
        logger.warning(f"Qlib数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# 从selection模块导入因子清洗、筛选和数据预处理功能
from alpha_agent.selection import (
    clean_factors,
    clean_factor_code,
    FactorCleaner,
    FactorSelector,
    SelectionResult,
    FactorWrapper,
    # 数据预处理
    add_derived_fields,
    prepare_train_test_data,
)

# 保留旧函数名以兼容
def filter_factors_by_available_columns(factors: List[Dict], available_columns: List[str]) -> List[Dict]:
    """兼容接口：改为调用clean_factors，不再过滤因子"""
    return clean_factors(factors, available_columns)


def load_factors_from_milvus(
    host: str = "localhost",
    port: int = 19530,
    collection_name: str = None,
    min_ic: float = None,
) -> List[Dict]:
    """
    从Milvus向量数据库加载因子
    
    Args:
        host: Milvus主机
        port: Milvus端口
        collection_name: 集合名称
        min_ic: 最小IC过滤
    
    Returns:
        因子列表
    """
    logger.info(f"从Milvus加载因子: {host}:{port}")
    
    try:
        from alpha_agent.memory.vector_store import MilvusStore
        from alpha_agent.config import vector_db_config
        
        store = MilvusStore(
            host=host,
            port=port,
            collection_name=collection_name or vector_db_config.collection_name,
        )
        
        if not store.connect():
            logger.error("Milvus连接失败")
            return []
        
        store.create_collection()
        
        # 获取因子数量
        count = store.count()
        logger.info(f"Milvus中共有 {count} 个因子")
        
        # 获取所有因子
        factors = store.get_all_factors(limit=10000, min_ic=min_ic)
        
        store.disconnect()
        
        logger.info(f"从Milvus加载 {len(factors)} 个因子")
        return factors
        
    except ImportError as e:
        logger.error(f"Milvus依赖未安装: {e}")
        logger.error("请安装: pip install pymilvus")
        return []
    except Exception as e:
        logger.error(f"从Milvus加载失败: {e}")
        return []


def load_factors_from_json(path: str) -> List[Dict]:
    """从JSON文件加载因子"""
    logger.info(f"从文件加载因子: {path}")
    
    if not os.path.exists(path):
        logger.error(f"文件不存在: {path}")
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        return []
    
    if isinstance(data, list):
        factors = data
    elif isinstance(data, dict) and 'factors' in data:
        # 支持{'factors': [...]} 或 {'factors': {'name': {...}, ...}}
        factors_data = data['factors']
        if isinstance(factors_data, list):
            factors = factors_data
        else:
            factors = list(factors_data.values())
    else:
        factors = [data]
    
    logger.info(f"从文件加载 {len(factors)} 个因子")
    return factors


def load_factors_from_library(
    factor_sets: List[str] = None,
    max_factors: int = None,
) -> List[Dict]:
    """
    从alpha_agent.factors因子库加载因子
    
    Args:
        factor_sets: 因子集列表，可选: alpha158, alpha360, worldquant101, gtja191, classic
        max_factors: 每个因子集最大加载数量
    
    Returns:
        因子列表
    """
    logger.info(f"从因子库加载因子: {factor_sets}")
    
    if factor_sets is None:
        factor_sets = ["alpha158"]
    
    factors = []
    
    try:
        from alpha_agent.factors import (
            ALPHA158_FACTORS,
            ALPHA360_FACTORS,
            WORLDQUANT_101_FACTORS,
            GTJA191_FACTORS,
            ALL_CLASSIC_FACTORS,
        )
        
        factor_map = {
            "alpha158": ALPHA158_FACTORS,
            "alpha360": ALPHA360_FACTORS,
            "worldquant101": WORLDQUANT_101_FACTORS,
            "gtja191": GTJA191_FACTORS,
            "classic": ALL_CLASSIC_FACTORS,
        }
        
        for factor_set in factor_sets:
            set_name = factor_set.lower()
            if set_name not in factor_map:
                logger.warning(f"未知因子集: {factor_set}，支持: {list(factor_map.keys())}")
                continue
            
            set_factors = factor_map[set_name]
            
            # 统一格式 - 支持ClassicFactor dataclass和字典格式
            for i, f in enumerate(set_factors):
                if max_factors and i >= max_factors:
                    break
                
                # 判断是dataclass还是字典
                if hasattr(f, 'id'):
                    # ClassicFactor dataclass
                    factor_dict = {
                        'id': getattr(f, 'id', f"{set_name}_{i:03d}"),
                        'name': getattr(f, 'name', getattr(f, 'id', f"{set_name}_{i:03d}")),
                        'code': getattr(f, 'code', ''),
                        'expression': getattr(f, 'code', ''),  # ClassicFactor用code存表达式
                        'description': getattr(f, 'description', ''),
                        'category': str(getattr(f, 'category', set_name)),
                        'source': f"library:{set_name}",
                    }
                else:
                    # 字典格式
                    factor_dict = {
                        'id': f.get('id', f"{set_name}_{i:03d}"),
                        'name': f.get('name', f.get('id', f"{set_name}_{i:03d}")),
                        'code': f.get('code', f.get('expression', '')),
                        'expression': f.get('expression', f.get('code', '')),
                        'description': f.get('description', ''),
                        'category': f.get('category', set_name),
                        'source': f"library:{set_name}",
                    }
                factors.append(factor_dict)
            
            logger.info(f"  - {set_name}: {min(len(set_factors), max_factors or len(set_factors))} 个因子")
        
        logger.info(f"从因子库共加载 {len(factors)} 个因子")
        return factors
        
    except ImportError as e:
        logger.error(f"因子库导入失败: {e}")
        return []
    except Exception as e:
        logger.error(f"从因子库加载失败: {e}")
        return []


# ============================================================
# 因子执行器 (复用现有Sandbox)
# ============================================================

def create_sandbox_executor(data: pd.DataFrame):
    """创建沙箱执行器 - 使用现有Sandbox"""
    from alpha_agent.core.sandbox import execute_code
    
    _error_count = [0]
    
    def executor(code: str, df: pd.DataFrame = None) -> Optional[pd.Series]:
        if df is None:
            df = data
        
        result, error = execute_code(code, df, timeout_seconds=30)
        if error:
            _error_count[0] += 1
            if _error_count[0] <= 3:
                # 只打印简短错误
                short_error = error.split('\n')[0] if '\n' in error else error
                logger.warning(f"执行失败: {short_error}")
            return None
        return result
    
    return executor


# ============================================================
# 回测结果数据类
# ============================================================

@dataclass
class BacktestResult:
    """回测结果"""
    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # 模型指标
    ic_mean: float = 0.0
    icir: float = 0.0
    
    # 分组收益
    top_group_return: float = 0.0
    bottom_group_return: float = 0.0
    long_short_return: float = 0.0
    
    # 其他
    factor_count: int = 0
    model_type: str = ""
    train_period: str = ""
    test_period: str = ""
    
    def to_dict(self) -> Dict:
        """转为字典，确保JSON可序列化"""
        d = asdict(self)
        # 转换numpy类型为Python原生类型
        for k, v in d.items():
            if hasattr(v, 'item'):  # numpy scalar
                d[k] = v.item()
            elif isinstance(v, float) and not isinstance(v, (int, bool)):
                d[k] = float(v)
        return d
    
    def summary(self) -> str:
        return f"""
============================================================
                    📈 回测结果
============================================================
模型: {self.model_type}
因子数: {self.factor_count}
训练期: {self.train_period}
测试期: {self.test_period}

收益指标:
  - 年化收益: {self.annual_return*100:.2f}%
  - 夏普比率: {self.sharpe_ratio:.2f}
  - 最大回撤: {self.max_drawdown*100:.2f}%

模型指标:
  - IC均值: {self.ic_mean:.4f}
  - ICIR: {self.icir:.2f}

分组收益 (年化):
  - Top组: {self.top_group_return*100:.2f}%
  - Bottom组: {self.bottom_group_return*100:.2f}%
  - 多空收益: {self.long_short_return*100:.2f}%
============================================================
"""


# ============================================================
# 模型训练与回测
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "lgb",
) -> Any:
    """
    训练预测模型
    
    Args:
        X_train: 特征
        y_train: 标签
        model_type: 模型类型 (lgb, linear, ridge)
    
    Returns:
        训练好的模型
    """
    if model_type == "lgb":
        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_jobs=-1,
                verbose=-1,
                random_state=42,
            )
        except ImportError:
            logger.warning("LightGBM未安装，使用Ridge回归")
            model_type = "ridge"
    
    if model_type == "linear":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    
    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
    
    # 训练
    valid_idx = ~(y_train.isna() | X_train.isna().any(axis=1))
    X_clean = X_train.loc[valid_idx]
    y_clean = y_train.loc[valid_idx]
    
    logger.info(f"训练模型: {model_type}, 样本数: {len(X_clean):,}")
    model.fit(X_clean, y_clean)
    
    return model


def run_backtest(
    wrapper,
    data: pd.DataFrame,
    target: pd.Series,
    train_start: str = "2022-01-01",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2023-12-31",
    model_type: str = "lgb",
    n_groups: int = 5,
) -> BacktestResult:
    """
    执行回测
    
    Args:
        wrapper: FactorWrapper实例
        data: 原始数据
        target: 目标收益
        train_start: 训练开始日期
        train_end: 训练结束日期
        test_start: 测试开始日期
        test_end: 测试结束日期
        model_type: 模型类型
        n_groups: 分组数
    
    Returns:
        BacktestResult
    """
    logger.info("="*60)
    logger.info("     📊 开始回测")
    logger.info("="*60)
    
    # 1. 计算因子值
    logger.info("\n📊 Step 1: 计算因子值")
    factor_df = wrapper.compute(data, n_workers=4)
    logger.info(f"计算完成: {factor_df.shape[1]} 个因子")
    
    if factor_df.empty or factor_df.shape[1] == 0:
        logger.error("无有效因子值")
        return BacktestResult()
    
    # 2. 准备数据（使用数据预处理模块）
    logger.info("\n📊 Step 2: 准备训练/测试数据")
    
    X_train, y_train, X_test, y_test = prepare_train_test_data(
        factor_df=factor_df,
        target=target,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        fill_strategy='fill_zero',
    )
    
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("训练集或测试集为空")
        return BacktestResult()
    
    # 3. 训练模型
    logger.info("\n📊 Step 3: 训练模型")
    model = train_model(X_train, y_train, model_type)
    
    # 4. 预测
    logger.info("\n📊 Step 4: 生成预测")
    y_pred = model.predict(X_test.fillna(0))
    pred_series = pd.Series(y_pred, index=X_test.index, name='prediction')
    
    # 5. 计算IC
    logger.info("\n📊 Step 5: 计算IC")
    ic_series = _compute_daily_ic(pred_series, y_test)
    ic_mean = ic_series.mean()
    icir = ic_mean / (ic_series.std() + 1e-8)
    logger.info(f"IC均值: {ic_mean:.4f}, ICIR: {icir:.2f}")
    
    # 6. 分组回测
    logger.info("\n📊 Step 6: 分组回测")
    group_returns = _compute_group_returns(pred_series, y_test, n_groups)
    
    top_return = group_returns.get(f'group_{n_groups}', 0)
    bottom_return = group_returns.get('group_1', 0)
    long_short = top_return - bottom_return
    
    logger.info(f"Top组年化: {top_return*100:.2f}%")
    logger.info(f"Bottom组年化: {bottom_return*100:.2f}%")
    logger.info(f"多空收益年化: {long_short*100:.2f}%")
    
    # 7. 计算组合收益
    logger.info("\n📊 Step 7: 计算组合收益")
    portfolio_return, sharpe, max_dd = _compute_portfolio_metrics(pred_series, y_test)
    
    result = BacktestResult(
        total_return=portfolio_return,
        annual_return=portfolio_return * 252 / len(ic_series) if len(ic_series) > 0 else 0,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        ic_mean=ic_mean,
        icir=icir,
        top_group_return=top_return,
        bottom_group_return=bottom_return,
        long_short_return=long_short,
        factor_count=len(X_train.columns),
        model_type=model_type,
        train_period=f"{train_start} ~ {train_end}",
        test_period=f"{test_start} ~ {test_end}",
    )
    
    logger.info(result.summary())
    
    return result


def _compute_daily_ic(pred: pd.Series, actual: pd.Series) -> pd.Series:
    """计算每日截面IC"""
    df = pd.concat([pred, actual], axis=1)
    df.columns = ['pred', 'actual']
    
    if hasattr(df.index, 'get_level_values'):
        # MultiIndex
        dates = df.index.get_level_values('datetime').unique()
        daily_ic = []
        for date in dates:
            try:
                day_data = df.xs(date, level='datetime')
                if len(day_data) > 10:
                    ic = day_data['pred'].corr(day_data['actual'], method='spearman')
                    if not np.isnan(ic):
                        daily_ic.append(ic)
            except Exception:
                continue
        return pd.Series(daily_ic)
    else:
        return pd.Series([pred.corr(actual, method='spearman')])


def _compute_group_returns(
    pred: pd.Series,
    actual: pd.Series,
    n_groups: int = 5,
) -> Dict[str, float]:
    """计算分组收益"""
    df = pd.concat([pred, actual], axis=1)
    df.columns = ['pred', 'actual']
    
    group_returns = {f'group_{i+1}': [] for i in range(n_groups)}
    
    if hasattr(df.index, 'get_level_values'):
        dates = df.index.get_level_values('datetime').unique()
        
        for date in dates:
            try:
                day_data = df.xs(date, level='datetime').dropna()
                if len(day_data) < n_groups * 5:
                    continue
                
                # 按预测值分组
                day_data['group'] = pd.qcut(day_data['pred'], n_groups, labels=False, duplicates='drop')
                
                for g in range(n_groups):
                    g_return = day_data[day_data['group'] == g]['actual'].mean()
                    if not np.isnan(g_return):
                        group_returns[f'group_{g+1}'].append(g_return)
            except Exception:
                continue
    
    # 计算年化收益
    result = {}
    for g, returns in group_returns.items():
        if returns:
            mean_daily = np.mean(returns)
            result[g] = mean_daily * 252  # 年化
        else:
            result[g] = 0
    
    return result


def _compute_portfolio_metrics(
    pred: pd.Series,
    actual: pd.Series,
    top_ratio: float = 0.2,
) -> Tuple[float, float, float]:
    """计算组合指标"""
    df = pd.concat([pred, actual], axis=1)
    df.columns = ['pred', 'actual']
    
    daily_returns = []
    
    if hasattr(df.index, 'get_level_values'):
        dates = df.index.get_level_values('datetime').unique()
        
        for date in dates:
            try:
                day_data = df.xs(date, level='datetime').dropna()
                if len(day_data) < 10:
                    continue
                
                # 选择预测值最高的top_ratio
                threshold = day_data['pred'].quantile(1 - top_ratio)
                selected = day_data[day_data['pred'] >= threshold]
                
                if len(selected) > 0:
                    daily_ret = selected['actual'].mean()
                    daily_returns.append(daily_ret)
            except Exception:
                continue
    
    if not daily_returns:
        return 0, 0, 0
    
    returns = pd.Series(daily_returns)
    
    # 总收益
    total_return = (1 + returns).prod() - 1
    
    # 夏普比率 (假设无风险利率为0)
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = abs(drawdown.min())
    
    return total_return, sharpe, max_dd


# ============================================================
# Qlib多模型回测 (使用现有的QlibBenchmark)
# ============================================================

def run_qlib_benchmark(
    models: List[str] = None,
    instruments: str = "csi300",
    train_period: Tuple[str, str] = ("2018-01-01", "2021-12-31"),
    valid_period: Tuple[str, str] = ("2022-01-01", "2022-06-30"),
    test_period: Tuple[str, str] = ("2022-07-01", "2023-12-31"),
    output_dir: str = "output/selection",
) -> Dict:
    """
    使用QlibBenchmark进行多模型回测
    
    Args:
        models: 模型列表 (默认: lgb, xgb, linear)
        instruments: 股票池
        train_period: 训练期
        valid_period: 验证期
        test_period: 测试期
        output_dir: 输出目录
    
    Returns:
        回测结果字典
    """
    logger.info("="*60)
    logger.info("     🔬 Qlib多模型基准测试")
    logger.info("="*60)
    
    try:
        from alpha_agent.modeling.qlib_model_zoo import QlibBenchmark, QlibModelZoo
        
        # 默认模型
        if models is None:
            models = ["lgb", "lgb_light", "xgb", "linear"]
        
        logger.info(f"模型: {models}")
        logger.info(f"训练期: {train_period}")
        logger.info(f"测试期: {test_period}")
        
        # 创建基准测试
        benchmark = QlibBenchmark(models=models)
        
        # 运行回测
        comparison_df = benchmark.run(
            instruments=instruments,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            experiment_name="factor_selection",
        )
        
        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"qlib_benchmark_{timestamp}.csv"
        comparison_df.to_csv(result_file, index=False)
        logger.info(f"结果已保存: {result_file}")
        
        # 与官方基准对比
        QlibModelZoo.compare_with_official(benchmark.results)
        
        return {
            'comparison': comparison_df.to_dict(),
            'results': {k: v.__dict__ for k, v in benchmark.results.items()},
            'file': str(result_file),
        }
        
    except ImportError as e:
        logger.error(f"QlibBenchmark导入失败: {e}")
        logger.info("请确保Qlib已安装: pip install pyqlib")
        return {}
    except Exception as e:
        logger.error(f"Qlib回测失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_qlib_with_custom_factors(
    wrapper,
    models: List[str] = None,
    instruments: str = "csi300",
    train_period: Tuple[str, str] = ("2018-01-01", "2021-12-31"),
    valid_period: Tuple[str, str] = ("2022-01-01", "2022-06-30"),
    test_period: Tuple[str, str] = ("2022-07-01", "2023-12-31"),
    output_dir: str = "output/selection",
) -> Dict:
    """
    使用筛选的因子 + Qlib模型进行回测
    
    Args:
        wrapper: FactorWrapper实例 (包含筛选的因子)
        models: Qlib模型列表
        instruments: 股票池
        train_period: 训练期
        valid_period: 验证期
        test_period: 测试期
        output_dir: 输出目录
    
    Returns:
        回测结果
    """
    logger.info("="*60)
    logger.info("     🧪 自定义因子 + Qlib模型回测")
    logger.info("="*60)
    
    try:
        import qlib
        from qlib.data import D
        from qlib.config import REG_CN
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
        
        # 初始化Qlib
        provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
        if not os.path.exists(provider_uri):
            logger.error(f"Qlib数据不存在: {provider_uri}")
            return {}
        
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        logger.info("Qlib已初始化")
        
        # 获取因子表达式
        expressions = wrapper.to_qlib_expressions()
        logger.info(f"因子数: {len(expressions)}")
        
        if not expressions:
            logger.error("无有效因子表达式")
            return {}
        
        # 构建自定义Handler配置
        fields = [e['expression'] for e in expressions]
        names = [e['name'] for e in expressions]
        
        # 创建数据集配置 (使用自定义因子)
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "DataHandlerLP",
                    "module_path": "qlib.data.dataset.handler",
                    "kwargs": {
                        "start_time": train_period[0],
                        "end_time": test_period[1],
                        "fit_start_time": train_period[0],
                        "fit_end_time": train_period[1],
                        "instruments": instruments,
                        "infer_processors": [
                            {"class": "RobustZScoreNorm", "kwargs": {"clip_outlier": True}},
                            {"class": "Fillna", "kwargs": {"fill_value": 0}},
                        ],
                        "learn_processors": [
                            {"class": "DropnaLabel"},
                            {"class": "CSRankNorm"},
                        ],
                        "data_loader": {
                            "class": "QlibDataLoader",
                            "kwargs": {
                                "config": {
                                    "feature": (fields, names),
                                    "label": (["Ref($close, -5) / $close - 1"], ["LABEL0"]),
                                },
                            },
                        },
                    },
                },
                "segments": {
                    "train": train_period,
                    "valid": valid_period,
                    "test": test_period,
                },
            },
        }
        
        # 导入模型配置
        from alpha_agent.modeling.qlib_model_zoo import QlibModelZoo
        
        if models is None:
            models = ["lgb", "linear"]
        
        results = {}
        
        for model_name in models:
            logger.info(f"\n训练模型: {model_name}")
            
            model_info = QlibModelZoo.get_model_info(model_name)
            if model_info is None:
                continue
            
            display_name, category, model_config = model_info
            
            try:
                # 初始化
                model = init_instance_by_config(model_config)
                dataset = init_instance_by_config(dataset_config)
                
                # 训练
                with R.start(experiment_name=f"custom_factors_{model_name}"):
                    model.fit(dataset)
                    pred = model.predict(dataset)
                    
                    recorder = R.get_recorder()
                    sr = SignalRecord(model, dataset, recorder)
                    sr.generate()
                    
                    # IC分析
                    try:
                        sar = SigAnaRecord(recorder)
                        sar.generate()
                        
                        ic_series = recorder.load_object("sig_analysis/ic.pkl")
                        if ic_series is not None:
                            ic = float(ic_series.mean())
                            icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0
                            logger.info(f"  ✓ {display_name}: IC={ic:.4f}, ICIR={icir:.2f}")
                            
                            results[model_name] = {
                                'name': display_name,
                                'ic': ic,
                                'icir': icir,
                                'status': 'success',
                            }
                    except Exception as e:
                        logger.warning(f"IC分析失败: {e}")
                        results[model_name] = {'name': display_name, 'status': 'error', 'error': str(e)}
                        
            except Exception as e:
                logger.error(f"模型 {model_name} 失败: {e}")
                results[model_name] = {'name': display_name, 'status': 'error', 'error': str(e)}
        
        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"custom_factor_benchmark_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n结果已保存: {result_file}")
        return results
        
    except ImportError as e:
        logger.error(f"Qlib导入失败: {e}")
        return {}
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================
# 主流程
# ============================================================

def run_full_pipeline(
    source: str = "milvus",
    input_file: str = None,
    max_factors: int = 30,
    quick_ic_threshold: float = 0.005,
    corr_threshold: float = 0.7,
    instruments: str = "csi300",
    model_type: str = "lgb",
    output_dir: str = "output/selection",
    run_backtest_flag: bool = True,
    train_start: str = "2022-01-01",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2023-12-31",
    library_sets: Optional[List[str]] = None,
) -> Dict:
    """
    运行完整Pipeline: 提取 → 筛选 → 回测
    
    Args:
        source: 因子来源 ("milvus", "file", "library")
        input_file: 输入文件路径 (source="file"时使用)
        max_factors: 最大因子数
        quick_ic_threshold: 快速IC阈值
        corr_threshold: 相关性阈值
        instruments: 股票池
        model_type: 模型类型 (lgb, linear, ridge)
        output_dir: 输出目录
        run_backtest_flag: 是否执行回测
    """
    logger.info("="*60)
    logger.info("     🚀 因子筛选与回测Pipeline")
    logger.info("="*60)
    logger.info(f"来源: {source}")
    logger.info(f"最大因子数: {max_factors}")
    logger.info(f"模型: {model_type}")
    logger.info("="*60)
    
    start_time = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================
    # Phase 1: 加载Qlib数据
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("  Phase 1: 加载Qlib数据")
    logger.info("="*60)
    
    data, target = load_qlib_data(instruments=instruments)
    if data is None:
        logger.error("Qlib数据加载失败，请确保:")
        logger.error("  1. 已安装Qlib: pip install pyqlib")
        logger.error("  2. 已下载数据: python -m qlib.run.data_collector qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn")
        return {}
    
    # ========================================
    # Phase 2: 提取候选因子
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("  Phase 2: 提取候选因子")
    logger.info("="*60)
    
    if source == "milvus":
        factors = load_factors_from_milvus()
        if not factors:
            logger.error("Milvus中无因子，请确保:")
            logger.error("  1. Milvus已启动: docker-compose up -d milvus")
            logger.error("  2. 已运行因子挖掘存储因子: python run_factor_mining.py")
            return {}
    elif source == "file" and input_file:
        factors = load_factors_from_json(input_file)
        if not factors:
            logger.error(f"文件中无因子: {input_file}")
            return {}
    elif source == "library":
        factors = load_factors_from_library(
            factor_sets=library_sets or ["alpha158"],
            max_factors=max_factors * 10,  # 加载更多用于筛选
        )
        if not factors:
            logger.error("因子库中无因子")
            return {}
    else:
        logger.error(f"未知来源: {source}，支持: milvus, file, library")
        return {}
    
    logger.info(f"提取到 {len(factors)} 个候选因子")
    
    # 清洗因子代码（移除不必要的import，适配字段名）
    factors = clean_factors(factors, list(data.columns))
    if not factors:
        logger.error("无可用因子")
        return {}
    
    # ========================================
    # Phase 3: 因子筛选
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("  Phase 3: 因子筛选")
    logger.info("="*60)
    
    executor = create_sandbox_executor(data)
    
    from alpha_agent.selection import FactorSelector, FactorWrapper
    
    selector = FactorSelector(
        quick_ic_threshold=quick_ic_threshold,
        max_factors=max_factors,
        corr_threshold=corr_threshold,
        enable_cluster=len(factors) > 100,
        n_clusters=min(20, max(5, len(factors) // 5)),
    )
    
    selection_result = selector.select(factors, data, target, executor)
    
    logger.info(selection_result.summary())
    
    # 保存筛选结果
    selected_file = output_path / f"selected_factors_{timestamp}.json"
    selected_data = []
    for f in selection_result.selected_factors:
        selected_data.append({
            'id': f.get('id', ''),
            'name': f.get('name', ''),
            'code': f.get('code', ''),
            'description': f.get('description', ''),
            'ic': float(f.get('ic', 0)),
            'icir': float(f.get('icir', 0)),
            'rank_ic': float(f.get('rank_ic', 0)),
            'category': f.get('category', ''),
            'source': f.get('source', source),
        })
    
    with open(selected_file, 'w', encoding='utf-8') as fp:
        json.dump({'factors': selected_data, 'timestamp': timestamp}, fp, ensure_ascii=False, indent=2)
    logger.info(f"筛选结果已保存: {selected_file}")
    
    # 封装因子
    wrapper = FactorWrapper.from_dict_list(selected_data)
    wrapper.set_executor(executor)
    wrapper.save(output_path / f"factor_wrapper_{timestamp}.json")
    
    # ========================================
    # Phase 4: 回测
    # ========================================
    backtest_result = None
    
    if run_backtest_flag and len(selected_data) > 0:
        logger.info("\n" + "="*60)
        logger.info("  Phase 4: 模型训练与回测")
        logger.info("="*60)
        
        backtest_result = run_backtest(
            wrapper=wrapper,
            data=data,
            target=target,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            model_type=model_type,
        )
        
        # 保存回测结果
        backtest_file = output_path / f"backtest_result_{timestamp}.json"
        with open(backtest_file, 'w', encoding='utf-8') as fp:
            json.dump(backtest_result.to_dict(), fp, ensure_ascii=False, indent=2)
        logger.info(f"回测结果已保存: {backtest_file}")
    
    # ========================================
    # 总结
    # ========================================
    elapsed = time.time() - start_time
    
    logger.info("\n" + "="*60)
    logger.info("     ✅ Pipeline完成")
    logger.info("="*60)
    logger.info(f"候选因子: {len(factors)}")
    logger.info(f"筛选因子: {len(selected_data)}")
    if backtest_result:
        logger.info(f"年化收益: {backtest_result.annual_return*100:.2f}%")
        logger.info(f"夏普比率: {backtest_result.sharpe_ratio:.2f}")
        logger.info(f"IC均值: {backtest_result.ic_mean:.4f}")
    logger.info(f"总耗时: {elapsed:.1f} 秒")
    logger.info("="*60)
    
    return {
        'selection': {
            'input_count': len(factors),
            'output_count': len(selected_data),
            'factors': selected_data,
            'file': str(selected_file),
        },
        'backtest': backtest_result.to_dict() if backtest_result else None,
        'elapsed': elapsed,
    }


def load_factors(config: PipelineConfig) -> List[Dict]:
    """
    根据配置加载因子
    
    Args:
        config: Pipeline配置
    
    Returns:
        因子列表
    """
    if config.source == FactorSource.MILVUS:
        return load_factors_from_milvus(
            host=config.milvus_host,
            port=config.milvus_port,
            collection_name=config.milvus_collection,
            min_ic=config.milvus_min_ic,
        )
    elif config.source == FactorSource.FILE:
        if not config.input_file:
            logger.error("未指定输入文件")
            return []
        return load_factors_from_json(config.input_file)
    elif config.source == FactorSource.LIBRARY:
        return load_factors_from_library(
            factor_sets=config.library_sets,
            max_factors=config.max_factors * 10,  # 加载更多用于筛选
        )
    else:
        logger.error(f"未知因子来源: {config.source}")
        return []


def run_pipeline(config: PipelineConfig) -> Dict:
    """
    统一的Pipeline入口函数
    
    根据配置运行不同模式的Pipeline
    
    Args:
        config: Pipeline配置
    
    Returns:
        结果字典
    """
    # 验证配置
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"配置错误: {error}")
        return {"error": errors}
    
    # 根据模式分发
    if config.mode == PipelineMode.FULL:
        return run_full_pipeline(
            source=config.source.value,
            input_file=config.input_file,
            max_factors=config.max_factors,
            quick_ic_threshold=config.quick_ic_threshold,
            corr_threshold=config.corr_threshold,
            instruments=config.instruments,
            model_type=config.model_type,
            output_dir=config.output_dir,
            run_backtest_flag=True,
            train_start=config.train_start,
            train_end=config.train_end,
            test_start=config.test_start,
            test_end=config.test_end,
            library_sets=config.library_sets,
        )
    
    elif config.mode == PipelineMode.SELECT:
        return run_full_pipeline(
            source=config.source.value,
            input_file=config.input_file,
            max_factors=config.max_factors,
            quick_ic_threshold=config.quick_ic_threshold,
            corr_threshold=config.corr_threshold,
            instruments=config.instruments,
            model_type=config.model_type,
            output_dir=config.output_dir,
            run_backtest_flag=False,
            library_sets=config.library_sets,
        )
    
    elif config.mode == PipelineMode.SELECT_MILVUS:
        return _run_select_milvus(config)
    
    elif config.mode == PipelineMode.BACKTEST:
        return _run_backtest_only(config)
    
    elif config.mode == PipelineMode.QLIB_BENCHMARK:
        return run_qlib_benchmark(
            models=config.qlib_models,
            instruments=config.instruments,
            output_dir=config.output_dir,
        )
    
    elif config.mode == PipelineMode.COMPARE:
        return _run_compare(config)
    
    elif config.mode == PipelineMode.QLIB_CUSTOM:
        return _run_qlib_custom(config)
    
    else:
        logger.error(f"未知模式: {config.mode}")
        return {"error": f"未知模式: {config.mode}"}


def _run_backtest_only(config: PipelineConfig) -> Dict:
    """仅运行回测模式"""
    from alpha_agent.selection import FactorWrapper
    
    # 加载Qlib数据
    data, target = load_qlib_data(
        instruments=config.instruments,
        return_days=config.return_days,
    )
    if data is None:
        return {"error": "Qlib数据加载失败"}
    
    # 加载因子
    wrapper = FactorWrapper.from_json(config.input_file)
    executor = create_sandbox_executor(data)
    wrapper.set_executor(executor)
    
    # 执行回测
    backtest_result = run_backtest(
        wrapper=wrapper,
        data=data,
        target=target,
        train_start=config.train_start,
        train_end=config.train_end,
        test_start=config.test_start,
        test_end=config.test_end,
        model_type=config.model_type,
    )
    
    return {"backtest": backtest_result.to_dict()}


def _run_qlib_custom(config: PipelineConfig) -> Dict:
    """自定义因子 + Qlib模型模式"""
    from alpha_agent.selection import FactorWrapper
    
    wrapper = FactorWrapper.from_json(config.input_file)
    
    return run_qlib_with_custom_factors(
        wrapper=wrapper,
        models=config.qlib_models,
        instruments=config.instruments,
        output_dir=config.output_dir,
    )


def _run_select_milvus(config: PipelineConfig) -> Dict:
    """Milvus因子筛选模式 - 调用FactorSelector进行多阶段筛选"""
    result = select_milvus_factors(
        instruments=config.instruments,
        max_factors=config.max_factors,
        output_dir=config.output_dir,
    )
    
    return {
        "selection": {
            "total_input": result.total_input,
            "final_count": result.final_count,
            "selected_factors": result.selected_factors,
            "output_count": result.final_count,
        },
        "status": "success" if result.final_count > 0 else "failed",
    }


def _run_compare(config: PipelineConfig) -> Dict:
    """因子集对比模式"""
    results = compare_factor_sets(
        factor_sets=config.compare_sets,
        custom_factors_path=config.input_file,
        max_factors_per_set=config.max_factors_per_set,
        instruments=config.instruments,
        model_type=config.model_type,
        train_start=config.train_start,
        train_end=config.train_end,
        test_start=config.test_start,
        test_end=config.test_end,
        output_dir=config.output_dir,
    )
    
    if not results:
        return {"error": "因子集对比失败"}
    
    # 转换结果为字典格式
    return {
        "comparison": {k: v.to_dict() for k, v in results.items()},
        "success_count": sum(1 for r in results.values() if r.status == "success"),
        "total_count": len(results),
    }


def run_selection(
    source: str = "milvus",
    input_file: Optional[str] = None,
    max_factors: int = 30,
    quick_ic_threshold: float = 0.005,
    corr_threshold: float = 0.7,
    instruments: str = "csi300",
    output_dir: str = "output/selection",
    library_sets: Optional[List[str]] = None,
) -> Dict:
    """
    仅运行因子筛选 (不回测)
    
    Args:
        source: 因子来源 ("milvus", "file", "library")
        input_file: 输入文件路径 (source="file"时使用)
        max_factors: 最大因子数
        quick_ic_threshold: 快速IC阈值
        corr_threshold: 相关性阈值
        instruments: 股票池
        output_dir: 输出目录
        library_sets: 因子库集合列表 (source="library"时使用)
    
    Returns:
        筛选结果字典
    """
    config = PipelineConfig(
        mode=PipelineMode.SELECT,
        source=FactorSource(source),
        input_file=input_file,
        max_factors=max_factors,
        quick_ic_threshold=quick_ic_threshold,
        corr_threshold=corr_threshold,
        instruments=instruments,
        output_dir=output_dir,
        library_sets=library_sets or ["alpha158"],
    )
    return run_pipeline(config)


# ============================================================
# 便捷API
# ============================================================

def quick_start(
    factor_sets: List[str] = None,
    max_factors: int = 30,
    instruments: str = "csi300",
    model_type: str = "lgb",
    run_backtest: bool = True,
) -> Dict:
    """
    快速启动因子筛选Pipeline
    
    使用因子库中的因子进行快速筛选和回测，适合快速原型验证
    
    Args:
        factor_sets: 因子集列表，默认 ["alpha158"]
                    可选: alpha158, alpha360, worldquant101, gtja191, classic
        max_factors: 最大因子数
        instruments: 股票池 (csi300, csi500, all)
        model_type: 模型类型 (lgb, linear, ridge)
        run_backtest: 是否运行回测
    
    Returns:
        结果字典，包含:
        - selection: 筛选结果
        - backtest: 回测结果 (如果run_backtest=True)
    
    Example:
        >>> from alpha_agent.run_factor_selection import quick_start
        >>> result = quick_start(
        ...     factor_sets=["alpha158"],
        ...     max_factors=20,
        ...     run_backtest=True
        ... )
        >>> print(f"IC: {result['backtest']['ic_mean']:.4f}")
    """
    config = PipelineConfig(
        mode=PipelineMode.FULL if run_backtest else PipelineMode.SELECT,
        source=FactorSource.LIBRARY,
        library_sets=factor_sets or ["alpha158"],
        max_factors=max_factors,
        instruments=instruments,
        model_type=model_type,
    )
    return run_pipeline(config)


def run_from_json(
    json_path: str,
    max_factors: int = 30,
    instruments: str = "csi300",
    model_type: str = "lgb",
    run_backtest: bool = True,
) -> Dict:
    """
    从JSON文件加载因子并运行Pipeline
    
    Args:
        json_path: JSON文件路径
        max_factors: 最大因子数
        instruments: 股票池
        model_type: 模型类型
        run_backtest: 是否运行回测
    
    Returns:
        结果字典
    """
    config = PipelineConfig(
        mode=PipelineMode.FULL if run_backtest else PipelineMode.SELECT,
        source=FactorSource.FILE,
        input_file=json_path,
        max_factors=max_factors,
        instruments=instruments,
        model_type=model_type,
    )
    return run_pipeline(config)


@dataclass
class ComparisonResult:
    """因子集对比结果"""
    name: str
    factor_count: int
    ic_mean: float = 0.0
    icir: float = 0.0
    sharpe_ratio: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    top_group_return: float = 0.0
    long_short_return: float = 0.0
    elapsed: float = 0.0
    status: str = "success"
    error: str = ""
    
    def to_dict(self) -> Dict:
        """转换为字典，确保所有数值类型可JSON序列化"""
        d = asdict(self)
        # 转换numpy类型为Python原生类型
        for k, v in d.items():
            if hasattr(v, 'item'):  # numpy数值类型
                d[k] = v.item()
            elif isinstance(v, float):
                d[k] = float(v)
        return d


def compare_factor_sets(
    factor_sets: List[str] = None,
    custom_factors_path: Optional[str] = None,
    max_factors_per_set: int = 50,
    instruments: str = "csi300",
    model_type: str = "lgb",
    train_start: str = "2022-01-01",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2023-12-31",
    output_dir: str = "output/comparison",
) -> Dict[str, ComparisonResult]:
    """
    对比不同因子集在同一模型上的回测效果
    
    Args:
        factor_sets: 要对比的因子集列表
                    可选: alpha158, alpha360, worldquant101, gtja191, classic
                    默认: ["alpha158", "worldquant101", "gtja191"]
        custom_factors_path: 自定义因子JSON文件路径 (可选)
        max_factors_per_set: 每个因子集使用的最大因子数
        instruments: 股票池
        model_type: 模型类型 (lgb, linear, ridge)
        train_start: 训练开始日期
        train_end: 训练结束日期
        test_start: 测试开始日期
        test_end: 测试结束日期
        output_dir: 输出目录
    
    Returns:
        Dict[str, ComparisonResult]: 每个因子集的对比结果
    
    Example:
        >>> from alpha_agent.run_factor_selection import compare_factor_sets
        >>> results = compare_factor_sets(
        ...     factor_sets=["alpha158", "worldquant101"],
        ...     custom_factors_path="output/selection/selected_factors.json",
        ...     model_type="lgb"
        ... )
        >>> for name, r in results.items():
        ...     print(f"{name}: IC={r.ic_mean:.4f}, Sharpe={r.sharpe_ratio:.2f}")
    """
    logger.info("="*70)
    logger.info("     📊 因子集对比测试")
    logger.info("="*70)
    
    if factor_sets is None:
        factor_sets = ["alpha158", "worldquant101", "gtja191"]
    
    logger.info(f"因子集: {factor_sets}")
    if custom_factors_path:
        logger.info(f"自定义因子: {custom_factors_path}")
    logger.info(f"模型: {model_type}")
    logger.info(f"训练期: {train_start} ~ {train_end}")
    logger.info(f"测试期: {test_start} ~ {test_end}")
    logger.info("="*70)
    
    # 1. 加载Qlib数据 (只加载一次)
    logger.info("\n📊 Step 1: 加载Qlib数据")
    data, target = load_qlib_data(instruments=instruments)
    if data is None:
        logger.error("Qlib数据加载失败")
        return {}
    
    # 添加派生字段（支持大模型生成的因子使用market_cap, market_ret等）
    data = add_derived_fields(data)
    logger.info(f"已添加派生字段: {list(data.columns)[-7:]}")
    
    executor = create_sandbox_executor(data)
    
    # 2. 准备结果
    results: Dict[str, ComparisonResult] = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 3. 测试每个因子集
    all_factor_sets = list(factor_sets)
    if custom_factors_path:
        all_factor_sets.append("custom")
    
    for i, set_name in enumerate(all_factor_sets, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"  [{i}/{len(all_factor_sets)}] 测试因子集: {set_name}")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 加载因子
            if set_name == "custom":
                factors = load_factors_from_json(custom_factors_path)
                display_name = f"custom ({Path(custom_factors_path).stem})"
            elif set_name == "milvus":
                # 从Milvus数据库加载因子（使用统一配置）
                from alpha_agent.config.settings import vector_db_config
                factors = load_factors_from_milvus(
                    host=vector_db_config.host,
                    port=vector_db_config.port,
                    collection_name=vector_db_config.collection_name,
                    min_ic=0.01,  # 只加载IC > 0.01的因子
                )
                if factors and max_factors_per_set:
                    factors = factors[:max_factors_per_set]
                display_name = f"milvus ({len(factors) if factors else 0})"
            elif set_name == "milvus-selected":
                # 从Milvus加载因子并通过FactorSelector进行多阶段筛选
                logger.info("使用FactorSelector进行多阶段筛选...")
                from alpha_agent.config.settings import vector_db_config
                milvus_factors = load_factors_from_milvus(
                    host=vector_db_config.host,
                    port=vector_db_config.port,
                    collection_name=vector_db_config.collection_name,
                )
                if milvus_factors:
                    # 清洗因子
                    milvus_factors = clean_factors(milvus_factors, list(data.columns))
                    # 使用FactorSelector筛选
                    selector = FactorSelector(max_factors=max_factors_per_set or 30)
                    selection_result = selector.select(
                        factors=milvus_factors,
                        data=data,
                        target=target,
                        sandbox_executor=executor,
                    )
                    factors = selection_result.selected_factors
                    display_name = f"milvus-selected ({len(factors) if factors else 0})"
                else:
                    factors = []
                    display_name = "milvus-selected (0)"
            else:
                factors = load_factors_from_library(
                    factor_sets=[set_name],
                    max_factors=max_factors_per_set,
                )
                display_name = set_name
            
            if not factors:
                logger.warning(f"因子集 {set_name} 无可用因子")
                results[set_name] = ComparisonResult(
                    name=display_name,
                    factor_count=0,
                    status="error",
                    error="无可用因子",
                )
                continue
            
            # 清洗因子代码（milvus-selected已经清洗过）
            if set_name != "milvus-selected":
                factors = clean_factors(factors, list(data.columns))
            if not factors:
                logger.warning(f"因子集 {set_name} 无可用因子")
                results[set_name] = ComparisonResult(
                    name=display_name,
                    factor_count=0,
                    status="error",
                    error="过滤后无可用因子",
                )
                continue
            
            logger.info(f"因子数: {len(factors)}")
            
            # 创建FactorWrapper
            from alpha_agent.selection import FactorWrapper
            wrapper = FactorWrapper.from_dict_list(factors[:max_factors_per_set])
            wrapper.set_executor(executor)
            
            # 运行回测
            backtest_result = run_backtest(
                wrapper=wrapper,
                data=data,
                target=target,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                model_type=model_type,
            )
            
            elapsed = time.time() - start_time
            
            results[set_name] = ComparisonResult(
                name=display_name,
                factor_count=len(factors[:max_factors_per_set]),
                ic_mean=backtest_result.ic_mean,
                icir=backtest_result.icir,
                sharpe_ratio=backtest_result.sharpe_ratio,
                annual_return=backtest_result.annual_return,
                max_drawdown=backtest_result.max_drawdown,
                top_group_return=backtest_result.top_group_return,
                long_short_return=backtest_result.long_short_return,
                elapsed=elapsed,
                status="success",
            )
            
            logger.info(f"✓ {display_name}: IC={backtest_result.ic_mean:.4f}, "
                       f"ICIR={backtest_result.icir:.2f}, "
                       f"Sharpe={backtest_result.sharpe_ratio:.2f}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"因子集 {set_name} 测试失败: {e}")
            results[set_name] = ComparisonResult(
                name=set_name,
                factor_count=0,
                elapsed=elapsed,
                status="error",
                error=str(e),
            )
    
    # 4. 生成对比报告
    logger.info("\n" + "="*70)
    logger.info("     📈 因子集对比结果")
    logger.info("="*70)
    
    # 按IC排序
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v.status == "success"],
        key=lambda x: x[1].ic_mean,
        reverse=True
    )
    
    # 打印表格
    print(f"\n{'因子集':<25} {'因子数':>6} {'IC':>8} {'ICIR':>8} {'Sharpe':>8} {'年化收益':>10} {'多空收益':>10}")
    print("-" * 85)
    
    for name, r in sorted_results:
        print(f"{r.name:<25} {r.factor_count:>6} {r.ic_mean:>8.4f} {r.icir:>8.2f} "
              f"{r.sharpe_ratio:>8.2f} {r.annual_return*100:>9.2f}% {r.long_short_return*100:>9.2f}%")
    
    print("-" * 85)
    
    # 打印失败的
    failed = [(k, v) for k, v in results.items() if v.status != "success"]
    if failed:
        print("\n失败的因子集:")
        for name, r in failed:
            print(f"  - {name}: {r.error}")
    
    # 5. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON
    result_file = output_path / f"comparison_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({k: v.to_dict() for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    logger.info(f"\n结果已保存: {result_file}")
    
    # 保存CSV
    csv_file = output_path / f"comparison_{timestamp}.csv"
    rows = []
    for name, r in results.items():
        rows.append({
            'factor_set': r.name,
            'factor_count': r.factor_count,
            'ic_mean': r.ic_mean,
            'icir': r.icir,
            'sharpe_ratio': r.sharpe_ratio,
            'annual_return': r.annual_return,
            'max_drawdown': r.max_drawdown,
            'top_group_return': r.top_group_return,
            'long_short_return': r.long_short_return,
            'elapsed': r.elapsed,
            'status': r.status,
        })
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    logger.info(f"CSV已保存: {csv_file}")
    
    # 找出最佳因子集
    if sorted_results:
        best_name, best_result = sorted_results[0]
        logger.info(f"\n🏆 最佳因子集: {best_result.name}")
        logger.info(f"   IC均值: {best_result.ic_mean:.4f}")
        logger.info(f"   夏普比率: {best_result.sharpe_ratio:.2f}")
        logger.info(f"   年化收益: {best_result.annual_return*100:.2f}%")
    
    logger.info("\n" + "="*70)
    
    return results


# ============================================================
# Milvus 因子筛选
# ============================================================

def select_milvus_factors(
    instruments: str = "csi300",
    max_factors: int = 50,
    output_dir: str = "output/selection",
) -> SelectionResult:
    """
    从 Milvus 加载因子并进行多阶段筛选
    
    Pipeline:
    1. 从 Milvus 加载所有因子
    2. 加载 Qlib 数据
    3. 调用 FactorSelector 进行筛选
    4. 保存筛选结果
    
    Args:
        instruments: 股票池
        max_factors: 最终选择的因子数上限
        output_dir: 输出目录
        
    Returns:
        SelectionResult
    """
    logger.info("="*70)
    logger.info("     📊 Milvus 因子筛选")
    logger.info("="*70)
    
    # 1. 加载 Qlib 数据
    logger.info("\n📊 Step 1: 加载 Qlib 数据")
    data, target = load_qlib_data(instruments=instruments)
    if data is None:
        logger.error("Qlib 数据加载失败")
        return SelectionResult()
    
    data = add_derived_fields(data)
    logger.info(f"数据维度: {data.shape}, 派生字段已添加")
    
    # 2. 从 Milvus 加载因子
    logger.info("\n📊 Step 2: 从 Milvus 加载因子")
    from alpha_agent.config.settings import vector_db_config
    factors = load_factors_from_milvus(
        host=vector_db_config.host,
        port=vector_db_config.port,
        collection_name=vector_db_config.collection_name,
    )
    
    if not factors:
        logger.error("Milvus 中无因子")
        return SelectionResult()
    
    logger.info(f"加载 {len(factors)} 个因子")
    
    # 3. 清洗因子代码
    logger.info("\n📊 Step 3: 清洗因子代码")
    factors = clean_factors(factors, list(data.columns))
    logger.info(f"清洗后: {len(factors)} 个因子")
    
    # 4. 创建沙箱执行器
    executor = create_sandbox_executor(data)
    
    # 5. 因子筛选
    logger.info("\n📊 Step 4: 开始因子筛选")
    selector = FactorSelector(max_factors=max_factors)
    
    result = selector.select(
        factors=factors,
        data=data,
        target=target,
        sandbox_executor=executor,
    )
    
    # 6. 保存结果
    logger.info("\n📊 Step 5: 保存筛选结果")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 清理不可序列化的对象
    def make_serializable(obj):
        """将对象转换为可JSON序列化的格式"""
        import pandas as pd
        import numpy as np
        if isinstance(obj, dict):
            # 处理字典键可能是元组的情况
            result = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    k = str(k)  # 元组键转为字符串
                result[k] = make_serializable(v)
            return result
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            # Series/DataFrame 转为简单列表或嵌套列表
            if isinstance(obj, pd.Series):
                return obj.tolist()
            return obj.values.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            return make_serializable(obj.__dict__)
        return obj
    
    # 保存选中因子
    selected_file = output_path / f"selected_factors_{timestamp}.json"
    with open(selected_file, 'w', encoding='utf-8') as f:
        import json
        serializable_factors = make_serializable(result.selected_factors)
        json.dump(serializable_factors, f, ensure_ascii=False, indent=2)
    logger.info(f"选中因子已保存: {selected_file}")
    
    # 保存因子详情
    details_file = output_path / f"factor_details_{timestamp}.json"
    with open(details_file, 'w', encoding='utf-8') as f:
        serializable_details = make_serializable(result.factor_details)
        json.dump(serializable_details, f, ensure_ascii=False, indent=2)
    logger.info(f"因子详情已保存: {details_file}")
    
    # 生成 Qlib 配置
    qlib_config = generate_qlib_config(result.selected_factors)
    config_file = output_path / f"qlib_factors_{timestamp}.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(qlib_config)
    logger.info(f"Qlib配置已保存: {config_file}")
    
    logger.info("="*70)
    logger.info(f"✅ 因子筛选完成: {result.total_input} → {result.final_count}")
    logger.info("="*70)
    
    return result


def generate_qlib_config(factors: List[Dict]) -> str:
    """生成Qlib因子配置"""
    lines = [
        "# Qlib因子配置 - 由因子筛选系统自动生成",
        f"# 生成时间: {datetime.now().isoformat()}",
        f"# 因子数量: {len(factors)}",
        "",
        "data_handler_config:",
        "  class: Alpha158",
        "  module_path: qlib.contrib.data.handler",
        "  kwargs:",
        "    instruments: csi300",
        "    start_time: '2018-01-01'",
        "    end_time: '2023-12-31'",
        "    fit_start_time: '2018-01-01'",
        "    fit_end_time: '2021-12-31'",
        "",
        "# 自定义因子表达式",
        "custom_factors:",
    ]
    
    for f in factors:
        name = f.get('name', f.get('id', 'unknown'))
        code = f.get('code', '')
        ic = f.get('ic', 0)
        
        # 转换为Qlib表达式格式
        qlib_expr = code.replace('df["', '$').replace('"]', '')
        qlib_expr = qlib_expr.replace("df['", '$').replace("']", '')
        
        lines.append(f"  - name: {name}")
        lines.append(f"    expression: \"{qlib_expr}\"")
        lines.append(f"    ic: {ic:.4f}")
        lines.append("")
    
    return "\n".join(lines)


# ============================================================
# 命令行接口
# ============================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="因子筛选与回测Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整Pipeline (从Milvus提取因子 → 筛选 → 回测)
  python run_factor_selection.py --mode full
  
  # 从因子库加载 Alpha158 因子进行筛选
  python run_factor_selection.py --mode full --source library --library-sets alpha158
  
  # 从文件加载因子进行筛选
  python run_factor_selection.py --mode select --source file --input factors.json
  
  # 自定义时间段回测
  python run_factor_selection.py --mode backtest --input output/selected_factors.json \
      --train-start 2021-01-01 --train-end 2022-06-30 --test-start 2022-07-01 --test-end 2023-12-31
  
  # Qlib多模型基准测试
  python run_factor_selection.py --mode qlib-benchmark --qlib-models lgb,xgb,linear
  
  # 自定义因子 + Qlib模型
  python run_factor_selection.py --mode qlib-custom --input output/selected_factors.json
  
  # 因子集对比测试
  python run_factor_selection.py --mode compare --compare-sets alpha158,worldquant101,gtja191
  
  # 对比内置因子集与自定义筛选因子
  python run_factor_selection.py --mode compare --compare-sets alpha158 --input output/selection/selected_factors.json
        """
    )
    
    # 模式
    parser.add_argument(
        "--mode",
        choices=["full", "select", "select-milvus", "backtest", "compare", "qlib-benchmark", "qlib-custom"],
        default="full",
        help="运行模式: full(完整), select(仅筛选), select-milvus(筛选Milvus因子), backtest(回测), compare(因子集对比), qlib-benchmark(Qlib多模型), qlib-custom(自定义因子+Qlib) (默认: full)"
    )
    
    # 数据来源
    parser.add_argument(
        "--source", "-s",
        choices=["milvus", "file", "library"],
        default="milvus",
        help="因子来源: milvus(向量数据库), file(JSON文件), library(因子库) (默认: milvus)"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入文件路径 (mode=backtest或source=file时使用)"
    )
    
    parser.add_argument(
        "--library-sets",
        type=str,
        default="alpha158",
        help="因子库集合 (逗号分隔): alpha158,alpha360,worldquant101,gtja191,classic (默认: alpha158)"
    )
    
    parser.add_argument(
        "--compare-sets",
        type=str,
        default="alpha158,worldquant101,gtja191",
        help="对比模式的因子集列表 (逗号分隔): alpha158,worldquant101,gtja191,milvus,milvus-selected,custom (默认: alpha158,worldquant101,gtja191)"
    )
    
    parser.add_argument(
        "--max-factors-per-set",
        type=int,
        default=50,
        help="对比模式中每个因子集的最大因子数 (默认: 50)"
    )
    
    # 筛选参数
    parser.add_argument(
        "--max-factors", "-m",
        type=int,
        default=30,
        help="最大因子数 (默认: 30)"
    )
    
    parser.add_argument(
        "--quick-ic",
        type=float,
        default=0.005,
        help="快速IC阈值 (默认: 0.005)"
    )
    
    parser.add_argument(
        "--corr-threshold", "-c",
        type=float,
        default=0.7,
        help="相关性阈值 (默认: 0.7)"
    )
    
    # 回测参数
    parser.add_argument(
        "--model",
        choices=["lgb", "linear", "ridge"],
        default="lgb",
        help="简单回测模型类型: lgb(LightGBM), linear, ridge (默认: lgb)"
    )
    
    parser.add_argument(
        "--qlib-models",
        type=str,
        default="lgb,lgb_light,xgb,linear",
        help="Qlib回测模型列表 (逗号分隔): lgb,xgb,catboost,linear,mlp,lstm,gru,transformer (默认: lgb,lgb_light,xgb,linear)"
    )
    
    # 数据参数
    parser.add_argument(
        "--instruments",
        default="csi300",
        help="股票池 (默认: csi300)"
    )
    
    # 时间段参数
    parser.add_argument(
        "--train-start",
        type=str,
        default="2022-01-01",
        help="训练开始日期 (默认: 2022-01-01)"
    )
    
    parser.add_argument(
        "--train-end",
        type=str,
        default="2022-12-31",
        help="训练结束日期 (默认: 2022-12-31)"
    )
    
    parser.add_argument(
        "--test-start",
        type=str,
        default="2023-01-01",
        help="测试开始日期 (默认: 2023-01-01)"
    )
    
    parser.add_argument(
        "--test-end",
        type=str,
        default="2023-12-31",
        help="测试结束日期 (默认: 2023-12-31)"
    )
    
    # 输出
    parser.add_argument(
        "--output", "-o",
        default="output/selection",
        help="输出目录 (默认: output/selection)"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 统一使用PipelineConfig管理所有模式
    try:
        config = PipelineConfig.from_args(args)
    except ValueError as e:
        logger.error(f"配置解析失败: {e}")
        return 1
    
    # 验证配置
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"配置错误: {error}")
        return 1
    
    # 运行Pipeline
    result = run_pipeline(config)
    
    # 处理错误
    if "error" in result:
        logger.error(f"Pipeline执行失败: {result['error']}")
        return 1
    
    # 根据模式处理结果
    if config.mode == PipelineMode.COMPARE:
        # 对比模式
        comparison = result.get('comparison', {})
        success_count = result.get('success_count', 0)
        total_count = result.get('total_count', 0)
        
        if success_count == 0:
            logger.error("所有因子集测试都失败")
            return 1
        
        logger.info(f"\n✅ 因子集对比完成，成功测试 {success_count}/{total_count} 个因子集")
        return 0
    
    if config.mode == PipelineMode.SELECT_MILVUS:
        # Milvus因子筛选模式
        selection = result.get('selection', {})
        total_input = selection.get('total_input', 0)
        final_count = selection.get('final_count', 0)
        
        if final_count > 0:
            logger.info(f"\n✅ Milvus因子筛选完成: {total_input} → {final_count}")
            return 0
        else:
            logger.warning("\n⚠️ 未筛选出有效因子")
            return 1
    
    # 其他模式
    selection = result.get('selection', {})
    backtest = result.get('backtest', {})
    
    if selection.get('output_count', 0) > 0:
        logger.info("\n✅ 因子筛选完成")
    
    if backtest:
        ic = backtest.get('ic_mean', 0)
        if abs(ic) > 0.01:
            logger.info("✅ 回测完成，模型有效")
            return 0
        else:
            logger.warning("⚠️ 回测完成，但IC较低")
            return 0
    
    if selection.get('output_count', 0) == 0:
        logger.warning("\n⚠️ 未筛选出有效因子")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
