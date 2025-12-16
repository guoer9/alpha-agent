"""
Qlib Model Zoo - 基于Qlib框架的多模型基准测试

使用Qlib的模型模板和工作流API:
- qlib.contrib.model 中的模型 (LGBModel, XGBModel, DNNModel等)
- qlib.workflow 进行实验管理
- init_instance_by_config 从配置初始化模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Qlib导入
try:
    import qlib
    from qlib.utils import init_instance_by_config, flatten_dict
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord
    from qlib.data.dataset import DatasetH
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("Qlib未安装，部分功能不可用")

from .config import OUTPUT_DIR, QLIB_CONFIG


# ===================== Qlib 模型配置模板 =====================

# LightGBM 模型配置
LGB_MODEL_CONFIG = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    },
}

# LightGBM 轻量版
LGB_LIGHT_CONFIG = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "n_estimators": 100,
        "num_threads": -1,
    },
}

# XGBoost 模型配置
XGB_MODEL_CONFIG = {
    "class": "XGBModel",
    "module_path": "qlib.contrib.model.xgboost",
    "kwargs": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
    },
}

# CatBoost 模型配置
CATBOOST_MODEL_CONFIG = {
    "class": "CatBoostModel",
    "module_path": "qlib.contrib.model.catboost",
    "kwargs": {
        "iterations": 200,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "thread_count": -1,
        "verbose": False,
    },
}

# 线性模型配置
LINEAR_MODEL_CONFIG = {
    "class": "LinearModel",
    "module_path": "qlib.contrib.model.linear",
    "kwargs": {
        "estimator": "ridge",
        "alpha": 0.05,
    },
}

# MLP 神经网络配置
MLP_MODEL_CONFIG = {
    "class": "DNNModelPytorch",
    "module_path": "qlib.contrib.model.pytorch_nn",
    "kwargs": {
        "d_feat": 158,  # 特征维度，需要根据实际调整
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 100,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2000,
        "metric": "loss",
        "loss": "mse",
        "GPU": 0,
    },
}

# LSTM 模型配置
LSTM_MODEL_CONFIG = {
    "class": "LSTM",
    "module_path": "qlib.contrib.model.pytorch_lstm",
    "kwargs": {
        "d_feat": 6,  # 每个时间步的特征数
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 100,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2000,
        "metric": "loss",
        "loss": "mse",
        "GPU": 0,
    },
}

# GRU 模型配置
GRU_MODEL_CONFIG = {
    "class": "GRU",
    "module_path": "qlib.contrib.model.pytorch_gru",
    "kwargs": {
        "d_feat": 6,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 100,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2000,
        "metric": "loss",
        "loss": "mse",
        "GPU": 0,
    },
}

# Transformer 模型配置
TRANSFORMER_MODEL_CONFIG = {
    "class": "Transformer",
    "module_path": "qlib.contrib.model.pytorch_transformer",
    "kwargs": {
        "d_feat": 6,
        "d_model": 64,
        "nhead": 2,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 100,
        "lr": 0.0001,
        "early_stop": 20,
        "batch_size": 2000,
        "metric": "loss",
        "loss": "mse",
        "GPU": 0,
    },
}

# TabNet 模型配置
TABNET_MODEL_CONFIG = {
    "class": "TabNetModel",
    "module_path": "qlib.contrib.model.pytorch_tabnet",
    "kwargs": {
        "d_feat": 158,
        "n_d": 64,
        "n_a": 64,
        "n_steps": 3,
        "gamma": 1.3,
        "n_epochs": 100,
        "lr": 0.02,
        "batch_size": 2000,
        "GPU": 0,
    },
}

# Double Ensemble 模型配置
DOUBLE_ENSEMBLE_CONFIG = {
    "class": "DEnsembleModel",
    "module_path": "qlib.contrib.model.double_ensemble",
    "kwargs": {
        "base_model": "gbm",
        "loss": "mse",
        "num_models": 6,
        "enable_sr": True,
        "enable_fs": True,
        "alpha1": 1.0,
        "alpha2": 1.0,
        "bins_sr": 10,
        "bins_fs": 5,
        "decay": 0.5,
        "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    },
}

# TCN 时间卷积网络
TCN_MODEL_CONFIG = {
    "class": "TCN",
    "module_path": "qlib.contrib.model.pytorch_tcn",
    "kwargs": {
        "d_feat": 6,
        "num_channels": [32, 32, 32],
        "kernel_size": 3,
        "dropout": 0.0,
        "n_epochs": 100,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2000,
        "GPU": 0,
    },
}

# ALSTM 注意力LSTM
ALSTM_MODEL_CONFIG = {
    "class": "ALSTM",
    "module_path": "qlib.contrib.model.pytorch_alstm",
    "kwargs": {
        "d_feat": 6,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 100,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2000,
        "GPU": 0,
    },
}

# GATs 图注意力网络
GATS_MODEL_CONFIG = {
    "class": "GATs",
    "module_path": "qlib.contrib.model.pytorch_gats",
    "kwargs": {
        "d_feat": 6,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 100,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2000,
        "GPU": 0,
    },
}

# TRA 时间路由适配器
TRA_MODEL_CONFIG = {
    "class": "TRA",
    "module_path": "qlib.contrib.model.pytorch_tra",
    "kwargs": {
        "d_feat": 6,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 100,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2000,
        "GPU": 0,
    },
}

# Localformer 局部注意力
LOCALFORMER_MODEL_CONFIG = {
    "class": "Localformer",
    "module_path": "qlib.contrib.model.pytorch_localformer",
    "kwargs": {
        "d_feat": 6,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 100,
        "lr": 0.0001,
        "early_stop": 20,
        "batch_size": 2000,
        "GPU": 0,
    },
}

# SFM 状态频率记忆
SFM_MODEL_CONFIG = {
    "class": "SFM",
    "module_path": "qlib.contrib.model.pytorch_sfm",
    "kwargs": {
        "d_feat": 6,
        "hidden_size": 64,
        "n_epochs": 100,
        "lr": 0.001,
        "early_stop": 20,
        "batch_size": 2000,
        "GPU": 0,
    },
}

# HIST 历史信息股票趋势
HIST_MODEL_CONFIG = {
    "class": "HIST",
    "module_path": "qlib.contrib.model.pytorch_hist",
    "kwargs": {
        "d_feat": 6,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "n_epochs": 100,
        "lr": 0.0002,
        "early_stop": 20,
        "batch_size": 2000,
        "GPU": 0,
    },
}


# ===================== 官方基准测试数据 (CSI300) =====================
# 数据来源: Qlib官方文档
# 测试区间: 2017-01-01 ~ 2020-08-01

OFFICIAL_BENCHMARK_ALPHA158 = {
    # 模型: (IC, ICIR, RankIC, RankICIR, AnnRet, IR, MaxDD)
    "double_ensemble": (0.0521, 0.4223, 0.0502, 0.4117, 0.1158, 1.3432, -0.0920),
    "lgb": (0.0448, 0.3660, 0.0469, 0.3877, 0.0901, 1.0164, -0.1038),
    "mlp": (0.0376, 0.2846, 0.0429, 0.3220, 0.0895, 1.1408, -0.1103),
    "xgb": (0.0498, 0.3779, 0.0505, 0.4131, 0.0780, 0.9070, -0.1168),
    "catboost": (0.0481, 0.3366, 0.0454, 0.3311, 0.0765, 0.8032, -0.1092),
    "tra": (0.0440, 0.3535, 0.0540, 0.4451, 0.0718, 1.0835, -0.0760),
    "linear": (0.0397, 0.3000, 0.0472, 0.3531, 0.0692, 0.9209, -0.1509),
    "gats": (0.0349, 0.2511, 0.0462, 0.3564, 0.0497, 0.7338, -0.0777),
    "alstm": (0.0362, 0.2789, 0.0463, 0.3661, 0.0470, 0.6992, -0.1072),
    "sfm": (0.0379, 0.2959, 0.0464, 0.3825, 0.0465, 0.5672, -0.1282),
    "localformer": (0.0356, 0.2756, 0.0468, 0.3784, 0.0438, 0.6600, -0.0952),
    "lstm": (0.0318, 0.2367, 0.0435, 0.3389, 0.0381, 0.5561, -0.1207),
    "gru": (0.0315, 0.2450, 0.0428, 0.3440, 0.0344, 0.5160, -0.1017),
    "transformer": (0.0264, 0.2053, 0.0407, 0.3273, 0.0273, 0.3970, -0.1101),
    "tcn": (0.0279, 0.2181, 0.0421, 0.3429, 0.0262, 0.4133, -0.1090),
    "tabnet": (0.0204, 0.1554, 0.0333, 0.2552, 0.0227, 0.3676, -0.1089),
}

OFFICIAL_BENCHMARK_ALPHA360 = {
    # 模型: (IC, ICIR, RankIC, RankICIR, AnnRet, IR, MaxDD)
    "hist": (0.0522, 0.3530, 0.0667, 0.4576, 0.0987, 1.3726, -0.0681),
    "igmtf": (0.0480, 0.3589, 0.0606, 0.4773, 0.0946, 1.3509, -0.0716),
    "tra": (0.0485, 0.3787, 0.0587, 0.4756, 0.0920, 1.2789, -0.0834),
    "tcts": (0.0508, 0.3931, 0.0599, 0.4756, 0.0893, 1.2256, -0.0857),
    "gats": (0.0476, 0.3508, 0.0598, 0.4604, 0.0824, 1.1079, -0.0894),
    "adarnn": (0.0464, 0.3619, 0.0539, 0.4287, 0.0753, 1.0200, -0.0936),
    "gru": (0.0493, 0.3772, 0.0584, 0.4638, 0.0720, 0.9730, -0.0821),
    "add": (0.0430, 0.3188, 0.0559, 0.4301, 0.0667, 0.8992, -0.0855),
    "lstm": (0.0448, 0.3474, 0.0549, 0.4366, 0.0647, 0.8963, -0.0875),
    "alstm": (0.0497, 0.3829, 0.0599, 0.4736, 0.0626, 0.8651, -0.0994),
    "tcn": (0.0441, 0.3301, 0.0519, 0.4130, 0.0604, 0.8295, -0.1018),
    "lgb": (0.0400, 0.3037, 0.0499, 0.4042, 0.0558, 0.7632, -0.0659),
    "double_ensemble": (0.0390, 0.2946, 0.0486, 0.3836, 0.0462, 0.6151, -0.0915),
    "xgb": (0.0394, 0.2909, 0.0448, 0.3679, 0.0344, 0.4527, -0.1004),
    "catboost": (0.0378, 0.2714, 0.0467, 0.3659, 0.0292, 0.3781, -0.0862),
    "localformer": (0.0404, 0.2932, 0.0542, 0.4110, 0.0246, 0.3211, -0.1095),
    "mlp": (0.0273, 0.1870, 0.0396, 0.2910, 0.0029, 0.0274, -0.1385),
    "transformer": (0.0114, 0.0716, 0.0327, 0.2248, -0.0270, -0.3378, -0.1653),
    "tabnet": (0.0099, 0.0593, 0.0290, 0.1887, -0.0369, -0.3892, -0.2145),
}

# 推荐模型组合
RECOMMENDED_MODEL_SETS = {
    "fast": ["lgb_light"],
    "standard": ["lgb", "xgb", "linear"],
    "full": ["lgb", "xgb", "catboost", "mlp", "double_ensemble"],
    "deep": ["lgb", "lstm", "gru", "transformer", "gats"],
    "sota": ["double_ensemble", "tra", "hist", "gats", "alstm"],
}


# ===================== 模型Zoo类 =====================

@dataclass
class QlibModelResult:
    """Qlib模型结果"""
    name: str
    category: str
    # 指标
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    rank_icir: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    # 回测指标
    annualized_return: float = 0.0
    information_ratio: float = 0.0
    max_drawdown: float = 0.0
    # 训练信息
    train_time: float = 0.0
    config: Dict = field(default_factory=dict)
    status: str = "pending"
    error: str = ""


class QlibModelZoo:
    """Qlib模型动物园"""
    
    # 模型配置映射
    MODEL_CONFIGS = {
        # 树模型
        "lgb": ("LightGBM", "boosting", LGB_MODEL_CONFIG),
        "lgb_light": ("LightGBM_Light", "boosting", LGB_LIGHT_CONFIG),
        "xgb": ("XGBoost", "boosting", XGB_MODEL_CONFIG),
        "catboost": ("CatBoost", "boosting", CATBOOST_MODEL_CONFIG),
        # 线性模型
        "linear": ("Linear", "linear", LINEAR_MODEL_CONFIG),
        # 神经网络 - 基础
        "mlp": ("MLP", "nn", MLP_MODEL_CONFIG),
        "lstm": ("LSTM", "nn", LSTM_MODEL_CONFIG),
        "gru": ("GRU", "nn", GRU_MODEL_CONFIG),
        "transformer": ("Transformer", "nn", TRANSFORMER_MODEL_CONFIG),
        "tabnet": ("TabNet", "nn", TABNET_MODEL_CONFIG),
        # 神经网络 - 高级
        "tcn": ("TCN", "nn", TCN_MODEL_CONFIG),
        "alstm": ("ALSTM", "nn", ALSTM_MODEL_CONFIG),
        "gats": ("GATs", "graph", GATS_MODEL_CONFIG),
        "tra": ("TRA", "nn", TRA_MODEL_CONFIG),
        "localformer": ("Localformer", "nn", LOCALFORMER_MODEL_CONFIG),
        "sfm": ("SFM", "nn", SFM_MODEL_CONFIG),
        "hist": ("HIST", "graph", HIST_MODEL_CONFIG),
        # 集成模型
        "double_ensemble": ("DoubleEnsemble", "ensemble", DOUBLE_ENSEMBLE_CONFIG),
    }
    
    # 官方基准数据
    OFFICIAL_BENCHMARK = {
        "alpha158": OFFICIAL_BENCHMARK_ALPHA158,
        "alpha360": OFFICIAL_BENCHMARK_ALPHA360,
    }
    
    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有模型"""
        return list(cls.MODEL_CONFIGS.keys())
    
    @classmethod
    def get_config(cls, name: str) -> Optional[Dict]:
        """获取模型配置"""
        if name in cls.MODEL_CONFIGS:
            return cls.MODEL_CONFIGS[name][2].copy()
        return None
    
    @classmethod
    def get_model_info(cls, name: str) -> Optional[Tuple[str, str, Dict]]:
        """获取模型信息: (显示名, 类别, 配置)"""
        return cls.MODEL_CONFIGS.get(name)
    
    @classmethod
    def get_official_benchmark(cls, dataset: str = "alpha158", model: str = None) -> Dict:
        """
        获取官方基准测试数据
        
        参数:
            dataset: "alpha158" 或 "alpha360"
            model: 模型名称 (可选，不指定则返回全部)
        
        返回:
            dict: {model: (IC, ICIR, RankIC, RankICIR, AnnRet, IR, MaxDD)}
        """
        benchmark = cls.OFFICIAL_BENCHMARK.get(dataset, {})
        if model:
            return {model: benchmark.get(model)}
        return benchmark
    
    @classmethod
    def get_recommended_models(cls, preset: str = "standard") -> List[str]:
        """
        获取推荐模型组合
        
        参数:
            preset: "fast", "standard", "full", "deep", "sota"
        """
        return RECOMMENDED_MODEL_SETS.get(preset, RECOMMENDED_MODEL_SETS["standard"])
    
    @classmethod
    def print_official_benchmark(cls, dataset: str = "alpha158"):
        """打印官方基准测试表"""
        benchmark = cls.OFFICIAL_BENCHMARK.get(dataset, {})
        if not benchmark:
            print(f"未找到数据集: {dataset}")
            return
        
        print(f"\n{'='*90}")
        print(f"【Qlib官方基准测试 - {dataset.upper()} (CSI300)】")
        print(f"{'='*90}")
        print(f"{'模型':<20} {'IC':>8} {'ICIR':>8} {'RankIC':>8} {'年化收益':>10} {'IR':>8} {'最大回撤':>10}")
        print("-" * 90)
        
        # 按年化收益排序
        sorted_models = sorted(benchmark.items(), key=lambda x: x[1][4], reverse=True)
        
        for model, metrics in sorted_models:
            ic, icir, rank_ic, rank_icir, ann_ret, ir, max_dd = metrics
            print(f"{model:<20} {ic:>8.4f} {icir:>8.4f} {rank_ic:>8.4f} {ann_ret:>10.2%} {ir:>8.4f} {max_dd:>10.2%}")
        
        print(f"{'='*90}")
        
        # Top 3
        top3 = sorted_models[:3]
        print(f"\n🏆 Top 3 模型 (按年化收益):")
        for i, (model, metrics) in enumerate(top3, 1):
            print(f"   {i}. {model}: 年化{metrics[4]:.2%}, IC={metrics[0]:.4f}, IR={metrics[5]:.4f}")
    
    @classmethod
    def compare_with_official(cls, results: Dict[str, 'QlibModelResult'], dataset: str = "alpha158"):
        """
        与官方基准对比
        
        参数:
            results: 实际测试结果
            dataset: 对比数据集
        """
        benchmark = cls.OFFICIAL_BENCHMARK.get(dataset, {})
        
        print(f"\n{'='*100}")
        print(f"【与官方基准对比 - {dataset.upper()}】")
        print(f"{'='*100}")
        print(f"{'模型':<15} {'实测IC':>10} {'官方IC':>10} {'差异':>10} {'实测ICIR':>10} {'官方ICIR':>10}")
        print("-" * 100)
        
        for model_name, result in results.items():
            if model_name in benchmark:
                official = benchmark[model_name]
                ic_diff = result.ic - official[0]
                print(f"{model_name:<15} {result.ic:>10.4f} {official[0]:>10.4f} {ic_diff:>+10.4f} {result.icir:>10.4f} {official[1]:>10.4f}")
            else:
                print(f"{model_name:<15} {result.ic:>10.4f} {'N/A':>10} {'':>10} {result.icir:>10.4f} {'N/A':>10}")
        
        print(f"{'='*100}")


class QlibBenchmark:
    """Qlib多模型基准测试"""
    
    def __init__(
        self,
        models: List[str] = None,
        qlib_initialized: bool = False,
    ):
        """
        初始化基准测试
        
        参数:
            models: 模型列表 (默认使用主要模型)
            qlib_initialized: Qlib是否已初始化
        """
        if not QLIB_AVAILABLE:
            raise ImportError("请先安装Qlib: pip install pyqlib")
        
        # 默认模型列表 (树模型为主，神经网络可选)
        self.model_names = models or ["lgb", "lgb_light", "xgb", "linear"]
        self.qlib_initialized = qlib_initialized
        
        self.results: Dict[str, QlibModelResult] = {}
        self.predictions: Dict[str, pd.DataFrame] = {}
    
    def init_qlib(self, provider_uri: str = None):
        """初始化Qlib"""
        if self.qlib_initialized:
            return
        
        provider_uri = provider_uri or QLIB_CONFIG.get('provider_uri', '~/.qlib/qlib_data/cn_data')
        region = QLIB_CONFIG.get('region', 'cn')
        
        qlib.init(provider_uri=provider_uri, region=region)
        self.qlib_initialized = True
        logger.info(f"Qlib已初始化: {provider_uri}")
    
    def create_dataset_config(
        self,
        handler_class: str = "Alpha158",
        handler_module: str = "qlib.contrib.data.handler",
        instruments: str = "csi300",
        train_period: Tuple[str, str] = ("2008-01-01", "2014-12-31"),
        valid_period: Tuple[str, str] = ("2015-01-01", "2016-12-31"),
        test_period: Tuple[str, str] = ("2017-01-01", "2020-08-01"),
    ) -> Dict:
        """创建数据集配置"""
        return {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": handler_class,
                    "module_path": handler_module,
                    "kwargs": {
                        "start_time": train_period[0],
                        "end_time": test_period[1],
                        "fit_start_time": train_period[0],
                        "fit_end_time": train_period[1],
                        "instruments": instruments,
                    },
                },
                "segments": {
                    "train": train_period,
                    "valid": valid_period,
                    "test": test_period,
                },
            },
        }
    
    def run_single_model(
        self,
        model_name: str,
        dataset_config: Dict,
        experiment_name: str = "benchmark",
    ) -> QlibModelResult:
        """运行单个模型"""
        model_info = QlibModelZoo.get_model_info(model_name)
        if model_info is None:
            return QlibModelResult(
                name=model_name, category="unknown",
                status="error", error="模型不存在"
            )
        
        display_name, category, model_config = model_info
        result = QlibModelResult(
            name=display_name,
            category=category,
            config=model_config.copy(),
        )
        
        try:
            start_time = time.time()
            
            # 初始化模型和数据集
            model = init_instance_by_config(model_config)
            dataset = init_instance_by_config(dataset_config)
            
            # 启动实验
            with R.start(experiment_name=f"{experiment_name}_{model_name}"):
                # 记录参数
                R.log_params(**flatten_dict({"model": model_config}))
                
                # 训练
                model.fit(dataset)
                
                # 预测
                pred = model.predict(dataset)
                self.predictions[model_name] = pred
                
                # 保存信号记录
                recorder = R.get_recorder()
                sr = SignalRecord(model, dataset, recorder)
                sr.generate()
                
                # 使用 SigAnaRecord 计算 IC 指标
                try:
                    from qlib.workflow.record_temp import SigAnaRecord
                    import pandas as pd
                    
                    sar = SigAnaRecord(recorder)
                    sar.generate()
                    
                    # ic.pkl 是每日 IC 的 Series，ric.pkl 是每日 Rank IC 的 Series
                    ic_series = recorder.load_object("sig_analysis/ic.pkl")
                    if ic_series is not None and isinstance(ic_series, pd.Series) and len(ic_series) > 0:
                        result.ic = float(ic_series.mean())
                        result.icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0
                    
                    # 加载 Rank IC
                    try:
                        ric_series = recorder.load_object("sig_analysis/ric.pkl")
                        if ric_series is not None and isinstance(ric_series, pd.Series) and len(ric_series) > 0:
                            result.rank_ic = float(ric_series.mean())
                            result.rank_icir = float(ric_series.mean() / ric_series.std()) if ric_series.std() > 0 else 0
                    except Exception:
                        pass  # Rank IC 文件可能不存在
                    
                    # 使用 PortAnaRecord 进行策略回测（计算年化收益、夏普、最大回撤）
                    try:
                        from qlib.workflow.record_temp import PortAnaRecord
                        
                        # PortAnaRecord 需要的配置格式
                        port_analysis_config = {
                            "strategy": {
                                "class": "TopkDropoutStrategy",
                                "module_path": "qlib.contrib.strategy",
                                "kwargs": {
                                    "signal": "<PRED>",
                                    "topk": 30,
                                    "n_drop": 5,
                                },
                            },
                            "backtest": {
                                "start_time": dataset_config["kwargs"]["segments"]["test"][0],
                                "end_time": dataset_config["kwargs"]["segments"]["test"][1],
                                "account": 100000000,
                                "benchmark": "SH000300",
                                "exchange_kwargs": {
                                    "freq": "day",
                                    "limit_threshold": 0.095,
                                    "deal_price": "close",
                                    "open_cost": 0.0005,
                                    "close_cost": 0.0015,
                                    "min_cost": 5,
                                },
                            },
                        }
                        
                        # 执行回测
                        par = PortAnaRecord(recorder, port_analysis_config)
                        par.generate()
                        
                        # 加载分析结果 (port_analysis_1day.pkl 包含年化收益等指标)
                        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
                        if analysis_df is not None:
                            # MultiIndex: (metric_type, metric_name) -> 'risk' column
                            try:
                                result.annualized_return = float(analysis_df.loc[("excess_return_with_cost", "annualized_return"), "risk"])
                                result.information_ratio = float(analysis_df.loc[("excess_return_with_cost", "information_ratio"), "risk"])
                                result.max_drawdown = float(abs(analysis_df.loc[("excess_return_with_cost", "max_drawdown"), "risk"]))
                                result.sharpe = result.information_ratio
                            except Exception:
                                # 备用：读取不含成本的
                                try:
                                    result.annualized_return = float(analysis_df.loc[("excess_return_without_cost", "annualized_return"), "risk"])
                                    result.information_ratio = float(analysis_df.loc[("excess_return_without_cost", "information_ratio"), "risk"])
                                    result.max_drawdown = float(abs(analysis_df.loc[("excess_return_without_cost", "max_drawdown"), "risk"]))
                                    result.sharpe = result.information_ratio
                                except Exception:
                                    pass
                        
                        # 加载换手率
                        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
                        if report_df is not None and "turnover" in report_df.columns:
                            result.turnover = float(report_df["turnover"].mean())
                    except Exception as port_err:
                        logger.debug(f"PortAnaRecord 回测跳过: {port_err}")
                        
                except Exception as sig_err:
                    # 备用方案：直接计算 IC
                    logger.warning(f"SigAnaRecord失败，使用备用方案: {sig_err}")
                    try:
                        test_label = dataset.prepare("test", col_set=["label"], data_key="infer")
                        if test_label is not None and len(pred) > 0:
                            label = test_label.iloc[:, 0]
                            common_idx = pred.index.intersection(label.index)
                            if len(common_idx) > 10:
                                p, l = pred.loc[common_idx], label.loc[common_idx]
                                daily_ic = p.groupby(level='datetime').apply(lambda x: x.corr(l.loc[x.index]))
                                result.ic = float(daily_ic.mean())
                                result.icir = float(daily_ic.mean() / daily_ic.std()) if daily_ic.std() > 0 else 0
                    except Exception as calc_err:
                        logger.warning(f"IC计算失败: {calc_err}")
            
            result.train_time = time.time() - start_time
            result.status = "success"
            
        except Exception as e:
            result.status = "error"
            result.error = str(e)
            logger.error(f"模型 {model_name} 训练失败: {e}")
        
        return result
    
    def run(
        self,
        dataset_config: Dict = None,
        experiment_name: str = "benchmark",
        **dataset_kwargs,
    ) -> pd.DataFrame:
        """
        运行所有模型基准测试
        
        参数:
            dataset_config: 数据集配置 (可选)
            experiment_name: 实验名称
            **dataset_kwargs: 传递给create_dataset_config的参数
        
        返回:
            模型对比表DataFrame
        """
        print("\n" + "="*70)
        print("【Qlib 多模型基准测试】")
        print("="*70)
        print(f"📊 模型数: {len(self.model_names)}")
        print(f"📊 模型: {', '.join(self.model_names)}")
        
        # 确保Qlib已初始化
        self.init_qlib()
        
        # 创建数据集配置
        if dataset_config is None:
            dataset_config = self.create_dataset_config(**dataset_kwargs)
        
        # 运行所有模型
        for i, model_name in enumerate(self.model_names, 1):
            print(f"\n[{i}/{len(self.model_names)}] 训练 {model_name}...")
            result = self.run_single_model(model_name, dataset_config, experiment_name)
            self.results[model_name] = result
            
            if result.status == "success":
                print(f"  ✓ IC={result.ic:.4f}, ICIR={result.icir:.4f}, 耗时={result.train_time:.1f}s")
            else:
                print(f"  ✗ {result.error[:50]}")
        
        # 生成对比表
        comparison = self._generate_comparison()
        self.print_summary()
        self.save_results()
        
        return comparison
    
    def _generate_comparison(self) -> pd.DataFrame:
        """生成对比表"""
        records = []
        for name, result in self.results.items():
            records.append({
                "模型": result.name,
                "类别": result.category,
                "IC": result.ic,
                "ICIR": result.icir,
                "Rank_IC": result.rank_ic,
                "Rank_ICIR": result.rank_icir,
                "年化收益": result.annualized_return,
                "信息比率": result.information_ratio,
                "最大回撤": result.max_drawdown,
                "训练时间": result.train_time,
                "状态": result.status,
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values("ICIR", ascending=False)
        return df
    
    def get_best_model(self, metric: str = "icir") -> str:
        """获取最佳模型"""
        best_name = None
        best_value = -np.inf
        
        for name, result in self.results.items():
            if result.status != "success":
                continue
            value = getattr(result, metric, 0)
            if value > best_value:
                best_value = value
                best_name = name
        
        return best_name
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "="*70)
        print("【Qlib 模型对比结果】")
        print("="*70)
        
        comparison = self._generate_comparison()
        print(comparison.to_string(index=False))
        
        best = self.get_best_model("icir")
        if best:
            print(f"\n🏆 最佳模型 (ICIR): {self.results[best].name}")
    
    def save_results(self, path: Path = None):
        """保存结果"""
        if path is None:
            path = OUTPUT_DIR / "qlib_benchmark_results.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "models": {
                name: {
                    "name": r.name,
                    "category": r.category,
                    "ic": r.ic,
                    "icir": r.icir,
                    "train_time": r.train_time,
                    "status": r.status,
                }
                for name, r in self.results.items()
            },
            "best_model": self.get_best_model("icir"),
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 结果已保存: {path}")


# ===================== 便捷函数 =====================

def run_qlib_benchmark(
    models: List[str] = None,
    instruments: str = "csi300",
    train_period: Tuple[str, str] = ("2008-01-01", "2014-12-31"),
    valid_period: Tuple[str, str] = ("2015-01-01", "2016-12-31"),
    test_period: Tuple[str, str] = ("2017-01-01", "2020-08-01"),
) -> Tuple[pd.DataFrame, str]:
    """
    便捷函数: 运行Qlib基准测试
    
    返回:
        comparison: 模型对比表
        best_model: 最佳模型名称
    """
    benchmark = QlibBenchmark(models=models)
    comparison = benchmark.run(
        instruments=instruments,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
    )
    best = benchmark.get_best_model("icir")
    
    return comparison, best


def generate_workflow_config(
    model_name: str,
    output_path: Path = None,
    instruments: str = "csi300",
    **kwargs,
) -> Dict:
    """
    生成Qlib workflow配置文件
    
    可用于 qrun 命令行工具
    """
    model_info = QlibModelZoo.get_model_info(model_name)
    if model_info is None:
        raise ValueError(f"未知模型: {model_name}")
    
    _, _, model_config = model_info
    
    config = {
        "qlib_init": {
            "provider_uri": QLIB_CONFIG.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
            "region": QLIB_CONFIG.get("region", "cn"),
        },
        "market": instruments,
        "benchmark": "SH000300" if instruments == "csi300" else "SH000905",
        "data_handler_config": {
            "start_time": kwargs.get("start_time", "2008-01-01"),
            "end_time": kwargs.get("end_time", "2020-08-01"),
            "fit_start_time": kwargs.get("fit_start_time", "2008-01-01"),
            "fit_end_time": kwargs.get("fit_end_time", "2014-12-31"),
            "instruments": instruments,
        },
        "task": {
            "model": model_config,
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                    },
                    "segments": {
                        "train": (kwargs.get("train_start", "2008-01-01"), 
                                 kwargs.get("train_end", "2014-12-31")),
                        "valid": (kwargs.get("valid_start", "2015-01-01"),
                                 kwargs.get("valid_end", "2016-12-31")),
                        "test": (kwargs.get("test_start", "2017-01-01"),
                                kwargs.get("test_end", "2020-08-01")),
                    },
                },
            },
        },
    }
    
    if output_path:
        import yaml
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"配置已保存: {output_path}")
    
    return config
