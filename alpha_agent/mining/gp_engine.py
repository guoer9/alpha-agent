"""
遗传规划搜索引擎 - 自动发现因子公式
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
import random
import logging

logger = logging.getLogger(__name__)

# 可选: gplearn
try:
    from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
    from gplearn.functions import make_function
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    logger.warning("gplearn未安装，GP功能受限")


@dataclass
class GPConfig:
    """GP配置"""
    population_size: int = 1000
    generations: int = 20
    tournament_size: int = 20
    stopping_criteria: float = 0.01
    
    p_crossover: float = 0.7
    p_subtree_mutation: float = 0.1
    p_hoist_mutation: float = 0.05
    p_point_mutation: float = 0.1
    
    max_samples: float = 0.9
    parsimony_coefficient: float = 0.01
    
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1


# 自定义算子
def _protected_div(x1, x2):
    """安全除法"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 1e-10, x1 / x2, 0.0)


def _protected_log(x1):
    """安全log"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 1e-10, np.log(np.abs(x1)), 0.0)


def _protected_sqrt(x1):
    """安全sqrt"""
    return np.sqrt(np.abs(x1))


def _ts_delay(x, period=5):
    """时序延迟"""
    return pd.Series(x).shift(period).fillna(0).values


def _ts_delta(x, period=5):
    """时序差分"""
    return pd.Series(x).diff(period).fillna(0).values


def _ts_mean(x, period=20):
    """时序均值"""
    return pd.Series(x).rolling(period, min_periods=1).mean().fillna(0).values


def _ts_std(x, period=20):
    """时序标准差"""
    return pd.Series(x).rolling(period, min_periods=1).std().fillna(0).values


def _ts_rank(x, period=20):
    """时序排名"""
    return pd.Series(x).rolling(period, min_periods=1).apply(
        lambda s: s.rank(pct=True).iloc[-1], raw=False
    ).fillna(0.5).values


def _cs_rank(x):
    """截面排名"""
    return pd.Series(x).rank(pct=True).fillna(0.5).values


class GPEngine:
    """遗传规划搜索引擎"""
    
    def __init__(self, config: GPConfig = None):
        self.config = config or GPConfig()
        self.model = None
        self.best_programs = []
        self.history = []
    
    def _create_function_set(self) -> List:
        """创建算子集"""
        if not GPLEARN_AVAILABLE:
            return []
        
        # 基础算子
        function_set = [
            'add', 'sub', 'mul', 
            'abs', 'neg', 'inv',
            'max', 'min',
        ]
        
        # 自定义算子
        div = make_function(_protected_div, 'div', arity=2)
        log = make_function(_protected_log, 'log', arity=1)
        sqrt = make_function(_protected_sqrt, 'sqrt', arity=1)
        
        function_set.extend([div, log, sqrt])
        
        return function_set
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str] = None,
    ) -> 'GPEngine':
        """
        拟合GP模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称
        """
        if not GPLEARN_AVAILABLE:
            raise ImportError("请安装gplearn: pip install gplearn")
        
        # 转换为numpy数组并处理缺失值
        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # 处理NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        feature_names = feature_names or [f'x{i}' for i in range(X.shape[1])]
        
        print(f"\n{'='*50}")
        print("【GP遗传规划搜索】")
        print(f"{'='*50}")
        print(f"📊 样本数: {len(X_valid)}")
        print(f"📊 特征数: {X_valid.shape[1]}")
        print(f"📊 种群大小: {self.config.population_size}")
        print(f"📊 进化代数: {self.config.generations}")
        
        # 创建模型
        self.model = SymbolicRegressor(
            population_size=self.config.population_size,
            generations=self.config.generations,
            tournament_size=self.config.tournament_size,
            stopping_criteria=self.config.stopping_criteria,
            
            p_crossover=self.config.p_crossover,
            p_subtree_mutation=self.config.p_subtree_mutation,
            p_hoist_mutation=self.config.p_hoist_mutation,
            p_point_mutation=self.config.p_point_mutation,
            
            max_samples=self.config.max_samples,
            parsimony_coefficient=self.config.parsimony_coefficient,
            
            function_set=self._create_function_set(),
            metric='spearman',  # 使用Spearman相关作为适应度
            
            feature_names=feature_names,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
        )
        
        # 拟合
        self.model.fit(X_valid, y_valid)
        
        # 保存最优程序
        self.best_programs = self._extract_best_programs()
        
        return self
    
    def _extract_best_programs(self, top_n: int = 10) -> List[Dict]:
        """提取最优程序"""
        if self.model is None:
            return []
        
        programs = []
        
        # 从Hall of Fame中提取
        if hasattr(self.model, '_programs'):
            hall_of_fame = sorted(
                [p for gen in self.model._programs for p in gen if p is not None],
                key=lambda p: p.fitness_,
                reverse=True
            )[:top_n]
            
            for i, prog in enumerate(hall_of_fame):
                programs.append({
                    'rank': i + 1,
                    'formula': str(prog),
                    'fitness': prog.fitness_,
                    'length': prog.length_,
                    'depth': prog.depth_,
                })
        
        return programs
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        pred = self.model.predict(X.values)
        return pd.Series(pred, index=X.index)
    
    def get_formula(self) -> str:
        """获取最优公式"""
        if self.model is None:
            return ""
        return str(self.model._program)
    
    def get_top_formulas(self, n: int = 5) -> List[str]:
        """获取Top N公式"""
        return [p['formula'] for p in self.best_programs[:n]]
    
    def print_summary(self):
        """打印摘要"""
        print(f"\n{'='*50}")
        print("【GP搜索结果】")
        print(f"{'='*50}")
        print(f"最优公式: {self.get_formula()}")
        print(f"\nTop 5 公式:")
        for i, prog in enumerate(self.best_programs[:5], 1):
            print(f"  {i}. {prog['formula']}")
            print(f"     适应度: {prog['fitness']:.4f}, 复杂度: {prog['length']}")
