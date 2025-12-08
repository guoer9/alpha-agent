"""
集成学习模块 (Ensemble / Alpha Synthesis)

功能:
1. Stacking / Blending - 多层模型融合
2. 权重优化 - 基于IC/Sharpe的动态权重
3. Alpha合成 - 生成综合信号
4. 集成策略 - 多种集成方法
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# 可选依赖
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from .config import OUTPUT_DIR


@dataclass
class EnsembleResult:
    """集成结果"""
    name: str
    method: str
    weights: Dict[str, float]
    # 性能指标
    ic: float = 0.0
    icir: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    # 相对单模型提升
    ic_improvement: float = 0.0
    sharpe_improvement: float = 0.0
    # 元信息
    n_models: int = 0
    timestamp: str = ""


class AlphaEnsemble:
    """Alpha因子集成器"""
    
    def __init__(
        self,
        method: str = "ic_weighted",
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        regularization: float = 0.1,
    ):
        """
        初始化集成器
        
        参数:
            method: 集成方法
                - "equal": 等权重
                - "ic_weighted": IC加权
                - "icir_weighted": ICIR加权
                - "sharpe_weighted": Sharpe加权
                - "optimize": 优化求解最优权重
                - "stacking": Stacking集成
                - "blending": Blending集成
            min_weight: 最小权重
            max_weight: 最大权重
            regularization: 正则化系数
        """
        self.method = method
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.regularization = regularization
        
        self.weights: Dict[str, float] = {}
        self.model_metrics: Dict[str, Dict] = {}
        self.meta_model = None
        self.result: EnsembleResult = None
    
    def fit(
        self,
        predictions: Dict[str, pd.Series],
        y_true: pd.Series,
        model_metrics: Dict[str, Dict] = None,
    ) -> 'AlphaEnsemble':
        """
        拟合集成器
        
        参数:
            predictions: 各模型预测 {model_name: prediction_series}
            y_true: 真实标签
            model_metrics: 模型指标 {model_name: {ic, icir, sharpe, ...}}
        """
        self.model_names = list(predictions.keys())
        self.n_models = len(self.model_names)
        self.model_metrics = model_metrics or {}
        
        print(f"\n{'='*60}")
        print(f"【Alpha集成】方法: {self.method}")
        print(f"{'='*60}")
        print(f"📊 模型数量: {self.n_models}")
        
        # 对齐数据
        pred_df = pd.DataFrame(predictions)
        valid_mask = pred_df.notna().all(axis=1) & y_true.notna()
        pred_df = pred_df[valid_mask]
        y_valid = y_true[valid_mask]
        
        # 根据方法计算权重
        if self.method == "equal":
            self.weights = self._equal_weights()
        elif self.method == "ic_weighted":
            self.weights = self._ic_weighted(pred_df, y_valid)
        elif self.method == "icir_weighted":
            self.weights = self._icir_weighted(pred_df, y_valid)
        elif self.method == "sharpe_weighted":
            self.weights = self._sharpe_weighted()
        elif self.method == "optimize":
            self.weights = self._optimize_weights(pred_df, y_valid)
        elif self.method == "stacking":
            self._fit_stacking(pred_df, y_valid)
        elif self.method == "blending":
            self._fit_blending(pred_df, y_valid)
        else:
            raise ValueError(f"未知方法: {self.method}")
        
        # 打印权重
        self._print_weights()
        
        return self
    
    def _equal_weights(self) -> Dict[str, float]:
        """等权重"""
        w = 1.0 / self.n_models
        return {name: w for name in self.model_names}
    
    def _ic_weighted(
        self, 
        pred_df: pd.DataFrame, 
        y_true: pd.Series
    ) -> Dict[str, float]:
        """IC加权"""
        ics = {}
        for col in pred_df.columns:
            ic = pred_df[col].corr(y_true, method='spearman')
            ics[col] = abs(ic) if not np.isnan(ic) else 0
        
        # 归一化
        total = sum(ics.values())
        if total > 0:
            weights = {k: v / total for k, v in ics.items()}
        else:
            weights = self._equal_weights()
        
        return self._clip_weights(weights)
    
    def _icir_weighted(
        self, 
        pred_df: pd.DataFrame, 
        y_true: pd.Series,
        window: int = 60
    ) -> Dict[str, float]:
        """ICIR加权"""
        icirs = {}
        
        for col in pred_df.columns:
            # 滚动IC
            rolling_ic = []
            for i in range(window, len(pred_df)):
                ic = pred_df[col].iloc[i-window:i].corr(
                    y_true.iloc[i-window:i], method='spearman'
                )
                if not np.isnan(ic):
                    rolling_ic.append(ic)
            
            if len(rolling_ic) > 0:
                ic_mean = np.mean(rolling_ic)
                ic_std = np.std(rolling_ic)
                icir = abs(ic_mean) / (ic_std + 1e-8)
            else:
                icir = 0
            
            icirs[col] = icir
        
        # 归一化
        total = sum(icirs.values())
        if total > 0:
            weights = {k: v / total for k, v in icirs.items()}
        else:
            weights = self._equal_weights()
        
        return self._clip_weights(weights)
    
    def _sharpe_weighted(self) -> Dict[str, float]:
        """Sharpe加权 (基于传入的metrics)"""
        sharpes = {}
        for name in self.model_names:
            metrics = self.model_metrics.get(name, {})
            sharpe = metrics.get('sharpe', 0)
            sharpes[name] = max(sharpe, 0)  # 只用正Sharpe
        
        total = sum(sharpes.values())
        if total > 0:
            weights = {k: v / total for k, v in sharpes.items()}
        else:
            weights = self._equal_weights()
        
        return self._clip_weights(weights)
    
    def _optimize_weights(
        self, 
        pred_df: pd.DataFrame, 
        y_true: pd.Series
    ) -> Dict[str, float]:
        """优化求解最优权重 (最大化IC)"""
        if not SCIPY_AVAILABLE:
            logger.warning("scipy未安装，使用IC加权替代")
            return self._ic_weighted(pred_df, y_true)
        
        X = pred_df.values
        y = y_true.values
        n_models = X.shape[1]
        
        # 目标函数: 负IC (最小化)
        def neg_ic(weights):
            ensemble_pred = X @ weights
            ic = np.corrcoef(ensemble_pred, y)[0, 1]
            # 添加正则化
            reg = self.regularization * np.sum(weights ** 2)
            return -abs(ic) + reg
        
        # 约束: 权重和为1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # 边界
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_models)]
        
        # 初始值: 等权重
        x0 = np.ones(n_models) / n_models
        
        # 优化
        result = minimize(
            neg_ic, x0, method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        
        if result.success:
            weights = {name: w for name, w in zip(self.model_names, result.x)}
        else:
            logger.warning("优化未收敛，使用IC加权")
            weights = self._ic_weighted(pred_df, y_true)
        
        return self._clip_weights(weights)
    
    def _fit_stacking(
        self, 
        pred_df: pd.DataFrame, 
        y_true: pd.Series
    ):
        """Stacking集成 - 使用元学习器"""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn未安装，使用IC加权替代")
            self.weights = self._ic_weighted(pred_df, y_true)
            return
        
        # 时序交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        oof_preds = np.zeros(len(pred_df))
        
        # 使用Ridge作为元学习器
        for train_idx, val_idx in tscv.split(pred_df):
            X_train = pred_df.iloc[train_idx].values
            y_train = y_true.iloc[train_idx].values
            X_val = pred_df.iloc[val_idx].values
            
            meta = Ridge(alpha=1.0)
            meta.fit(X_train, y_train)
            oof_preds[val_idx] = meta.predict(X_val)
        
        # 最终元学习器
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(pred_df.values, y_true.values)
        
        # 权重从系数获取
        coefs = self.meta_model.coef_
        coefs = np.maximum(coefs, 0)  # 限制非负
        total = coefs.sum()
        if total > 0:
            self.weights = {name: c / total for name, c in zip(self.model_names, coefs)}
        else:
            self.weights = self._equal_weights()
    
    def _fit_blending(
        self, 
        pred_df: pd.DataFrame, 
        y_true: pd.Series,
        holdout_ratio: float = 0.2
    ):
        """Blending集成 - 使用holdout集训练元模型"""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn未安装，使用IC加权替代")
            self.weights = self._ic_weighted(pred_df, y_true)
            return
        
        # 分割数据
        split_idx = int(len(pred_df) * (1 - holdout_ratio))
        X_train = pred_df.iloc[:split_idx].values
        y_train = y_true.iloc[:split_idx].values
        X_holdout = pred_df.iloc[split_idx:].values
        y_holdout = y_true.iloc[split_idx:].values
        
        # 在holdout上训练元模型
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(X_holdout, y_holdout)
        
        # 权重
        coefs = self.meta_model.coef_
        coefs = np.maximum(coefs, 0)
        total = coefs.sum()
        if total > 0:
            self.weights = {name: c / total for name, c in zip(self.model_names, coefs)}
        else:
            self.weights = self._equal_weights()
    
    def _clip_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """裁剪权重到范围内并归一化"""
        clipped = {k: np.clip(v, self.min_weight, self.max_weight) 
                   for k, v in weights.items()}
        total = sum(clipped.values())
        if total > 0:
            return {k: v / total for k, v in clipped.items()}
        return weights
    
    def _print_weights(self):
        """打印权重"""
        print("\n📊 集成权重:")
        print("-" * 40)
        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        for name, weight in sorted_weights:
            bar = "█" * int(weight * 20)
            print(f"  {name:20s}: {weight:.3f} {bar}")
    
    def predict(self, predictions: Dict[str, pd.Series]) -> pd.Series:
        """
        生成集成预测
        
        参数:
            predictions: 各模型预测
        
        返回:
            集成后的预测Series
        """
        if self.meta_model is not None:
            # Stacking/Blending使用元模型
            pred_df = pd.DataFrame(predictions)
            return pd.Series(
                self.meta_model.predict(pred_df.values),
                index=pred_df.index
            )
        
        # 加权平均
        result = None
        for name, weight in self.weights.items():
            if name in predictions:
                pred = predictions[name] * weight
                if result is None:
                    result = pred
                else:
                    result = result + pred
        
        return result
    
    def evaluate(
        self,
        predictions: Dict[str, pd.Series],
        y_true: pd.Series,
    ) -> EnsembleResult:
        """
        评估集成效果
        
        参数:
            predictions: 各模型预测
            y_true: 真实标签
        
        返回:
            EnsembleResult
        """
        # 生成集成预测
        ensemble_pred = self.predict(predictions)
        
        # 对齐
        valid_mask = ensemble_pred.notna() & y_true.notna()
        pred = ensemble_pred[valid_mask]
        y = y_true[valid_mask]
        
        # 计算指标
        ic = pred.corr(y, method='spearman')
        
        # 滚动ICIR
        rolling_ic = []
        window = 60
        for i in range(window, len(pred)):
            r_ic = pred.iloc[i-window:i].corr(y.iloc[i-window:i], method='spearman')
            if not np.isnan(r_ic):
                rolling_ic.append(r_ic)
        
        icir = np.mean(rolling_ic) / (np.std(rolling_ic) + 1e-8) if rolling_ic else 0
        
        # Sharpe (假设预测即收益)
        sharpe = np.sqrt(252) * pred.mean() / (pred.std() + 1e-8)
        
        # 最大回撤
        cumsum = pred.cumsum()
        running_max = cumsum.cummax()
        drawdown = running_max - cumsum
        max_drawdown = drawdown.max()
        
        # 对比单模型最优
        best_single_ic = 0
        best_single_sharpe = 0
        for name in self.model_names:
            if name in predictions:
                single_ic = predictions[name][valid_mask].corr(y, method='spearman')
                single_sharpe = np.sqrt(252) * predictions[name][valid_mask].mean() / (predictions[name][valid_mask].std() + 1e-8)
                best_single_ic = max(best_single_ic, abs(single_ic))
                best_single_sharpe = max(best_single_sharpe, single_sharpe)
        
        ic_improvement = (abs(ic) - best_single_ic) / (best_single_ic + 1e-8) * 100
        sharpe_improvement = (sharpe - best_single_sharpe) / (abs(best_single_sharpe) + 1e-8) * 100
        
        self.result = EnsembleResult(
            name=f"ensemble_{self.method}",
            method=self.method,
            weights=self.weights.copy(),
            ic=ic,
            icir=icir,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            ic_improvement=ic_improvement,
            sharpe_improvement=sharpe_improvement,
            n_models=self.n_models,
            timestamp=datetime.now().isoformat(),
        )
        
        return self.result
    
    def print_summary(self):
        """打印摘要"""
        if self.result is None:
            print("请先调用 evaluate() 方法")
            return
        
        r = self.result
        print(f"\n{'='*60}")
        print(f"【集成结果摘要】")
        print(f"{'='*60}")
        print(f"📊 方法: {r.method}")
        print(f"📊 模型数: {r.n_models}")
        print(f"\n📈 性能指标:")
        print(f"  IC: {r.ic:.4f}")
        print(f"  ICIR: {r.icir:.4f}")
        print(f"  Sharpe: {r.sharpe:.4f}")
        print(f"  最大回撤: {r.max_drawdown:.4f}")
        print(f"\n📊 相对单模型提升:")
        print(f"  IC提升: {r.ic_improvement:+.2f}%")
        print(f"  Sharpe提升: {r.sharpe_improvement:+.2f}%")
    
    def save_weights(self, path: Path = None):
        """保存权重"""
        if path is None:
            path = OUTPUT_DIR / f"ensemble_weights_{self.method}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "method": self.method,
            "weights": self.weights,
            "result": {
                "ic": self.result.ic if self.result else None,
                "icir": self.result.icir if self.result else None,
                "sharpe": self.result.sharpe if self.result else None,
            } if self.result else None
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"📁 权重已保存: {path}")
    
    @classmethod
    def load_weights(cls, path: Path) -> 'AlphaEnsemble':
        """加载权重"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ensemble = cls(method=data['method'])
        ensemble.weights = data['weights']
        ensemble.model_names = list(data['weights'].keys())
        ensemble.n_models = len(ensemble.model_names)
        
        return ensemble


class MultiMethodEnsemble:
    """多方法集成对比"""
    
    METHODS = [
        "equal",
        "ic_weighted", 
        "icir_weighted",
        "optimize",
        "stacking",
        "blending",
    ]
    
    def __init__(self, methods: List[str] = None):
        """初始化"""
        self.methods = methods or self.METHODS
        self.ensembles: Dict[str, AlphaEnsemble] = {}
        self.results: Dict[str, EnsembleResult] = {}
    
    def run(
        self,
        predictions: Dict[str, pd.Series],
        y_true: pd.Series,
        model_metrics: Dict[str, Dict] = None,
    ) -> pd.DataFrame:
        """
        运行多方法集成对比
        
        返回:
            方法对比表DataFrame
        """
        print("\n" + "="*70)
        print("【多方法集成对比】")
        print("="*70)
        
        for method in self.methods:
            print(f"\n▶ 测试方法: {method}")
            try:
                ensemble = AlphaEnsemble(method=method)
                ensemble.fit(predictions, y_true, model_metrics)
                result = ensemble.evaluate(predictions, y_true)
                
                self.ensembles[method] = ensemble
                self.results[method] = result
                
                print(f"  ✓ IC={result.ic:.4f}, ICIR={result.icir:.4f}")
            except Exception as e:
                print(f"  ✗ {str(e)[:50]}")
        
        # 生成对比表
        comparison = self._generate_comparison()
        self.print_summary()
        
        return comparison
    
    def _generate_comparison(self) -> pd.DataFrame:
        """生成对比表"""
        records = []
        for method, result in self.results.items():
            records.append({
                "方法": method,
                "IC": result.ic,
                "ICIR": result.icir,
                "Sharpe": result.sharpe,
                "最大回撤": result.max_drawdown,
                "IC提升%": result.ic_improvement,
                "Sharpe提升%": result.sharpe_improvement,
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values("ICIR", ascending=False)
        return df
    
    def get_best_method(self, metric: str = "icir") -> str:
        """获取最佳方法"""
        best = None
        best_value = -np.inf
        
        for method, result in self.results.items():
            value = getattr(result, metric, 0)
            if value > best_value:
                best_value = value
                best = method
        
        return best
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "="*60)
        print("【集成方法对比结果】")
        print("="*60)
        
        comparison = self._generate_comparison()
        print(comparison.to_string(index=False))
        
        best = self.get_best_method("icir")
        if best:
            print(f"\n🏆 最佳方法: {best}")
            print(f"  IC提升: {self.results[best].ic_improvement:+.2f}%")


# ===================== 便捷函数 =====================

def ensemble_alpha(
    predictions: Dict[str, pd.Series],
    y_true: pd.Series,
    method: str = "ic_weighted",
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    便捷函数: Alpha集成
    
    返回:
        ensemble_pred: 集成预测
        weights: 权重字典
    """
    ensemble = AlphaEnsemble(method=method)
    ensemble.fit(predictions, y_true)
    ensemble_pred = ensemble.predict(predictions)
    
    return ensemble_pred, ensemble.weights


def compare_ensemble_methods(
    predictions: Dict[str, pd.Series],
    y_true: pd.Series,
) -> Tuple[pd.DataFrame, str]:
    """
    便捷函数: 对比多种集成方法
    
    返回:
        comparison: 对比表
        best_method: 最佳方法
    """
    multi = MultiMethodEnsemble()
    comparison = multi.run(predictions, y_true)
    best = multi.get_best_method("icir")
    
    return comparison, best
