"""
特征选择与解释模块 (Feature Selection & XAI)

功能:
1. IC分析 - 信息系数计算
2. 共线性分析 - 相关性矩阵与VIF
3. 特征重要性 - SHAP/Permutation Importance
4. 自动特征筛选 - 综合评分过滤噪声特征
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
import json
from datetime import datetime

# 可选依赖
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .config import OUTPUT_DIR


@dataclass
class FeatureReport:
    """单个特征的解释性报告"""
    name: str
    ic: float
    ic_ir: float  # IC信息比率
    abs_ic: float
    importance_shap: float
    importance_perm: float
    importance_lgb: float
    correlation_max: float  # 与其他特征的最大相关性
    vif: float  # 方差膨胀因子
    stability: float  # IC稳定性 (滚动IC标准差)
    final_score: float  # 综合评分
    recommendation: str  # 保留/剔除/待定


class FeatureSelector:
    """特征选择与解释器"""
    
    def __init__(
        self,
        ic_threshold: float = 0.02,
        correlation_threshold: float = 0.85,
        vif_threshold: float = 10.0,
        min_importance: float = 0.01,
        top_k: int = None,
    ):
        """
        初始化特征选择器
        
        参数:
            ic_threshold: IC阈值，低于此值的特征被剔除
            correlation_threshold: 共线性阈值
            vif_threshold: VIF阈值
            min_importance: 最小重要性阈值
            top_k: 保留top K个特征 (None表示不限制)
        """
        self.ic_threshold = ic_threshold
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.min_importance = min_importance
        self.top_k = top_k
        
        self.feature_reports: Dict[str, FeatureReport] = {}
        self.correlation_matrix: pd.DataFrame = None
        self.selected_features: List[str] = []
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str] = None,
    ) -> 'FeatureSelector':
        """
        拟合特征选择器
        
        参数:
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称列表
        """
        if feature_names is None:
            feature_names = list(X.columns)
        
        print("\n" + "="*70)
        print("【特征选择与解释分析】")
        print("="*70)
        
        # 1. 计算IC
        print("\n📊 Step 1: 计算信息系数 (IC)...")
        ic_results = self._compute_ic(X[feature_names], y)
        
        # 2. 计算共线性
        print("📊 Step 2: 计算共线性矩阵...")
        self.correlation_matrix = self._compute_correlation(X[feature_names])
        vif_results = self._compute_vif(X[feature_names])
        
        # 3. 计算特征重要性
        print("📊 Step 3: 计算特征重要性 (SHAP/Permutation/LGB)...")
        importance_results = self._compute_importance(X[feature_names], y)
        
        # 4. 生成特征报告
        print("📊 Step 4: 生成特征报告...")
        self._generate_reports(
            feature_names, ic_results, vif_results, importance_results
        )
        
        # 5. 特征筛选
        print("📊 Step 5: 特征筛选...")
        self._select_features()
        
        return self
    
    def _compute_ic(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        rolling_window: int = 60
    ) -> Dict[str, Dict]:
        """计算IC及相关指标"""
        results = {}
        
        for col in X.columns:
            valid_mask = X[col].notna() & y.notna()
            if valid_mask.sum() < 30:
                results[col] = {'ic': 0, 'ic_ir': 0, 'stability': 0}
                continue
            
            x_valid = X.loc[valid_mask, col]
            y_valid = y[valid_mask]
            
            # 整体IC
            ic = x_valid.corr(y_valid, method='spearman')
            
            # 滚动IC
            rolling_ic = []
            for i in range(rolling_window, len(x_valid)):
                window_x = x_valid.iloc[i-rolling_window:i]
                window_y = y_valid.iloc[i-rolling_window:i]
                if len(window_x) >= 20:
                    rolling_ic.append(window_x.corr(window_y, method='spearman'))
            
            rolling_ic = pd.Series(rolling_ic)
            ic_mean = rolling_ic.mean() if len(rolling_ic) > 0 else ic
            ic_std = rolling_ic.std() if len(rolling_ic) > 0 else 0.1
            ic_ir = ic_mean / (ic_std + 1e-8)  # IC信息比率
            stability = 1 - ic_std  # 稳定性
            
            results[col] = {
                'ic': ic if not np.isnan(ic) else 0,
                'ic_ir': ic_ir if not np.isnan(ic_ir) else 0,
                'stability': stability if not np.isnan(stability) else 0,
            }
        
        return results
    
    def _compute_correlation(self, X: pd.DataFrame) -> pd.DataFrame:
        """计算相关性矩阵"""
        return X.corr(method='spearman')
    
    def _compute_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        """计算方差膨胀因子 (VIF)"""
        if not SKLEARN_AVAILABLE:
            return {col: 1.0 for col in X.columns}
        
        vif_results = {}
        X_clean = X.dropna()
        
        if len(X_clean) < 100:
            return {col: 1.0 for col in X.columns}
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        for col in X.columns:
            try:
                other_cols = [c for c in X.columns if c != col]
                if len(other_cols) == 0:
                    vif_results[col] = 1.0
                    continue
                
                # 用其他特征预测当前特征
                y_col = X_scaled[col]
                X_other = X_scaled[other_cols]
                
                # 简化计算: 使用R²
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_other, y_col)
                r_squared = model.score(X_other, y_col)
                
                vif = 1 / (1 - r_squared + 1e-8)
                vif_results[col] = min(vif, 100)  # 限制最大值
            except Exception:
                vif_results[col] = 1.0
        
        return vif_results
    
    def _compute_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Dict]:
        """计算多种特征重要性"""
        results = {col: {'shap': 0, 'perm': 0, 'lgb': 0} for col in X.columns}
        
        # 准备数据
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        if len(X_clean) < 100:
            return results
        
        # 填充剩余NaN
        X_clean = X_clean.fillna(X_clean.median())
        
        # 1. LightGBM 特征重要性
        if LGB_AVAILABLE:
            try:
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    verbose=-1,
                    n_jobs=-1
                )
                model.fit(X_clean, y_clean)
                importance = model.feature_importances_
                importance = importance / (importance.sum() + 1e-8)
                
                for i, col in enumerate(X_clean.columns):
                    results[col]['lgb'] = importance[i]
            except Exception as e:
                print(f"  LGB importance failed: {e}")
        
        # 2. Permutation Importance
        if SKLEARN_AVAILABLE:
            try:
                # 使用简单RF
                rf = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1)
                rf.fit(X_clean, y_clean)
                
                perm_result = permutation_importance(
                    rf, X_clean, y_clean, 
                    n_repeats=5, random_state=42, n_jobs=-1
                )
                importance = perm_result.importances_mean
                importance = np.maximum(importance, 0)
                importance = importance / (importance.sum() + 1e-8)
                
                for i, col in enumerate(X_clean.columns):
                    results[col]['perm'] = importance[i]
            except Exception as e:
                print(f"  Permutation importance failed: {e}")
        
        # 3. SHAP (仅在特征数量合理时)
        if SHAP_AVAILABLE and LGB_AVAILABLE and len(X_clean.columns) <= 50:
            try:
                model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, verbose=-1)
                model.fit(X_clean, y_clean)
                
                # 采样以加速
                sample_size = min(500, len(X_clean))
                X_sample = X_clean.sample(sample_size, random_state=42)
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                importance = np.abs(shap_values).mean(axis=0)
                importance = importance / (importance.sum() + 1e-8)
                
                for i, col in enumerate(X_clean.columns):
                    results[col]['shap'] = importance[i]
            except Exception as e:
                print(f"  SHAP failed: {e}")
        
        return results
    
    def _generate_reports(
        self,
        feature_names: List[str],
        ic_results: Dict,
        vif_results: Dict,
        importance_results: Dict,
    ):
        """生成特征报告"""
        for name in feature_names:
            ic_data = ic_results.get(name, {})
            imp_data = importance_results.get(name, {})
            
            ic = ic_data.get('ic', 0)
            ic_ir = ic_data.get('ic_ir', 0)
            stability = ic_data.get('stability', 0)
            
            # 计算与其他特征的最大相关性
            if self.correlation_matrix is not None and name in self.correlation_matrix.columns:
                corr_row = self.correlation_matrix[name].drop(name, errors='ignore')
                max_corr = corr_row.abs().max() if len(corr_row) > 0 else 0
            else:
                max_corr = 0
            
            # 综合评分
            score = self._compute_final_score(
                ic=ic,
                ic_ir=ic_ir,
                stability=stability,
                importance_shap=imp_data.get('shap', 0),
                importance_lgb=imp_data.get('lgb', 0),
                max_corr=max_corr,
                vif=vif_results.get(name, 1)
            )
            
            # 推荐
            recommendation = self._get_recommendation(
                ic, max_corr, vif_results.get(name, 1), score
            )
            
            self.feature_reports[name] = FeatureReport(
                name=name,
                ic=ic,
                ic_ir=ic_ir,
                abs_ic=abs(ic),
                importance_shap=imp_data.get('shap', 0),
                importance_perm=imp_data.get('perm', 0),
                importance_lgb=imp_data.get('lgb', 0),
                correlation_max=max_corr,
                vif=vif_results.get(name, 1),
                stability=stability,
                final_score=score,
                recommendation=recommendation
            )
    
    def _compute_final_score(
        self,
        ic: float,
        ic_ir: float,
        stability: float,
        importance_shap: float,
        importance_lgb: float,
        max_corr: float,
        vif: float,
    ) -> float:
        """计算综合评分"""
        # 权重配置
        weights = {
            'ic': 0.25,
            'ic_ir': 0.15,
            'stability': 0.10,
            'importance': 0.30,
            'independence': 0.20,
        }
        
        # IC分数 (绝对值)
        ic_score = min(abs(ic) / 0.1, 1.0)
        
        # IC_IR分数
        ic_ir_score = min(abs(ic_ir) / 1.0, 1.0)
        
        # 稳定性分数
        stability_score = max(0, min(stability, 1.0))
        
        # 重要性分数 (取多种方法的平均)
        importance_score = (importance_shap + importance_lgb) / 2 * 10  # 放大
        importance_score = min(importance_score, 1.0)
        
        # 独立性分数 (相关性和VIF)
        corr_penalty = max(0, 1 - max_corr)
        vif_penalty = max(0, 1 - (vif - 1) / 10)
        independence_score = (corr_penalty + vif_penalty) / 2
        
        # 综合分数
        final_score = (
            weights['ic'] * ic_score +
            weights['ic_ir'] * ic_ir_score +
            weights['stability'] * stability_score +
            weights['importance'] * importance_score +
            weights['independence'] * independence_score
        )
        
        return final_score
    
    def _get_recommendation(
        self,
        ic: float,
        max_corr: float,
        vif: float,
        score: float,
    ) -> str:
        """生成推荐"""
        if abs(ic) < self.ic_threshold:
            return "剔除 (IC过低)"
        if max_corr > self.correlation_threshold:
            return "待定 (高共线性)"
        if vif > self.vif_threshold:
            return "待定 (VIF过高)"
        if score > 0.5:
            return "保留 (优质特征)"
        if score > 0.3:
            return "保留 (可用特征)"
        return "待定 (边缘特征)"
    
    def _select_features(self):
        """执行特征筛选"""
        # 按综合评分排序
        sorted_reports = sorted(
            self.feature_reports.values(),
            key=lambda x: x.final_score,
            reverse=True
        )
        
        selected = []
        selected_set = set()
        
        for report in sorted_reports:
            # 跳过明确剔除的
            if "剔除" in report.recommendation:
                continue
            
            # 检查与已选特征的相关性
            too_correlated = False
            if self.correlation_matrix is not None:
                for selected_name in selected_set:
                    if selected_name in self.correlation_matrix.columns:
                        corr = abs(self.correlation_matrix.loc[report.name, selected_name])
                        if corr > self.correlation_threshold:
                            too_correlated = True
                            break
            
            if too_correlated:
                continue
            
            selected.append(report.name)
            selected_set.add(report.name)
            
            # top_k限制
            if self.top_k and len(selected) >= self.top_k:
                break
        
        self.selected_features = selected
        
        print(f"\n✅ 筛选结果: {len(selected)}/{len(self.feature_reports)} 个特征")
    
    def get_selected_features(self) -> List[str]:
        """获取筛选后的特征列表"""
        return self.selected_features
    
    def get_feature_report(self, name: str) -> Optional[FeatureReport]:
        """获取单个特征的报告"""
        return self.feature_reports.get(name)
    
    def get_all_reports(self) -> pd.DataFrame:
        """获取所有特征报告的DataFrame"""
        records = []
        for report in self.feature_reports.values():
            records.append({
                '特征': report.name,
                'IC': f"{report.ic:.4f}",
                'IC_IR': f"{report.ic_ir:.2f}",
                '|IC|': f"{report.abs_ic:.4f}",
                'SHAP': f"{report.importance_shap:.3f}",
                'Perm': f"{report.importance_perm:.3f}",
                'LGB': f"{report.importance_lgb:.3f}",
                '最大相关': f"{report.correlation_max:.2f}",
                'VIF': f"{report.vif:.1f}",
                '稳定性': f"{report.stability:.2f}",
                '综合分': f"{report.final_score:.3f}",
                '推荐': report.recommendation,
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values('综合分', ascending=False)
        return df
    
    def print_summary(self, top_n: int = 20):
        """打印摘要"""
        print("\n" + "="*70)
        print("【特征筛选结果摘要】")
        print("="*70)
        
        df = self.get_all_reports()
        
        print(f"\n📊 总特征数: {len(df)}")
        print(f"📊 保留特征: {len(self.selected_features)}")
        print(f"📊 剔除特征: {len(df) - len(self.selected_features)}")
        
        print(f"\n🏆 Top {top_n} 特征:")
        print("-"*70)
        print(df.head(top_n).to_string(index=False))
        
        print(f"\n✅ 最终选择的特征:")
        for i, name in enumerate(self.selected_features[:20], 1):
            report = self.feature_reports[name]
            print(f"  {i:2d}. {name}: IC={report.ic:.4f}, Score={report.final_score:.3f}")
        
        if len(self.selected_features) > 20:
            print(f"  ... 共 {len(self.selected_features)} 个")
    
    def save_report(self, path: Path = None):
        """保存完整报告"""
        if path is None:
            path = OUTPUT_DIR / 'feature_selection_report.json'
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'ic_threshold': self.ic_threshold,
                'correlation_threshold': self.correlation_threshold,
                'vif_threshold': self.vif_threshold,
                'top_k': self.top_k,
            },
            'summary': {
                'total_features': len(self.feature_reports),
                'selected_features': len(self.selected_features),
            },
            'selected_features': self.selected_features,
            'feature_reports': {
                name: {
                    'ic': report.ic,
                    'ic_ir': report.ic_ir,
                    'importance_shap': report.importance_shap,
                    'importance_lgb': report.importance_lgb,
                    'correlation_max': report.correlation_max,
                    'vif': report.vif,
                    'final_score': report.final_score,
                    'recommendation': report.recommendation,
                }
                for name, report in self.feature_reports.items()
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存CSV
        csv_path = path.with_suffix('.csv')
        self.get_all_reports().to_csv(csv_path, index=False)
        
        print(f"\n📁 报告已保存: {path}")
        print(f"📁 CSV已保存: {csv_path}")


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    ic_threshold: float = 0.02,
    correlation_threshold: float = 0.85,
    top_k: int = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    便捷函数: 特征选择
    
    返回:
        selected_features: 筛选后的特征列表
        report_df: 特征报告DataFrame
    """
    selector = FeatureSelector(
        ic_threshold=ic_threshold,
        correlation_threshold=correlation_threshold,
        top_k=top_k,
    )
    selector.fit(X, y)
    selector.print_summary()
    selector.save_report()
    
    return selector.get_selected_features(), selector.get_all_reports()
