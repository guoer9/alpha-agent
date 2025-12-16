"""
建模模块

- feature_selector: 特征选择 (SHAP/IC/VIF)
- ensemble: 集成学习
- qlib_model_zoo: Qlib模型库
"""

from .feature_selector import FeatureSelector, select_features
from .ensemble import AlphaEnsemble, ensemble_alpha

try:
    from .qlib_model_zoo import QlibModelZoo, QlibBenchmark
    QLIB_ZOO_AVAILABLE = True
except ImportError:
    QLIB_ZOO_AVAILABLE = False

__all__ = [
    'FeatureSelector', 'select_features',
    'AlphaEnsemble', 'ensemble_alpha',
    'QlibModelZoo', 'QlibBenchmark',
    'QLIB_ZOO_AVAILABLE',
]
