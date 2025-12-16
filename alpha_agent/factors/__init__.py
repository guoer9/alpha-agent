"""
大型因子库 - 整合多个权威因子集合

包含:
- Barra风格因子 (CNE5/CNE6) - MSCI Barra
- Fama-French因子 - 学术经典
- 技术分析因子 - 经典技术指标
- 量价因子 - 微观结构
- 基本面因子 - 财务指标
- Alpha158 - Microsoft Qlib
- Alpha360 - Microsoft Qlib (扩展版)
- WorldQuant 101 - Kakushadze学术论文
- 国泰君安 191 - 短周期量价因子
- Academic Premia - 学术溢价因子

总计: 300+ 因子
"""

from .classic_factors import (
    BARRA_FACTORS,
    TECHNICAL_FACTORS,
    FUNDAMENTAL_FACTORS,
    VOLUME_PRICE_FACTORS,
    ALL_CLASSIC_FACTORS,
    ClassicFactor,
    FactorCategory,
)

from .alpha158 import ALPHA158_FACTORS, get_alpha158_factors
from .alpha360 import ALPHA360_FACTORS, get_alpha360_factors
from .worldquant101 import WORLDQUANT_101_FACTORS, get_worldquant101_factors
from .gtja191 import GTJA191_FACTORS, get_gtja191_factors
from .academic_premia import ACADEMIC_PREMIA_FACTORS, get_academic_premia_factors

from .factor_library import FactorLibrary, create_factor_library

# 汇总所有因子
ALL_FACTORS = (
    ALL_CLASSIC_FACTORS + 
    ALPHA158_FACTORS + 
    ALPHA360_FACTORS + 
    WORLDQUANT_101_FACTORS +
    GTJA191_FACTORS +
    ACADEMIC_PREMIA_FACTORS
)

__all__ = [
    # 经典因子
    'BARRA_FACTORS',
    'TECHNICAL_FACTORS', 
    'FUNDAMENTAL_FACTORS',
    'VOLUME_PRICE_FACTORS',
    'ALL_CLASSIC_FACTORS',
    # Qlib因子
    'ALPHA158_FACTORS',
    'ALPHA360_FACTORS',
    # WorldQuant
    'WORLDQUANT_101_FACTORS',
    # 国泰君安
    'GTJA191_FACTORS',
    # 学术溢价
    'ACADEMIC_PREMIA_FACTORS',
    # 汇总
    'ALL_FACTORS',
    # 类
    'ClassicFactor',
    'FactorCategory',
    'FactorLibrary',
    'create_factor_library',
    # 函数
    'get_alpha158_factors',
    'get_alpha360_factors',
    'get_worldquant101_factors',
    'get_gtja191_factors',
    'get_academic_premia_factors',
]
