"""
因子筛选模块

包含：
- FactorSelector: 因子筛选器（IC筛选、去重、聚类、正交化）
- FactorWrapper: 因子封装器（代码执行、批量计算）
- FactorCleaner: 因子代码清洗器（格式化、字段适配）
"""

from .selector import (
    FactorSelector,
    SelectionResult,
    select_factors,
    quick_filter,
    orthogonal_select,
)
from .factor_wrapper import (
    FactorWrapper,
    FactorMeta,
    load_factors,
    create_factor_wrapper,
)
from .factor_cleaner import (
    FactorCleaner,
    CleaningStats,
    clean_factor_code,
    clean_factors,
    adapt_field_references,
    remove_safe_imports,
    extract_used_fields,
    check_fields_available,
    FIELD_ALIASES,
)
from .data_preprocessor import (
    add_derived_fields,
    prepare_train_test_data,
    split_by_date,
    handle_missing_values,
    QLIB_AVAILABLE_FIELDS,
    DERIVED_FIELDS,
)

__all__ = [
    # 筛选器
    'FactorSelector',
    'SelectionResult',
    'select_factors',
    'quick_filter',
    'orthogonal_select',
    # 因子封装
    'FactorWrapper',
    'FactorMeta',
    'load_factors',
    'create_factor_wrapper',
    # 清洗器
    'FactorCleaner',
    'CleaningStats',
    'clean_factor_code',
    'clean_factors',
    'adapt_field_references',
    'remove_safe_imports',
    # 数据预处理
    'add_derived_fields',
    'prepare_train_test_data',
    'split_by_date',
    'handle_missing_values',
    # 常量
    'FIELD_ALIASES',
    'DERIVED_FIELDS',
    'QLIB_AVAILABLE_FIELDS',
]
