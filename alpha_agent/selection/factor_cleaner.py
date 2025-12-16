"""
因子代码清洗模块

功能：
1. 清洗因子代码格式（移除不必要的import等）
2. 适配字段引用（将不存在的字段映射到可用字段）
3. 标准化代码结构
4. 保留所有因子，不做过滤

设计理念：
- 大模型生成的因子代码可能格式不统一
- 沙箱环境已预置 numpy, pandas 等库
- 通过清洗而非过滤来保留所有因子
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# 配置常量
# ============================================================

# 沙箱已预置的模块，这些import可以安全移除
SANDBOX_PRELOADED_MODULES = {
    'numpy', 'np',
    'pandas', 'pd', 
    'math',
    'scipy',
}

# 字段别名映射：将常见别名映射到标准字段名
FIELD_ALIASES = {
    # 价格相关
    'Close': 'close',
    'CLOSE': 'close',
    'Open': 'open', 
    'OPEN': 'open',
    'High': 'high',
    'HIGH': 'high',
    'Low': 'low',
    'LOW': 'low',
    # 成交量相关
    'Volume': 'volume',
    'VOLUME': 'volume',
    'vol': 'volume',
    'Vol': 'volume',
    'VOL': 'volume',
    # 成交额
    'Amount': 'amount',
    'AMOUNT': 'amount',
    'amt': 'amount',
    # 换手率
    'Turnover': 'turnover',
    'TURNOVER': 'turnover',
    'turn': 'turnover',
    'Turn': 'turnover',
    # 收益率
    'ret': 'returns',
    'Ret': 'returns',
    'RET': 'returns',
    'Return': 'returns',
    'RETURN': 'returns',
    # VWAP
    'VWAP': 'vwap',
    'Vwap': 'vwap',
}

# 需要动态计算的派生字段
DERIVED_FIELDS = {
    'market_cap',      # 市值 = close * volume * 100
    'market_ret',      # 市场收益 = 所有股票平均收益
    'returns',         # 日收益率 = close.pct_change()
    'amount',          # 成交额 = close * volume
    'amplitude',       # 振幅 = (high - low) / close.shift(1)
    'turnover',        # 换手率 = turn
    'adj_factor',      # 复权因子
}


# ============================================================
# 清洗统计
# ============================================================

@dataclass
class CleaningStats:
    """清洗统计信息"""
    total_factors: int = 0
    imports_removed: int = 0
    fields_adapted: int = 0
    code_reformatted: int = 0
    
    def summary(self) -> str:
        return (
            f"清洗统计: 共{self.total_factors}个因子, "
            f"移除{self.imports_removed}个import, "
            f"适配{self.fields_adapted}个字段, "
            f"格式化{self.code_reformatted}个代码"
        )


# ============================================================
# 代码清洗函数
# ============================================================

def remove_safe_imports(code: str) -> tuple[str, int]:
    """
    移除沙箱已预置的import语句
    
    Args:
        code: 原始代码
        
    Returns:
        (清洗后代码, 移除的import数量)
    """
    if not code:
        return code, 0
    
    lines = code.split('\n')
    cleaned_lines = []
    removed_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        # 检查是否是可移除的import
        if stripped.startswith(('import ', 'from ')):
            # 检查是否引用预置模块
            is_preloaded = any(
                mod in stripped 
                for mod in SANDBOX_PRELOADED_MODULES
            )
            if is_preloaded:
                removed_count += 1
                continue  # 跳过这行
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines), removed_count


def adapt_field_references(code: str) -> tuple[str, int]:
    """
    适配字段引用，将别名转换为标准字段名
    
    Args:
        code: 原始代码
        
    Returns:
        (适配后代码, 适配的字段数量)
    """
    if not code:
        return code, 0
    
    adapted_count = 0
    result = code
    
    for alias, standard in FIELD_ALIASES.items():
        # 匹配 df['alias'] 或 df["alias"] 格式
        patterns = [
            (f"['{alias}']", f"['{standard}']"),
            (f'["{alias}"]', f'["{standard}"]'),
        ]
        
        for old, new in patterns:
            if old in result:
                result = result.replace(old, new)
                adapted_count += 1
    
    return result, adapted_count


def ensure_function_wrapper(code: str) -> str:
    """
    确保代码有compute_alpha函数包装
    
    如果代码是裸表达式，则包装成函数
    """
    if not code:
        return code
    
    # 检查是否已有函数定义
    if 'def compute_alpha' in code or 'def alpha' in code:
        return code
    
    # 检查是否是简单表达式（没有def）
    if 'def ' not in code:
        # 尝试包装成函数
        lines = code.strip().split('\n')
        
        # 如果只有一行且像表达式
        if len(lines) == 1 and not lines[0].startswith('#'):
            expr = lines[0].strip()
            return f'''
def compute_alpha(df):
    """自动包装的因子表达式"""
    return {expr}
'''
    
    return code


def clean_factor_code(code: str) -> tuple[str, Dict[str, int]]:
    """
    清洗单个因子代码
    
    Args:
        code: 原始因子代码
        
    Returns:
        (清洗后代码, 清洗统计字典)
    """
    stats = {
        'imports_removed': 0,
        'fields_adapted': 0,
        'reformatted': False,
    }
    
    if not code:
        return code, stats
    
    result = code
    
    # 1. 移除预置模块的import
    result, removed = remove_safe_imports(result)
    stats['imports_removed'] = removed
    
    # 2. 适配字段引用
    result, adapted = adapt_field_references(result)
    stats['fields_adapted'] = adapted
    
    # 3. 确保有函数包装
    original = result
    result = ensure_function_wrapper(result)
    if result != original:
        stats['reformatted'] = True
    
    return result, stats


# ============================================================
# 字段检查
# ============================================================

# Qlib默认可用的技术指标字段
QLIB_AVAILABLE_FIELDS = {
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'change', 'factor',  # 复权因子
}

# 基本面字段（Qlib默认数据中通常不包含）
FUNDAMENTAL_FIELDS = {
    'pe_ttm', 'pb', 'ps_ttm', 'pcf_ttm',  # 估值
    'roe_ttm', 'roa_ttm', 'gross_margin',  # 盈利能力
    'debt_ratio', 'current_ratio',  # 偿债能力
    'revenue_yoy', 'net_profit_yoy', 'total_assets_yoy',  # 成长性
    'net_profit', 'total_assets', 'total_equity',  # 规模
    'ocf', 'dividend_yield',  # 现金流/股息
}


def extract_used_fields(code: str) -> set:
    """
    从因子代码中提取使用的字段名
    
    Args:
        code: 因子代码
        
    Returns:
        使用的字段名集合
    """
    import re
    fields = set()
    # 匹配 df['xxx'] 模式
    fields.update(re.findall(r"df\[['\"]([\w]+)['\"]\]", code))
    # 匹配 df.xxx 模式 (排除方法调用)
    for match in re.findall(r"df\.(\w+)", code):
        if match not in {'rolling', 'shift', 'pct_change', 'diff', 'rank', 
                         'mean', 'std', 'min', 'max', 'sum', 'corr', 'cov',
                         'apply', 'groupby', 'fillna', 'dropna', 'copy'}:
            fields.add(match)
    return fields


def check_fields_available(
    code: str, 
    available_columns: Optional[List[str]] = None,
) -> tuple[bool, set]:
    """
    检查因子代码中使用的字段是否可用
    
    Args:
        code: 因子代码
        available_columns: 可用列列表
        
    Returns:
        (是否全部可用, 不可用字段集合)
    """
    used_fields = extract_used_fields(code)
    
    if available_columns:
        available = set(available_columns)
    else:
        # 使用默认可用字段 + 派生字段
        available = QLIB_AVAILABLE_FIELDS | DERIVED_FIELDS
    
    unavailable = used_fields - available
    # 移除可能的别名（在FIELD_ALIASES中的会被转换）
    for alias in list(unavailable):
        if alias in FIELD_ALIASES:
            unavailable.discard(alias)
    
    return len(unavailable) == 0, unavailable


# ============================================================
# 批量清洗
# ============================================================

def clean_factors(
    factors: List[Dict],
    available_columns: Optional[List[str]] = None,
    verbose: bool = True,
    filter_unavailable: bool = False,  # 默认不过滤，保留所有因子
) -> List[Dict]:
    """
    批量清洗因子列表
    
    Args:
        factors: 因子列表，每个因子是包含'code'键的字典
        available_columns: 可用的数据列
        verbose: 是否输出详细日志
        filter_unavailable: 是否过滤掉使用不可用字段的因子
        
    Returns:
        清洗后的因子列表
    """
    if not factors:
        return factors
    
    cleaned = []
    filtered_count = 0
    total_stats = CleaningStats(total_factors=len(factors))
    
    for factor in factors:
        factor_copy = factor.copy()
        original_code = factor_copy.get('code', '')
        
        # 清洗代码
        cleaned_code, stats = clean_factor_code(original_code)
        
        # 检查字段可用性
        if filter_unavailable and cleaned_code:
            is_available, unavailable_fields = check_fields_available(
                cleaned_code, available_columns
            )
            if not is_available:
                filtered_count += 1
                factor_name = factor_copy.get('name', factor_copy.get('id', 'unknown'))
                logger.debug(f"过滤因子 {factor_name}: 使用不可用字段 {unavailable_fields}")
                continue
        
        # 更新统计
        total_stats.imports_removed += stats['imports_removed']
        total_stats.fields_adapted += stats['fields_adapted']
        if stats['reformatted']:
            total_stats.code_reformatted += 1
        
        factor_copy['code'] = cleaned_code
        factor_copy['_cleaned'] = True  # 标记已清洗
        cleaned.append(factor_copy)
    
    # 输出统计
    if verbose:
        msg = total_stats.summary()
        if filtered_count > 0:
            msg += f", 过滤{filtered_count}个不兼容因子"
        if total_stats.imports_removed > 0 or total_stats.fields_adapted > 0 or \
           total_stats.code_reformatted > 0 or filtered_count > 0:
            logger.info(msg)
    
    return cleaned


# ============================================================
# 因子清洗器类（面向对象接口）
# ============================================================

class FactorCleaner:
    """
    因子代码清洗器
    
    提供面向对象的清洗接口，支持自定义配置
    
    Example:
        >>> cleaner = FactorCleaner()
        >>> cleaned_factors = cleaner.clean(factors)
        >>> print(cleaner.stats.summary())
    """
    
    def __init__(
        self,
        remove_imports: bool = True,
        adapt_fields: bool = True,
        ensure_wrapper: bool = True,
        custom_aliases: Optional[Dict[str, str]] = None,
    ):
        """
        初始化清洗器
        
        Args:
            remove_imports: 是否移除预置模块的import
            adapt_fields: 是否适配字段别名
            ensure_wrapper: 是否确保函数包装
            custom_aliases: 自定义字段别名映射
        """
        self.remove_imports = remove_imports
        self.adapt_fields = adapt_fields
        self.ensure_wrapper = ensure_wrapper
        self.stats = CleaningStats()
        
        # 合并自定义别名
        self.field_aliases = FIELD_ALIASES.copy()
        if custom_aliases:
            self.field_aliases.update(custom_aliases)
    
    def clean_code(self, code: str) -> str:
        """清洗单个因子代码"""
        if not code:
            return code
        
        result = code
        
        if self.remove_imports:
            result, removed = remove_safe_imports(result)
            self.stats.imports_removed += removed
        
        if self.adapt_fields:
            result, adapted = adapt_field_references(result)
            self.stats.fields_adapted += adapted
        
        if self.ensure_wrapper:
            original = result
            result = ensure_function_wrapper(result)
            if result != original:
                self.stats.code_reformatted += 1
        
        return result
    
    def clean(self, factors: List[Dict]) -> List[Dict]:
        """
        批量清洗因子
        
        Args:
            factors: 因子列表
            
        Returns:
            清洗后的因子列表
        """
        self.stats = CleaningStats(total_factors=len(factors))
        
        cleaned = []
        for factor in factors:
            factor_copy = factor.copy()
            factor_copy['code'] = self.clean_code(factor_copy.get('code', ''))
            factor_copy['_cleaned'] = True
            cleaned.append(factor_copy)
        
        return cleaned
    
    def get_stats(self) -> CleaningStats:
        """获取清洗统计"""
        return self.stats


# add_derived_fields 已移动到 data_preprocessor.py
