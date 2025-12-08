"""
因子封装器 - 将筛选出的因子封装为可回测的格式

功能:
1. 因子代码执行与值计算
2. 转换为Qlib特征格式
3. 批量因子计算
4. 因子持久化存储

使用示例:
    # 从筛选结果创建
    wrapper = FactorWrapper.from_selection_result(result)
    
    # 计算因子值
    factor_df = wrapper.compute(data)
    
    # 导出为Qlib格式
    wrapper.to_qlib_handler(save_path)
"""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorMeta:
    """因子元数据"""
    id: str
    name: str
    name_en: str = ""
    code: str = ""
    description: str = ""
    category: str = ""
    source: str = ""
    
    # 评估指标
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    
    # 状态
    is_valid: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FactorMeta":
        return cls(
            id=data.get('id', data.get('factor_id', '')),
            name=data.get('name', ''),
            name_en=data.get('name_en', ''),
            code=data.get('code', ''),
            description=data.get('description', ''),
            category=data.get('category', ''),
            source=data.get('source', ''),
            ic=float(data.get('ic', 0)),
            icir=float(data.get('icir', 0)),
            rank_ic=float(data.get('rank_ic', 0)),
            is_valid=data.get('is_valid', True),
        )


class FactorWrapper:
    """
    因子封装器
    
    将筛选出的因子封装为统一接口，支持:
    - 因子值计算
    - Qlib集成
    - 批量处理
    - 持久化
    """
    
    def __init__(self, factors: List[FactorMeta] = None):
        self.factors: List[FactorMeta] = factors or []
        self._executor: Optional[Callable] = None
        self._cache: Dict[str, pd.Series] = {}
    
    # ============================================================
    # 创建方法
    # ============================================================
    
    @classmethod
    def from_selection_result(cls, result) -> "FactorWrapper":
        """从SelectionResult创建"""
        factors = []
        for f in result.selected_factors:
            meta = FactorMeta.from_dict(f)
            factors.append(meta)
        
        wrapper = cls(factors)
        logger.info(f"从筛选结果创建FactorWrapper: {len(factors)} 个因子")
        return wrapper
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "FactorWrapper":
        """从JSON文件加载"""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        factors = [FactorMeta.from_dict(d) for d in data.get('factors', data)]
        wrapper = cls(factors)
        logger.info(f"从 {path} 加载 {len(factors)} 个因子")
        return wrapper
    
    @classmethod
    def from_dict_list(cls, factor_list: List[Dict]) -> "FactorWrapper":
        """从字典列表创建"""
        factors = [FactorMeta.from_dict(d) for d in factor_list]
        return cls(factors)
    
    # ============================================================
    # 因子计算
    # ============================================================
    
    def set_executor(self, executor: Callable):
        """设置因子执行器"""
        self._executor = executor
    
    def compute_single(
        self,
        factor: FactorMeta,
        data: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """
        计算单个因子值
        
        Args:
            factor: 因子元数据
            data: 输入数据 (需包含 close, open, high, low, volume 等列)
        
        Returns:
            因子值Series，索引与data相同
        """
        if not factor.code:
            return None
        
        try:
            # 使用自定义执行器
            if self._executor:
                values = self._executor(factor.code, data)
                if isinstance(values, pd.Series):
                    return values
                return pd.Series(values, index=data.index)
            
            # 默认执行器
            return self._execute_factor_code(factor.code, data)
            
        except Exception as e:
            logger.warning(f"计算因子 {factor.name} 失败: {e}")
            return None
    
    def compute(
        self,
        data: pd.DataFrame,
        n_workers: int = 4,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        批量计算所有因子值
        
        Args:
            data: 输入数据
            n_workers: 并行工作线程数
            use_cache: 是否使用缓存
        
        Returns:
            DataFrame，列为因子名，索引与data相同
        """
        if not self.factors:
            return pd.DataFrame(index=data.index)
        
        # 数据哈希用于缓存
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        
        results = {}
        
        # 并行计算
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            
            for factor in self.factors:
                # 检查缓存
                cache_key = f"{factor.id}_{data_hash}"
                if use_cache and cache_key in self._cache:
                    results[factor.name] = self._cache[cache_key]
                    continue
                
                future = pool.submit(self.compute_single, factor, data)
                futures[future] = factor
            
            for future in as_completed(futures):
                factor = futures[future]
                try:
                    values = future.result(timeout=60)
                    if values is not None:
                        results[factor.name] = values
                        # 缓存
                        if use_cache:
                            cache_key = f"{factor.id}_{data_hash}"
                            self._cache[cache_key] = values
                except Exception as e:
                    logger.warning(f"因子 {factor.name} 计算超时或失败")
        
        factor_df = pd.DataFrame(results, index=data.index)
        logger.info(f"计算完成: {len(results)}/{len(self.factors)} 个因子")
        return factor_df
    
    def _execute_factor_code(
        self,
        code: str,
        data: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """默认因子代码执行器"""
        try:
            # 准备执行环境
            local_vars = {
                'df': data,
                'np': np,
                'pd': pd,
                # 常用数据
                'close': data.get('close', data.get('$close')),
                'open': data.get('open', data.get('$open')),
                'high': data.get('high', data.get('$high')),
                'low': data.get('low', data.get('$low')),
                'volume': data.get('volume', data.get('$volume')),
                'vwap': data.get('vwap', data.get('$vwap')),
                'turn': data.get('turn', data.get('$turn')),
            }
            
            # 执行代码
            exec(code, {"__builtins__": {}}, local_vars)
            
            # 获取结果
            result = local_vars.get('result', local_vars.get('factor'))
            
            if result is not None:
                if isinstance(result, pd.Series):
                    return result
                return pd.Series(result, index=data.index)
            
            return None
            
        except Exception as e:
            logger.debug(f"执行因子代码失败: {e}")
            return None
    
    # ============================================================
    # Qlib集成
    # ============================================================
    
    def to_qlib_expressions(self) -> List[Dict]:
        """
        转换为Qlib表达式格式
        
        Returns:
            [{'name': 'factor1', 'expression': '$close/$open'}, ...]
        """
        expressions = []
        
        for factor in self.factors:
            # 转换代码为Qlib表达式
            qlib_expr = self._code_to_qlib_expr(factor.code)
            
            expressions.append({
                'name': factor.name,
                'expression': qlib_expr,
                'ic': factor.ic,
                'icir': factor.icir,
            })
        
        return expressions
    
    def _code_to_qlib_expr(self, code: str) -> str:
        """将Python代码转换为Qlib表达式"""
        if not code:
            return ""
        
        expr = code
        
        # 替换DataFrame访问为Qlib字段
        replacements = [
            ('df["close"]', '$close'),
            ('df["open"]', '$open'),
            ('df["high"]', '$high'),
            ('df["low"]', '$low'),
            ('df["volume"]', '$volume'),
            ('df["vwap"]', '$vwap'),
            ('df["turn"]', '$turn'),
            ("df['close']", '$close'),
            ("df['open']", '$open'),
            ("df['high']", '$high'),
            ("df['low']", '$low'),
            ("df['volume']", '$volume'),
            ("df['vwap']", '$vwap'),
            ("df['turn']", '$turn'),
            ('close', '$close'),
            ('open', '$open'),
            ('high', '$high'),
            ('low', '$low'),
            ('volume', '$volume'),
        ]
        
        for old, new in replacements:
            expr = expr.replace(old, new)
        
        # 简化处理
        if 'result' in expr:
            # 提取result = xxx 后面的表达式
            lines = expr.strip().split('\n')
            for line in lines:
                if 'result' in line and '=' in line:
                    expr = line.split('=', 1)[1].strip()
                    break
        
        return expr
    
    def to_qlib_handler_config(self) -> Dict:
        """
        生成Qlib DataHandler配置
        
        Returns:
            可直接用于Qlib的配置字典
        """
        fields = []
        names = []
        
        for factor in self.factors:
            expr = self._code_to_qlib_expr(factor.code)
            if expr and not expr.startswith('def '):
                fields.append(expr)
                names.append(factor.name)
        
        config = {
            'class': 'Alpha158',
            'module_path': 'qlib.contrib.data.handler',
            'kwargs': {
                'instruments': 'csi300',
                'start_time': '2018-01-01',
                'end_time': '2023-12-31',
                'fit_start_time': '2018-01-01',
                'fit_end_time': '2021-12-31',
                'infer_processors': [
                    {'class': 'RobustZScoreNorm', 'kwargs': {'clip_outlier': True}},
                    {'class': 'Fillna', 'kwargs': {'fill_value': 0}},
                ],
                'learn_processors': [
                    {'class': 'DropnaLabel'},
                    {'class': 'CSRankNorm'},
                ],
                'label': ['Ref($close, -5) / $close - 1'],
            },
            # 自定义因子
            'custom_fields': fields,
            'custom_names': names,
        }
        
        return config
    
    def create_qlib_dataset(
        self,
        data: pd.DataFrame,
        target: pd.Series = None,
    ) -> Dict[str, Any]:
        """
        创建Qlib格式的数据集
        
        Args:
            data: 原始数据
            target: 目标收益 (可选)
        
        Returns:
            {'features': DataFrame, 'label': Series}
        """
        # 计算因子值
        features = self.compute(data)
        
        # 标准化
        features = (features - features.mean()) / (features.std() + 1e-8)
        features = features.clip(-3, 3)  # 截断异常值
        features = features.fillna(0)
        
        result = {'features': features}
        
        if target is not None:
            result['label'] = target
        
        return result
    
    # ============================================================
    # 持久化
    # ============================================================
    
    def save(self, path: Union[str, Path]):
        """保存到JSON文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'factor_count': len(self.factors),
            'factors': [f.to_dict() for f in self.factors],
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"因子保存到: {path}")
    
    def save_qlib_config(self, path: Union[str, Path]):
        """保存Qlib配置文件"""
        import yaml
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = self.to_qlib_handler_config()
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Qlib配置保存到: {path}")
    
    # ============================================================
    # 属性和工具方法
    # ============================================================
    
    @property
    def names(self) -> List[str]:
        """获取所有因子名称"""
        return [f.name for f in self.factors]
    
    @property
    def codes(self) -> List[str]:
        """获取所有因子代码"""
        return [f.code for f in self.factors]
    
    def __len__(self) -> int:
        return len(self.factors)
    
    def __iter__(self):
        return iter(self.factors)
    
    def __getitem__(self, idx) -> FactorMeta:
        if isinstance(idx, str):
            for f in self.factors:
                if f.name == idx or f.id == idx:
                    return f
            raise KeyError(f"因子 {idx} 不存在")
        return self.factors[idx]
    
    def get_by_name(self, name: str) -> Optional[FactorMeta]:
        """按名称获取因子"""
        for f in self.factors:
            if f.name == name:
                return f
        return None
    
    def filter_by_ic(self, min_ic: float = 0.01) -> "FactorWrapper":
        """按IC过滤因子"""
        filtered = [f for f in self.factors if abs(f.ic) >= min_ic]
        return FactorWrapper(filtered)
    
    def sort_by_ic(self, descending: bool = True) -> "FactorWrapper":
        """按IC排序"""
        sorted_factors = sorted(
            self.factors, 
            key=lambda x: abs(x.ic), 
            reverse=descending
        )
        return FactorWrapper(sorted_factors)
    
    def summary(self) -> pd.DataFrame:
        """生成因子摘要表"""
        data = []
        for f in self.factors:
            data.append({
                'name': f.name,
                'category': f.category,
                'ic': f.ic,
                'icir': f.icir,
                'rank_ic': f.rank_ic,
                'source': f.source,
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('ic', key=abs, ascending=False)
        return df
    
    def clear_cache(self):
        """清除计算缓存"""
        self._cache.clear()


# ============================================================
# 便捷函数
# ============================================================

def load_factors(path: Union[str, Path]) -> FactorWrapper:
    """加载因子"""
    return FactorWrapper.from_json(path)


def create_factor_wrapper(
    selected_factors: List[Dict],
    executor: Callable = None,
) -> FactorWrapper:
    """
    创建因子封装器
    
    Args:
        selected_factors: 筛选出的因子列表
        executor: 可选的因子执行器
    
    Returns:
        FactorWrapper实例
    """
    wrapper = FactorWrapper.from_dict_list(selected_factors)
    if executor:
        wrapper.set_executor(executor)
    return wrapper
