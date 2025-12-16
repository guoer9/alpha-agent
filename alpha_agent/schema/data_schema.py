"""
数据字典 (Data Schema) - 让LLM理解数据结构和约束
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import pandas as pd
import numpy as np


class DataFrequency(Enum):
    TICK = "tick"
    MINUTE = "1min"
    DAILY = "daily"
    WEEKLY = "weekly"


class DataType(Enum):
    PRICE = "price"
    VOLUME = "volume"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    ALTERNATIVE = "alternative"


@dataclass
class FieldSchema:
    """单个字段的完整定义"""
    name: str
    dtype: str
    description: str
    data_type: DataType
    
    # 数据质量
    missing_rate: float = 0.0
    update_frequency: DataFrequency = DataFrequency.DAILY
    lag_days: int = 0  # 防止未来函数
    
    # 值域约束
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    typical_range: tuple = (None, None)
    
    # 使用指南
    usage_examples: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)
    related_fields: List[str] = field(default_factory=list)
    
    # 衍生信息
    is_adjusted: bool = False
    lookback_required: int = 0


@dataclass
class DataSchema:
    """完整数据字典"""
    name: str
    version: str
    description: str
    fields: Dict[str, FieldSchema]
    
    primary_key: List[str] = field(default_factory=lambda: ['date', 'instrument'])
    time_column: str = 'date'
    entity_column: str = 'instrument'
    
    start_date: str = ""
    end_date: str = ""
    universe_size: int = 0
    
    def to_llm_prompt(self) -> str:
        """生成给LLM的数据字典描述"""
        lines = [
            f"## 数据字典: {self.name} (v{self.version})",
            f"\n{self.description}",
            f"\n### 数据范围",
            f"- 时间: {self.start_date} ~ {self.end_date}",
            f"- 股票数: {self.universe_size}",
            f"- 主键: {self.primary_key}",
            "\n### 字段详情\n"
        ]
        
        for name, f in self.fields.items():
            lines.append(f"#### `{name}` ({f.dtype})")
            lines.append(f"- **含义**: {f.description}")
            lines.append(f"- **类别**: {f.data_type.value}")
            
            if f.missing_rate > 0:
                lines.append(f"- **缺失率**: {f.missing_rate:.1%}")
            
            if f.min_value is not None or f.max_value is not None:
                lines.append(f"- **值域**: [{f.min_value}, {f.max_value}]")
            
            if f.lag_days > 0:
                lines.append(f"- **⚠️ 数据滞后**: {f.lag_days}天")
            
            if f.lookback_required > 0:
                lines.append(f"- **需要回溯**: {f.lookback_required}天")
            
            if f.usage_examples:
                lines.append("- **使用示例**:")
                for ex in f.usage_examples[:2]:
                    lines.append(f"  ```python\n  {ex}\n  ```")
            
            if f.common_pitfalls:
                lines.append("- **⚠️ 注意事项**:")
                for pit in f.common_pitfalls:
                    lines.append(f"  - {pit}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def get_field(self, name: str) -> Optional[FieldSchema]:
        return self.fields.get(name)
    
    def list_fields_by_type(self, data_type: DataType) -> List[str]:
        return [name for name, f in self.fields.items() if f.data_type == data_type]


class DataValidator:
    """数据质量验证器"""
    
    def __init__(self, schema: DataSchema):
        self.schema = schema
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证数据质量"""
        report = {'is_valid': True, 'errors': [], 'warnings': [], 'stats': {}}
        
        # 检查必要字段
        missing = set(self.schema.fields.keys()) - set(df.columns)
        if missing:
            report['warnings'].append(f"缺少字段: {missing}")
        
        # 检查每个字段
        for name, fs in self.schema.fields.items():
            if name not in df.columns:
                continue
            
            col = df[name]
            actual_missing = col.isna().mean()
            
            # 缺失率告警
            if actual_missing > fs.missing_rate * 2:
                report['warnings'].append(
                    f"{name}: 缺失率 {actual_missing:.2%} 超预期"
                )
            
            # 值域检查
            if fs.min_value is not None and (col < fs.min_value).any():
                report['warnings'].append(f"{name}: 存在低于最小值的数据")
            
            if fs.max_value is not None and (col > fs.max_value).any():
                report['warnings'].append(f"{name}: 存在超过最大值的数据")
            
            # 统计
            if col.dtype in ['float64', 'int64']:
                report['stats'][name] = {
                    'mean': col.mean(),
                    'std': col.std(),
                    'missing': actual_missing,
                }
        
        return report
    
    def generate_llm_context(self, df: pd.DataFrame) -> str:
        """生成给LLM的完整上下文"""
        validation = self.validate(df)
        context = self.schema.to_llm_prompt()
        
        context += "\n## 当前数据统计\n"
        for name, stats in validation['stats'].items():
            context += f"- {name}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}\n"
        
        if validation['warnings']:
            context += "\n## ⚠️ 数据质量警告\n"
            for w in validation['warnings']:
                context += f"- {w}\n"
        
        return context
