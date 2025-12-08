"""
因子库管理器 - 管理所有因子的生命周期

功能:
- 初始化经典因子
- 添加/更新因子
- 因子检索
- 导出到GraphRAG/RAPTOR
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

from .classic_factors import (
    ClassicFactor, 
    ALL_CLASSIC_FACTORS,
    FactorCategory,
    get_factor_by_id,
)
from .alpha158 import ALPHA158_FACTORS
from .alpha360 import ALPHA360_FACTORS
from .worldquant101 import WORLDQUANT_101_FACTORS


@dataclass
class FactorRecord:
    """因子记录 - 包含运行时信息"""
    # 基础信息
    id: str
    name: str
    code: str
    category: str
    
    # 来源
    source: str = "classic"         # classic/llm/gp/human
    version: int = 1
    parent_ids: List[str] = field(default_factory=list)
    
    # 评估指标
    ic: float = 0.0
    icir: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    fitness: float = 0.0
    
    # 反思
    reflection: str = ""
    suggestions: List[str] = field(default_factory=list)
    
    # 元信息
    description: str = ""
    tags: List[str] = field(default_factory=list)
    reference: str = ""
    
    # 状态
    status: str = "active"          # active/deprecated/experimental
    created_at: str = ""
    updated_at: str = ""
    evaluated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
    
    @property
    def code_hash(self) -> str:
        """代码哈希，用于去重"""
        return hashlib.md5(self.code.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_classic(cls, factor: ClassicFactor) -> 'FactorRecord':
        """从经典因子创建记录"""
        return cls(
            id=factor.id,
            name=factor.name,
            code=factor.code,
            category=factor.category.value,
            source="classic",
            ic=factor.historical_ic,
            icir=factor.historical_icir,
            turnover=factor.typical_turnover,
            description=factor.description,
            tags=factor.tags,
            reference=factor.reference,
            reflection=factor.logic,
            suggestions=factor.pitfalls,
        )


class FactorLibrary:
    """因子库管理器"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.factors: Dict[str, FactorRecord] = {}
        self._code_hashes: Dict[str, str] = {}  # hash -> factor_id
        
    def initialize_classic_factors(self) -> int:
        """初始化经典因子库"""
        count = 0
        for classic_factor in ALL_CLASSIC_FACTORS:
            record = FactorRecord.from_classic(classic_factor)
            self.add_factor(record)
            count += 1
        return count
    
    def add_factor(self, factor: FactorRecord, check_duplicate: bool = True) -> bool:
        """
        添加因子
        
        Args:
            factor: 因子记录
            check_duplicate: 是否检查代码重复
            
        Returns:
            是否成功添加
        """
        # 检查重复
        if check_duplicate:
            code_hash = factor.code_hash
            if code_hash in self._code_hashes:
                existing_id = self._code_hashes[code_hash]
                print(f"警告: 因子代码与 {existing_id} 重复")
                return False
            self._code_hashes[code_hash] = factor.id
        
        # 检查ID冲突
        if factor.id in self.factors:
            # 更新版本
            factor.version = self.factors[factor.id].version + 1
            factor.updated_at = datetime.now().isoformat()
        
        self.factors[factor.id] = factor
        return True
    
    def get_factor(self, factor_id: str) -> Optional[FactorRecord]:
        """获取因子"""
        return self.factors.get(factor_id)
    
    def search_factors(
        self,
        category: str = None,
        tags: List[str] = None,
        min_ic: float = None,
        max_turnover: float = None,
        status: str = "active",
    ) -> List[FactorRecord]:
        """
        搜索因子
        """
        results = []
        
        for factor in self.factors.values():
            # 状态过滤
            if status and factor.status != status:
                continue
            
            # 类别过滤
            if category and factor.category != category:
                continue
            
            # 标签过滤
            if tags:
                if not any(tag in factor.tags for tag in tags):
                    continue
            
            # IC过滤
            if min_ic and factor.ic < min_ic:
                continue
            
            # 换手率过滤
            if max_turnover and factor.turnover > max_turnover:
                continue
            
            results.append(factor)
        
        return results
    
    def get_factors_for_graphrag(self) -> List[Dict]:
        """
        导出因子数据用于GraphRAG构建
        """
        nodes = []
        for factor in self.factors.values():
            nodes.append({
                'id': factor.id,
                'name': factor.name,
                'type': 'Factor',
                'properties': {
                    'code': factor.code,
                    'category': factor.category,
                    'ic': factor.ic,
                    'icir': factor.icir,
                    'sharpe': factor.sharpe,
                    'turnover': factor.turnover,
                    'fitness': factor.fitness,
                    'tags': factor.tags,
                    'status': factor.status,
                },
                'reflection': factor.reflection,
                'suggestions': factor.suggestions,
            })
        return nodes
    
    def get_factors_for_raptor(self) -> List[Dict]:
        """
        导出因子数据用于RAPTOR构建
        """
        documents = []
        for factor in self.factors.values():
            # 构建因子描述文档
            doc = f"""
## {factor.name} ({factor.id})

### 代码
```python
{factor.code}
```

### 指标
- IC: {factor.ic:.4f}
- ICIR: {factor.icir:.4f}
- 夏普: {factor.sharpe:.2f}
- 换手率: {factor.turnover:.1%}

### 描述
{factor.description}

### 投资逻辑
{factor.reflection}

### 注意事项
{chr(10).join('- ' + s for s in factor.suggestions)}

### 标签
{', '.join(factor.tags)}
"""
            documents.append({
                'id': factor.id,
                'category': factor.category,
                'content': doc,
                'metadata': {
                    'ic': factor.ic,
                    'turnover': factor.turnover,
                    'tags': factor.tags,
                }
            })
        return documents
    
    def save(self, path: str = None):
        """保存因子库到文件"""
        save_path = Path(path) if path else self.storage_path
        if not save_path:
            raise ValueError("未指定存储路径")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'factors': {fid: f.to_dict() for fid, f in self.factors.items()}
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"因子库已保存: {save_path} ({len(self.factors)}个因子)")
    
    def load(self, path: str = None):
        """从文件加载因子库"""
        load_path = Path(path) if path else self.storage_path
        if not load_path or not load_path.exists():
            raise FileNotFoundError(f"因子库文件不存在: {load_path}")
        
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.factors = {}
        self._code_hashes = {}
        
        for fid, fdata in data.get('factors', {}).items():
            record = FactorRecord(**fdata)
            self.factors[fid] = record
            self._code_hashes[record.code_hash] = fid
        
        print(f"因子库已加载: {load_path} ({len(self.factors)}个因子)")
    
    def summary(self) -> str:
        """因子库统计摘要"""
        total = len(self.factors)
        by_category = {}
        by_source = {}
        active_count = 0
        
        for f in self.factors.values():
            by_category[f.category] = by_category.get(f.category, 0) + 1
            by_source[f.source] = by_source.get(f.source, 0) + 1
            if f.status == 'active':
                active_count += 1
        
        lines = [
            "=" * 50,
            "因子库统计",
            "=" * 50,
            f"总因子数: {total}",
            f"活跃因子: {active_count}",
            "",
            "按类别:",
        ]
        for cat, count in sorted(by_category.items()):
            lines.append(f"  - {cat}: {count}")
        
        lines.append("")
        lines.append("按来源:")
        for src, count in sorted(by_source.items()):
            lines.append(f"  - {src}: {count}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.factors)
    
    def __iter__(self):
        return iter(self.factors.values())


def create_factor_library(
    include_classic: bool = True,
    include_alpha158: bool = True,
    include_alpha360: bool = True,
    include_worldquant: bool = True,
    storage_path: str = None,
) -> FactorLibrary:
    """
    创建因子库的便捷函数
    
    Args:
        include_classic: 是否包含经典因子 (Barra/技术/基本面)
        include_alpha158: 是否包含Qlib Alpha158
        include_alpha360: 是否包含Qlib Alpha360
        include_worldquant: 是否包含WorldQuant 101
        storage_path: 存储路径
    """
    library = FactorLibrary(storage_path)
    total = 0
    
    if include_classic:
        count = library.initialize_classic_factors()
        total += count
        print(f"  ✓ 经典因子: {count} 个 (Barra/技术/基本面/量价)")
    
    if include_alpha158:
        for factor in ALPHA158_FACTORS:
            record = FactorRecord.from_classic(factor)
            library.add_factor(record, check_duplicate=False)
        total += len(ALPHA158_FACTORS)
        print(f"  ✓ Alpha158: {len(ALPHA158_FACTORS)} 个 (Microsoft Qlib)")
    
    if include_alpha360:
        for factor in ALPHA360_FACTORS:
            record = FactorRecord.from_classic(factor)
            library.add_factor(record, check_duplicate=False)
        total += len(ALPHA360_FACTORS)
        print(f"  ✓ Alpha360: {len(ALPHA360_FACTORS)} 个 (Microsoft Qlib)")
    
    if include_worldquant:
        for factor in WORLDQUANT_101_FACTORS:
            record = FactorRecord.from_classic(factor)
            library.add_factor(record, check_duplicate=False)
        total += len(WORLDQUANT_101_FACTORS)
        print(f"  ✓ WorldQuant 101: {len(WORLDQUANT_101_FACTORS)} 个 (Kakushadze 2016)")
    
    print(f"\n总计加载 {total} 个因子")
    
    return library


# ============================================================
# 命令行测试
# ============================================================

if __name__ == "__main__":
    # 创建因子库
    library = create_factor_library(include_classic=True)
    
    # 打印统计
    print(library.summary())
    
    # 搜索示例
    print("\n动量类因子:")
    momentum_factors = library.search_factors(tags=["momentum"])
    for f in momentum_factors:
        print(f"  - {f.name}: IC={f.ic:.4f}, 换手={f.turnover:.1%}")
    
    print("\n低换手因子 (换手<30%):")
    low_turnover = library.search_factors(max_turnover=0.30)
    for f in low_turnover:
        print(f"  - {f.name}: IC={f.ic:.4f}, 换手={f.turnover:.1%}")
