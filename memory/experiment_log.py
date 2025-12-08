"""
实验日志 - 记录因子生成和评估历史
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import logging

from ..config import LOGS_DIR

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = LOGS_DIR / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.records: List[Dict] = []
        self.metadata: Dict = {
            'name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
        }
    
    def log_factor(
        self,
        name: str,
        code: str,
        ic: float,
        icir: float = 0.0,
        status: str = "pending",
        error: str = "",
        **kwargs,
    ):
        """记录因子"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'code': code,
            'ic': ic,
            'icir': icir,
            'status': status,
            'error': error,
            **kwargs,
        }
        self.records.append(record)
        
        # 实时保存
        self._save_record(record)
    
    def _save_record(self, record: Dict):
        """保存单条记录"""
        log_file = self.log_dir / "factors.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def get_summary(self) -> Dict:
        """获取实验摘要"""
        if not self.records:
            return {'total': 0}
        
        df = pd.DataFrame(self.records)
        
        return {
            'total': len(df),
            'valid': len(df[df['status'].isin(['excellent', 'good', 'marginal'])]),
            'excellent': len(df[df['status'] == 'excellent']),
            'failed': len(df[df['status'] == 'failed']),
            'avg_ic': df['ic'].mean(),
            'max_ic': df['ic'].max(),
            'best_factor': df.loc[df['ic'].abs().idxmax(), 'name'] if len(df) > 0 else None,
        }
    
    def save(self):
        """保存实验"""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['status'] = 'completed'
        self.metadata['summary'] = self.get_summary()
        
        # 保存元数据
        meta_file = self.log_dir / "metadata.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # 保存CSV
        if self.records:
            df = pd.DataFrame(self.records)
            df.to_csv(self.log_dir / "factors.csv", index=False)
        
        logger.info(f"实验已保存: {self.log_dir}")
    
    def print_summary(self):
        """打印摘要"""
        summary = self.get_summary()
        print(f"\n{'='*50}")
        print(f"实验摘要: {self.experiment_name}")
        print(f"{'='*50}")
        print(f"总生成: {summary['total']}")
        print(f"有效:   {summary['valid']}")
        print(f"优秀:   {summary['excellent']}")
        print(f"失败:   {summary['failed']}")
        print(f"平均IC: {summary['avg_ic']:.4f}")
        print(f"最高IC: {summary['max_ic']:.4f}")
        print(f"最佳:   {summary['best_factor']}")
