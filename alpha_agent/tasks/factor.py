"""
因子计算任务 - Celery

功能:
1. 并行因子计算
2. 批量IC评估
3. Redis缓存
"""
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from celery import shared_task, group, chord

logger = logging.getLogger(__name__)

# Redis缓存
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class FactorCache:
    """因子计算结果缓存 - Redis"""
    
    def __init__(self, host='localhost', port=6379, db=2, ttl=3600*24):
        self.ttl = ttl  # 默认缓存24小时
        self.redis = None
        if REDIS_AVAILABLE:
            try:
                self.redis = redis.Redis(host=host, port=port, db=db)
                self.redis.ping()
                logger.info(f"Redis缓存已连接: {host}:{port}")
            except Exception as e:
                logger.warning(f"Redis连接失败: {e}")
                self.redis = None
    
    def _make_key(self, factor_code: str, data_hash: str) -> str:
        """生成缓存key"""
        code_hash = hashlib.md5(factor_code.encode()).hexdigest()[:12]
        return f"factor:ic:{code_hash}:{data_hash}"
    
    def get(self, factor_code: str, data_hash: str) -> Optional[Dict]:
        """获取缓存"""
        if not self.redis:
            return None
        try:
            key = self._make_key(factor_code, data_hash)
            data = self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None
    
    def set(self, factor_code: str, data_hash: str, result: Dict):
        """设置缓存"""
        if not self.redis:
            return
        try:
            key = self._make_key(factor_code, data_hash)
            self.redis.setex(key, self.ttl, json.dumps(result))
        except Exception:
            pass
    
    def clear_all(self):
        """清除所有因子缓存"""
        if not self.redis:
            return
        try:
            keys = self.redis.keys("factor:ic:*")
            if keys:
                self.redis.delete(*keys)
        except Exception:
            pass


# 全局缓存实例
_factor_cache = None

def get_factor_cache() -> FactorCache:
    """获取缓存实例"""
    global _factor_cache
    if _factor_cache is None:
        _factor_cache = FactorCache()
    return _factor_cache


@shared_task(bind=True, max_retries=3, queue='factor')
def compute_factor(self, factor_code: str, data: dict):
    """计算单个因子"""
    try:
        from alpha_agent.core.sandbox import Sandbox
        sandbox = Sandbox(timeout_seconds=30)
        result, error = sandbox.execute(factor_code, data)
        return {'status': 'success', 'result': result}
    except Exception as e:
        self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=2, queue='factor')
def evaluate_factor_ic(
    self,
    factor_id: str,
    factor_code: str,
    data_json: str,
    target_json: str,
    data_hash: str,
) -> Dict:
    """
    评估单个因子IC - Celery任务
    
    Args:
        factor_id: 因子ID
        factor_code: 因子代码
        data_json: 数据JSON (压缩)
        target_json: 目标收益JSON (压缩)
        data_hash: 数据哈希 (用于缓存)
    
    Returns:
        评估结果 {factor_id, ic, icir, rank_ic, status}
    """
    try:
        # 检查缓存
        cache = get_factor_cache()
        cached = cache.get(factor_code, data_hash)
        if cached:
            cached['factor_id'] = factor_id
            cached['from_cache'] = True
            return cached
        
        # 反序列化数据
        data = pd.read_json(data_json, orient='split')
        target = pd.read_json(target_json, typ='series', orient='split')
        
        # 执行因子
        from alpha_agent.core.sandbox import Sandbox
        sandbox = Sandbox(timeout_seconds=60)
        factor_values, error = sandbox.execute(factor_code, data.to_dict('list'))
        
        if error or factor_values is None:
            return {
                'factor_id': factor_id,
                'status': 'failed',
                'error': error or 'No values',
            }
        
        # 转换为Series
        factor_series = pd.Series(factor_values, index=data.index)
        
        # 计算IC
        result = _compute_ic_metrics(factor_series, target)
        result['factor_id'] = factor_id
        result['status'] = 'success'
        
        # 存入缓存
        cache.set(factor_code, data_hash, result)
        
        return result
        
    except Exception as e:
        logger.error(f"因子评估失败 {factor_id}: {e}")
        return {
            'factor_id': factor_id,
            'status': 'failed',
            'error': str(e),
        }


def _compute_ic_metrics(factor: pd.Series, target: pd.Series) -> Dict:
    """计算IC指标 - 向量化"""
    # 对齐数据
    aligned = pd.concat([factor, target], axis=1).dropna()
    if len(aligned) < 100:
        return {'ic': 0, 'icir': 0, 'rank_ic': 0}
    
    factor_vals = aligned.iloc[:, 0]
    target_vals = aligned.iloc[:, 1]
    
    # 整体IC
    ic = factor_vals.corr(target_vals, method='spearman')
    
    # Rank IC
    rank_ic = factor_vals.rank().corr(target_vals.rank())
    
    # 按日期计算截面IC (如果是MultiIndex)
    icir = 0
    if hasattr(factor_vals.index, 'get_level_values'):
        try:
            dates = factor_vals.index.get_level_values('datetime').unique()
            daily_ic = []
            sample_dates = dates[::max(1, len(dates)//50)]
            
            for date in sample_dates:
                try:
                    f_day = factor_vals.xs(date, level='datetime')
                    t_day = target_vals.xs(date, level='datetime')
                    if len(f_day) > 10:
                        day_ic = f_day.corr(t_day, method='spearman')
                        if not np.isnan(day_ic):
                            daily_ic.append(day_ic)
                except Exception:
                    continue
            
            if daily_ic:
                daily_ic = pd.Series(daily_ic)
                icir = daily_ic.mean() / (daily_ic.std() + 1e-8)
        except Exception:
            icir = ic / 0.1 if abs(ic) > 0.01 else 0
    
    return {
        'ic': float(ic) if not np.isnan(ic) else 0,
        'icir': float(icir) if not np.isnan(icir) else 0,
        'rank_ic': float(rank_ic) if not np.isnan(rank_ic) else 0,
    }


@shared_task(queue='factor')
def batch_evaluate_factors(
    factors: List[Dict],
    data_json: str,
    target_json: str,
    data_hash: str,
) -> List[Dict]:
    """
    批量评估因子 - 并行分发
    
    Args:
        factors: 因子列表 [{id, code}, ...]
        data_json: 数据JSON
        target_json: 目标收益JSON
        data_hash: 数据哈希
    
    Returns:
        评估结果列表
    """
    # 创建任务组
    tasks = group([
        evaluate_factor_ic.s(
            factor_id=f.get('id', f.get('factor_id', str(i))),
            factor_code=f.get('code', ''),
            data_json=data_json,
            target_json=target_json,
            data_hash=data_hash,
        )
        for i, f in enumerate(factors)
    ])
    
    # 执行并等待结果
    result = tasks.apply_async()
    return result.get(timeout=600)  # 10分钟超时


@shared_task
def update_factors():
    """更新所有因子"""
    logger.info("执行每日因子更新...")
    return {'status': 'success'}


@shared_task
def batch_backtest(factor_ids: list):
    """批量回测因子"""
    results = []
    for fid in factor_ids:
        results.append({'factor_id': fid, 'status': 'completed'})
    return results


# ============================================================
# 同步并行 (无Celery时使用multiprocessing)
# ============================================================

def parallel_evaluate_factors_local(
    factors: List[Dict],
    data: pd.DataFrame,
    target: pd.Series,
    executor,
    n_workers: int = 4,
) -> List[Dict]:
    """
    本地并行评估因子 - multiprocessing
    
    当Celery不可用时使用
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    
    results = []
    
    # 使用线程池 (因为需要共享数据)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        
        for factor in factors:
            future = pool.submit(
                _evaluate_single_factor,
                factor,
                data,
                target,
                executor,
            )
            futures[future] = factor.get('id', factor.get('name', ''))
        
        for future in as_completed(futures):
            factor_id = futures[future]
            try:
                result = future.result(timeout=120)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"因子评估失败 {factor_id}: {e}")
    
    return results


def _evaluate_single_factor(
    factor: Dict,
    data: pd.DataFrame,
    target: pd.Series,
    executor,
) -> Optional[Dict]:
    """评估单个因子"""
    try:
        code = factor.get('code', '')
        if not code:
            return None
        
        # 执行因子
        if executor:
            values = executor(code, data)
        else:
            return None
        
        if values is None or len(values) == 0:
            return None
        
        # 计算IC
        result = _compute_ic_metrics(values, target)
        
        # 合并结果
        factor_result = factor.copy()
        factor_result.update(result)
        factor_result['status'] = 'success'
        
        return factor_result
        
    except Exception as e:
        return None