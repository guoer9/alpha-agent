"""
分布式计算 - Ray

功能:
1. 分布式回测
2. 并行因子计算
3. 分布式超参搜索
4. 模型并行训练

参考: https://docs.ray.io/
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

# Ray可选
try:
    import ray
    from ray import tune
    from ray.util.multiprocessing import Pool
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.info("Ray未安装，使用本地多进程")

# 多进程备选
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp


@dataclass
class DistributedResult:
    """分布式计算结果"""
    task_id: str
    results: List[Any]
    elapsed_time: float
    num_workers: int
    errors: List[str] = None


class RayExecutor:
    """Ray分布式执行器"""
    
    def __init__(
        self,
        num_cpus: int = None,
        num_gpus: int = 0,
        address: str = None,
    ):
        """
        参数:
            num_cpus: CPU数量 (None=自动)
            num_gpus: GPU数量
            address: Ray集群地址 (None=本地)
        """
        self.num_cpus = num_cpus or mp.cpu_count()
        self.num_gpus = num_gpus
        self.address = address
        self.initialized = False
    
    def init(self):
        """初始化Ray"""
        if self.initialized:
            return
        
        if RAY_AVAILABLE:
            try:
                if self.address:
                    ray.init(address=self.address)
                else:
                    ray.init(
                        num_cpus=self.num_cpus,
                        num_gpus=self.num_gpus,
                        ignore_reinit_error=True,
                    )
                self.initialized = True
                logger.info(f"Ray已初始化: {self.num_cpus} CPUs, {self.num_gpus} GPUs")
            except Exception as e:
                logger.warning(f"Ray初始化失败: {e}")
                self.initialized = False
        else:
            logger.info("Ray未安装，使用多进程模式")
    
    def shutdown(self):
        """关闭Ray"""
        if RAY_AVAILABLE and self.initialized:
            ray.shutdown()
            self.initialized = False
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        num_returns: int = 1,
    ) -> List[Any]:
        """
        并行Map操作
        
        参数:
            func: 处理函数
            items: 待处理项目列表
            num_returns: 每个任务返回值数量
        """
        self.init()
        
        if RAY_AVAILABLE and self.initialized:
            return self._ray_map(func, items)
        else:
            return self._local_map(func, items)
    
    def _ray_map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Ray并行Map"""
        @ray.remote
        def remote_func(item):
            return func(item)
        
        futures = [remote_func.remote(item) for item in items]
        results = ray.get(futures)
        return results
    
    def _local_map(self, func: Callable, items: List[Any]) -> List[Any]:
        """本地多进程Map"""
        with ProcessPoolExecutor(max_workers=self.num_cpus) as executor:
            results = list(executor.map(func, items))
        return results
    
    def parallel_apply(
        self,
        df: pd.DataFrame,
        func: Callable,
        axis: int = 0,
        num_partitions: int = None,
    ) -> pd.DataFrame:
        """
        DataFrame并行Apply
        
        参数:
            df: 输入DataFrame
            func: 处理函数
            axis: 应用轴 (0=行, 1=列)
            num_partitions: 分区数
        """
        num_partitions = num_partitions or self.num_cpus
        
        if axis == 0:
            # 按行分割
            chunks = np.array_split(df, num_partitions)
        else:
            # 按列分割
            chunks = [df.iloc[:, i::num_partitions] for i in range(num_partitions)]
        
        results = self.map(func, chunks)
        
        if axis == 0:
            return pd.concat(results, axis=0)
        else:
            return pd.concat(results, axis=1)


def distributed_backtest(
    factor_func: Callable,
    param_grid: List[Dict],
    data: pd.DataFrame,
    returns: pd.Series,
    num_workers: int = None,
) -> List[Dict]:
    """
    分布式回测
    
    参数:
        factor_func: 因子计算函数 f(data, **params) -> pd.Series
        param_grid: 参数网格
        data: 特征数据
        returns: 收益数据
        num_workers: 并行数
    
    返回:
        回测结果列表
    """
    from ..mining import run_backtest, BacktestResult
    
    num_workers = num_workers or mp.cpu_count()
    
    def backtest_single(params: Dict) -> Dict:
        """单次回测"""
        try:
            # 计算因子
            factor = factor_func(data, **params)
            
            # 运行回测
            result = run_backtest(factor, returns, method='simple')
            
            return {
                'params': params,
                'sharpe': result.sharpe_ratio,
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'status': 'success',
            }
        except Exception as e:
            return {
                'params': params,
                'status': 'failed',
                'error': str(e),
            }
    
    start_time = time.time()
    
    # 选择执行器
    if RAY_AVAILABLE:
        executor = RayExecutor(num_cpus=num_workers)
        results = executor.map(backtest_single, param_grid)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(backtest_single, param_grid))
    
    elapsed = time.time() - start_time
    logger.info(f"分布式回测完成: {len(param_grid)}个参数, {elapsed:.2f}秒")
    
    return results


def distributed_factor_compute(
    factor_funcs: List[Callable],
    data: pd.DataFrame,
    num_workers: int = None,
) -> pd.DataFrame:
    """
    分布式因子计算
    
    参数:
        factor_funcs: 因子函数列表
        data: 输入数据
        num_workers: 并行数
    
    返回:
        因子DataFrame
    """
    num_workers = num_workers or mp.cpu_count()
    
    def compute_factor(func: Callable) -> Tuple[str, pd.Series]:
        """计算单个因子"""
        try:
            result = func(data)
            name = getattr(func, '__name__', 'unknown')
            return name, result
        except Exception as e:
            logger.warning(f"因子计算失败: {e}")
            return None, None
    
    start_time = time.time()
    
    # 并行计算
    if RAY_AVAILABLE:
        executor = RayExecutor(num_cpus=num_workers)
        results = executor.map(compute_factor, factor_funcs)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(compute_factor, factor_funcs))
    
    # 合并结果
    factor_dict = {}
    for name, series in results:
        if name is not None and series is not None:
            factor_dict[name] = series
    
    elapsed = time.time() - start_time
    logger.info(f"因子计算完成: {len(factor_dict)}个因子, {elapsed:.2f}秒")
    
    return pd.DataFrame(factor_dict)


def hyperparameter_search(
    train_func: Callable,
    param_space: Dict,
    num_samples: int = 10,
    metric: str = "sharpe",
    mode: str = "max",
) -> Dict:
    """
    超参数搜索 (使用Ray Tune)
    
    参数:
        train_func: 训练函数 f(config) -> {metric: value}
        param_space: 参数空间
        num_samples: 采样数
        metric: 优化指标
        mode: max或min
    
    返回:
        最优参数
    """
    if not RAY_AVAILABLE:
        logger.warning("Ray未安装，使用随机搜索")
        return _random_search(train_func, param_space, num_samples, metric, mode)
    
    try:
        from ray import tune
        from ray.tune.search.basic_variant import BasicVariantGenerator
        
        analysis = tune.run(
            train_func,
            config=param_space,
            num_samples=num_samples,
            metric=metric,
            mode=mode,
            verbose=1,
        )
        
        best_config = analysis.best_config
        best_result = analysis.best_result
        
        return {
            'best_config': best_config,
            'best_metric': best_result.get(metric),
            'all_results': analysis.results_df.to_dict(),
        }
    except Exception as e:
        logger.warning(f"Ray Tune失败: {e}, 使用随机搜索")
        return _random_search(train_func, param_space, num_samples, metric, mode)


def _random_search(
    train_func: Callable,
    param_space: Dict,
    num_samples: int,
    metric: str,
    mode: str,
) -> Dict:
    """随机搜索"""
    import random
    
    best_config = None
    best_metric = float('-inf') if mode == 'max' else float('inf')
    all_results = []
    
    for _ in range(num_samples):
        # 随机采样参数
        config = {}
        for key, space in param_space.items():
            if isinstance(space, list):
                config[key] = random.choice(space)
            elif isinstance(space, tuple) and len(space) == 2:
                config[key] = random.uniform(space[0], space[1])
            else:
                config[key] = space
        
        # 训练
        try:
            result = train_func(config)
            metric_value = result.get(metric, 0)
            all_results.append({'config': config, metric: metric_value})
            
            # 更新最优
            if mode == 'max' and metric_value > best_metric:
                best_metric = metric_value
                best_config = config
            elif mode == 'min' and metric_value < best_metric:
                best_metric = metric_value
                best_config = config
        except Exception as e:
            logger.warning(f"训练失败: {e}")
    
    return {
        'best_config': best_config,
        'best_metric': best_metric,
        'all_results': all_results,
    }
