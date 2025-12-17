"""
因子筛选器 - 核心实现

算法:
1. 快速预筛选: 采样计算IC，淘汰无效因子
2. 语义去重: 利用Milvus向量相似度去除冗余
3. 正交化组合: Greedy Forward Selection + 相关性约束

参考:
- AlphaForge (AAAI 2024): 动态因子组合
- Warm Start GP (2024): 因子去冗余
"""

from __future__ import annotations

import logging
import hashlib
from typing import List, Dict, Tuple, Callable, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from ..config.settings import selection_config

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SelectionResult:
    """筛选结果"""
    # 各阶段数量
    total_input: int = 0
    after_quick_filter: int = 0
    after_dedup: int = 0
    after_cluster: int = 0
    final_count: int = 0
    
    # 最终选中的因子
    selected_factors: List = field(default_factory=list)
    
    # 相关性矩阵
    correlation_matrix: pd.DataFrame = None
    
    # 各因子详细信息
    factor_details: List[Dict] = field(default_factory=list)
    
    # 统计
    elapsed_time: float = 0.0
    
    def summary(self) -> str:
        """摘要"""
        return f"""
{'='*60}
因子筛选结果
{'='*60}
输入因子: {self.total_input}
├── 快速筛选后: {self.after_quick_filter} ({self.after_quick_filter/max(1,self.total_input):.1%})
├── 去重后: {self.after_dedup} ({self.after_dedup/max(1,self.after_quick_filter):.1%})
├── 聚类后: {self.after_cluster} ({self.after_cluster/max(1,self.after_dedup):.1%})
└── 最终选中: {self.final_count}

耗时: {self.elapsed_time:.1f}秒
{'='*60}
"""


# ============================================================
# 核心筛选器
# ============================================================

class FactorSelector:
    """
    因子筛选器
    
    核心功能:
    1. 快速预筛选 (采样IC)
    2. 语义去重 (可选Milvus)
    3. 聚类代表选择
    4. 正交化组合优化
    """
    
    def __init__(
        self,
        # 快速筛选
        quick_sample_ratio: float = None,
        quick_ic_threshold: float = None,
        random_seed: int = None,
        # 去重
        enable_dedup: bool = None,
        dedup_threshold: float = None,
        # 聚类
        enable_cluster: bool = None,
        n_clusters: int = None,
        reps_per_cluster: int = None,
        cluster_method: str = None,
        cluster_sample_ratio: float = None,
        cluster_sample_size: int = None,
        cluster_corr_threshold: float = None,
        # 正交化
        enable_orthogonal: bool = None,
        max_factors: int = None,
        corr_threshold: float = None,
        min_marginal_ic: float = None,
        # 完整评估
        enable_full_eval: bool = None,
        full_ic_threshold: float = None,
        full_icir_threshold: float = None,
        full_eval_parallel: bool = None,
        full_eval_n_workers: int = None,
    ):
        # 使用配置文件默认值
        cfg = selection_config
        self.quick_sample_ratio = quick_sample_ratio if quick_sample_ratio is not None else cfg.quick_sample_ratio
        self.quick_ic_threshold = quick_ic_threshold if quick_ic_threshold is not None else cfg.quick_ic_threshold
        self.random_seed = random_seed if random_seed is not None else getattr(cfg, 'random_seed', 42)
        self.enable_dedup = enable_dedup if enable_dedup is not None else cfg.enable_dedup
        self.dedup_threshold = dedup_threshold if dedup_threshold is not None else cfg.dedup_threshold
        self.enable_cluster = enable_cluster if enable_cluster is not None else cfg.enable_cluster
        self.n_clusters = n_clusters if n_clusters is not None else cfg.n_clusters
        self.reps_per_cluster = reps_per_cluster if reps_per_cluster is not None else cfg.reps_per_cluster
        self.cluster_method = cluster_method if cluster_method is not None else getattr(cfg, 'cluster_method', 'kmeans')
        self.cluster_sample_ratio = (
            cluster_sample_ratio
            if cluster_sample_ratio is not None
            else getattr(cfg, 'cluster_sample_ratio', 1.0)
        )
        self.cluster_sample_size = (
            cluster_sample_size
            if cluster_sample_size is not None
            else getattr(cfg, 'cluster_sample_size', 0)
        )
        self.cluster_corr_threshold = (
            cluster_corr_threshold
            if cluster_corr_threshold is not None
            else getattr(cfg, 'cluster_corr_threshold', 0.8)
        )
        self.enable_orthogonal = enable_orthogonal if enable_orthogonal is not None else cfg.enable_orthogonal
        self.max_factors = max_factors if max_factors is not None else cfg.max_factors
        self.corr_threshold = corr_threshold if corr_threshold is not None else cfg.corr_threshold
        self.min_marginal_ic = min_marginal_ic if min_marginal_ic is not None else cfg.min_marginal_ic
        self.enable_full_eval = enable_full_eval if enable_full_eval is not None else cfg.enable_full_eval
        self.full_ic_threshold = full_ic_threshold if full_ic_threshold is not None else cfg.full_ic_threshold
        self.full_icir_threshold = full_icir_threshold if full_icir_threshold is not None else cfg.full_icir_threshold
        self.full_eval_parallel = (
            full_eval_parallel
            if full_eval_parallel is not None
            else getattr(cfg, 'full_eval_parallel', False)
        )
        self.full_eval_n_workers = (
            full_eval_n_workers
            if full_eval_n_workers is not None
            else getattr(cfg, 'full_eval_n_workers', 4)
        )
        
        # Milvus连接 (可选)
        self._milvus_store = None
    
    def set_milvus(self, milvus_store):
        """设置Milvus存储用于语义去重"""
        self._milvus_store = milvus_store
    
    def select(
        self,
        factors: List[Dict],
        data: pd.DataFrame,
        target: pd.Series,
        sandbox_executor: Callable = None,
    ) -> SelectionResult:
        """
        执行因子筛选
        
        Args:
            factors: 因子列表，每个因子为dict包含 'code', 'name', 'id' 等
            data: 特征数据 (Qlib格式: MultiIndex [instrument, datetime])
            target: 目标收益
            sandbox_executor: 沙箱执行器 (code -> pd.Series)
        
        Returns:
            SelectionResult
        """
        import time
        start_time = time.time()
        
        result = SelectionResult(total_input=len(factors))
        
        if len(factors) == 0:
            logger.warning("输入因子为空")
            return result
        
        logger.info(f"开始因子筛选: {len(factors)} 个候选因子")
        logger.info(
            "Selector配置: "
            f"enable_dedup={self.enable_dedup}, "
            f"enable_cluster={self.enable_cluster}, "
            f"enable_full_eval={self.enable_full_eval}, "
            f"enable_orthogonal={self.enable_orthogonal}, "
            f"quick_sample_ratio={self.quick_sample_ratio}, "
            f"quick_ic_threshold={self.quick_ic_threshold}, "
            f"random_seed={self.random_seed}, "
            f"dedup_threshold={self.dedup_threshold}, "
            f"n_clusters={self.n_clusters}, reps_per_cluster={self.reps_per_cluster}, "
            f"cluster_method={self.cluster_method}, "
            f"max_factors={self.max_factors}, corr_threshold={self.corr_threshold}, "
            f"min_marginal_ic={self.min_marginal_ic}, "
            f"full_ic_threshold={self.full_ic_threshold}, full_icir_threshold={self.full_icir_threshold}, "
            f"full_eval_parallel={self.full_eval_parallel}, full_eval_n_workers={self.full_eval_n_workers}"
        )

        values_cache: Dict[Tuple[str, str], pd.Series] = {}
        
        # Stage 1: 快速预筛选
        logger.info("Stage 1: 快速预筛选...")
        candidates = self._quick_filter(factors, data, target, sandbox_executor, values_cache)
        result.after_quick_filter = len(candidates)
        logger.info(f"  快速筛选: {len(factors)} → {len(candidates)}")
        
        if len(candidates) == 0:
            logger.warning("快速筛选后无有效因子")
            result.elapsed_time = time.time() - start_time
            return result
        
        # Stage 2: 语义去重
        logger.info("Stage 2: 语义去重...")
        if self.enable_dedup:
            candidates = self._semantic_dedup(candidates)
        else:
            logger.info("  跳过去重")
        result.after_dedup = len(candidates)
        logger.info(f"  语义去重: {result.after_quick_filter} → {len(candidates)}")
        
        # Stage 3: 聚类代表选择 (可选)
        if self.enable_cluster and len(candidates) > self.n_clusters * self.reps_per_cluster:
            logger.info("Stage 3: 聚类代表选择...")
            candidates = self._cluster_select(candidates, data, target, sandbox_executor, values_cache)
            result.after_cluster = len(candidates)
            logger.info(f"  聚类选择: {result.after_dedup} → {len(candidates)}")
        else:
            if not self.enable_cluster:
                logger.info("Stage 3: 聚类代表选择... 跳过(未启用)")
            else:
                logger.info(
                    "Stage 3: 聚类代表选择... 跳过(候选因子数不足) "
                    f"candidates={len(candidates)}, need>{self.n_clusters * self.reps_per_cluster}"
                )
            result.after_cluster = len(candidates)
        
        # Stage 4: 完整评估
        logger.info("Stage 4: 完整评估...")
        candidates_before_full_eval = list(candidates)
        if self.enable_full_eval:
            candidates = self._full_evaluate(
                candidates,
                data,
                target,
                sandbox_executor,
                use_parallel=self.full_eval_parallel,
                n_workers=self.full_eval_n_workers,
                values_cache=values_cache,
            )
            logger.info(f"  完整评估后: {len(candidates)} 个有效因子")
            if len(candidates) == 0 and len(candidates_before_full_eval) > 0:
                logger.warning(
                    "  完整评估后无因子通过阈值，回退到完整评估前候选(按 quick_ic 排序截断)"
                )
                candidates_before_full_eval = sorted(
                    candidates_before_full_eval,
                    key=lambda x: abs(x.get('quick_ic', 0)),
                    reverse=True,
                )
                candidates = candidates_before_full_eval[: self.max_factors]
        else:
            logger.info("  跳过完整评估")
        
        # Stage 5: 正交化组合优化
        logger.info("Stage 5: 正交化组合优化...")
        if self.enable_orthogonal:
            selected, corr_matrix = self._orthogonal_select(
                candidates, data, target, sandbox_executor
            )
        else:
            logger.info("  跳过正交化，直接截断到 max_factors")
            selected = candidates[: self.max_factors]
            corr_matrix = pd.DataFrame()
        result.final_count = len(selected)
        result.selected_factors = selected
        result.correlation_matrix = corr_matrix
        logger.info(f"  最终选中: {len(selected)} 个因子")
        
        # 整理详细信息
        result.factor_details = [
            {
                'id': f.get('id', ''),
                'name': f.get('name', ''),
                'ic': f.get('ic', 0),
                'icir': f.get('icir', 0),
                'rank_ic': f.get('rank_ic', 0),
            }
            for f in selected
        ]
        
        result.elapsed_time = time.time() - start_time
        logger.info(result.summary())
        
        return result
    
    # ============================================================
    # Stage 1: 快速预筛选
    # ============================================================
    
    def _quick_filter(
        self,
        factors: List[Dict],
        data: pd.DataFrame,
        target: pd.Series,
        executor: Callable,
        values_cache: Optional[Dict[Tuple[str, str], pd.Series]] = None,
    ) -> List[Dict]:
        """
        快速预筛选 - 采样计算IC
        
        核心思想: 用10%数据快速计算IC，淘汰明显无效的因子
        效率提升: 约10倍
        """
        # 采样数据
        sample_size = max(1000, int(len(data) * self.quick_sample_ratio))
        if len(data) > sample_size:
            rs = np.random.RandomState(self.random_seed)
            sample_idx = rs.choice(len(data), sample_size, replace=False)
            data_sample = data.iloc[sample_idx]
            target_sample = target.iloc[sample_idx]
        else:
            data_sample = data
            target_sample = target
        
        valid_factors = []
        
        for factor in factors:
            try:
                # 执行因子代码
                factor_values = self._get_factor_values(
                    factor,
                    data_sample,
                    executor,
                    values_cache,
                    cache_tag='quick',
                )
                
                if factor_values is None or len(factor_values) == 0:
                    continue
                
                # 快速IC计算
                ic = self._compute_ic(factor_values, target_sample)
                
                if abs(ic) >= self.quick_ic_threshold:
                    factor['quick_ic'] = ic
                    valid_factors.append(factor)
                    
            except Exception as e:
                logger.debug(f"因子执行失败: {factor.get('name', '')}: {e}")
                continue
        
        return valid_factors
    
    # ============================================================
    # Stage 2: 语义去重
    # ============================================================
    
    def _semantic_dedup(self, factors: List[Dict]) -> List[Dict]:
        """
        语义去重
        
        方法1: 如果有Milvus，使用向量相似度
        方法2: 否则使用代码哈希 + 简单相似度
        """
        if len(factors) <= 1:
            return factors
        
        # 按IC排序，保留IC高的
        factors = sorted(factors, key=lambda x: abs(x.get('quick_ic', 0)), reverse=True)
        
        unique_factors = []
        seen_hashes = set()
        
        for factor in factors:
            code = factor.get('code', '')
            
            # 方法1: 代码哈希去重
            code_hash = hashlib.md5(code.strip().encode()).hexdigest()[:12]
            if code_hash in seen_hashes:
                continue
            
            # 方法2: 如果有Milvus，检查语义相似度
            if self._milvus_store:
                try:
                    is_dup = self._check_milvus_duplicate(code)
                    if is_dup:
                        continue
                except Exception:
                    pass
            
            # 方法3: 简单代码相似度检查
            is_similar = False
            for existing in unique_factors:
                sim = self._code_similarity(code, existing.get('code', ''))
                if sim > self.dedup_threshold:
                    is_similar = True
                    break
            
            if is_similar:
                continue
            
            seen_hashes.add(code_hash)
            unique_factors.append(factor)
        
        return unique_factors
    
    def _check_milvus_duplicate(self, code: str) -> bool:
        """检查Milvus中是否有重复"""
        if not self._milvus_store:
            return False
        
        try:
            from ..memory.rag import check_factor_duplicate
            is_dup, _ = check_factor_duplicate(code, self._milvus_store, self.dedup_threshold)
            return is_dup
        except Exception:
            return False
    
    def _code_similarity(self, code1: str, code2: str) -> float:
        """简单的代码相似度计算"""
        # 归一化
        c1 = code1.strip().lower().replace(' ', '').replace('\n', '')
        c2 = code2.strip().lower().replace(' ', '').replace('\n', '')
        
        if c1 == c2:
            return 1.0
        
        # Jaccard相似度 (基于字符n-gram)
        n = 3
        ngrams1 = set(c1[i:i+n] for i in range(len(c1)-n+1))
        ngrams2 = set(c2[i:i+n] for i in range(len(c2)-n+1))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    # ============================================================
    # Stage 3: 聚类代表选择
    # ============================================================
    
    def _cluster_select(
        self,
        factors: List[Dict],
        data: pd.DataFrame,
        target: pd.Series,
        executor: Callable,
        values_cache: Optional[Dict[Tuple[str, str], pd.Series]] = None,
    ) -> List[Dict]:
        """
        聚类代表选择
        
        核心思想: 将相似因子聚类，每簇选IC最高的代表
        """
        if len(factors) <= self.n_clusters:
            return factors

        method = (self.cluster_method or 'kmeans').strip().lower()
        if method == 'corr_greedy':
            return self._cluster_select_corr_greedy(
                factors,
                data,
                executor,
                values_cache,
            )
        
        cluster_data = self._get_cluster_data(data)

        # 计算因子值矩阵
        factor_matrix = []
        valid_factors = []
        
        for factor in factors:
            try:
                values = self._get_factor_values(
                    factor,
                    cluster_data,
                    executor,
                    values_cache,
                    cache_tag='cluster',
                )
                
                if values is not None and len(values) > 0:
                    # 标准化
                    values_clean = values.fillna(0).values
                    factor_matrix.append(values_clean[:1000])  # 取前1000个样本
                    valid_factors.append(factor)
            except Exception:
                continue
        
        if len(valid_factors) <= self.n_clusters:
            return valid_factors
        
        # 转换为矩阵
        factor_matrix = np.array(factor_matrix)
        
        # 处理NaN和Inf
        factor_matrix = np.nan_to_num(factor_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # K-Means聚类
        n_clusters = min(self.n_clusters, len(valid_factors))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        try:
            labels = kmeans.fit_predict(factor_matrix)
        except Exception as e:
            logger.warning(f"聚类失败: {e}")
            return valid_factors
        
        # 每簇选择IC最高的代表
        selected = []
        for cluster_id in range(n_clusters):
            cluster_factors = [
                (f, abs(f.get('quick_ic', 0)))
                for f, label in zip(valid_factors, labels)
                if label == cluster_id
            ]
            
            # 按IC排序，选top
            cluster_factors.sort(key=lambda x: x[1], reverse=True)
            for f, _ in cluster_factors[:self.reps_per_cluster]:
                selected.append(f)
        
        return selected

    def _cluster_select_corr_greedy(
        self,
        factors: List[Dict],
        data: pd.DataFrame,
        executor: Callable,
        values_cache: Optional[Dict[Tuple[str, str], pd.Series]] = None,
    ) -> List[Dict]:
        if len(factors) <= 1:
            return factors

        cluster_data = self._get_cluster_data(data)
        target_count = max(1, int(self.n_clusters * self.reps_per_cluster))
        factors_sorted = sorted(
            factors,
            key=lambda x: abs(x.get('quick_ic', 0)),
            reverse=True,
        )

        selected: List[Dict] = []
        selected_values: Dict[str, pd.Series] = {}

        for factor in factors_sorted:
            if len(selected) >= target_count:
                break

            values = self._get_factor_values(
                factor,
                cluster_data,
                executor,
                values_cache,
                cache_tag='cluster',
            )
            if values is None or len(values) == 0:
                continue

            max_corr = 0.0
            for existing_values in selected_values.values():
                try:
                    aligned = pd.concat([values, existing_values], axis=1).dropna()
                    if len(aligned) < 20:
                        continue
                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method='spearman')
                    corr = abs(corr) if not np.isnan(corr) else 0.0
                    if corr > max_corr:
                        max_corr = corr
                        if max_corr > self.cluster_corr_threshold:
                            break
                except Exception:
                    continue

            if max_corr > self.cluster_corr_threshold:
                continue

            key = self._factor_key(factor)
            selected.append(factor)
            selected_values[key] = values

        return selected

    def _get_cluster_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.cluster_sample_size and self.cluster_sample_size > 0:
            sample_size = int(self.cluster_sample_size)
        else:
            ratio = float(self.cluster_sample_ratio) if self.cluster_sample_ratio is not None else 1.0
            if ratio >= 1.0:
                return data
            sample_size = int(len(data) * ratio)

        sample_size = max(1000, min(len(data), sample_size))
        if len(data) <= sample_size:
            return data

        rs = np.random.RandomState(self.random_seed)
        sample_idx = rs.choice(len(data), sample_size, replace=False)
        return data.iloc[sample_idx]

    def _factor_key(self, factor: Dict) -> str:
        fid = factor.get('id')
        if fid is not None and str(fid) != '':
            return str(fid)
        code = (factor.get('code', '') or '').strip()
        code_norm = " ".join(code.split())
        return hashlib.md5(code_norm.encode('utf-8')).hexdigest()

    def _get_factor_values(
        self,
        factor: Dict,
        data: pd.DataFrame,
        executor: Callable,
        values_cache: Optional[Dict[Tuple[str, str], pd.Series]],
        cache_tag: str,
    ) -> Optional[pd.Series]:
        if values_cache is None:
            if executor:
                return executor(factor.get('code', ''), data)
            return self._execute_factor(factor.get('code', ''), data)

        key = self._factor_key(factor)
        cache_key = (key, cache_tag)
        if cache_key in values_cache:
            return values_cache.get(cache_key)

        try:
            if executor:
                values = executor(factor.get('code', ''), data)
            else:
                values = self._execute_factor(factor.get('code', ''), data)
        except Exception:
            values_cache[cache_key] = None
            return None

        values_cache[cache_key] = values
        return values
    
    # ============================================================
    # Stage 4: 完整评估
    # ============================================================
    
    def _full_evaluate(
        self,
        factors: List[Dict],
        data: pd.DataFrame,
        target: pd.Series,
        executor: Callable,
        use_parallel: bool = False,  # 默认禁用并行，避免多线程问题
        n_workers: int = 4,
        values_cache: Optional[Dict[Tuple[str, str], pd.Series]] = None,
    ) -> List[Dict]:
        """
        完整评估 - 支持并行
        
        计算完整的IC、ICIR、RankIC等指标
        
        Args:
            factors: 待评估因子列表
            data: 数据
            target: 目标收益
            executor: 因子执行器
            use_parallel: 是否使用并行 (Celery/ThreadPool)
            n_workers: 并行工作线程数
        """
        if len(factors) == 0:
            return []
        
        # 尝试并行评估
        if use_parallel and len(factors) >= 5:
            try:
                return self._parallel_evaluate(
                    factors,
                    data,
                    target,
                    executor,
                    n_workers,
                    values_cache=values_cache,
                )
            except Exception as e:
                logger.warning(f"并行评估失败，回退到串行: {e}")
        
        # 串行评估
        return self._sequential_evaluate(
            factors,
            data,
            target,
            executor,
            values_cache=values_cache,
        )
    
    def _parallel_evaluate(
        self,
        factors: List[Dict],
        data: pd.DataFrame,
        target: pd.Series,
        executor: Callable,
        n_workers: int = 4,
        values_cache: Optional[Dict[Tuple[str, str], pd.Series]] = None,
    ) -> List[Dict]:
        """并行评估 - 使用ThreadPool或Celery"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        evaluated = []
        stats = {'exec_fail': 0, 'ic_nan': 0, 'ic_low': 0, 'icir_low': 0, 'success': 0}
        
        logger.info(f"  并行评估 {len(factors)} 个因子 (workers={n_workers})...")
        logger.info(f"  阈值: IC>{self.full_ic_threshold}, ICIR>{self.full_icir_threshold}")
        
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            
            for factor in factors:
                future = pool.submit(
                    self._evaluate_single_with_stats,
                    factor,
                    data,
                    target,
                    executor,
                    values_cache,
                )
                futures[future] = factor.get('name', factor.get('id', ''))
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"  进度: {completed}/{len(factors)}")
                
                try:
                    result, reason = future.result(timeout=120)
                    if result:
                        evaluated.append(result)
                        stats['success'] += 1
                    else:
                        stats[reason] = stats.get(reason, 0) + 1
                except Exception as e:
                    stats['exec_fail'] += 1
        
        # 打印统计
        logger.info(f"  评估统计:")
        logger.info(f"    - 执行失败: {stats.get('exec_fail', 0)}")
        logger.info(f"    - IC为NaN: {stats.get('ic_nan', 0)}")
        logger.info(f"    - IC过低: {stats.get('ic_low', 0)}")
        logger.info(f"    - ICIR过低: {stats.get('icir_low', 0)}")
        logger.info(f"    - 通过筛选: {stats.get('success', 0)}")
        
        return evaluated
    
    def _evaluate_single_with_stats(
        self,
        factor: Dict,
        data: pd.DataFrame,
        target: pd.Series,
        executor: Callable,
        values_cache: Optional[Dict[Tuple[str, str], pd.Series]] = None,
    ) -> Tuple[Optional[Dict], str]:
        """评估单个因子并返回失败原因"""
        name = factor.get('name', factor.get('id', 'unknown'))
        code = factor.get('code', '')
        
        try:
            # 执行因子
            values = self._get_factor_values(
                factor,
                data,
                executor,
                values_cache,
                cache_tag='full',
            )
            
            if values is None or len(values) == 0:
                # 打印前3个失败因子的详细信息
                if not hasattr(self, '_fail_count'):
                    self._fail_count = 0
                self._fail_count += 1
                if self._fail_count <= 3:
                    logger.warning(f"  因子执行失败 [{name}]: 返回空值")
                    logger.warning(f"    代码前100字符: {code[:100]}...")
                return None, 'exec_fail'
            
            # 完整IC计算
            eval_result = self._full_ic_eval(values, target)
            
            ic = eval_result.get('ic', 0)
            icir = eval_result.get('icir', 0)
            
            # 检查NaN
            if np.isnan(ic) or np.isnan(icir):
                return None, 'ic_nan'
            
            # 阈值检查
            if abs(ic) < self.full_ic_threshold:
                return None, 'ic_low'
            if abs(icir) < self.full_icir_threshold:
                return None, 'icir_low'
            
            # 通过筛选
            logger.info(f"  ✓ 因子 {name}: IC={ic:.4f}, ICIR={icir:.2f}")
            
            # 更新因子信息
            factor_result = factor.copy()
            factor_result.update(eval_result)
            factor_result['values'] = values
            
            return factor_result, 'success'
            
        except Exception as e:
            return None, 'exec_fail'
    
    def _evaluate_single(
        self,
        factor: Dict,
        data: pd.DataFrame,
        target: pd.Series,
        executor: Callable,
        debug: bool = False,
        values_cache: Optional[Dict[Tuple[str, str], pd.Series]] = None,
    ) -> Optional[Dict]:
        """评估单个因子"""
        name = factor.get('name', factor.get('id', 'unknown'))
        
        try:
            # 执行因子
            values = self._get_factor_values(
                factor,
                data,
                executor,
                values_cache,
                cache_tag='full',
            )
            
            if values is None or len(values) == 0:
                if debug:
                    logger.debug(f"  因子 {name}: 执行失败或返回空值")
                return None
            
            # 完整IC计算
            eval_result = self._full_ic_eval(values, target)
            
            ic = eval_result.get('ic', 0)
            icir = eval_result.get('icir', 0)
            
            # 检查NaN
            if np.isnan(ic) or np.isnan(icir):
                if debug:
                    logger.debug(f"  因子 {name}: IC={ic}, ICIR={icir} (NaN)")
                return None
            
            # 阈值检查
            if abs(ic) < self.full_ic_threshold:
                if debug:
                    logger.debug(f"  因子 {name}: IC={ic:.4f} < {self.full_ic_threshold}")
                return None
            if abs(icir) < self.full_icir_threshold:
                if debug:
                    logger.debug(f"  因子 {name}: ICIR={icir:.4f} < {self.full_icir_threshold}")
                return None
            
            # 通过筛选
            logger.info(f"  ✓ 因子 {name}: IC={ic:.4f}, ICIR={icir:.2f}")
            
            # 更新因子信息
            factor_result = factor.copy()
            factor_result.update(eval_result)
            factor_result['values'] = values
            
            return factor_result
            
        except Exception as e:
            if debug:
                logger.debug(f"  因子 {name}: 异常 - {e}")
            return None
    
    def _sequential_evaluate(
        self,
        factors: List[Dict],
        data: pd.DataFrame,
        target: pd.Series,
        executor: Callable,
        values_cache: Optional[Dict[Tuple[str, str], pd.Series]] = None,
    ) -> List[Dict]:
        """串行评估"""
        evaluated = []
        
        for i, factor in enumerate(factors):
            if (i + 1) % 10 == 0:
                logger.info(f"  进度: {i+1}/{len(factors)}")
            
            result = self._evaluate_single(
                factor,
                data,
                target,
                executor,
                values_cache=values_cache,
            )
            if result:
                evaluated.append(result)
        
        return evaluated
    
    def _full_ic_eval(self, factor: pd.Series, target: pd.Series) -> Dict:
        """
        完整IC评估 - 向量化优化版本
        
        速度优化:
        1. 使用pandas rolling.corr() 代替Python循环
        2. 按日期分组计算截面IC
        """
        # 对齐数据
        aligned = pd.concat([factor, target], axis=1).dropna()
        if len(aligned) < 100:
            return {'ic': 0, 'icir': 0, 'rank_ic': 0, 'rank_icir': 0}
        
        factor_vals = aligned.iloc[:, 0]
        target_vals = aligned.iloc[:, 1]
        
        # Rank IC (Spearman)
        ic = factor_vals.corr(target_vals, method='spearman')
        rank_ic = ic
        
        # 滚动IC计算ICIR - 向量化版本
        # 按日期分组计算每日截面IC
        if hasattr(factor_vals.index, 'get_level_values'):
            try:
                # MultiIndex (instrument, datetime)
                dates = factor_vals.index.get_level_values('datetime').unique()
                daily_ic = []
                
                # 采样计算（加速）
                sample_dates = dates[::max(1, len(dates)//100)]  # 最多100个日期
                
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
                else:
                    icir = 0
            except Exception:
                # 退化到简单计算
                icir = ic / 0.1 if abs(ic) > 0.01 else 0
        else:
            # 简单Index - 使用rolling
            factor_rank = factor_vals.rank()
            target_rank = target_vals.rank()
            rolling_corr = factor_rank.rolling(20).corr(target_rank)
            rolling_corr = rolling_corr.dropna()
            if len(rolling_corr) > 0:
                icir = rolling_corr.mean() / (rolling_corr.std() + 1e-8)
            else:
                icir = 0
        
        return {
            'ic': ic if not np.isnan(ic) else 0,
            'icir': icir if not np.isnan(icir) else 0,
            'rank_ic': rank_ic if not np.isnan(rank_ic) else 0,
            'rank_icir': icir,
        }
    
    # ============================================================
    # Stage 5: 正交化组合优化
    # ============================================================
    
    def _orthogonal_select(
        self,
        factors: List[Dict],
        data: pd.DataFrame,
        target: pd.Series,
        executor: Callable,
    ) -> Tuple[List[Dict], pd.DataFrame]:
        """
        正交化组合优化 - Greedy Forward Selection
        
        核心算法:
        1. 按IC排序因子
        2. 依次添加因子，检查:
           - 与已选因子相关性 < threshold
           - 正交化后的边际IC > min_marginal_ic
        3. 返回最优组合
        """
        if len(factors) == 0:
            return [], pd.DataFrame()
        
        # 按IC排序
        factors = sorted(
            factors,
            key=lambda x: abs(x.get('ic', x.get('quick_ic', 0))),
            reverse=True,
        )
        
        selected = []
        selected_names = []
        selected_values = pd.DataFrame()
        
        for factor in factors:
            if len(selected) >= self.max_factors:
                break
            
            name = factor.get('name', factor.get('id', f'factor_{len(selected)}'))
            values = factor.get('values')
            
            if values is None:
                try:
                    code = factor.get('code', '')
                    if executor:
                        values = executor(code, data)
                    else:
                        values = self._execute_factor(code, data)
                    if values is None or len(values) == 0:
                        continue
                    factor['values'] = values
                except Exception:
                    continue
            
            # 检查相关性约束
            if len(selected) > 0:
                max_corr = 0
                for existing_name in selected_names:
                    if existing_name in selected_values.columns:
                        corr = abs(values.corr(selected_values[existing_name]))
                        max_corr = max(max_corr, corr if not np.isnan(corr) else 0)
                
                if max_corr > self.corr_threshold:
                    logger.debug(f"跳过 {name}: 相关性过高 ({max_corr:.2f})")
                    continue
            
            # 检查边际IC贡献 (正交化)
            if len(selected) > 0 and self.min_marginal_ic > 0:
                marginal_ic = self._compute_marginal_ic(
                    values, selected_values, target
                )
                
                if abs(marginal_ic) < self.min_marginal_ic:
                    logger.debug(f"跳过 {name}: 边际IC不足 ({marginal_ic:.4f})")
                    continue
            
            # 添加因子
            selected.append(factor)
            selected_names.append(name)
            selected_values[name] = values
            
            logger.info(f"  选中: {name}, IC={factor.get('ic', 0):.4f}, "
                       f"累计: {len(selected)}")
        
        # 计算相关性矩阵
        corr_matrix = selected_values.corr() if len(selected_values.columns) > 0 else pd.DataFrame()
        
        return selected, corr_matrix
    
    def _compute_marginal_ic(
        self,
        new_factor: pd.Series,
        existing: pd.DataFrame,
        target: pd.Series,
    ) -> float:
        """
        计算正交化后的边际IC
        
        方法: 用已选因子回归新因子，计算残差与target的IC
        """
        if existing.empty:
            return new_factor.corr(target, method='spearman')
        
        try:
            # 对齐
            aligned = pd.concat([new_factor, existing, target], axis=1).dropna()
            if len(aligned) < 100:
                return 0
            
            y = aligned.iloc[:, 0].values
            X = aligned.iloc[:, 1:-1].values
            t = aligned.iloc[:, -1].values
            
            # 回归
            model = LinearRegression()
            model.fit(X, y)
            residual = y - model.predict(X)
            
            # 残差IC
            return pd.Series(residual).corr(pd.Series(t), method='spearman')
        except Exception:
            return 0
    
    # ============================================================
    # 工具方法
    # ============================================================
    
    def _execute_factor(self, code: str, data: pd.DataFrame) -> Optional[pd.Series]:
        """默认因子执行器"""
        if not code or not code.strip():
            return None
        
        try:
            # 准备环境
            local_env = {
                'df': data,
                'np': np,
                'pd': pd,
            }
            
            # 执行代码
            if 'def compute_alpha' in code:
                exec(code, local_env)
                result = local_env['compute_alpha'](data)
            else:
                result = eval(code, local_env)
            
            if isinstance(result, pd.Series):
                return result
            elif isinstance(result, pd.DataFrame):
                return result.iloc[:, 0]
            else:
                return pd.Series(result)
                
        except Exception as e:
            logger.debug(f"因子执行失败: {e}")
            return None
    
    def _compute_ic(self, factor: pd.Series, target: pd.Series) -> float:
        """计算IC"""
        try:
            aligned = pd.concat([factor, target], axis=1).dropna()
            if len(aligned) < 30:
                return 0
            return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method='spearman')
        except Exception:
            return 0


# ============================================================
# 便捷函数
# ============================================================

def select_factors(
    candidates: List[Dict],
    data: pd.DataFrame,
    target: pd.Series,
    max_factors: int = 30,
    quick_ic_threshold: float = 0.01,
    corr_threshold: float = 0.7,
    executor: Callable = None,
) -> SelectionResult:
    """
    便捷函数: 因子筛选
    
    Args:
        candidates: 因子列表
        data: 特征数据
        target: 目标收益
        max_factors: 最大因子数
        quick_ic_threshold: 快速IC阈值
        corr_threshold: 相关性阈值
        executor: 因子执行器
    
    Returns:
        SelectionResult
    """
    selector = FactorSelector(
        max_factors=max_factors,
        quick_ic_threshold=quick_ic_threshold,
        corr_threshold=corr_threshold,
    )
    
    return selector.select(candidates, data, target, executor)


def quick_filter(
    factors: List[Dict],
    data: pd.DataFrame,
    target: pd.Series,
    sample_ratio: float = 0.1,
    ic_threshold: float = 0.01,
    executor: Callable = None,
) -> List[Dict]:
    """
    便捷函数: 快速预筛选
    """
    selector = FactorSelector(
        quick_sample_ratio=sample_ratio,
        quick_ic_threshold=ic_threshold,
    )
    return selector._quick_filter(factors, data, target, executor)


def orthogonal_select(
    factors: List[Dict],
    data: pd.DataFrame,
    target: pd.Series,
    max_factors: int = 30,
    corr_threshold: float = 0.7,
    executor: Callable = None,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    便捷函数: 正交化组合选择
    """
    selector = FactorSelector(
        max_factors=max_factors,
        corr_threshold=corr_threshold,
    )
    return selector._orthogonal_select(factors, data, target, executor)


# ============================================================
# 命令行测试
# ============================================================

if __name__ == "__main__":
    # 测试示例
    print("因子筛选模块测试")
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.abs(np.random.randn(n_samples)) * 1e6,
    })
    target = pd.Series(np.random.randn(n_samples) * 0.02)
    
    # 模拟因子
    factors = [
        {'id': 'mom_5', 'name': '动量5日', 'code': 'df["close"].pct_change(5)'},
        {'id': 'mom_10', 'name': '动量10日', 'code': 'df["close"].pct_change(10)'},
        {'id': 'vol_std', 'name': '波动率', 'code': 'df["close"].pct_change().rolling(20).std()'},
    ]
    
    # 测试
    result = select_factors(factors, data, target, max_factors=2)
    print(result.summary())
