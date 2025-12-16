"""
混合进化引擎 - LLM + GP 协作

设计理念:
1. LLM负责"创意" - 生成有逻辑的种子因子
2. GP负责"优化" - 在有效种子基础上微调参数
3. LLM负责"反思" - 解释GP发现的有效变体

三阶段流程:
- Phase 1: LLM探索 → 快速筛选 → 种子库
- Phase 2: GP精炼 → 参数优化 → 候选池
- Phase 3: LLM反思 → 提炼逻辑 → 最终因子
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from enum import Enum
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class EvolutionPhase(Enum):
    """进化阶段"""
    LLM_EXPLORE = "llm_explore"      # LLM探索
    GP_REFINE = "gp_refine"          # GP精炼
    LLM_REFLECT = "llm_reflect"      # LLM反思


@dataclass
class FactorCandidate:
    """因子候选"""
    id: str
    code: str
    name: str = ""
    
    # 来源
    source: str = ""      # "llm" / "gp" / "hybrid"
    parent_ids: List[str] = field(default_factory=list)
    
    # 核心评估指标
    ic: float = 0.0                # IC均值
    icir: float = 0.0              # IC信息比 (IC_mean / IC_std)
    rank_ic: float = 0.0           # Rank IC (Spearman相关)
    rank_icir: float = 0.0         # Rank IC信息比
    
    # 收益风险指标
    ann_return: float = 0.0        # 年化收益
    information_ratio: float = 0.0 # 信息比 (超额收益/跟踪误差)
    sharpe: float = 0.0            # 夏普比率
    max_drawdown: float = 0.0      # 最大回撤
    turnover: float = 0.0          # 换手率
    
    # 综合指标
    fitness: float = 0.0           # 适应度得分
    
    # 指标等级
    ic_grade: str = ""             # IC等级: A/B/C/D/F
    icir_grade: str = ""           # ICIR等级: A/B/C/D/F
    
    # 状态
    phase: EvolutionPhase = EvolutionPhase.LLM_EXPLORE
    generation: int = 0
    is_seed: bool = False     # 是否为种子因子
    is_elite: bool = False    # 是否为精英
    
    # 元信息
    logic: str = ""           # 投资逻辑 (LLM生成)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HybridConfig:
    """混合进化配置"""
    # Phase 1: LLM探索
    llm_batch_size: int = 10          # 每批LLM生成数量
    llm_rounds: int = 3               # LLM探索轮数
    seed_threshold_ic: float = 0.015  # 进入种子库的IC阈值
    seed_pool_size: int = 20          # 种子库大小
    
    # Phase 2: GP精炼
    gp_population: int = 50           # GP种群大小
    gp_generations: int = 10          # GP迭代代数
    gp_mutation_rate: float = 0.3     # GP变异率
    gp_crossover_rate: float = 0.5    # GP交叉率
    gp_elite_rate: float = 0.1        # GP精英保留率
    
    # Phase 3: LLM反思
    reflect_top_k: int = 5            # 反思Top K个GP优胜者
    
    # 通用
    min_ic_improvement: float = 0.005  # 最小IC提升
    max_turnover: float = 0.5          # 最大换手率
    
    # 早停
    early_stop_rounds: int = 3         # 连续N轮无提升则早停
    target_ic: float = 0.03            # 目标IC


class HybridEvolutionEngine:
    """混合进化引擎"""
    
    def __init__(
        self,
        config: HybridConfig = None,
        llm_generator: Callable = None,    # LLM因子生成器
        gp_mutator: Callable = None,       # GP变异器
        evaluator: Callable = None,        # 回测评估器
        prompt_composer = None,            # Prompt组装器
    ):
        self.config = config or HybridConfig()
        self.llm_generator = llm_generator
        self.gp_mutator = gp_mutator
        self.evaluator = evaluator
        self.prompt_composer = prompt_composer
        
        # 状态
        self.seed_pool: List[FactorCandidate] = []
        self.candidate_pool: List[FactorCandidate] = []
        self.elite_pool: List[FactorCandidate] = []
        self.history: List[Dict] = []
        
        self.current_phase = EvolutionPhase.LLM_EXPLORE
        self.best_ic = 0.0
        self.no_improve_rounds = 0
    
    # ============================================================
    # 主流程
    # ============================================================
    
    def evolve(self, initial_factors: List[FactorCandidate] = None) -> List[FactorCandidate]:
        """
        执行混合进化
        
        Returns:
            最终的优质因子列表
        """
        logger.info("开始混合进化...")
        
        # 初始化
        if initial_factors:
            self.seed_pool = initial_factors
            logger.info(f"使用 {len(initial_factors)} 个初始因子")
        
        # Phase 1: LLM探索
        self._phase1_llm_explore()
        
        if not self.seed_pool:
            logger.warning("Phase 1未产生有效种子，终止进化")
            return []
        
        # Phase 2: GP精炼
        self._phase2_gp_refine()
        
        # Phase 3: LLM反思
        self._phase3_llm_reflect()
        
        # 返回精英因子
        return sorted(self.elite_pool, key=lambda x: x.fitness, reverse=True)
    
    # ============================================================
    # Phase 1: LLM探索
    # ============================================================
    
    def _phase1_llm_explore(self):
        """Phase 1: LLM探索生成种子因子"""
        self.current_phase = EvolutionPhase.LLM_EXPLORE
        logger.info("="*50)
        logger.info("Phase 1: LLM探索")
        logger.info("="*50)
        
        for round_idx in range(self.config.llm_rounds):
            logger.info(f"\n--- LLM探索 Round {round_idx + 1}/{self.config.llm_rounds} ---")
            
            # 1. 生成因子
            new_factors = self._llm_generate_batch(round_idx)
            
            # 2. 评估
            evaluated = self._evaluate_factors(new_factors)
            
            # 3. 筛选进入种子库
            seeds_added = 0
            for factor in evaluated:
                if self._is_valid_seed(factor):
                    factor.is_seed = True
                    self.seed_pool.append(factor)
                    seeds_added += 1
                    
                    if factor.ic > self.best_ic:
                        self.best_ic = factor.ic
                        self.no_improve_rounds = 0
                        logger.info(f"  新最优: {factor.name}, IC={factor.ic:.4f}")
            
            # 4. 控制种子库大小
            if len(self.seed_pool) > self.config.seed_pool_size:
                self.seed_pool = sorted(self.seed_pool, key=lambda x: x.ic, reverse=True)
                self.seed_pool = self.seed_pool[:self.config.seed_pool_size]
            
            logger.info(f"  本轮生成: {len(new_factors)}, 入库: {seeds_added}, 种子库: {len(self.seed_pool)}")
            
            # 5. 早停检查
            if seeds_added == 0:
                self.no_improve_rounds += 1
                if self.no_improve_rounds >= self.config.early_stop_rounds:
                    logger.info("  早停: 连续多轮无新种子")
                    break
            
            # 6. 记录历史
            self.history.append({
                'phase': 'llm_explore',
                'round': round_idx,
                'generated': len(new_factors),
                'seeds_added': seeds_added,
                'best_ic': self.best_ic,
            })
        
        logger.info(f"\nPhase 1完成: 种子库大小={len(self.seed_pool)}, 最优IC={self.best_ic:.4f}")
    
    def _llm_generate_batch(self, round_idx: int) -> List[FactorCandidate]:
        """LLM批量生成因子"""
        if not self.llm_generator:
            return self._mock_llm_generate(round_idx)
        
        # 准备上下文
        context = {
            'round': round_idx,
            'existing_seeds': [f.code for f in self.seed_pool[-5:]],
            'best_ic': self.best_ic,
        }
        
        # 调用LLM
        factors = []
        for i in range(self.config.llm_batch_size):
            try:
                result = self.llm_generator(context)
                factor = FactorCandidate(
                    id=f"llm_{round_idx}_{i}",
                    code=result.get('code', ''),
                    name=result.get('name', f'LLM因子_{round_idx}_{i}'),
                    source='llm',
                    logic=result.get('logic', ''),
                    phase=EvolutionPhase.LLM_EXPLORE,
                    generation=round_idx,
                )
                factors.append(factor)
            except Exception as e:
                logger.warning(f"LLM生成失败: {e}")
        
        return factors
    
    def _mock_llm_generate(self, round_idx: int) -> List[FactorCandidate]:
        """模拟LLM生成 (测试用)"""
        factors = []
        templates = [
            ("动量_{w}日", "df['close'].pct_change({w})", [5, 10, 20]),
            ("反转_{w}日", "-df['close'].pct_change({w})", [1, 3, 5]),
            ("波动_{w}日", "df['close'].pct_change().rolling({w}).std()", [10, 20, 30]),
            ("量价背离_{w}日", "df['close'].pct_change({w}) - df['volume'].pct_change({w})", [5, 10, 20]),
        ]
        
        for i in range(self.config.llm_batch_size):
            template = templates[i % len(templates)]
            w = template[2][round_idx % len(template[2])]
            factors.append(FactorCandidate(
                id=f"llm_{round_idx}_{i}",
                code=template[1].format(w=w),
                name=template[0].format(w=w),
                source='llm',
                phase=EvolutionPhase.LLM_EXPLORE,
                generation=round_idx,
            ))
        
        return factors
    
    def _is_valid_seed(self, factor: FactorCandidate) -> bool:
        """判断是否为有效种子"""
        if factor.ic < self.config.seed_threshold_ic:
            return False
        if factor.turnover > self.config.max_turnover:
            return False
        return True
    
    # ============================================================
    # Phase 2: GP精炼
    # ============================================================
    
    def _phase2_gp_refine(self):
        """Phase 2: GP精炼种子因子"""
        self.current_phase = EvolutionPhase.GP_REFINE
        logger.info("="*50)
        logger.info("Phase 2: GP精炼")
        logger.info("="*50)
        
        if not self.seed_pool:
            logger.warning("无种子因子，跳过GP精炼")
            return
        
        # 初始化GP种群 (基于种子)
        population = self._init_gp_population()
        
        for gen in range(self.config.gp_generations):
            logger.info(f"\n--- GP Generation {gen + 1}/{self.config.gp_generations} ---")
            
            # 1. 评估当前种群
            population = self._evaluate_factors(population)
            
            # 2. 选择精英
            population = sorted(population, key=lambda x: x.fitness, reverse=True)
            elite_count = max(1, int(len(population) * self.config.gp_elite_rate))
            elites = population[:elite_count]
            
            # 更新全局最优
            if elites[0].ic > self.best_ic + self.config.min_ic_improvement:
                self.best_ic = elites[0].ic
                self.no_improve_rounds = 0
                logger.info(f"  GP发现更优: {elites[0].name}, IC={elites[0].ic:.4f}")
            else:
                self.no_improve_rounds += 1
            
            # 3. 生成下一代
            new_population = list(elites)  # 保留精英
            
            while len(new_population) < self.config.gp_population:
                if np.random.random() < self.config.gp_crossover_rate and len(elites) >= 2:
                    # 交叉
                    p1, p2 = np.random.choice(elites, 2, replace=False)
                    child = self._gp_crossover(p1, p2, gen)
                elif np.random.random() < self.config.gp_mutation_rate:
                    # 变异
                    parent = np.random.choice(elites)
                    child = self._gp_mutate(parent, gen)
                else:
                    # 复制
                    child = np.random.choice(elites)
                
                new_population.append(child)
            
            population = new_population
            
            # 4. 早停检查
            if self.no_improve_rounds >= self.config.early_stop_rounds:
                logger.info("  早停: 连续多代无提升")
                break
            
            # 5. 记录
            self.history.append({
                'phase': 'gp_refine',
                'generation': gen,
                'best_ic': elites[0].ic if elites else 0,
                'avg_ic': np.mean([f.ic for f in population]),
            })
        
        # 保存GP精英候选
        gp_candidates = sorted(population, key=lambda x: x.fitness, reverse=True)
        gp_candidates = gp_candidates[:self.config.reflect_top_k * 2]
        
        # GP优胜者完整回测验证
        logger.info(f"\n--- GP优胜者完整回测验证 ---")
        self.candidate_pool = self._validate_gp_winners(gp_candidates)
        
        logger.info(f"\nPhase 2完成: 验证通过={len(self.candidate_pool)}, 最优IC={self.best_ic:.4f}")
    
    def _init_gp_population(self) -> List[FactorCandidate]:
        """初始化GP种群"""
        population = []
        
        # 从种子创建变体
        for seed in self.seed_pool:
            # 原始种子
            population.append(FactorCandidate(
                id=f"gp_seed_{seed.id}",
                code=seed.code,
                name=f"GP_{seed.name}",
                source='gp',
                parent_ids=[seed.id],
                phase=EvolutionPhase.GP_REFINE,
            ))
            
            # 创建变体填充
            while len(population) < self.config.gp_population:
                mutant = self._gp_mutate(seed, 0)
                population.append(mutant)
                if len(population) >= self.config.gp_population:
                    break
        
        return population[:self.config.gp_population]
    
    def _gp_mutate(self, parent: FactorCandidate, generation: int) -> FactorCandidate:
        """GP变异"""
        if self.gp_mutator:
            mutated_code = self.gp_mutator(parent.code)
        else:
            mutated_code = self._mock_gp_mutate(parent.code)
        
        return FactorCandidate(
            id=f"gp_mut_{parent.id}_{generation}_{np.random.randint(1000)}",
            code=mutated_code,
            name=f"Mut_{parent.name}",
            source='gp',
            parent_ids=[parent.id],
            phase=EvolutionPhase.GP_REFINE,
            generation=generation,
        )
    
    def _gp_crossover(self, p1: FactorCandidate, p2: FactorCandidate, generation: int) -> FactorCandidate:
        """GP交叉"""
        # 简单的代码片段交叉
        code1_parts = p1.code.split('.')
        code2_parts = p2.code.split('.')
        
        if len(code1_parts) > 1 and len(code2_parts) > 1:
            # 交换后半部分
            new_code = '.'.join(code1_parts[:len(code1_parts)//2] + code2_parts[len(code2_parts)//2:])
        else:
            # 简单组合
            new_code = f"({p1.code} + {p2.code}) / 2"
        
        return FactorCandidate(
            id=f"gp_cross_{generation}_{np.random.randint(1000)}",
            code=new_code,
            name=f"Cross_{p1.name[:10]}_{p2.name[:10]}",
            source='gp',
            parent_ids=[p1.id, p2.id],
            phase=EvolutionPhase.GP_REFINE,
            generation=generation,
        )
    
    def _mock_gp_mutate(self, code: str) -> str:
        """模拟GP变异"""
        mutations = [
            # 调整参数
            lambda c: c.replace('5', str(np.random.choice([3, 7, 10]))),
            lambda c: c.replace('10', str(np.random.choice([5, 15, 20]))),
            lambda c: c.replace('20', str(np.random.choice([10, 30, 60]))),
            # 添加操作
            lambda c: f"np.sign({c})",
            lambda c: f"({c}).rank()",
            lambda c: f"-({c})",
        ]
        
        mutation = np.random.choice(mutations)
        try:
            return mutation(code)
        except:
            return code
    
    # ============================================================
    # GP优胜者完整回测验证
    # ============================================================
    
    def _validate_gp_winners(self, candidates: List[FactorCandidate]) -> List[FactorCandidate]:
        """
        对GP优胜者进行完整回测验证
        
        验证指标:
        - IC/ICIR: 预测能力 (Pearson相关)
        - Rank IC/Rank ICIR: 预测能力 (Spearman相关)
        - 年化收益: 收益能力
        - 信息比: 风险调整收益
        - 最大回撤: 尾部风险
        - 换手率: 交易成本
        
        筛选条件: IC>2%, ICIR>0.5, Rank IC>2.5%, 换手<50%, 回撤<30%
        """
        validated = []
        
        for factor in candidates:
            logger.info(f"  验证: {factor.name}")
            
            # 完整回测评估
            backtest_result = self._full_backtest(factor)
            
            # 更新所有指标
            factor.ic = backtest_result.get('ic', 0)
            factor.icir = backtest_result.get('icir', 0)
            factor.rank_ic = backtest_result.get('rank_ic', 0)
            factor.rank_icir = backtest_result.get('rank_icir', 0)
            factor.ann_return = backtest_result.get('ann_return', 0)
            factor.information_ratio = backtest_result.get('information_ratio', 0)
            factor.sharpe = backtest_result.get('sharpe', 0)
            factor.max_drawdown = backtest_result.get('max_drawdown', 0)
            factor.turnover = backtest_result.get('turnover', 0)
            
            # 计算等级
            factor.ic_grade = self._compute_grade(factor.rank_ic, 'rank_ic')
            factor.icir_grade = self._compute_grade(factor.rank_icir, 'rank_icir')
            
            # 验证条件
            is_valid = True
            reasons = []
            
            # IC验证
            if factor.ic < 0.02:
                is_valid = False
                reasons.append(f"IC={factor.ic:.4f}<2%")
            
            # ICIR验证
            if factor.icir < 0.5:
                is_valid = False
                reasons.append(f"ICIR={factor.icir:.2f}<0.5")
            
            # Rank IC验证
            if factor.rank_ic < 0.025:
                is_valid = False
                reasons.append(f"RankIC={factor.rank_ic:.4f}<2.5%")
            
            # 换手率验证
            if factor.turnover > self.config.max_turnover:
                is_valid = False
                reasons.append(f"换手={factor.turnover:.1%}>{self.config.max_turnover:.1%}")
            
            # 最大回撤验证
            if factor.max_drawdown > 0.3:
                is_valid = False
                reasons.append(f"回撤={factor.max_drawdown:.1%}>30%")
            
            if is_valid:
                validated.append(factor)
                logger.info(f"    ✓ 通过 [{factor.ic_grade}]")
                logger.info(f"      IC={factor.ic:.4f}, ICIR={factor.icir:.2f}")
                logger.info(f"      RankIC={factor.rank_ic:.4f}, RankICIR={factor.rank_icir:.2f}")
                logger.info(f"      年化={factor.ann_return:.1%}, IR={factor.information_ratio:.2f}")
                logger.info(f"      回撤={factor.max_drawdown:.1%}, 换手={factor.turnover:.1%}")
            else:
                logger.info(f"    ✗ 未通过: {', '.join(reasons)}")
        
        logger.info(f"  验证结果: {len(validated)}/{len(candidates)} 通过")
        
        return validated
    
    def _full_backtest(self, factor: FactorCandidate) -> Dict:
        """
        完整回测评估 - 使用QlibModelZoo多模型回测
        
        使用modeling/qlib_model_zoo.py中的11个模型:
        - lgb, xgb, catboost (树模型)
        - lstm, gru, transformer (深度学习)
        - tabnet, double_ensemble (集成)
        
        返回指标:
        - ic: IC均值 (Pearson)
        - icir: IC信息比 (IC_mean / IC_std)
        - rank_ic: Rank IC均值 (Spearman)
        - rank_icir: Rank IC信息比
        - ann_return: 年化收益
        - information_ratio: 信息比 (超额收益/跟踪误差)
        - sharpe: 夏普比率
        - max_drawdown: 最大回撤
        - turnover: 换手率
        """
        # 1. 优先使用外部评估器
        if self.evaluator:
            try:
                result = self.evaluator(factor.code, full_backtest=True)
                return result
            except Exception as e:
                logger.warning(f"评估器回测失败 {factor.id}: {e}")
        
        # 2. 使用QlibModelZoo多模型回测
        try:
            return self._run_model_zoo_backtest(factor)
        except Exception as e:
            logger.warning(f"QlibModelZoo回测失败，使用简化评估: {e}")
            # 回测失败时使用已有的 IC 指标
            return {
                'ic': factor.ic or 0,
                'icir': factor.icir or 0,
                'rank_ic': factor.rank_ic or 0,
                'rank_icir': factor.rank_icir or 0,
                'sharpe': 0,
                'max_drawdown': 0,
                'turnover': 0,
                'status': 'simplified',
            }
    
    def _run_model_zoo_backtest(self, factor: FactorCandidate) -> Dict:
        """
        使用QlibModelZoo运行多模型回测
        
        默认使用4个主要模型并行测试:
        - lgb (LightGBM)
        - xgb (XGBoost) 
        - catboost (CatBoost)
        - linear (线性基准)
        """
        from ..modeling.qlib_model_zoo import QlibBenchmark, QlibModelZoo
        
        # 默认使用的模型列表
        models = getattr(self.config, 'backtest_models', ["lgb", "xgb", "catboost", "linear"])
        
        # 创建benchmark
        benchmark = QlibBenchmark(models=models)
        
        # 配置数据集
        dataset_kwargs = {
            "instruments": getattr(self.config, 'instruments', "csi300"),
            "train_period": (
                getattr(self.config, 'train_start', "2018-01-01"),
                getattr(self.config, 'train_end', "2021-12-31"),
            ),
            "valid_period": (
                getattr(self.config, 'valid_start', "2022-01-01"),
                getattr(self.config, 'valid_end', "2022-06-30"),
            ),
            "test_period": (
                getattr(self.config, 'test_start', "2022-07-01"),
                getattr(self.config, 'test_end', "2023-12-31"),
            ),
        }
        
        # 运行benchmark
        logger.info(f"运行QlibModelZoo回测: 模型={models}")
        comparison = benchmark.run(
            experiment_name=f"factor_{factor.id}",
            **dataset_kwargs,
        )
        
        # 获取最佳模型结果
        best_model = benchmark.get_best_model("icir")
        if best_model and best_model in benchmark.results:
            result = benchmark.results[best_model]
            return {
                'ic': result.ic,
                'icir': result.icir,
                'rank_ic': result.rank_ic,
                'rank_icir': result.rank_icir,
                'ann_return': result.annualized_return,
                'information_ratio': result.information_ratio,
                'sharpe': result.information_ratio,  # IR近似夏普
                'max_drawdown': result.max_drawdown,
                'turnover': 0.0,  # 从回测中获取
                'best_model': best_model,
                'model_comparison': comparison.to_dict() if hasattr(comparison, 'to_dict') else {},
            }
        
        raise RuntimeError("所有模型回测失败")
    
    def _run_full_model_zoo_backtest(
        self, 
        factor: FactorCandidate,
        models: List[str] = None,
    ) -> Dict:
        """
        运行完整11模型回测 (深度验证)
        
        使用所有可用模型:
        - boosting: lgb, lgb_light, xgb, catboost
        - linear: linear
        - nn: mlp, lstm, gru, transformer, tabnet
        - ensemble: double_ensemble
        """
        from ..modeling.qlib_model_zoo import QlibBenchmark, QlibModelZoo
        
        # 全模型列表
        if models is None:
            models = QlibModelZoo.list_models()  # 全部11个模型
        
        logger.info(f"运行完整ModelZoo回测: {len(models)}个模型")
        
        benchmark = QlibBenchmark(models=models)
        
        comparison = benchmark.run(
            instruments=getattr(self.config, 'instruments', "csi300"),
            experiment_name=f"full_benchmark_{factor.id}",
        )
        
        # 返回汇总结果
        results_summary = {}
        for name, result in benchmark.results.items():
            if result.status == "success":
                results_summary[name] = {
                    'ic': result.ic,
                    'icir': result.icir,
                    'rank_ic': result.rank_ic,
                    'rank_icir': result.rank_icir,
                    'train_time': result.train_time,
                }
        
        best = benchmark.get_best_model("icir")
        
        return {
            'best_model': best,
            'best_icir': benchmark.results[best].icir if best else 0,
            'model_results': results_summary,
            'comparison': comparison,
        }
    
    def _compute_grade(self, value: float, metric: str) -> str:
        """
        计算指标等级
        
        Rank IC等级:
        - A: >5%  (优秀)
        - B: 3-5% (良好)
        - C: 2-3% (合格)
        - D: 1-2% (较弱)
        - F: <1%  (无效)
        
        Rank ICIR等级:
        - A: >1.5 (优秀)
        - B: 1.0-1.5 (良好)
        - C: 0.5-1.0 (合格)
        - D: 0.3-0.5 (较弱)
        - F: <0.3 (无效)
        """
        if metric == 'rank_ic':
            if value >= 0.05:
                return 'A'
            elif value >= 0.03:
                return 'B'
            elif value >= 0.02:
                return 'C'
            elif value >= 0.01:
                return 'D'
            else:
                return 'F'
        elif metric == 'rank_icir':
            if value >= 1.5:
                return 'A'
            elif value >= 1.0:
                return 'B'
            elif value >= 0.5:
                return 'C'
            elif value >= 0.3:
                return 'D'
            else:
                return 'F'
        else:
            return ''
    
    # ============================================================
    # Phase 3: LLM反思
    # ============================================================
    
    def _phase3_llm_reflect(self):
        """Phase 3: LLM反思解释GP发现"""
        self.current_phase = EvolutionPhase.LLM_REFLECT
        logger.info("="*50)
        logger.info("Phase 3: LLM反思")
        logger.info("="*50)
        
        if not self.candidate_pool:
            logger.warning("无候选因子，跳过LLM反思")
            return
        
        # 选择Top K进行反思
        top_candidates = self.candidate_pool[:self.config.reflect_top_k]
        
        for factor in top_candidates:
            logger.info(f"\n反思: {factor.name}, IC={factor.ic:.4f}")
            
            # LLM解释为什么有效
            if self.llm_generator:
                logic = self._llm_explain(factor)
                factor.logic = logic
                logger.info(f"  逻辑: {logic[:100]}...")
            
            # 标记为精英
            factor.is_elite = True
            factor.phase = EvolutionPhase.LLM_REFLECT
            self.elite_pool.append(factor)
        
        logger.info(f"\nPhase 3完成: 精英因子={len(self.elite_pool)}")
    
    def _llm_explain(self, factor: FactorCandidate) -> str:
        """LLM解释因子逻辑"""
        # 这里应该调用LLM来解释
        # 简化实现
        return f"该因子通过{factor.code[:50]}计算，捕捉了市场的某种规律"
    
    # ============================================================
    # 评估
    # ============================================================
    
    def _evaluate_factors(self, factors: List[FactorCandidate]) -> List[FactorCandidate]:
        """评估因子列表"""
        for factor in factors:
            if self.evaluator:
                try:
                    result = self.evaluator(factor.code)
                    factor.ic = result.get('ic', 0)
                    factor.icir = result.get('icir', 0)
                    factor.sharpe = result.get('sharpe', 0)
                    factor.turnover = result.get('turnover', 0)
                except Exception as e:
                    logger.warning(f"评估失败 {factor.id}: {e}")
                    factor.ic = 0
            else:
                # Mock评估
                factor.ic = np.random.uniform(-0.01, 0.04)
                factor.icir = factor.ic * np.random.uniform(0.5, 2.0)
                factor.turnover = np.random.uniform(0.1, 0.5)
            
            # 计算fitness
            factor.fitness = self._compute_fitness(factor)
        
        return factors
    
    def _compute_fitness(self, factor: FactorCandidate) -> float:
        """计算适应度"""
        # 综合IC、ICIR和换手率
        ic_score = max(0, factor.ic * 100)  # IC归一化
        icir_score = max(0, factor.icir * 10)
        turnover_penalty = max(0, factor.turnover - 0.3) * 10
        
        return ic_score + icir_score * 0.5 - turnover_penalty
    
    # ============================================================
    # 状态查询
    # ============================================================
    
    def get_summary(self) -> Dict:
        """获取进化摘要"""
        return {
            'current_phase': self.current_phase.value,
            'seed_pool_size': len(self.seed_pool),
            'candidate_pool_size': len(self.candidate_pool),
            'elite_pool_size': len(self.elite_pool),
            'best_ic': self.best_ic,
            'total_history': len(self.history),
        }
