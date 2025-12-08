"""
进化引擎核心
"""

import re
import random
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import numpy as np

from .config import EvolutionConfig
from .individual import Individual, EvolutionHistory, GenerationStats


class EvolutionaryEngine:
    """进化式因子生成引擎"""
    
    def __init__(
        self,
        llm_generator: Any,           # LLM生成器
        evaluator: Any,               # 因子评估器
        sandbox: Any,                 # 代码沙箱
        config: EvolutionConfig = None,
    ):
        self.llm = llm_generator
        self.evaluator = evaluator
        self.sandbox = sandbox
        self.config = config or EvolutionConfig()
        
        self.population: List[Individual] = []
        self.history = EvolutionHistory()
        self.generation = 0
    
    def run(
        self,
        task: str,
        data: Any,                    # DataFrame
        target: Any,                  # Series
        context: Dict = None,
    ) -> EvolutionHistory:
        """
        运行进化流程
        
        Args:
            task: 用户任务描述
            data: 股票数据
            target: 目标收益率
            context: 附加上下文 {data_schema, historical, ...}
        """
        context = context or {}
        
        # 初始化历史
        self.history = EvolutionHistory(
            task=task,
            config=self.config.__dict__,
            start_time=datetime.now().isoformat(),
        )
        
        print(f"[进化引擎] 开始任务: {task[:50]}...")
        print(f"[进化引擎] 配置: pop={self.config.population_size}, elite={self.config.elite_size}, max_gen={self.config.max_generations}")
        
        # Step 1: 初始化种群
        print(f"\n[Gen 0] 初始化种群...")
        self.population = self._initialize_population(task, context)
        
        # Step 2: 评估初始种群
        self._evaluate_population(data, target)
        self._log_generation_stats()
        self.history.add_generation(self.population, 0)
        
        best_fitness = max(ind.fitness for ind in self.population)
        no_improve_count = 0
        
        # Step 3: 进化循环
        for gen in range(1, self.config.max_generations + 1):
            self.generation = gen
            print(f"\n[Gen {gen}] 开始进化...")
            
            # 3.1 选择精英
            elites = self._select_elites()
            print(f"  精英: {[e.id for e in elites]}")
            
            # 3.2 基于反馈生成后代
            offspring = self._generate_offspring(elites, task, context)
            print(f"  后代: {len(offspring)} 个")
            
            # 3.3 注入随机个体
            random_inds = self._inject_random(task, context)
            print(f"  随机: {len(random_inds)} 个")
            
            # 3.4 组成新种群
            self.population = elites + offspring + random_inds
            
            # 3.5 评估
            self._evaluate_population(data, target)
            self._log_generation_stats()
            self.history.add_generation(self.population, gen)
            
            # 3.6 检查终止条件
            current_best = max(ind.fitness for ind in self.population)
            
            if current_best >= self.config.min_fitness:
                print(f"\n✓ 达到目标适应度 {current_best:.4f} >= {self.config.min_fitness}")
                break
            
            if current_best <= best_fitness:
                no_improve_count += 1
                if no_improve_count >= self.config.patience:
                    print(f"\n✗ 连续 {self.config.patience} 代无改进，早停")
                    break
            else:
                best_fitness = current_best
                no_improve_count = 0
        
        # Step 4: 收尾
        self.history.final_elites = self._select_elites()
        self.history.end_time = datetime.now().isoformat()
        
        print("\n" + "=" * 50)
        print("进化完成!")
        print(self.history.summary())
        
        return self.history
    
    def _initialize_population(self, task: str, context: Dict) -> List[Individual]:
        """初始化种群"""
        prompt = self._build_init_prompt(task, context)
        
        # 调用LLM生成多个变体
        response = self.llm.generate(prompt)
        codes = self._extract_code_blocks(response)
        
        # 补充不足的
        attempts = 0
        while len(codes) < self.config.population_size and attempts < 3:
            extra_prompt = f"再生成 {self.config.population_size - len(codes)} 个不同的因子变体，要求与前面不同"
            extra_response = self.llm.generate(extra_prompt)
            codes.extend(self._extract_code_blocks(extra_response))
            attempts += 1
        
        return [
            Individual(code=code, generation=0)
            for code in codes[:self.config.population_size]
        ]
    
    def _evaluate_population(self, data: Any, target: Any):
        """评估种群"""
        def evaluate_one(ind: Individual) -> Individual:
            if ind.is_evaluated:
                return ind
            
            try:
                # 执行代码
                factor_values = self.sandbox.execute(ind.code, data)
                
                # 评估指标
                metrics = self.evaluator.evaluate(factor_values, target)
                
                # 计算适应度
                fitness = self._compute_fitness(metrics)
                
                # 生成反思
                reflection = self._generate_reflection(metrics)
                
                ind.mark_evaluated(metrics, fitness, reflection)
                
            except Exception as e:
                ind.mark_failed(str(e))
            
            return ind
        
        # 并行评估
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            self.population = list(executor.map(evaluate_one, self.population))
    
    def _compute_fitness(self, metrics: Dict) -> float:
        """计算综合适应度"""
        score = 0.0
        weights = self.config.fitness_weights
        targets = self.config.fitness_targets
        
        # IC
        ic = metrics.get('ic', 0)
        score += weights.get('ic', 0) * min(max(ic, 0) / targets['ic'], 1.0)
        
        # ICIR
        icir = metrics.get('icir', 0)
        score += weights.get('icir', 0) * min(max(icir, 0) / targets['icir'], 1.0)
        
        # Sharpe
        sharpe = metrics.get('sharpe', 0)
        score += weights.get('sharpe', 0) * min(max(sharpe, 0) / targets['sharpe'], 1.0)
        
        # MaxDrawdown (负向)
        mdd = abs(metrics.get('max_drawdown', 0.5))
        mdd_score = max(1 - mdd / targets['max_drawdown'], 0)
        score += weights.get('max_drawdown', 0) * mdd_score
        
        # Turnover (负向)
        turnover = metrics.get('turnover', 1.0)
        turnover_score = max(1 - turnover / targets['turnover'], 0)
        score += weights.get('turnover', 0) * turnover_score
        
        return score
    
    def _generate_reflection(self, metrics: Dict) -> str:
        """生成诊断反思"""
        lines = []
        
        ic = metrics.get('ic', 0)
        if ic < 0.02:
            lines.append(f"⚠️ IC过低({ic:.4f})：预测能力弱，建议尝试不同信号构造")
        elif ic > 0.05:
            lines.append(f"✓ IC优秀({ic:.4f})，注意过拟合风险")
        
        icir = metrics.get('icir', 0)
        if icir < 0.3:
            lines.append(f"⚠️ ICIR过低({icir:.4f})：IC不稳定，建议增加平滑处理")
        
        sharpe = metrics.get('sharpe', 0)
        if sharpe < 0.5:
            lines.append(f"⚠️ 夏普偏低({sharpe:.2f})：风险收益比差")
        
        mdd = abs(metrics.get('max_drawdown', 0))
        if mdd > 0.2:
            lines.append(f"⚠️ 回撤过大({mdd:.1%})：建议添加风控机制")
        
        turnover = metrics.get('turnover', 0)
        if turnover > 0.5:
            lines.append(f"⚠️ 换手率过高({turnover:.1%})：建议延长持仓周期")
        
        if not lines:
            lines.append("✓ 各项指标均衡")
        
        return "\n".join(lines)
    
    def _select_elites(self) -> List[Individual]:
        """选择精英"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        elites = []
        for ind in sorted_pop:
            if len(elites) >= self.config.elite_size:
                break
            
            # 多样性检查
            is_diverse = all(
                self._code_similarity(ind.code, e.code) < self.config.diversity_threshold
                for e in elites
            )
            
            if is_diverse or len(elites) == 0:
                ind.is_elite = True
                elites.append(ind)
        
        return elites
    
    def _generate_offspring(self, elites: List[Individual], task: str, context: Dict) -> List[Individual]:
        """基于精英生成后代"""
        offspring = []
        
        for elite in elites[:2]:
            prompt = self._build_improve_prompt(elite, task, context)
            response = self.llm.generate(prompt)
            codes = self._extract_code_blocks(response)
            
            for code in codes:
                offspring.append(Individual(
                    code=code,
                    generation=self.generation,
                    parent_ids=[elite.id],
                ))
        
        return offspring[:self.config.offspring_size]
    
    def _inject_random(self, task: str, context: Dict) -> List[Individual]:
        """注入随机个体"""
        n = max(1, int(self.config.population_size * self.config.random_injection_rate))
        
        prompt = f"""
生成 {n} 个完全创新的因子，不基于常规模式。
尝试：组合不相关指标、非线性变换、另类思路。

任务: {task}
"""
        
        if 'data_schema' in context:
            prompt += f"\n数据: {context['data_schema'][:500]}..."
        
        response = self.llm.generate(prompt)
        codes = self._extract_code_blocks(response)
        
        return [
            Individual(code=code, generation=self.generation, parent_ids=['random'])
            for code in codes[:n]
        ]
    
    def _build_init_prompt(self, task: str, context: Dict) -> str:
        """构建初始化Prompt"""
        prompt = f"""
## 任务
生成 {self.config.population_size} 个**完全不同**的因子变体:
{task}

## 多样性要求
每个变体应尝试不同的:
1. 数据字段组合
2. 时间窗口 (5/10/20/60日)
3. 数学运算 (均值/标准差/排名/比值)
4. 信号逻辑 (动量/反转/波动/价值)
"""
        
        if 'data_schema' in context:
            prompt += f"\n## 数据字典\n{context['data_schema']}\n"
        
        if 'historical' in context:
            prompt += f"\n## 历史经验\n{context['historical']}\n"
        
        prompt += """
## 输出格式
每个因子用 ```python 包裹，函数签名: def compute_alpha(df) -> pd.Series
"""
        
        return prompt
    
    def _build_improve_prompt(self, elite: Individual, task: str, context: Dict) -> str:
        """构建改进Prompt"""
        return f"""
## 任务
基于以下因子进行改进优化。

## 原始因子
```python
{elite.code}
```

## 评估结果
- 综合得分: {elite.fitness:.2f}/1.00
- IC: {elite.metrics.get('ic', 0):.4f}
- ICIR: {elite.metrics.get('icir', 0):.4f}
- 夏普: {elite.metrics.get('sharpe', 0):.2f}
- 回撤: {elite.metrics.get('max_drawdown', 0):.1%}
- 换手: {elite.metrics.get('turnover', 0):.1%}

## 诊断报告
{elite.reflection}

## 请生成 {self.config.offspring_size // 2} 个改进版本
针对诊断报告中的问题进行改进。
每个版本用 ```python 包裹。
"""
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """提取代码块"""
        pattern = r'```python\s*(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        valid_codes = []
        for code in matches:
            code = code.strip()
            if 'def compute_alpha' in code or 'def alpha' in code:
                valid_codes.append(code)
        
        return valid_codes
    
    def _code_similarity(self, code1: str, code2: str) -> float:
        """计算代码相似度"""
        return SequenceMatcher(None, code1, code2).ratio()
    
    def _log_generation_stats(self):
        """记录当代统计"""
        fitnesses = [ind.fitness for ind in self.population]
        valid_count = sum(1 for f in fitnesses if f > 0)
        
        print(f"  适应度: max={max(fitnesses):.4f}, avg={np.mean(fitnesses):.4f}")
        print(f"  有效: {valid_count}/{len(fitnesses)}")
