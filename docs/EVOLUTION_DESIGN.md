# 进化式因子生成设计

> LLM + 遗传算法思想的因子进化引擎

## 1. 设计理念

### 传统方式的问题

```
用户: "写一个动量因子"
LLM:  返回1个因子
结果: 可能不好用，需要人工反复调整

问题:
1. 单次生成质量不稳定
2. LLM没有"试错"的机会
3. 缺乏系统性的改进机制
```

### 进化式生成的优势

```
用户: "写一个动量因子"
系统: 
  Gen 0: LLM生成16个变体 → 全部评估 → 选出Top 4
  Gen 1: Top 4 + 评估反馈 → LLM改进 → 新变体 → 评估
  Gen 2: 继续迭代...
  ...
  Gen N: 返回最优因子

优势:
1. 多样性探索 - 初始种群覆盖不同思路
2. 优胜劣汰 - 只有好因子能"繁衍"
3. 反馈驱动 - LLM基于评估报告改进
4. 防止局部最优 - 随机注入保持创新
```

---

## 2. 核心概念

### 2.1 个体 (Individual)

```yaml
Individual:
  id: string              # 唯一ID
  code: string            # 因子代码
  generation: int         # 所属代数
  parent_ids: list        # 父代ID(追溯进化路径)
  
  # 评估结果
  fitness: float          # 综合适应度 [0, 1]
  metrics: dict           # 详细指标 {ic, icir, sharpe, mdd, turnover}
  reflection: string      # 诊断反思报告
  
  # 状态
  is_evaluated: bool
  is_elite: bool          # 是否为精英
```

### 2.2 种群 (Population)

```yaml
Population:
  individuals: List[Individual]
  generation: int
  
  # 统计
  best_fitness: float
  avg_fitness: float
  diversity: float        # 种群多样性
```

### 2.3 适应度 (Fitness)

```
多目标优化，加权求和:

fitness = Σ(weight_i × normalize(metric_i))

权重分配:
┌─────────────┬────────┬───────────┬───────────────┐
│    指标     │  权重  │  目标值   │    归一化     │
├─────────────┼────────┼───────────┼───────────────┤
│ IC          │  0.25  │  > 0.05   │ min(IC/0.05, 1) │
│ ICIR        │  0.25  │  > 1.0    │ min(ICIR/1.0, 1)│
│ Sharpe      │  0.25  │  > 2.0    │ min(Sharpe/2, 1)│
│ MaxDrawdown │  0.15  │  < 20%    │ max(1-MDD/0.3,0)│
│ Turnover    │  0.10  │  < 50%    │ max(1-TO/1.0, 0)│
└─────────────┴────────┴───────────┴───────────────┘
```

---

## 3. 进化流程

### 3.1 整体流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      进化式因子生成                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 1: 初始化种群                                   │   │
│  │   - LLM生成16个不同变体                              │   │
│  │   - 提示词要求多样性                                 │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 2: 并行评估                                     │   │
│  │   - 沙箱执行每个因子                                 │   │
│  │   - 计算IC/ICIR/Sharpe/MDD/Turnover                │   │
│  │   - 生成诊断反思报告                                 │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 3: 选择精英                                     │   │
│  │   - 按适应度排序                                     │   │
│  │   - 保留Top K (通常K=4)                             │   │
│  │   - 保证多样性(去除过于相似的)                       │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 4: 生成后代                                     │   │
│  │   - 精英 + 诊断报告 → 喂给LLM                       │   │
│  │   - LLM基于反馈改进                                  │   │
│  │   - 生成8个改进变体                                  │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 5: 随机注入                                     │   │
│  │   - 生成1-2个完全随机的新因子                        │   │
│  │   - 防止陷入局部最优                                 │   │
│  │   - 保持创新探索                                     │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 6: 检查终止条件                                 │   │
│  │   - 达到目标适应度? → 结束                          │   │
│  │   - 达到最大代数? → 结束                            │   │
│  │   - 连续N代无改进? → 早停                           │   │
│  │   - 否则 → 回到Step 2                               │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 7: 返回结果                                     │   │
│  │   - 返回最终精英因子                                 │   │
│  │   - 包含完整进化历史                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 配置参数

```yaml
EvolutionConfig:
  # 种群参数
  population_size: 16      # 初始种群大小
  elite_size: 4            # 精英保留数
  offspring_size: 8        # 每代新生成数

  # 迭代参数
  max_generations: 10      # 最大代数
  min_fitness: 0.6         # 目标适应度
  patience: 3              # 早停耐心(连续N代无改进)

  # 多样性控制
  diversity_threshold: 0.3 # 代码相似度阈值(超过则去重)
  random_injection_rate: 0.1  # 随机注入比例

  # 变异控制
  mutation_types:          # 变异方向
    - parameter_tuning     # 参数调整(窗口大小等)
    - operator_change      # 算子替换(mean→median)
    - feature_addition     # 添加特征
    - risk_control         # 添加风控
```

---

## 4. 关键组件设计

### 4.1 初始化种群

```python
def initialize_population(task: str, schema: DataSchema) -> List[Individual]:
    """
    种群初始化 - 生成多样化的初始因子
    """
    prompt = f"""
## 任务
生成 16 个**完全不同**的因子变体来完成以下需求:
{task}

## 数据字典
{schema.to_llm_prompt()}

## 多样性要求
每个变体应尝试不同的:
1. 数据字段组合 (close/volume/turnover等)
2. 时间窗口 (5日/10日/20日/60日)
3. 数学运算 (均值/标准差/排名/分位数/比值)
4. 信号逻辑 (动量/反转/突破/均值回归)

## 输出格式
用 ```python 包裹每个因子，共16个代码块
"""
    
    response = llm.generate(prompt)
    codes = extract_code_blocks(response)
    
    return [Individual(code=code, generation=0) for code in codes]
```

### 4.2 精英选择

```python
def select_elites(population: List[Individual], elite_size: int) -> List[Individual]:
    """
    精英选择 - 保留最优且多样的个体
    """
    # 按适应度排序
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    
    elites = []
    for ind in sorted_pop:
        if len(elites) >= elite_size:
            break
        
        # 检查与已选精英的相似度
        is_diverse = all(
            code_similarity(ind.code, e.code) < DIVERSITY_THRESHOLD
            for e in elites
        )
        
        if is_diverse:
            elites.append(ind)
    
    return elites
```

### 4.3 后代生成（核心）

```python
def generate_offspring(elites: List[Individual], task: str) -> List[Individual]:
    """
    基于精英和反馈生成后代 - 这是LLM改进的核心
    """
    offspring = []
    
    for elite in elites[:2]:  # 对最好的2个精英改进
        prompt = f"""
## 任务
基于以下因子进行改进优化。

## 原始因子
```python
{elite.code}
```

## 评估结果
- 综合得分: {elite.fitness:.2f} / 1.00
- IC: {elite.metrics['ic']:.4f} (目标>0.03)
- ICIR: {elite.metrics['icir']:.4f} (目标>0.5)
- 夏普: {elite.metrics['sharpe']:.2f} (目标>1.0)
- 回撤: {elite.metrics['max_drawdown']:.1%} (目标<20%)
- 换手: {elite.metrics['turnover']:.1%} (目标<50%)

## 诊断报告
{elite.reflection}

## 改进方向
请生成4个改进版本，针对诊断报告中指出的问题:
1. 如果IC低 → 尝试不同信号构造
2. 如果回撤大 → 添加风控条件
3. 如果换手高 → 增加平滑或延长周期
4. 如果夏普低 → 优化风险收益比

每个改进版本用 ```python 包裹。
"""
        
        response = llm.generate(prompt)
        codes = extract_code_blocks(response)
        
        for code in codes:
            offspring.append(Individual(
                code=code,
                generation=current_generation,
                parent_ids=[elite.id],
            ))
    
    return offspring
```

### 4.4 诊断反思生成

```python
def generate_reflection(metrics: dict) -> str:
    """
    生成LLM可读的诊断反思报告
    """
    reflection = []
    
    ic = metrics.get('ic', 0)
    if ic < 0.02:
        reflection.append(
            f"⚠️ IC过低({ic:.4f}): 因子预测能力弱。"
            f"建议: 1)检查未来函数 2)尝试非线性变换 3)行业中性化"
        )
    
    icir = metrics.get('icir', 0)
    if icir < 0.3:
        reflection.append(
            f"⚠️ ICIR过低({icir:.4f}): IC不稳定。"
            f"建议: 1)增加平滑 2)使用更长窗口"
        )
    
    mdd = abs(metrics.get('max_drawdown', 0))
    if mdd > 0.2:
        mdd_period = metrics.get('mdd_period', '未知')
        reflection.append(
            f"⚠️ 回撤过大({mdd:.1%})，发生于{mdd_period}。"
            f"建议: 1)添加波动率过滤 2)设置止损 3)分析失败场景"
        )
    
    turnover = metrics.get('turnover', 0)
    if turnover > 0.5:
        reflection.append(
            f"⚠️ 换手过高({turnover:.1%}): 交易成本会侵蚀收益。"
            f"建议: 1)延长持仓周期 2)信号平滑 3)设置调仓阈值"
        )
    
    if not reflection:
        reflection.append("✓ 各项指标均衡，可考虑纳入组合")
    
    return "\n".join(reflection)
```

### 4.5 随机探索

```python
def inject_random(task: str, schema: DataSchema, n: int = 2) -> List[Individual]:
    """
    注入随机个体 - 防止局部最优，保持创新
    """
    prompt = f"""
## 任务
生成{n}个**完全创新**的因子，不要基于常规模式。

尝试非常规思路:
- 组合看似不相关的指标
- 使用罕见的数学变换
- 逆向思维（别人看空时可能是机会）
- 另类数据的创新用法

## 数据字典
{schema.to_llm_prompt()}

用 ```python 包裹每个因子。
"""
    
    response = llm.generate(prompt)
    codes = extract_code_blocks(response)
    
    return [
        Individual(code=code, generation=current_generation, parent_ids=['random'])
        for code in codes[:n]
    ]
```

---

## 5. 随机探索机制

### 5.1 为什么需要随机

```
问题: LLM有偏见
  - 倾向于生成"正确"但平庸的代码
  - 避免风险，不愿尝试激进策略
  - 容易陷入局部最优

解决: 保留随机通过率
  - 10%的个体是完全随机生成的
  - 低分因子有一定概率被保留
  - 高新颖度因子获得额外机会
```

### 5.2 接受策略

```python
def should_accept(individual: Individual, is_novel: bool) -> bool:
    """
    决定是否接受一个个体进入下一代
    """
    base_threshold = 0.3  # 基准通过线
    
    # 新颖度奖励
    if is_novel:
        base_threshold *= 0.7  # 降低门槛30%
    
    # 随机通过
    if random.random() < 0.1:  # 10%概率
        base_threshold *= 0.5  # 再降低50%
    
    return individual.fitness > base_threshold


def calculate_novelty(new_code: str, existing_codes: List[str]) -> float:
    """
    计算新颖度 = 1 - 最大相似度
    """
    if not existing_codes:
        return 1.0
    
    max_similarity = max(
        code_similarity(new_code, existing)
        for existing in existing_codes
    )
    
    return 1.0 - max_similarity
```

---

## 6. 进化历史追踪

### 6.1 进化树

```
Gen 0:  [A] [B] [C] [D] [E] [F] [G] [H] ...
         │   │   │   │
         └───┼───┼───┘
             │   │
Gen 1:      [A'] [B'] [R1] [R2]    (R=随机)
             │    │
             └────┼────┐
                  │    │
Gen 2:          [A''] [B''] [R3]
                  │
                  │
Gen 3:          [A'''] ← 最终结果
```

### 6.2 历史记录

```python
@dataclass
class EvolutionHistory:
    """进化历史记录"""
    
    # 所有个体
    all_individuals: List[Individual]
    
    # 每代统计
    generation_stats: List[dict]  # [{gen, best, avg, diversity}, ...]
    
    # 最佳进化路径
    best_lineage: List[str]  # [ancestor_id, ..., best_id]
    
    # 改进分析
    improvements: List[dict]  # [{from, to, delta_fitness, change_type}, ...]
    
    def get_evolution_path(self, final_id: str) -> List[Individual]:
        """追溯某个因子的完整进化路径"""
        path = []
        current = self.get_by_id(final_id)
        
        while current:
            path.append(current)
            if current.parent_ids:
                current = self.get_by_id(current.parent_ids[0])
            else:
                break
        
        return list(reversed(path))
```

---

## 7. 与其他模块的集成

### 7.1 数据流

```
┌─────────────┐
│  DataSchema │ ─── 数据字典 ──────────────────┐
└─────────────┘                                │
                                               ▼
┌─────────────┐                      ┌─────────────────┐
│  GraphRAG   │ ─── 历史经验 ──────> │  Evolution      │
└─────────────┘                      │  Engine         │
                                     │                 │
┌─────────────┐                      │  LLM生成        │
│  RAPTOR     │ ─── 类别摘要 ──────> │  ↓              │
└─────────────┘                      │  评估           │
                                     │  ↓              │
┌─────────────┐                      │  选择           │
│  Evaluator  │ <── 因子评估 ─────── │  ↓              │
└─────────────┘                      │  迭代           │
                                     └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │  最优因子       │
                                     │  + 进化历史     │
                                     └─────────────────┘
```

### 7.2 调用示例

```python
# 完整流程
from alpha_agent.evolution import EvolutionaryEngine, EvolutionConfig
from alpha_agent.schema import CN_STOCK_SCHEMA
from alpha_agent.graph import GraphRAG
from alpha_agent.memory import RaptorTree

# 1. 准备上下文
data_context = CN_STOCK_SCHEMA.to_llm_prompt()
historical_context = graph_rag.retrieve("动量因子失败经验")
category_context = raptor.retrieve("动量策略整体分析")

# 2. 配置进化引擎
config = EvolutionConfig(
    population_size=16,
    elite_size=4,
    max_generations=10,
    min_fitness=0.6,
)

engine = EvolutionaryEngine(
    llm_generator=llm,
    evaluator=evaluator,
    sandbox=sandbox,
    config=config,
)

# 3. 运行进化
result = engine.run(
    task="生成一个低换手的动量因子",
    data=stock_data,
    target=future_returns,
    context={
        'data_schema': data_context,
        'historical': historical_context,
        'category': category_context,
    }
)

# 4. 获取结果
best_factors = result.elites
evolution_history = result.history
```

---

## 8. 文件结构

```
alpha_agent/
├── evolution/
│   ├── __init__.py
│   ├── config.py           # 进化配置
│   ├── individual.py       # 个体定义
│   ├── population.py       # 种群管理
│   ├── engine.py           # 进化引擎
│   ├── selection.py        # 选择策略
│   ├── mutation.py         # 变异/后代生成
│   ├── fitness.py          # 适应度计算
│   └── history.py          # 历史追踪
```

---

*文档版本: v1.0 | 更新时间: 2024-12-05*
