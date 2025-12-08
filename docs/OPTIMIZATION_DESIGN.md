# Alpha Agent 优化设计方案

> LLM-Driven Evolutionary Factor Mining with Reflection & GraphRAG

## 目录

1. [数据字典设计](#1-数据字典设计)
2. [进化式因子生成](#2-进化式因子生成)
3. [奖励反思机制](#3-奖励反思机制)
4. [GraphRAG知识图谱](#4-graphrag知识图谱)
5. [RAPTOR递归抽象](#5-raptor递归抽象)
6. [随机探索机制](#6-随机探索机制)

---

## 1. 数据字典设计

### 设计原则

数据字典不是简单的字段列表，而是完整的数据契约：

| 维度 | 内容 |
|------|------|
| **字段语义** | 金融含义、计算方式 |
| **数据约束** | 值域、缺失率、更新频率 |
| **使用示例** | 典型用法、代码片段 |
| **常见陷阱** | 易错点、注意事项 |
| **关联关系** | 字段之间的逻辑关系 |

### 核心类

```python
@dataclass
class FieldSchema:
    name: str                      # 字段名
    dtype: str                     # 数据类型
    description: str               # 金融语义
    missing_rate: float            # 缺失率
    min_value: float               # 最小值
    max_value: float               # 最大值
    lag_days: int                  # 数据滞后(防未来函数)
    usage_examples: List[str]      # 使用示例
    common_pitfalls: List[str]     # 常见陷阱
    related_fields: List[str]      # 关联字段
```

### 文件位置

- `alpha_agent/schema/data_schema.py` - 数据字典基类
- `alpha_agent/schema/cn_stock_schema.py` - A股数据字典

---

## 2. 进化式因子生成

### 流程图

```
┌─────────────────────────────────────────────────────────┐
│                   Generation 0                          │
│  LLM生成16个变体 → 并行评估 → 筛选Top-4精英            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Generation 1~N                        │
│  精英 + 反馈报告 → LLM改进 → 新变体 → 评估 → 筛选     │
│         ↑                                    │          │
│         └────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

### 核心配置

```python
@dataclass
class EvolutionConfig:
    population_size: int = 16      # 种群大小
    elite_size: int = 4            # 精英数量
    max_generations: int = 10      # 最大代数
    min_fitness: float = 0.6       # 目标适应度
    random_injection_rate: float = 0.1  # 随机注入比例
```

### 适应度计算

```python
fitness = (
    0.25 * normalize(IC, target=0.05) +
    0.25 * normalize(ICIR, target=1.0) +
    0.25 * normalize(Sharpe, target=2.0) +
    0.15 * (1 - MaxDD/0.3) +
    0.10 * (1 - Turnover/1.0)
)
```

### 文件位置

- `alpha_agent/evolution/evolutionary_engine.py`
- `alpha_agent/evolution/config.py`

---

## 3. 奖励反思机制

### 诊断报告结构

不只是Sharpe Ratio，而是完整的诊断：

```python
@dataclass
class DetailedEvaluation:
    # 基础指标
    ic, icir, sharpe, max_drawdown, turnover
    
    # 时序分析
    ic_series: pd.Series           # IC时间序列
    drawdown_series: pd.Series     # 回撤序列
    
    # 风格归因
    style_exposures: Dict          # 市值/动量/波动暴露
    
    # 分时期表现
    period_performance: Dict       # 牛市/熊市/震荡
    
    # 市场环境分析
    regime_performance: Dict       # 高波动/低波动
    
    # IC衰减
    ic_decay: List[float]          # 1日/5日/10日/20日
```

### 反思报告示例

```markdown
## 因子诊断报告

### 整体表现
| 指标 | 值 | 评级 |
|------|-----|------|
| IC | 0.042 | ⭐⭐ 良好 |
| 最大回撤 | -28% | ❌ 不及格 |

### 最大回撤分析
- 发生日期: 2020-03-16
- 原因分析: 因子做空波动率，在疫情恐慌中大幅亏损
- 恢复天数: 45天

### 改进建议
1. 添加VIX阈值：当VIX > 30时停止交易
2. 增加止损：单日亏损 > 5%时平仓
```

### 文件位置

- `alpha_agent/evaluation/detailed_evaluator.py`
- `alpha_agent/evaluation/reflection_generator.py`

---

## 4. GraphRAG知识图谱

### 图结构设计

```
节点类型:
├── Factor          因子节点
├── Reflection      反思报告
├── MarketRegime    市场环境
├── Concept         策略概念
└── DataField       数据字段

边类型:
├── CORRELATES_WITH     因子相关性
├── DERIVED_FROM        衍生关系
├── HAS_REFLECTION      有反思报告
├── FAILED_IN           失败场景
├── SUCCEEDED_IN        成功场景
├── USES_FIELD          使用字段
└── IMPROVED_BY         改进关系
```

### 图示例

```
[动量因子_v1]
    │
    ├──(FAILED_IN)──→ [2020年3月暴跌]
    │                      │
    │                      └──(HAS_REFLECTION)──→ [动量在极端下跌中失效]
    │
    ├──(IMPROVED_BY)──→ [动量因子_v2]
    │                      │
    │                      └──(添加波动率过滤)
    │
    ├──(CORRELATES_WITH, 0.85)──→ [5日收益率因子]
    │
    └──(USES_FIELD)──→ [close] [returns]
```

### 查询示例

```cypher
// 查找与当前任务相关的历史失败经验
MATCH (f:Factor)-[:USES_FIELD]->(d:DataField {name: 'momentum'})
MATCH (f)-[:HAS_REFLECTION]->(r:Reflection)
WHERE r.type = 'failure'
RETURN f.name, r.content, r.suggestions
```

### 文件位置

- `alpha_agent/graph/schema.py`
- `alpha_agent/graph/neo4j_store.py`

---

## 5. RAPTOR递归抽象

### 层次结构

```
Level 3: 全局策略总结
         "动量类因子在A股整体有效，但在极端行情需要风控"
              │
Level 2: 类别聚类摘要
         ├── "短期动量(5日)换手高但IC高"
         ├── "中期动量(20日)更稳定"
         └── "长期动量(60日)接近反转"
              │
Level 1: 因子组聚类
         ├── [5日动量, 5日收益率, 周动量]
         └── [20日动量, 月动量, MACD]
              │
Level 0: 原始因子 + Reflection
         [因子代码, 评估报告, 失败经验]
```

### 聚类算法

```python
def build_raptor_tree(factors: List[Factor], max_levels: int = 3):
    """
    递归构建RAPTOR树
    1. 对因子代码做embedding
    2. GMM聚类分组
    3. 每组用LLM生成摘要
    4. 递归向上聚合
    """
    current_level = factors
    tree = {0: current_level}
    
    for level in range(1, max_levels + 1):
        # 聚类
        clusters = gmm_cluster(current_level, n_clusters=len(current_level)//3)
        
        # 每个簇生成摘要
        summaries = []
        for cluster in clusters:
            summary = llm_summarize(cluster)
            summaries.append(summary)
        
        tree[level] = summaries
        current_level = summaries
    
    return tree
```

### 检索策略

```python
def raptor_retrieve(query: str, tree: Dict, top_k: int = 5):
    """
    从RAPTOR树检索
    1. 先在高层摘要中定位相关类别
    2. 深入到该类别的底层获取具体因子
    """
    # Top-down: 从高层定位
    relevant_branch = find_relevant_branch(query, tree[max_level])
    
    # Bottom-up: 获取该分支下的所有因子
    factors = get_leaf_factors(relevant_branch)
    
    return rank_by_relevance(factors, query)[:top_k]
```

### 文件位置

- `alpha_agent/memory/raptor.py`
- `alpha_agent/memory/clustering.py`

---

## 6. 随机探索机制

### 设计原理

防止LLM偏见扼杀创新：

```python
class ExplorationStrategy:
    def __init__(self):
        self.random_rate = 0.1         # 10%完全随机
        self.mutation_rate = 0.2       # 20%随机变异
        self.guided_rate = 0.7         # 70%引导生成
    
    def should_accept(self, fitness: float, is_novel: bool) -> bool:
        """
        接受策略：
        1. fitness > threshold → 必接受
        2. 完全随机个体 → 降低threshold
        3. 高新颖度 → 给予bonus
        """
        threshold = 0.3  # 基准
        
        if is_novel:
            threshold *= 0.7  # 新颖因子降低门槛
        
        # 随机接受一部分低分因子
        if random.random() < self.random_rate:
            threshold *= 0.5
        
        return fitness > threshold
```

### 新颖度计算

```python
def novelty_score(new_factor: str, existing_factors: List[str]) -> float:
    """
    计算新颖度：与现有因子的最大相似度的补数
    """
    similarities = [code_similarity(new_factor, f) for f in existing_factors]
    max_sim = max(similarities) if similarities else 0
    return 1 - max_sim
```

---

## 完整流程

```
用户: "挖掘一个动量类因子"
          │
          ▼
    ┌─────────────────┐
    │  1. 加载数据字典  │ → 生成完整的数据上下文
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  2. GraphRAG检索 │ → 历史动量因子 + 失败经验
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  3. 进化初始化   │ → LLM生成16个变体
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────────────┐
    │  4. 进化循环 (最多10代)                  │
    │     ├─ 并行评估所有个体                 │
    │     ├─ 生成详细诊断报告                 │
    │     ├─ 选择精英 + 注入随机个体          │
    │     └─ LLM基于反馈改进                  │
    └────────┬────────────────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  5. 存储到Graph  │ → 因子 + Reflection + 关系
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  6. RAPTOR更新   │ → 重新聚类和摘要
    └────────┬────────┘
             │
             ▼
       输出最佳因子
```

---

## 文件结构

```
alpha_agent/
├── schema/
│   ├── data_schema.py       # 数据字典基类
│   └── cn_stock_schema.py   # A股数据字典
├── evolution/
│   ├── config.py            # 进化配置
│   ├── evolutionary_engine.py  # 进化引擎
│   └── population.py        # 种群管理
├── evaluation/
│   ├── detailed_evaluator.py   # 详细评估
│   └── reflection_generator.py # 反思生成
├── graph/
│   ├── schema.py            # 图Schema
│   └── neo4j_store.py       # Neo4j操作
└── memory/
    ├── raptor.py            # RAPTOR实现
    └── clustering.py        # 聚类算法
```

---

*文档版本: v1.0 | 更新时间: 2024-12-05*
