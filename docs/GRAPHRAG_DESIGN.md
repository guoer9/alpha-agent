# GraphRAG 知识图谱设计

> 用图结构存储因子知识，实现多跳推理和结构化检索

## 1. 为什么需要GraphRAG

### 传统RAG的局限

```
传统RAG: 
  因子代码 → Embedding → 向量库 → 相似度检索

问题:
1. 丢失结构关系 - 因子A和因子B高度相关，但向量可能不相似
2. 无法多跳推理 - "动量因子失败时，哪些因子可以替代？"
3. 反思知识碎片化 - 失败经验散落在各处，无法系统利用
```

### GraphRAG的优势

```
GraphRAG:
  因子 → 节点 + 关系 → 图数据库 → 图遍历 + 推理

优势:
1. 显式关系 - 因子相关性、衍生关系、互补关系
2. 多跳推理 - 从失败案例追溯到改进方案
3. 结构化反思 - 按场景、时期、策略类型组织经验
```

---

## 2. 图Schema设计

### 2.1 节点类型 (Nodes)

```
┌─────────────────────────────────────────────────────────┐
│                      节点类型                            │
├─────────────────────────────────────────────────────────┤
│ Factor          因子节点 - 存储代码、指标、元信息         │
│ Reflection      反思节点 - 存储诊断报告、改进建议         │
│ MarketRegime    市场环境 - 牛市/熊市/震荡/高波动         │
│ FailureCase     失败案例 - 具体的失败场景和原因          │
│ SuccessCase     成功案例 - 表现优异的场景               │
│ Concept         策略概念 - 动量/价值/质量/低波动         │
│ DataField       数据字段 - 使用的数据列                 │
│ TimeWindow      时间窗口 - 5日/20日/60日                │
└─────────────────────────────────────────────────────────┘
```

### 2.2 节点属性详情

#### Factor (因子节点)

```yaml
Factor:
  # 标识
  id: string              # 唯一ID, e.g., "factor_001"
  name: string            # 因子名称, e.g., "5日动量因子"
  version: int            # 版本号
  
  # 代码
  code: string            # Python代码
  code_hash: string       # 代码哈希(去重用)
  
  # 核心指标
  ic: float               # 信息系数
  icir: float             # IC信息比率
  sharpe: float           # 夏普比率
  max_drawdown: float     # 最大回撤
  turnover: float         # 换手率
  fitness: float          # 综合适应度 [0,1]
  
  # 元信息
  generation: int         # 进化代数
  parent_ids: list        # 父代因子ID
  created_at: datetime    # 创建时间
  author: string          # 创建者 (llm/gp/human)
  
  # 状态
  status: enum            # active/deprecated/experimental
  tags: list              # 标签 [momentum, short_term, high_turnover]
```

#### Reflection (反思节点)

```yaml
Reflection:
  id: string
  type: enum              # diagnosis/suggestion/failure_analysis/success_analysis
  
  # 内容
  content: string         # 反思内容(Markdown格式)
  summary: string         # 一句话摘要
  
  # 关联
  factor_id: string       # 关联的因子
  trigger: string         # 触发原因 (low_ic/high_drawdown/regime_change)
  
  # 建议
  suggestions: list       # 改进建议列表
  confidence: float       # 置信度
  
  # 元信息
  created_at: datetime
  validated: bool         # 是否经过验证
```

#### MarketRegime (市场环境)

```yaml
MarketRegime:
  id: string
  name: string            # e.g., "高波动下跌"
  
  # 定义
  description: string
  conditions: dict        # {volatility: "high", trend: "down"}
  
  # 时间范围
  typical_periods: list   # ["2015-06~2015-09", "2020-03"]
  
  # 统计
  occurrence_rate: float  # 历史出现频率
  avg_duration_days: int  # 平均持续天数
```

#### Concept (策略概念)

```yaml
Concept:
  id: string
  name: string            # e.g., "动量策略"
  category: string        # factor_type/risk_type/style
  
  # 描述
  description: string
  typical_holding_period: string  # short/medium/long
  
  # 特征
  characteristics: list   # ["追涨杀跌", "高换手", "趋势跟随"]
  known_weaknesses: list  # ["反转市场表现差", "极端行情风险"]
  
  # 关联概念
  related_concepts: list  # ["趋势跟踪", "相对强弱"]
  opposite_concepts: list # ["反转策略", "均值回归"]
```

### 2.3 边类型 (Relationships)

```
┌─────────────────────────────────────────────────────────┐
│                      边类型                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Factor ──CORRELATES_WITH──> Factor                    │
│         属性: correlation(float), period(string)        │
│         含义: 因子相关性                                 │
│                                                         │
│  Factor ──DERIVED_FROM──> Factor                       │
│         属性: modification(string)                      │
│         含义: 衍生/改进关系                              │
│                                                         │
│  Factor ──HAS_REFLECTION──> Reflection                 │
│         属性: created_at(datetime)                      │
│         含义: 因子的反思记录                             │
│                                                         │
│  Factor ──FAILED_IN──> MarketRegime                    │
│         属性: loss(float), period(string)               │
│         含义: 在某环境中失败                             │
│                                                         │
│  Factor ──SUCCEEDED_IN──> MarketRegime                 │
│         属性: gain(float), period(string)               │
│         含义: 在某环境中成功                             │
│                                                         │
│  Factor ──USES_FIELD──> DataField                      │
│         属性: usage_type(string)                        │
│         含义: 使用某数据字段                             │
│                                                         │
│  Factor ──BELONGS_TO──> Concept                        │
│         属性: confidence(float)                         │
│         含义: 属于某策略类型                             │
│                                                         │
│  Factor ──SIMILAR_TO──> Factor                         │
│         属性: similarity(float), type(string)           │
│         含义: 相似因子(代码/逻辑/表现)                   │
│                                                         │
│  Reflection ──SUGGESTS──> Concept                      │
│         属性: action(string)                            │
│         含义: 建议的改进方向                             │
│                                                         │
│  Concept ──CONTRADICTS──> Concept                      │
│         含义: 互斥的策略概念                             │
│                                                         │
│  Concept ──COMPLEMENTS──> Concept                      │
│         含义: 互补的策略概念                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 图结构示例

### 3.1 单个因子的完整图

```
                        ┌──────────────────┐
                        │  Concept:动量策略  │
                        └────────┬─────────┘
                                 │ BELONGS_TO
                                 ▼
┌─────────────┐  USES_FIELD  ┌──────────────────┐  DERIVED_FROM  ┌─────────────────┐
│DataField:   │◄─────────────│  Factor:         │───────────────►│ Factor:         │
│  close      │              │  5日动量_v2      │                │ 5日动量_v1      │
│  returns    │              │  IC=0.045        │                │ IC=0.032        │
└─────────────┘              │  Sharpe=1.8      │                └─────────────────┘
                             └────────┬─────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  HAS_REFLECTION │    │   FAILED_IN     │    │  SUCCEEDED_IN   │
    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
             │                      │                      │
             ▼                      ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Reflection:    │    │ MarketRegime:   │    │ MarketRegime:   │
    │  "换手率优化后  │    │  高波动下跌     │    │  低波动上涨     │
    │   IC提升40%"    │    │  (2020-03)      │    │  (2019全年)     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 3.2 因子关联网络

```
                    ┌─────────────────┐
                    │ 20日动量因子    │
                    │ IC=0.038        │
                    └────────┬────────┘
                             │
            CORRELATES_WITH  │  0.75
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ 5日动量因子 │      │ 月度动量因子 │      │ MACD因子    │
│ IC=0.045   │      │ IC=0.041    │      │ IC=0.035    │
└──────┬──────┘      └─────────────┘      └─────────────┘
       │
       │ SIMILAR_TO (0.85)
       ▼
┌─────────────┐
│ 5日收益率   │
│ 因子        │
└─────────────┘
```

---

## 4. 查询模式

### 4.1 基础查询

```cypher
// 1. 查找高质量因子
MATCH (f:Factor)
WHERE f.ic > 0.04 AND f.sharpe > 1.5 AND f.status = 'active'
RETURN f
ORDER BY f.fitness DESC
LIMIT 10

// 2. 查找某类型的所有因子
MATCH (f:Factor)-[:BELONGS_TO]->(c:Concept {name: '动量策略'})
RETURN f.name, f.ic, f.sharpe

// 3. 查找使用某字段的因子
MATCH (f:Factor)-[:USES_FIELD]->(d:DataField {name: 'close'})
RETURN f.name, f.code
```

### 4.2 关系查询

```cypher
// 4. 查找因子的所有反思
MATCH (f:Factor {id: 'factor_001'})-[:HAS_REFLECTION]->(r:Reflection)
RETURN r.type, r.content, r.suggestions

// 5. 查找与某因子相关的因子
MATCH (f:Factor {id: 'factor_001'})-[:CORRELATES_WITH]-(related:Factor)
RETURN related.name, related.ic

// 6. 查找因子的进化路径
MATCH path = (f:Factor {id: 'factor_latest'})-[:DERIVED_FROM*]->(ancestor:Factor)
RETURN path
```

### 4.3 场景查询（关键）

```cypher
// 7. 【核心】查找当前任务相关的历史失败经验
MATCH (f:Factor)-[:BELONGS_TO]->(c:Concept {name: $concept_type})
MATCH (f)-[:FAILED_IN]->(regime:MarketRegime)
MATCH (f)-[:HAS_REFLECTION]->(r:Reflection {type: 'failure_analysis'})
RETURN 
  f.name,
  regime.name AS failed_regime,
  r.content AS failure_reason,
  r.suggestions AS how_to_avoid
ORDER BY f.created_at DESC
LIMIT 5

// 8. 【核心】查找在特定环境表现好的因子
MATCH (f:Factor)-[:SUCCEEDED_IN]->(regime:MarketRegime {name: $current_regime})
WHERE f.status = 'active'
RETURN f.name, f.code, f.ic, f.sharpe
ORDER BY f.fitness DESC
LIMIT 3

// 9. 【核心】查找可替代因子
MATCH (failed:Factor {id: $failed_factor_id})
MATCH (failed)-[:BELONGS_TO]->(c:Concept)
MATCH (alternative:Factor)-[:BELONGS_TO]->(c)
WHERE alternative.id <> failed.id
  AND NOT (alternative)-[:CORRELATES_WITH {correlation: >0.8}]-(failed)
  AND alternative.fitness > failed.fitness
RETURN alternative.name, alternative.code
ORDER BY alternative.fitness DESC
LIMIT 3
```

### 4.4 多跳推理查询

```cypher
// 10. 从失败因子找到改进思路
MATCH (failed:Factor {id: $factor_id})
MATCH (failed)-[:FAILED_IN]->(regime:MarketRegime)
MATCH (success:Factor)-[:SUCCEEDED_IN]->(regime)
MATCH (success)-[:HAS_REFLECTION]->(r:Reflection {type: 'success_analysis'})
WHERE success.fitness > failed.fitness
RETURN 
  failed.name AS original,
  success.name AS reference,
  r.content AS why_it_worked,
  regime.name AS target_regime

// 11. 概念推理 - 找互补策略
MATCH (c1:Concept {name: $current_concept})
MATCH (c1)-[:COMPLEMENTS]->(c2:Concept)
MATCH (f:Factor)-[:BELONGS_TO]->(c2)
WHERE f.status = 'active'
RETURN c2.name, f.name, f.ic
```

---

## 5. 图构建流程

### 5.1 新因子入图

```python
def add_factor_to_graph(factor: Factor, evaluation: DetailedEvaluation):
    """
    将新因子添加到知识图谱
    """
    # 1. 创建因子节点
    create_factor_node(factor)
    
    # 2. 提取使用的数据字段
    used_fields = extract_used_fields(factor.code)
    for field in used_fields:
        create_edge(factor, "USES_FIELD", get_or_create_field(field))
    
    # 3. 判断策略类型
    concept = classify_factor_concept(factor.code)
    create_edge(factor, "BELONGS_TO", get_or_create_concept(concept))
    
    # 4. 计算与现有因子的相关性
    for existing in get_active_factors():
        corr = calculate_correlation(factor, existing)
        if abs(corr) > 0.5:
            create_edge(factor, "CORRELATES_WITH", existing, correlation=corr)
    
    # 5. 分析市场环境表现
    for regime, performance in evaluation.regime_performance.items():
        regime_node = get_or_create_regime(regime)
        if performance['sharpe'] > 0.5:
            create_edge(factor, "SUCCEEDED_IN", regime_node, gain=performance['return'])
        elif performance['sharpe'] < -0.5:
            create_edge(factor, "FAILED_IN", regime_node, loss=performance['return'])
    
    # 6. 创建反思节点
    reflection = generate_reflection(factor, evaluation)
    create_edge(factor, "HAS_REFLECTION", reflection)
    
    # 7. 如果有父代，创建衍生关系
    if factor.parent_ids:
        for parent_id in factor.parent_ids:
            parent = get_factor_by_id(parent_id)
            create_edge(factor, "DERIVED_FROM", parent)
```

### 5.2 反思更新

```python
def update_reflection_on_failure(factor_id: str, failure_context: dict):
    """
    因子失败后更新反思
    """
    factor = get_factor_by_id(factor_id)
    
    # 生成失败分析
    failure_analysis = llm_analyze_failure(
        code=factor.code,
        metrics=factor.metrics,
        context=failure_context,
    )
    
    # 创建失败反思节点
    reflection = Reflection(
        type="failure_analysis",
        content=failure_analysis.content,
        summary=failure_analysis.summary,
        suggestions=failure_analysis.suggestions,
    )
    
    create_edge(factor, "HAS_REFLECTION", reflection)
    
    # 更新失败场景
    regime = get_or_create_regime(failure_context['regime'])
    create_edge(factor, "FAILED_IN", regime, 
                loss=failure_context['loss'],
                period=failure_context['period'])
```

---

## 6. RAG集成

### 6.1 GraphRAG检索流程

```python
def graph_rag_retrieve(query: str, context: dict) -> str:
    """
    基于图的RAG检索
    """
    # 1. 解析查询意图
    intent = parse_query_intent(query)
    # e.g., {type: "generate", concept: "momentum", constraint: "low_turnover"}
    
    # 2. 基于意图构建图查询
    if intent['type'] == 'generate':
        # 查找相关历史因子
        similar_factors = query_similar_factors(intent['concept'])
        
        # 查找失败经验
        failures = query_failure_cases(intent['concept'])
        
        # 查找当前市场环境的成功案例
        current_regime = detect_current_regime(context['market_data'])
        successes = query_success_in_regime(current_regime)
    
    # 3. 组装RAG上下文
    rag_context = f"""
## 相关历史因子 (参考)
{format_factors(similar_factors[:3])}

## 历史失败经验 (避免重蹈覆辙)
{format_failures(failures[:3])}

## 当前市场环境 ({current_regime}) 的成功案例
{format_successes(successes[:2])}

## 注意事项
- 上次写{intent['concept']}类因子时，{failures[0].summary}
- 当前市场处于{current_regime}，{successes[0].name}表现好的原因是{successes[0].reflection}
"""
    
    return rag_context
```

### 6.2 检索示例

**用户输入**: "生成一个动量因子，要求换手率低"

**GraphRAG返回**:

```markdown
## 相关历史因子 (参考)

### 20日动量因子 (IC=0.041, 换手率=35%)
```python
def compute_alpha(df):
    return df['close'].pct_change(20).rank(pct=True)
```

## 历史失败经验 (避免重蹈覆辙)

### 5日动量因子_v1 失败分析
- **失败场景**: 2020年3月高波动下跌
- **原因**: 短期动量在极端行情中反转
- **建议**: 添加波动率过滤，VIX>30时降低信号强度

### 高频动量因子 失败分析  
- **失败场景**: 换手率达120%
- **原因**: 信号周期太短，每日换仓
- **建议**: 使用20日以上窗口，或添加信号平滑

## 当前市场环境 (低波动震荡) 的成功案例

### 月度动量因子 (当前环境夏普=2.1)
- 成功原因: 低波动环境下趋势延续性好
- 代码特点: 使用60日窗口，月度调仓
```

---

## 7. 文件结构

```
alpha_agent/
├── graph/
│   ├── __init__.py
│   ├── schema.py           # 节点和边的定义
│   ├── neo4j_store.py      # Neo4j操作封装
│   ├── graph_builder.py    # 图构建逻辑
│   ├── graph_query.py      # 图查询封装
│   └── graph_rag.py        # GraphRAG集成
```

---

*文档版本: v1.0 | 更新时间: 2024-12-05*
