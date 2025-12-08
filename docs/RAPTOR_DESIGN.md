# RAPTOR 递归抽象设计

> Recursive Abstractive Processing for Tree-Organized Retrieval

## 1. 什么是RAPTOR

### 核心思想

```
传统RAG: 扁平存储 → 向量检索 → 返回原文

RAPTOR: 
  原始文档 → 聚类 → 摘要 → 递归聚类 → 层次树
  检索时从高层定位 → 深入底层获取细节
```

### 为什么因子挖掘需要RAPTOR

```
问题场景:
  数据库有1000个历史因子
  用户问: "动量策略在熊市的表现如何？"

传统方法:
  检索 → 返回10个最相似的因子 → 可能都是牛市的

RAPTOR方法:
  Level 3: "量化因子整体表现"
  Level 2: "动量类因子在不同市场环境的总结"  ← 定位到这里
  Level 1: "熊市期间的动量因子组"
  Level 0: 具体因子 + 反思报告  ← 深入获取
```

---

## 2. 层次结构设计

### 2.1 四层金字塔

```
                    ┌─────────────────────┐
                    │      Level 3        │
                    │   全局策略总结      │
                    │  "因子库整体画像"   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌────────────┐   ┌────────────┐   ┌────────────┐
     │  Level 2   │   │  Level 2   │   │  Level 2   │
     │ 动量类因子 │   │ 价值类因子 │   │ 质量类因子 │
     │  整体总结  │   │  整体总结  │   │  整体总结  │
     └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
           │                │                │
     ┌─────┴─────┐    ┌─────┴─────┐    ┌─────┴─────┐
     │           │    │           │    │           │
     ▼           ▼    ▼           ▼    ▼           ▼
┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐
│ Level 1 ││ Level 1 ││ Level 1 ││ Level 1 ││ Level 1 │
│短期动量 ││中期动量 ││  低PE   ││  高ROE  ││ 低波动  │
│因子组   ││因子组   ││ 因子组  ││ 因子组  ││ 因子组  │
└────┬────┘└────┬────┘└────┬────┘└────┬────┘└────┬────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────┐
│                      Level 0                         │
│     [因子代码] + [评估报告] + [反思记录]             │
│     原始数据，不做任何摘要                           │
└─────────────────────────────────────────────────────┘
```

### 2.2 各层内容

#### Level 0: 原始因子

```yaml
内容:
  - 因子代码 (完整Python代码)
  - 评估报告 (IC/Sharpe/回撤/换手等)
  - 反思记录 (失败分析/成功经验/改进建议)
  - 元信息 (创建时间/作者/版本)

数量: 可能有1000+个因子

示例:
  - 5日动量因子_v3
  - 20日动量因子_v1
  - PE倒数因子
  - ROE稳定性因子
  - ...
```

#### Level 1: 因子组摘要

```yaml
聚类依据:
  - 代码相似度 (AST/Embedding)
  - 使用数据字段相似
  - 时间窗口相近
  - 评估指标接近

摘要内容:
  - 该组因子的共同特征
  - 整体表现统计 (平均IC/最佳Sharpe等)
  - 主要风险点
  - 典型代表因子

示例:
  "短期动量因子组 (5-10日窗口)"
  - 包含12个因子
  - 平均IC: 0.038, 最佳: 0.052
  - 共同特点: 使用close字段，pct_change计算
  - 主要风险: 换手率高(平均65%)，在反转市场表现差
  - 代表因子: 5日动量_v3 (IC=0.045)
```

#### Level 2: 策略类别摘要

```yaml
聚类依据:
  - 策略类型 (动量/价值/质量/低波动)
  - 风格因子暴露
  - 适用市场环境

摘要内容:
  - 该类策略的整体特征
  - 在不同市场环境的表现
  - 与其他策略的相关性
  - 历史上的重大失败/成功
  - 使用建议

示例:
  "动量策略整体分析"
  - 下辖3个因子组: 短期动量(12个)/中期动量(8个)/长期动量(5个)
  - 牛市表现: 平均夏普1.8 ✓
  - 熊市表现: 平均夏普-0.3 ✗
  - 震荡市表现: 平均夏普0.5
  - 历史最大失败: 2015年股灾 (-35%)
  - 建议: 配合波动率择时使用，高波动时降低动量权重
```

#### Level 3: 全局总结

```yaml
内容:
  - 因子库整体画像
  - 各类策略的配置建议
  - 当前市场环境下的推荐
  - 需要补充的因子方向

示例:
  "Alpha因子库总览 (v2024.12)"
  - 总因子数: 1,234个
  - 活跃因子: 456个 (IC>0.02 & 状态active)
  - 类别分布: 动量35% / 价值25% / 质量20% / 其他20%
  - 整体特点: 偏向短期信号，换手率整体偏高
  - 薄弱环节: 缺少另类数据因子，基本面因子覆盖不足
  - 当前推荐: 市场处于低波动上涨，建议侧重动量+质量
```

---

## 3. 构建流程

### 3.1 整体流程

```
┌─────────────────────────────────────────────────────────┐
│                    RAPTOR 构建流程                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: 收集所有因子                                    │
│     └─> [Factor_1, Factor_2, ..., Factor_N]             │
│                         │                               │
│                         ▼                               │
│  Step 2: 向量化                                          │
│     └─> 代码Embedding + 指标向量 → 综合表示              │
│                         │                               │
│                         ▼                               │
│  Step 3: 聚类 (Level 0 → Level 1)                       │
│     └─> GMM/K-Means → 形成因子组                        │
│                         │                               │
│                         ▼                               │
│  Step 4: 摘要生成 (Level 1)                             │
│     └─> LLM总结每个因子组的特征                          │
│                         │                               │
│                         ▼                               │
│  Step 5: 递归聚类 (Level 1 → Level 2)                   │
│     └─> 对Level 1的摘要向量聚类                          │
│                         │                               │
│                         ▼                               │
│  Step 6: 摘要生成 (Level 2)                             │
│     └─> LLM总结每个策略类别                              │
│                         │                               │
│                         ▼                               │
│  Step 7: 全局摘要 (Level 3)                             │
│     └─> LLM生成全局总结                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 聚类算法

```python
class RaptorClusterer:
    """RAPTOR聚类器"""
    
    def __init__(self, 
                 max_cluster_size: int = 10,
                 min_cluster_size: int = 3,
                 similarity_threshold: float = 0.7):
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
    
    def cluster(self, embeddings: np.ndarray, n_clusters: int = None) -> List[List[int]]:
        """
        GMM软聚类
        返回: [[idx1, idx2], [idx3, idx4, idx5], ...] 每个簇的索引
        """
        if n_clusters is None:
            # 自动确定簇数
            n_clusters = self._estimate_n_clusters(embeddings)
        
        # GMM聚类
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(embeddings)
        
        # 按簇分组
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        # 过滤太小的簇
        valid_clusters = [
            indices for indices in clusters.values()
            if len(indices) >= self.min_cluster_size
        ]
        
        return valid_clusters
    
    def _estimate_n_clusters(self, embeddings: np.ndarray) -> int:
        """使用肘部法则估计簇数"""
        max_k = min(len(embeddings) // self.min_cluster_size, 20)
        
        inertias = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # 找肘部点
        return self._find_elbow(inertias) + 2
```

### 3.3 摘要生成

```python
class RaptorSummarizer:
    """RAPTOR摘要生成器"""
    
    def __init__(self, llm_generator):
        self.llm = llm_generator
    
    def summarize_factor_group(self, factors: List[Factor]) -> str:
        """
        Level 1: 因子组摘要
        """
        prompt = f"""
请总结以下{len(factors)}个因子的共同特征：

## 因子列表
{self._format_factors(factors)}

## 请按以下格式输出:

### 组名
[给这组因子起一个描述性的名字]

### 共同特征
- 使用的数据字段: 
- 计算方式:
- 时间窗口:

### 整体表现
- 平均IC: 
- IC范围: 
- 平均换手率:

### 主要风险
[这组因子的共同风险点]

### 代表因子
[选出表现最好的1-2个代表]

### 使用建议
[何时使用/避免使用]
"""
        return self.llm.generate(prompt)
    
    def summarize_category(self, group_summaries: List[str]) -> str:
        """
        Level 2: 策略类别摘要
        """
        prompt = f"""
请总结以下因子组的整体特征，形成策略级别的分析：

## 因子组列表
{chr(10).join(group_summaries)}

## 请按以下格式输出:

### 策略类型
[动量/价值/质量/其他]

### 整体特征
[这类策略的核心逻辑]

### 市场环境表现
- 牛市: 
- 熊市:
- 震荡市:
- 高波动:
- 低波动:

### 与其他策略的关系
- 正相关: 
- 负相关:
- 互补策略:

### 历史重大事件
[这类策略历史上的重大成功/失败]

### 配置建议
[在组合中的建议权重和使用条件]
"""
        return self.llm.generate(prompt)
    
    def summarize_global(self, category_summaries: List[str]) -> str:
        """
        Level 3: 全局摘要
        """
        prompt = f"""
请基于以下策略类别分析，生成因子库的全局总结：

## 策略类别
{chr(10).join(category_summaries)}

## 请按以下格式输出:

### 因子库概览
- 总体规模:
- 活跃因子数:
- 类别分布:

### 整体特点
[这个因子库的优势和特点]

### 薄弱环节
[缺失或表现不佳的领域]

### 当前市场建议
[基于当前市场环境的配置建议]

### 发展方向
[建议重点研究的方向]
"""
        return self.llm.generate(prompt)
```

---

## 4. 检索策略

### 4.1 Top-Down检索

```python
def raptor_retrieve_topdown(query: str, tree: RaptorTree, top_k: int = 5) -> List[Factor]:
    """
    自顶向下检索
    
    流程:
    1. 在Level 3找到最相关的全局主题
    2. 下钻到Level 2找相关策略类别
    3. 下钻到Level 1找相关因子组
    4. 最终返回Level 0的具体因子
    """
    # 向量化查询
    query_embedding = embed(query)
    
    # Level 3: 定位主题
    # (通常只有1个全局摘要，可跳过)
    
    # Level 2: 找相关策略类别
    category_scores = []
    for category in tree.level_2:
        score = cosine_similarity(query_embedding, category.embedding)
        category_scores.append((category, score))
    
    top_categories = sorted(category_scores, key=lambda x: x[1], reverse=True)[:2]
    
    # Level 1: 在相关类别下找因子组
    relevant_groups = []
    for category, _ in top_categories:
        for group in category.children:
            score = cosine_similarity(query_embedding, group.embedding)
            relevant_groups.append((group, score))
    
    top_groups = sorted(relevant_groups, key=lambda x: x[1], reverse=True)[:3]
    
    # Level 0: 获取具体因子
    factors = []
    for group, _ in top_groups:
        factors.extend(group.children)
    
    # 最终排序
    factor_scores = [
        (f, cosine_similarity(query_embedding, f.embedding))
        for f in factors
    ]
    
    return [f for f, _ in sorted(factor_scores, key=lambda x: x[1], reverse=True)[:top_k]]
```

### 4.2 Tree Traversal检索

```python
def raptor_retrieve_traversal(query: str, tree: RaptorTree, top_k: int = 5) -> List[Factor]:
    """
    树遍历检索 - 同时搜索所有层级
    
    优势: 可以同时获取高层摘要（概览）和底层细节
    """
    query_embedding = embed(query)
    
    all_nodes = []
    
    # 收集所有层级的节点
    for level in range(4):
        for node in tree.get_level(level):
            score = cosine_similarity(query_embedding, node.embedding)
            all_nodes.append((node, score, level))
    
    # 按分数排序
    all_nodes.sort(key=lambda x: x[1], reverse=True)
    
    # 智能选择: 高层摘要 + 底层细节
    results = {
        'summaries': [],  # Level 2-3的摘要
        'factors': [],    # Level 0的因子
    }
    
    for node, score, level in all_nodes:
        if level >= 2 and len(results['summaries']) < 2:
            results['summaries'].append(node)
        elif level == 0 and len(results['factors']) < top_k:
            results['factors'].append(node)
        
        if len(results['summaries']) >= 2 and len(results['factors']) >= top_k:
            break
    
    return results
```

### 4.3 检索示例

**用户查询**: "动量策略在熊市表现如何？有什么改进建议？"

**检索过程**:

```
Step 1: Level 2 匹配
  → 找到 "动量策略整体分析" (相似度 0.89)

Step 2: Level 1 深入
  → "动量策略整体分析" 下的子节点:
    - "短期动量因子组" (0.72)
    - "中期动量因子组" (0.68)
    - "长期动量因子组" (0.61)

Step 3: Level 0 获取
  → 从相关因子组中获取具体因子和反思
```

**返回结果**:

```markdown
## 高层摘要 (Level 2)

### 动量策略在熊市的整体表现
- 平均夏普: -0.3
- 最大回撤: -35% (2015年股灾)
- 失效原因: 趋势反转导致追涨杀跌
- 历史教训: 2015年6月、2018年全年、2020年3月都出现较大亏损

### 改进建议
1. 添加波动率过滤: VIX > 25时降低仓位
2. 结合反转信号: 极端超卖时暂停动量策略
3. 配合市场状态识别: 使用HMM识别牛熊切换

---

## 具体因子 (Level 0)

### 5日动量因子_v3
- IC: 0.045, Sharpe: 1.2
- 熊市表现: Sharpe = -0.8 ❌
- 反思: "在2020年3月暴跌中亏损28%，主要因为..."

### 波动率调整动量因子
- IC: 0.038, Sharpe: 1.5
- 熊市表现: Sharpe = 0.2 ✓
- 反思: "添加波动率惩罚后，熊市回撤减少60%..."
```

---

## 5. 增量更新

### 5.1 新因子加入

```python
def add_factor_to_raptor(new_factor: Factor, tree: RaptorTree):
    """
    增量添加新因子
    """
    # 1. 添加到Level 0
    tree.level_0.append(new_factor)
    
    # 2. 找到最近的因子组
    new_embedding = embed_factor(new_factor)
    best_group = None
    best_score = 0
    
    for group in tree.level_1:
        score = cosine_similarity(new_embedding, group.embedding)
        if score > best_score:
            best_score = score
            best_group = group
    
    # 3. 判断是否加入现有组
    if best_score > 0.8:
        # 加入现有组
        best_group.children.append(new_factor)
        # 更新组摘要
        best_group.summary = summarize_factor_group(best_group.children)
        best_group.embedding = embed(best_group.summary)
    else:
        # 创建新组 (如果累积够了就创建)
        orphan_factors.append(new_factor)
        if len(orphan_factors) >= MIN_CLUSTER_SIZE:
            new_group = create_new_group(orphan_factors)
            tree.level_1.append(new_group)
            orphan_factors.clear()
    
    # 4. 向上传播更新 (可选，定期批量做)
    # update_level_2_summaries(tree)
```

### 5.2 定期重构

```python
def rebuild_raptor_tree(factors: List[Factor], schedule: str = "weekly"):
    """
    定期完全重构RAPTOR树
    
    触发条件:
    - 新增因子超过100个
    - 距离上次重构超过7天
    - 手动触发
    """
    print(f"开始RAPTOR重构, 共{len(factors)}个因子")
    
    # Level 0: 原始因子
    level_0 = factors
    
    # Level 1: 聚类成因子组
    embeddings = np.array([embed_factor(f) for f in factors])
    clusters = clusterer.cluster(embeddings)
    level_1 = [
        FactorGroup(
            factors=[factors[i] for i in cluster],
            summary=summarizer.summarize_factor_group([factors[i] for i in cluster])
        )
        for cluster in clusters
    ]
    print(f"Level 1: {len(level_1)}个因子组")
    
    # Level 2: 聚类成策略类别
    group_embeddings = np.array([embed(g.summary) for g in level_1])
    category_clusters = clusterer.cluster(group_embeddings, n_clusters=5)
    level_2 = [
        StrategyCategory(
            groups=[level_1[i] for i in cluster],
            summary=summarizer.summarize_category([level_1[i].summary for i in cluster])
        )
        for cluster in category_clusters
    ]
    print(f"Level 2: {len(level_2)}个策略类别")
    
    # Level 3: 全局摘要
    level_3 = GlobalSummary(
        categories=level_2,
        summary=summarizer.summarize_global([c.summary for c in level_2])
    )
    
    return RaptorTree(level_0, level_1, level_2, level_3)
```

---

## 6. 文件结构

```
alpha_agent/
├── memory/
│   ├── raptor/
│   │   ├── __init__.py
│   │   ├── tree.py           # RAPTOR树结构
│   │   ├── clusterer.py      # 聚类算法
│   │   ├── summarizer.py     # 摘要生成
│   │   ├── retriever.py      # 检索策略
│   │   └── builder.py        # 树构建/更新
```

---

## 7. 与GraphRAG的配合

```
GraphRAG: 存储因子之间的显式关系（相关性、衍生、失败场景）
RAPTOR:   存储因子的层次化摘要（从具体到抽象）

配合使用:
1. 先用RAPTOR定位相关类别（快速缩小范围）
2. 再用GraphRAG查找具体关系（失败经验、替代方案）

示例流程:
  用户: "帮我改进这个动量因子"
  
  RAPTOR: 定位到"短期动量因子组"
        → 获取该组的整体表现和常见问题
  
  GraphRAG: 查找该因子的失败场景
          → 查找成功的相似因子
          → 查找改进路径
```

---

*文档版本: v1.0 | 更新时间: 2024-12-05*
