# Alpha Agent 核心算法实现逻辑说明书

> 最后更新: 2025-12-16
>
> 适用范围: `alpha_agent/`（因子挖掘/筛选/评估/回测/记忆检索）
>
> 推荐配合阅读:
> - `docs/SYSTEM_FLOW.md`
> - `docs/PIPELINE.md`
> - `docs/EVOLUTION_DESIGN.md`

---

## 0. 你将从本文获得什么

本文聚焦 **“核心算法如何在代码中落地”**，按“输入→处理→输出”拆解 4 条关键链路：

- **因子挖掘/进化（LLM → GP → LLM反思）**
- **因子筛选（快速IC → 去重 → 聚类 → 完整评估 → 正交化）**
- **评估与回测（轻量评估 vs Qlib完整回测 vs 多模型benchmark）**
- **记忆/检索（Milvus向量库 + RAG提示增强 + 去重）**

并补充：关键参数、扩展点、工程坑位与风险。

---

## 1. 核心入口与模块地图（建议先记住这张表）

| 目标 | 入口/模块 | 作用 | 输入 | 输出 |
|---|---|---|---|---|
| 因子挖掘（跑通一条闭环） | `run_factor_mining.py` | 组织：数据加载→LLM生成→沙箱执行→IC评估→Hybrid进化 | 用户选择的模式/时间段/股票池 | `FactorCandidate` 列表 + JSON结果 |
| 混合进化核心算法 | `evolution/hybrid_engine.py` | 三阶段：LLM探索→GP精炼→LLM反思 | `HybridConfig` + `llm_generator/evaluator` | `elite_pool`（按fitness排序） |
| 因子筛选（核心算法） | `selection/selector.py` | 分层筛选+正交化 | `factors/data/target` | `SelectionResult` |
| 轻量IC评估 | `core/evaluator.py` | IC/ICIR/分组收益/评级 | `factor`/`target` | `EvaluationResult` |
| 安全执行 | `core/sandbox.py` | 校验+白名单+超时，执行 `compute_alpha` | `code + df` | `pd.Series` 或错误 |
| 记忆向量库 | `memory/vector_store.py` | Milvus schema、插入、检索相似 | embedding + metadata | 相似因子列表 |
| RAG增强提示 | `memory/rag.py` | 检索相似因子→拼prompt；重复检测 | instruction/code | prompt / duplicate info |
| 多模型benchmark（Qlib） | `modeling/qlib_model_zoo.py` | Qlib workflow 多模型训练/对比 | 时间段、股票池、模型列表 | 各模型回测/对比表 |
| 指标体系（更完整） | `evaluation/metrics.py` | IC/收益/风险/综合评分（偏通用） | 数组序列 | `BacktestMetrics` |
| 回测（含Qlib集成） | `mining/backtest.py` | simple分组/qlib回测接口 | factor + returns | `BacktestResult` |

---

## 2. 数据结构与接口约定（所有链路的“胶水”）

### 2.1 数据 `df` 与 `target` 的常见格式

- **`df: pd.DataFrame`**
  - 通常为 **MultiIndex**：`(instrument, datetime)`（在 `run_factor_selection.load_qlib_data()` 与 `FactorSelector.select()` 的注释中明确）
  - 常见列：`close/open/high/low/volume/vwap/turn/adj_factor`，以及派生列 `returns/market_cap/market_ret/turnover/amount/amplitude`（见 `run_factor_selection.load_qlib_data()`）
- **`target: pd.Series`**
  - 与 `df.index` 对齐（同一 MultiIndex）
  - 典型定义：未来 `N` 日收益，示例：
    - `df['close'].groupby(level='instrument').pct_change(N).shift(-N)`

> 重要：项目内不同模块对 MultiIndex 的 level 顺序可能有差异（例如 `mining/backtest.py` 中 `prepare_qlib_prediction()`要求 `(datetime, instrument)`），在接入前要做显式转换。

### 2.2 因子代码统一接口：`compute_alpha(df)`

- 所有 LLM/GP 生成出来的因子最终都应能落到这一个接口：

```
# 伪代码
def compute_alpha(df):
    return pd.Series(..., index=df.index)
```

- `core/sandbox.validate_code()` 强制要求：
  - 代码必须包含 `def compute_alpha`
  - 且不得包含危险模式（`import os`、`eval(`、`open(` 等）

> 这意味着：
> - **LLM输出**不满足该接口时，需要在上游做“代码提取/包裹”（`run_factor_mining._extract_code()`就做了类似处理）。

---

## 3. 核心链路 A：因子挖掘/混合进化（LLM → GP → LLM反思）

### 3.1 入口：`run_factor_mining.py` 的系统编排

核心类：`FactorMiningSystem`

1. **环境准备**
   - `_check_api_key()`：优先读 `DASHSCOPE_API_KEY`，其次尝试从 `config.settings.LLMConfig` 读取
   - `_init_qlib()`：尝试 `qlib.init(provider_uri=~/.qlib/qlib_data/cn_data)`
   - `_preload_data()`：使用 `qlib.data.D.features()` 取字段，构造 `df/target`
   - 若 Qlib 不可用则 `_create_mock_data()` 生成模拟数据（用于“流程测试”）

2. **组件构建**
   - `create_llm_generator()`：通过 DashScope `Generation.call()` + `PromptComposer`（提示词组装）生成因子
   - `create_evaluator()`：`Sandbox.execute()` → `core/FactorEvaluator.evaluate()` → 返回字典型评估结果

3. **进入混合进化**
   - 构造 `HybridConfig`（将 CLI/RunConfig 映射成进化参数）
   - 实例化 `HybridEvolutionEngine(config, llm_generator, evaluator)`
   - `engine.evolve()` 返回 `elite_pool`（按 `fitness` 排序）

> 注意：这里的 `create_evaluator()` 是 **轻量评估**（IC/ICIR为主，部分收益/风险指标是近似或占位），并非完整 Qlib 回测。

---

### 3.2 混合进化引擎：`evolution/hybrid_engine.py`

核心数据结构：

- `FactorCandidate`：候选因子（包含代码、来源、IC/ICIR、收益风险指标、fitness、阶段、父代信息等）
- `HybridConfig`：进化参数（LLM探索轮数、GP种群/代数、早停、换手上限等）

核心流程：

```
Phase 1 (LLM_EXPLORE)
  生成 -> 评估 -> 过阈值入 seed_pool -> 维护 seed_pool_size -> 早停
Phase 2 (GP_REFINE)
  seed_pool 初始化种群 -> 变异/交叉 -> 评估 -> 精英保留 -> 更新 candidate_pool/elite_pool
Phase 3 (LLM_REFLECT)
  取 top_k 优胜者 -> LLM生成解释/逻辑 -> 写回 FactorCandidate.logic
```

#### Phase 1：LLM探索（“创意生成”）

- `_phase1_llm_explore()`
  - `for round_idx in range(llm_rounds)`
  - `_llm_generate_batch(round_idx)`
    - 上下文包含：
      - `round`
      - `existing_seeds`（最近若干种子代码）
      - `best_ic`
  - `_evaluate_factors(new_factors)`：调用外部 `evaluator` 回调（由 `run_factor_mining.create_evaluator()`提供）
  - `_is_valid_seed()`：满足阈值（如 `seed_threshold_ic`）则进入 `seed_pool`
  - 维护 `seed_pool_size`：按 IC 排序裁剪
  - 早停：连续 `early_stop_rounds` 轮无新 seed 则停止

#### Phase 2：GP精炼（“局部搜索”）

- 目标：在有效种子附近做**结构/参数微调**，提升 IC 或稳定性。
- 配置：`gp_population/gp_generations/gp_mutation_rate/gp_crossover_rate/gp_elite_rate`
- 关键点（从接口与字段推断）：
  - GP会生成来自 `source="gp"` 或 `source="hybrid"` 的候选
  - 会维护 `elite_pool`（更高适应度、更稳定的个体）
  - `min_ic_improvement` 用于判定“是否真正改进”

> 实现侧扩展：你可以替换 `gp_mutator` / `evaluator` 来定义“允许的变异空间”与“真实的回测指标”。

#### Phase 3：LLM反思（“解释与提炼”）

- `reflect_top_k`：挑选前 K 个 GP 优胜者
- LLM生成：
  - 因子逻辑解释（写入 `FactorCandidate.logic`）
  - 可用于后续 RAG、GraphRAG、以及“失败教训”沉淀

---

### 3.3 评估回调 `evaluator` 的输入输出契约（非常关键）

`HybridEvolutionEngine` 中评估器是一个 **可插拔回调**：

- **输入**：`factor_code: str`（以及可选 `full_backtest`）
- **输出**：`dict`，典型键（见 `run_factor_mining.create_evaluator()`）
  - `ic, icir, rank_ic, rank_icir`
  - `ann_return, information_ratio, sharpe, max_drawdown, turnover`

如果你替换成“真实Qlib回测”，建议：

- 保持这些键存在（否则 `FactorCandidate` 无法填充完整字段）
- 同时把 `turnover/max_drawdown` 等值做成真实测算（否则 fitness 会被“假指标”误导）

---

### 3.4 关键参数表（混合进化）

| 参数 | 位置 | 作用 | 风险/建议 |
|---|---|---|---|
| `llm_rounds` | `HybridConfig` | 探索轮数 | 过大→成本高；过小→多样性不足 |
| `llm_batch_size` | `HybridConfig` | 每轮生成数 | 与沙箱评估耗时线性相关 |
| `seed_threshold_ic` | `HybridConfig` | 入 seed_pool 阈值 | 太高→种子稀缺；太低→垃圾种子污染GP |
| `seed_pool_size` | `HybridConfig` | 种子库上限 | 太小→多样性差；太大→GP计算爆炸 |
| `gp_population/gp_generations` | `HybridConfig` | GP搜索强度 | 直接决定算力消耗 |
| `max_turnover` | `HybridConfig` | 换手上限 | 强约束有助于实盘可用 |
| `early_stop_rounds` | `HybridConfig` | 早停 | 防止“无效LLM循环烧钱” |

---

## 4. 核心链路 B：因子筛选（快速IC → 去重 → 聚类 → 完整评估 → 正交化）

### 4.1 入口：`FactorSelector.select()`

文件：`selection/selector.py`

输入：
- `factors: List[Dict]`（每个 dict 包含 `code/name/id` 等）
- `data: pd.DataFrame`（MultiIndex 数据）
- `target: pd.Series`（未来收益）
- `sandbox_executor: Callable`（可选；否则内部直接执行）

输出：`SelectionResult`
- 记录各阶段数量、最终因子、相关性矩阵、耗时等

### 4.2 Stage 1：快速预筛选（采样IC）

方法：`_quick_filter()`

核心思想：
- 用 `quick_sample_ratio` 在数据上随机采样
- 对每个因子：执行因子 → 计算 IC（快速）→ IC 过阈值保留

关键点：
- 采样规模：`sample_size = max(1000, int(len(data) * quick_sample_ratio))`
- 执行方式：优先 `executor(code, data_sample)`，否则 `_execute_factor(code, data_sample)`

常见坑：
- 采样 `np.random.choice` 未设随机种子 → 结果不可复现
- 因子输出与 `target_sample` 对齐失败 → IC计算异常

### 4.3 Stage 2：语义去重（Milvus可选）

方法：`_semantic_dedup()`

实现策略（从模块接口推断）：
- 若配置了 `self._milvus_store`：用向量相似度去掉“语义重复”的因子
- 否则至少会做“字符串/哈希级别”的去重（防止同一代码重复进入后续阶段）

> 扩展点：
> - 你可以用 `FactorMemory + OpenAIEmbeddings` 做更强的去重
> - 或结合 AST/Token 相似度（避免仅凭 embedding）

### 4.4 Stage 3：聚类代表选择（可选）

方法：`_cluster_select()`

触发条件：
- `enable_cluster` 为真
- 且候选数显著大于 `n_clusters * reps_per_cluster`

思路：
- 用 KMeans 做聚类
- 每个簇选 `reps_per_cluster` 个代表（一般按IC排序取Top）

风险：
- 聚类特征如何定义会显著影响结果（常见选择是相关性向量/因子值统计特征）

### 4.5 Stage 4：完整评估（全量IC/ICIR）

方法：`_full_evaluate()`

区别于 Stage1：
- 不再采样，用全量数据评估
- 可能包含 `full_ic_threshold/full_icir_threshold` 的过滤

### 4.6 Stage 5：正交化组合优化（Greedy Forward + 相关性约束）

方法：`_orthogonal_select()`

目标：
- 在候选因子中选出一个集合，使得：
  - 个体质量（IC/ICIR）高
  - 两两相关性不超过 `corr_threshold`
  - 新增因子带来的边际贡献不低于 `min_marginal_ic`
  - 最终数量不超过 `max_factors`

实现风格：
- 典型是 greedy forward selection
- 文件中引入了 `LinearRegression`，通常用于残差化/正交化评估（以“新增解释力”来衡量边际贡献）

---

### 4.7 关键参数表（筛选链路）

> 参数名以 `FactorSelector.__init__()` 内部字段为准。

| 参数 | 作用 | 经验建议 |
|---|---|---|
| `quick_sample_ratio` | 快速筛选采样比例 | 0.05~0.2，越大越准但越慢 |
| `quick_ic_threshold` | 快速IC阈值 | 可设为 0.003~0.01（与收益定义有关） |
| `dedup_threshold` | 去重相似度阈值 | embedding相似度通常 0.9~0.97 之间试验 |
| `enable_cluster` | 是否聚类 | 候选>几百时建议开启 |
| `n_clusters` | 聚类簇数 | 10~50 视候选规模 |
| `reps_per_cluster` | 每簇保留代表数 | 1~5 |
| `max_factors` | 最终因子数 | 建模/回测常用 20~100 |
| `corr_threshold` | 相关性上限 | 常用 0.6~0.8 |
| `min_marginal_ic` | 最小边际IC | 防止“加了没用”的因子 |
| `full_ic_threshold/full_icir_threshold` | 全量评估阈值 | 用于提高最终质量 |

---

## 5. 核心链路 C：评估与回测

### 5.1 轻量评估：`core/evaluator.py`

`FactorEvaluator.evaluate(factor, target)` 主要做：

1. **对齐与有效样本检查**
   - `valid_mask = factor.notna() & target.notna()`
   - 样本少于 100 → `FAILED`

2. **IC/Rank IC**
   - `ic`：Spearman
   - `rank_ic`：对 factor/target 分别 rank 后相关（实现里是简化版本）

3. **滚动IC与ICIR**
   - `_compute_rolling_ic()`：滑窗计算 IC
   - `icir = mean / std`

4. **分组收益**
   - `pd.qcut(factor, n_groups)` 分组
   - 各组平均收益，得到 `long_short_return`

5. **状态评级**
   - 根据 `factor_config.ic_excellent/ic_good/ic_minimum` 给出 `EXCELLENT/GOOD/MARGINAL/POOR`

适用场景：
- 大规模候选的快速过滤
- 进化过程的 fitness 信号

不适用场景：
- 严谨的实盘级评估（需要考虑交易成本、滑点、可交易性、行业中性、风格暴露等）

### 5.2 指标体系：`evaluation/metrics.py`

该模块提供更系统的指标结构：
- `ICMetrics`：Pearson/Rank IC + 稳定性 + 等级
- `ReturnMetrics`：收益/胜率/换手
- `RiskMetrics`：波动/回撤/夏普/信息比/VAR/CVAR
- `BacktestMetrics`：汇总 + 综合评分

适用场景：
- 作为统一报告格式
- 与 Qlib 输出对齐

### 5.3 回测模块：`mining/backtest.py`

提供 3 类路径：
- `method='simple'`：不依赖Qlib的分组回测（`compute_simple_backtest`）
- `method in ['qlib', 'qlib_topk']`：对接 Qlib 回测（要求 `pyqlib`）
- `method='qlib_weight'`：权重策略回测（`FactorStrategy` 目前为占位，需补全）

重点提醒：
- `prepare_qlib_prediction()` 要求因子 index 为 `(datetime, instrument)`，和部分数据加载模块默认的 `(instrument, datetime)` 可能相反。

### 5.4 多模型benchmark：`modeling/qlib_model_zoo.py`

该模块组织 Qlib 的 workflow（`qlib.workflow.R`）和模型配置模板，提供多模型训练/对比。

`run_factor_selection.py` 中：
- `run_qlib_benchmark(models=[...])` 会调用 `QlibBenchmark(models=models).run(...)`

适用场景：
- 用标准化方式验证“因子集合 + 模型”在不同模型下的稳健性

---

## 6. 核心链路 D：记忆/检索（Milvus + RAG）

### 6.1 Milvus存储：`memory/vector_store.py`

- `MilvusStore`
  - `connect()/disconnect()`
  - `create_collection()`：schema 包含 `factor_id/name/code/description/ic/icir/status/tags/embedding`
  - `insert(records)`：批量入库
  - `search(query_embedding, top_k, min_score)`：向量检索相似因子

实现细节：
- 索引：`IVF_FLAT`
- 距离：`metric_type='IP'`（Inner Product）

风险点：
- embedding 模型/维度必须与 `VectorDBConfig.embedding_dim` 一致（默认 1536）
- IP 通常需要向量归一化才能近似 cosine，否则分数分布会漂移

### 6.2 RAG：`memory/rag.py`

- `RAGGenerator.build_prompt(instruction)`：
  - `retrieve()` 用 `FactorMemory.search_similar()` 检索相似因子
  - `format_context()` 把历史因子代码 + IC/相似度拼成上下文
  - `PromptTemplate` 套用 `RAG_PROMPT_TEMPLATE`

- `FactorDeduplicator`：
  - `is_duplicate(code)`：调用 `FactorMemory.check_duplicate(code, threshold)`
  - `deduplicate_batch(factors)`：批量去重（注意：内部用 `hash(code.strip())`，在不同 Python 进程中可能不稳定）

> 推荐用法：
> - 在 LLM 生成前先 RAG（提高“差异性”和“借鉴历史经验”）
> - 在入库前再 dedup（防止重复污染记忆库）

---

## 7. 配置与运行前置

### 7.1 环境变量

- LLM：
  - `DASHSCOPE_API_KEY`（DashScope）
  - `OPENAI_API_KEY`、`OPENAI_BASE_URL`（OpenAI/兼容接口）
  - `LLM_PROVIDER`（默认 `dashscope`）
  - `LLM_MODEL`（默认 `qwen-max`）

### 7.2 Qlib数据

- 默认路径：`~/.qlib/qlib_data/cn_data`
- 若未下载：
  - 参考 Qlib 官方 get_data/collector 文档（以你当前 Qlib 版本为准）

### 7.3 项目内配置对象

`config/settings.py` 中提供了：
- `qlib_config/llm_config/factor_config/model_config/sandbox_config/gp_config/vector_db_config`

---

## 8. 扩展点清单（你要二次开发通常从这里下手）

- **替换 LLM**：
  - `core/llm.py::LLMGenerator` 或 `run_factor_mining.create_llm_generator()`
- **替换评估器**：
  - 给 `HybridEvolutionEngine(evaluator=...)` 传入“真实回测”
  - 给 `FactorSelector.select(..., sandbox_executor=...)` 传入自定义执行器
- **接入记忆系统**：
  - 在 LLM 生成前用 `RAGGenerator.build_prompt()`
  - 或把历史失败/成功案例写入向量库/图谱
- **替换去重策略**：
  - `FactorDeduplicator` + 自定义 threshold
  - 或加入 AST 相似度 + 向量相似度的双阈值
- **替换选择策略**：
  - `FactorSelector` 的 Stage3/Stage5 可换成更严格的“组合优化”（如正交残差IC、风险暴露约束、交易成本模型）

---

## 9. 常见坑与排查清单（强烈建议收藏）

### 9.1 指标看似很好但不可交易

- 因子未做滞后（未来函数）
- 只在训练区间有效，出样后崩
- 未考虑交易成本/滑点/停牌/涨跌停

建议：
- 在 `compute_alpha()` 内显式 `.shift(1)` / `.shift(N)`
- 引入 `turnover` 惩罚与 Qlib 完整回测

### 9.2 MultiIndex 顺序/命名不一致导致对齐失败

典型症状：
- IC 计算为 NaN
- 分组回测没有有效日期

建议：
- 统一 index level 名：`instrument`、`datetime`
- 明确 `(instrument, datetime)` 与 `(datetime, instrument)` 的转换点

### 9.3 沙箱执行失败

常见原因：
- LLM 输出缺少 `compute_alpha`
- 代码触发危险模式（import/eval/open 等）
- 超时（复杂 rolling/corr）

建议：
- 上游做“代码提取+包裹”
- 增加超时、减少 rolling 窗口、避免双层 for

### 9.4 Milvus 相似度阈值不稳定

原因：
- embedding 模型/维度不一致
- IP 未归一化

建议：
- 固定 embedding 模型
- 统一归一化策略

### 9.5 配置引用不一致

当前代码中存在“配置名引用与 `config/settings.py` 实际定义不一致”的迹象（例如 `selection/selector.py` 引用了 `selection_config`，但需以你本地代码为准核对）。

建议：
- 以 `run_factor_selection.PipelineConfig` 作为运行时参数权威来源
- 或补齐统一的 `selection_config` 定义（若确实缺失）

---

## 10. 一句话总结

- **进化**解决“从 0 到 1 造因子”：LLM 提供多样性，GP 做局部提升，LLM反思沉淀可解释性。
- **筛选**解决“从 1 到 N 组因子”：快速过滤→去冗余→聚类降维→全量评估→正交化组合。
- **评估/回测/记忆**决定系统能否从“实验”走向“可复用与可迭代”。
