# Alpha Agent - LLM驱动的智能量化因子系统

> 基于LLM+GP混合进化的智能因子挖掘与筛选框架，集成Qlib多模型回测

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qlib](https://img.shields.io/badge/qlib-0.9+-green.svg)](https://github.com/microsoft/qlib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 核心特性

- 🧬 **混合进化引擎**: LLM探索 → GP精炼 → LLM反思 三阶段策略
- 🎯 **多阶段因子筛选**: 快速预筛 → 语义去重 → 聚类 → 正交化组合
- 🤖 **11模型回测**: LightGBM/XGBoost/LSTM/Transformer等并行验证
- 📊 **完整指标体系**: IC/ICIR/夏普/回撤/信息比率等
- 🧠 **GraphRAG+RAPTOR**: 层次化知识检索
- 🔄 **自动去重**: 语义相似度过滤冗余因子
- 📦 **6大因子库**: Alpha158/Alpha360/WorldQuant101/GTJA191/Classic/Academic
- 🗄️ **Milvus向量库**: 因子存储、检索与管理

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| **智能体框架** | LangChain | Agent编排、工具调用 |
| **向量数据库** | Milvus | 因子代码嵌入、相似检索 |
| **图数据库** | Neo4j | 风险知识图谱 |
| **回测框架** | Qlib | 11模型ML回测 |
| **进化引擎** | LLM+GP | 混合因子生成 |
| **部署** | Docker | 容器化部署 |

## 目录结构

```
alpha_agent/
├── README.md                  # 项目文档
├── MODULE_API_REFERENCE.md    # API参考文档
├── PROGRESS.md                # 开发进度
├── requirements.txt           # 依赖
│
├── config/                    # ⚙️ 配置管理
│   └── settings.py            # 全局配置 (Qlib/LLM/Selection/Cache)
│
├── core/                      # 🔧 核心组件
│   ├── llm.py                 # LLM生成器
│   ├── sandbox.py             # 安全沙箱执行器
│   └── evaluator.py           # 因子评估
│
├── selection/                 # 🎯 因子筛选系统 ⭐ NEW
│   ├── selector.py            # 多阶段筛选器 (5阶段Pipeline)
│   ├── factor_wrapper.py      # 因子包装器
│   └── data_preprocessor.py   # 数据预处理
│
├── factors/                   # 📦 因子库 (6大库)
│   ├── alpha158.py            # Qlib Alpha158
│   ├── alpha360.py            # Qlib Alpha360
│   ├── worldquant101.py       # WorldQuant 101
│   ├── gtja191.py             # 国泰君安191
│   ├── classic_factors.py     # 经典因子
│   ├── academic_premia.py     # 学术因子溢价
│   └── factor_library.py      # 因子管理器
│
├── evolution/                 # 🧬 进化引擎
│   ├── engine.py              # GP遗传算法
│   └── hybrid_engine.py       # LLM+GP混合进化 ⭐
│
├── evaluation/                # � 回测评估
│   ├── metrics.py             # 完整指标体系 (IC/夏普/回撤)
│   └── evaluator.py           # 因子评估器 + 报告生成
│
├── modeling/                  # 🤖 模型层
│   ├── qlib_model_zoo.py      # 11模型Zoo ⭐
│   ├── feature_selector.py    # 特征选择
│   └── ensemble.py            # 集成学习
│
├── memory/                    # 🧠 记忆系统
│   ├── vector_store.py        # Milvus存储
│   └── rag.py                 # RAG检索
│
├── agents/                    # 🤖 多Agent系统
│   ├── mining_agent.py        # 挖掘Agent
│   ├── analysis_agent.py      # 分析Agent
│   └── orchestrator.py        # 协调器
│
├── graph/                     # � GraphRAG
│   └── ...
│
├── raptor/                    # 🌲 RAPTOR层次检索
│   └── ...
│
├── run_factor_mining.py       # 🚀 因子挖掘入口
├── run_factor_selection.py    # 🚀 因子筛选入口 ⭐ NEW
│
├── output/                    # 📁 输出目录
│   ├── factors/               # 挖掘的因子
│   ├── selection/             # 筛选结果
│   └── models/                # 模型文件
│
└── docker/                    # 🐳 部署
    └── docker-compose.yml
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 下载Qlib数据 (~5GB)
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data

# 设置API Key
export DASHSCOPE_API_KEY=your-api-key
```

### 2. 因子筛选与对比 (推荐) ⭐

```bash
# 从Milvus筛选因子 (多阶段筛选)
python run_factor_selection.py --mode select-milvus --max-factors 30

# 因子集对比测试
python run_factor_selection.py --mode compare \
    --compare-sets alpha158,alpha360,worldquant101,milvus-selected \
    --max-factors-per-set 100

# 完整Pipeline: 筛选 + 回测
python run_factor_selection.py --mode full --source milvus --instruments csi300

# 单独回测
python run_factor_selection.py --mode backtest --input output/selection/factors.json
```

### 3. 因子挖掘

```bash
# 快速测试 (~3分钟)
python run_factor_mining.py --mode quick -y

# 标准运行 (~15分钟)
python run_factor_mining.py --mode standard

# 深度挖掘 (~30分钟)
python run_factor_mining.py --mode deep

# 自定义参数
python run_factor_mining.py --llm-rounds 5 --batch-size 3 --gp-generations 10
```

### 4. (可选) 启动Docker服务

```bash
cd docker
docker-compose up -d
```

这将启动:
- Milvus (向量数据库): localhost:19530
- Neo4j (图数据库): localhost:7474
- Redis (缓存): localhost:6379

### 5. 因子筛选API

```python
from alpha_agent.selection import FactorSelector, SelectionResult

# 创建筛选器
selector = FactorSelector(
    max_factors=30,
    corr_threshold=0.7,
    enable_cluster=True,
)

# 执行5阶段筛选
# Stage 1: 快速预筛选 (采样IC)
# Stage 2: 语义去重 (代码相似度)
# Stage 3: 聚类代表选择
# Stage 4: 完整评估 (IC/ICIR)
# Stage 5: 正交化组合优化
result: SelectionResult = selector.select(
    factors=factor_list,
    data=df,
    target=target,
    sandbox_executor=executor,
)

print(f"输入: {result.total_input} → 输出: {result.final_count}")
for factor in result.selected_factors:
    print(f"  {factor['name']}: IC={factor['ic']:.4f}")
```

### 6. Python API使用

```python
import pandas as pd
from alpha_agent import MiningAgent

# 加载数据
df = pd.read_csv('data/features.csv')
target = df['returns']

# 初始化Agent
agent = MiningAgent(api_key="your-openai-key")
agent.setup(df, target, experiment_name="exp_001")

# 对话模式
response = agent.chat("生成一个基于成交量的动量因子")
print(response)

# 批量挖掘
result = agent.run("挖掘5个与现有因子低相关的有效因子")
print(f"生成因子: {result.total_generated}")
print(f"有效因子: {result.total_valid}")
```

### 7. 使用混合进化引擎 ⭐

```python
from alpha_agent.evolution import HybridEvolutionEngine, HybridEvolutionConfig

# 配置
config = HybridEvolutionConfig(
    # 回测模型
    backtest_models=["lgb", "xgb", "catboost", "linear"],
    instruments="csi300",
    
    # 进化参数
    llm_seeds_per_round=10,
    gp_generations=10,
    min_ic_threshold=0.02,
)

# 创建引擎
engine = HybridEvolutionEngine(
    llm_client=your_llm_client,
    config=config,
)

# 运行三阶段进化
# Phase 1: LLM探索 → Phase 2: GP精炼 → Phase 3: LLM反思
results = engine.evolve(max_iterations=5)

# 获取最佳因子
for factor in results.best_factors:
    print(f"因子: {factor.code}")
    print(f"  IC: {factor.ic:.4f}, ICIR: {factor.icir:.2f}")
    print(f"  夏普: {factor.sharpe:.2f}, 回撤: {factor.max_drawdown:.1%}")
```

### 8. 使用11模型Zoo回测

```python
from alpha_agent.modeling import QlibBenchmark, QlibModelZoo

# 查看可用模型
print(QlibModelZoo.list_models())
# ['lgb', 'lgb_light', 'xgb', 'catboost', 'linear', 
#  'mlp', 'lstm', 'gru', 'transformer', 'tabnet', 'double_ensemble']

# 运行多模型基准测试
benchmark = QlibBenchmark(
    models=["lgb", "xgb", "catboost", "lstm"]
)

comparison = benchmark.run(
    instruments="csi300",
    train_period=("2018-01-01", "2021-12-31"),
    test_period=("2022-01-01", "2023-12-31"),
)

# 获取最佳模型
best = benchmark.get_best_model("icir")
print(f"最佳模型: {best}, ICIR: {benchmark.results[best].icir:.2f}")
```

### 9. 使用完整回测指标

```python
from alpha_agent.evaluation import FactorEvaluator, EvaluatorConfig

config = EvaluatorConfig(
    min_ic=0.02,
    min_sharpe=0.5,
    max_drawdown=0.30,
)

evaluator = FactorEvaluator(config)

# 完整回测
metrics = evaluator.full_evaluate(factor_code)

# 查看指标
print(f"IC: {metrics.ic.ic_mean:.4f} [{metrics.ic.ic_grade}]")
print(f"ICIR: {metrics.ic.icir:.2f} [{metrics.ic.icir_grade}]")
print(f"夏普: {metrics.risk.sharpe_ratio:.2f} [{metrics.risk.sharpe_grade}]")
print(f"回撤: {metrics.risk.max_drawdown:.1%} [{metrics.risk.drawdown_grade}]")

# 验证是否通过筛选
passed, reasons = evaluator.validate(metrics)

# 生成报告
print(evaluator.generate_report(metrics))
```

### 10. 使用风险图谱

```python
from alpha_agent import RiskKnowledgeGraph

# 连接Neo4j
kg = RiskKnowledgeGraph(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
kg.connect()
kg.init_schema()

# 添加因子和风险关系
kg.add_factor("momentum_5d", ic=0.03, category="momentum")
kg.add_exposure("momentum_5d", "market", weight=0.5)

# 查询
risks = kg.get_factor_risks("momentum_5d")
```

## 阶段目标

| 阶段 | 目标 | 状态 |
|------|------|------|
| 0 | 环境与底座 | ✅ 完成 |
| 1 | Mining-Agent MVP | ✅ 完成 |
| 2 | Memory & RAG | ✅ 完成 |
| 3 | Modeling-Agent | ✅ 完成 |
| 4 | Analysis-Agent | ✅ 完成 |
| 5 | Multi-Agent协作 | ✅ 完成 |
| 6 | 混合进化引擎 | ✅ 完成 |
| 7 | 11模型回测 | ✅ 完成 |
| 8 | 多阶段因子筛选 | ✅ 完成 |
| 9 | 因子集对比框架 | ✅ 完成 |
| 10 | 筛选算法优化 | 🟡 进行中 |

## 因子库说明

| 因子库 | 数量 | 来源 | 说明 |
|--------|------|------|------|
| Alpha158 | 158 | Qlib | 量价技术指标 |
| Alpha360 | 27 | Qlib | 扩展技术因子 |
| WorldQuant101 | 101 | WorldQuant | Alpha公式集 |
| GTJA191 | 191 | 国泰君安 | A股研报因子 |
| Classic | 25 | Academic | 经典学术因子 |
| Academic Premia | 10 | Fama-French | 风险溢价因子 |

## 环境变量

创建 `.env` 文件:

```bash
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
MILVUS_HOST=localhost
MILVUS_PORT=19530
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## License

MIT
