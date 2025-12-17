# Alpha-Agent: LLM驱动的智能量化因子系统

> 基于 **LLM + GP 混合进化** 的智能因子挖掘、多阶段筛选与回测框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qlib](https://img.shields.io/badge/qlib-0.9+-green.svg)](https://github.com/microsoft/qlib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 核心特性

| 特性 | 描述 |
|------|------|
| 🧬 **混合进化引擎** | LLM探索 → GP精炼 → LLM反思 三阶段策略 |
| 🎯 **多阶段因子筛选** | 快速预筛 → 语义去重 → 聚类 → 完整评估 → 正交化组合 |
| 🤖 **11模型回测** | LightGBM / XGBoost / CatBoost / LSTM / Transformer 等并行验证 |
| 📊 **完整指标体系** | IC / ICIR / Sharpe / MaxDrawdown / 信息比率 / 多空收益 |
| 🧠 **GraphRAG + RAPTOR** | 层次化知识检索与风险图谱 |
| 🔄 **自动去重** | 语义相似度过滤冗余因子 |
| 📦 **6大因子库** | Alpha158 / Alpha360 / WorldQuant101 / GTJA191 / Classic / Academic |
| 🗄️ **Milvus向量库** | 因子存储、检索与管理 |

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Alpha-Agent 系统架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │  LLM 生成   │──▶│  GP 进化    │──▶│  LLM 反思   │           │
│  │  (探索)     │   │  (精炼)     │   │  (优化)     │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│         │                │                 │                    │
│         ▼                ▼                 ▼                    │
│  ┌──────────────────────────────────────────────────┐          │
│  │              因子筛选 Pipeline (5阶段)             │          │
│  │  快速预筛 → 语义去重 → 聚类选择 → 完整评估 → 正交化  │          │
│  └──────────────────────────────────────────────────┘          │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │              Qlib 多模型回测 (11模型)              │          │
│  │  LightGBM / XGBoost / CatBoost / LSTM / Transformer │          │
│  └──────────────────────────────────────────────────┘          │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   Milvus    │   │   Neo4j     │   │   Redis     │           │
│  │  (向量库)   │   │  (图谱)     │   │  (缓存)     │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
qlib_trading/
├── alpha_agent/          # 🧠 因子挖掘/筛选/回测智能体 (核心)
│   ├── config/           #    配置管理
│   ├── core/             #    核心组件 (LLM/沙箱/评估)
│   ├── selection/        #    多阶段因子筛选系统 ⭐
│   ├── evolution/        #    LLM+GP混合进化引擎
│   ├── evaluation/       #    回测评估与指标
│   ├── modeling/         #    11模型Zoo
│   ├── factors/          #    6大因子库
│   ├── memory/           #    Milvus向量存储
│   ├── graph/            #    GraphRAG知识图谱
│   ├── raptor/           #    RAPTOR层次检索
│   ├── run_factor_mining.py      # 因子挖掘入口
│   └── run_factor_selection.py   # 因子筛选入口 ⭐
├── data_manager/         # 📊 数据更新/处理/特征工程
├── kaggle/               # 🏆 竞赛/实验代码
├── output/               # 📁 输出目录 (gitignore)
└── README.md
```

---

## 快速开始

### 1) 环境准备

- Python: **3.10+**
- 推荐：使用虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r alpha_agent/requirements.txt
```

### 2) Qlib 数据准备（必需）

本项目的很多流程依赖本地 Qlib 数据（例如 `cn_data`）。你需要先准备 Qlib 数据目录，并确保与 `alpha_agent/config/settings.py` 中的 `provider_uri` 配置一致。

> 你可以参考 Qlib 官方文档的 `get_data` / `collector` 流程下载数据。

### 3) LLM Key（必需）

因子挖掘/进化流程需要 LLM Key。

- DashScope：`DASHSCOPE_API_KEY`
- OpenAI：`OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`）

示例：

```bash
export DASHSCOPE_API_KEY=YOUR_KEY
# 或
export OPENAI_API_KEY=YOUR_KEY
```

### 4) 启动可选依赖（Milvus/Neo4j/Redis 等）

如果你要使用向量库检索、GraphRAG、服务化组件，可使用：

```bash
docker compose -f alpha_agent/docker/docker-compose.yml up -d
```

---

## 运行入口

### 因子挖掘（LLM → GP → LLM反思）

入口：`alpha_agent/run_factor_mining.py`

### 因子筛选 + 回测 Pipeline

入口：`alpha_agent/run_factor_selection.py`

### 数据更新与处理

入口示例：`data_manager/daily_updater.py`

---

## 文档

- `alpha_agent/docs/CORE_ALGO_MANUAL.md`：核心算法实现逻辑说明书（挖掘/筛选/评估/回测/记忆检索）
- `alpha_agent/docs/SYSTEM_FLOW.md`：系统整体流程概览
- `alpha_agent/README.md`：Alpha Agent 子系统详细说明

---

## 因子库说明

| 因子库 | 数量 | 来源 | 说明 |
|--------|------|------|------|
| Alpha158 | 158 | Qlib | 量价技术指标 |
| Alpha360 | 27 | Qlib | 扩展技术因子 |
| WorldQuant101 | 101 | WorldQuant | Alpha公式集 |
| GTJA191 | 191 | 国泰君安 | A股研报因子 |
| Classic | 25 | Academic | 经典学术因子 |
| Academic Premia | 10 | Fama-French | 风险溢价因子 |

## 因子筛选 Pipeline

```
输入因子 (N个)
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 1: 快速预筛选 (采样IC)         │  ← 10%采样，IC > 0.01
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 2: 语义去重 (代码相似度)        │  ← 相似度 > 0.85 去重
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 3: 聚类代表选择                │  ← corr_greedy / kmeans
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 4: 完整评估 (IC/ICIR)          │  ← IC > 0.02, ICIR > 0.2
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 5: 正交化组合优化              │  ← 贪心选择，相关性 < 0.7
└─────────────────────────────────────┘
    │
    ▼
输出因子 (M个, M << N)
```

## 命令行示例

```bash
# 因子集对比测试 (推荐)
python alpha_agent/run_factor_selection.py --mode compare \
    --compare-sets alpha158,worldquant101,milvus-selected \
    --instruments csi300 \
    --max-factors-per-set 50

# 完整Pipeline: 筛选 + 回测
python alpha_agent/run_factor_selection.py --mode full \
    --source milvus \
    --instruments all

# 因子挖掘 (LLM + GP)
python alpha_agent/run_factor_mining.py --mode standard
```

## 关于 Kaggle 数据

为了避免把大体积数据推送到 GitHub（单文件 >100MB 会被拒绝），仓库默认忽略 `kaggle/**/data/`。

如果你需要运行 Kaggle 实验，请自行下载对应数据集到各自的 `kaggle/**/data/` 目录。

---

## 安全提示

- 请勿把 `.env`、API Key、私钥等敏感信息提交到仓库。
- 回测/实验产生的大文件（例如 `mlruns/`、输出目录）默认已在 `.gitignore` 中忽略。

## License

MIT
