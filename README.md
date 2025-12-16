# Quantitative Trading System (qlib_trading)

本仓库是一个 **量化交易系统的基础实现**（monorepo），核心包含：

- `alpha_agent/`：LLM 驱动的智能量化因子挖掘/筛选/回测框架（LLM + GP + Qlib + Milvus/RAG）
- `data_manager/`：Qlib 数据更新/处理与特征工程（本地 Qlib 二进制数据 + 可选线上数据源）
- `kaggle/`：竞赛/研究实验代码（默认不包含大型数据文件）

---

## 目录结构

```
qlib_trading/
  alpha_agent/          # 因子挖掘/筛选/回测智能体
  data_manager/         # 数据更新/处理/特征工程
  kaggle/               # 竞赛/实验（数据目录默认被 gitignore）
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

## 关于 Kaggle 数据

为了避免把大体积数据推送到 GitHub（单文件 >100MB 会被拒绝），仓库默认忽略 `kaggle/**/data/`。

如果你需要运行 Kaggle 实验，请自行下载对应数据集到各自的 `kaggle/**/data/` 目录。

---

## 安全提示

- 请勿把 `.env`、API Key、私钥等敏感信息提交到仓库。
- 回测/实验产生的大文件（例如 `mlruns/`、输出目录）默认已在 `.gitignore` 中忽略。
