# 项目文件清单

## 📂 完整文件列表

### 🔧 核心部署文件（生产使用）

| 文件 | 说明 | 重要性 |
|------|------|--------|
| `deploy_with_limits.py` | 生产部署脚本（带并发限制和速率限制） | ⭐⭐⭐ |
| `start_production.sh` | 生产环境启动脚本（Gunicorn） | ⭐⭐⭐ |
| `merge_lora.py` | LoRA适配器合并脚本 | ⭐⭐⭐ |
| `test_api.py` | API功能测试脚本 | ⭐⭐⭐ |

### 🛠️ 辅助工具文件

| 文件 | 说明 | 用途 |
|------|------|------|
| `deploy_with_transformers.py` | 基础部署脚本（开发用） | 开发测试 |
| `start_deploy.sh` | 开发环境启动脚本 | 开发测试 |
| `download_model.py` | 原始模型下载脚本 | 模型下载 |
| `download_from_modelscope.py` | 魔塔下载脚本 | 模型下载 |
| `download_base_model.py` | 基础模型下载 | 模型下载 |
| `quantize_model.py` | 模型量化脚本（未完成） | 未使用 |

### ⚙️ 环境配置文件

| 文件 | 说明 | 用途 |
|------|------|------|
| `setup_deploy_env.sh` | 自动环境安装脚本 | 环境配置 |
| `requirements.txt` | Python依赖列表 | 环境配置 |
| `verify_setup.py` | 环境验证脚本 | 环境检查 |

### 🎓 训练相关文件

| 文件 | 说明 | 状态 |
|------|------|------|
| `train_qwen.py` | 模型训练脚本 | 已完成 |
| `inference_qwen.py` | 推理测试脚本 | 已完成 |
| `prepare_data.py` | 数据准备脚本 | 已完成 |
| `download_datasets.py` | 数据集下载脚本 | 已完成 |
| `train.log` | 训练日志文件 | 历史记录 |

### 📚 文档文件（重要）

| 文件 | 说明 | 优先级 |
|------|------|--------|
| `README_DEPLOYMENT.md` | 部署快速开始指南 | ⭐⭐⭐ |
| `API_USAGE.md` | API接口详细使用文档 | ⭐⭐⭐ |
| `PROJECT_ARCHITECTURE.md` | 项目架构说明 | ⭐⭐⭐ |
| `DEPLOYMENT_SUMMARY.md` | 完整部署总结 | ⭐⭐ |
| `CONCURRENCY_CONFIG.md` | 并发配置详细说明 | ⭐⭐ |
| `PROJECT_FILES.md` | 本文档 | ⭐ |
| `README.md` | 项目原始说明 | ⭐ |
| `QUICKSTART.md` | 快速开始指南 | ⭐ |
| `INSTALL_5090.md` | RTX 5090安装指南 | 参考 |
| `DEPLOY_5090.md` | RTX 5090部署指南 | 参考 |

### 📂 目录结构

| 目录 | 说明 | 大小 |
|------|------|------|
| `models/` | 模型文件目录 | ~32GB |
| `├── Qwen/Qwen3-8B/` | 基础模型 | 16GB |
| `├── qwen-news-classifier/` | LoRA适配器 | 700MB |
| `└── qwen-news-classifier-merged/` | 合并模型（部署用） | 16GB |
| `data/` | 训练数据目录 | - |
| `logs/` | 训练日志目录 | - |

## 🎯 文件使用指南

### 新用户快速开始
```bash
1. 阅读: README_DEPLOYMENT.md
2. 启动: bash start_production.sh
3. 测试: python test_api.py
4. 使用: 参考 API_USAGE.md
```

### 开发者深入了解
```bash
1. 架构: PROJECT_ARCHITECTURE.md
2. 配置: CONCURRENCY_CONFIG.md
3. 代码: deploy_with_limits.py
4. 测试: test_api.py
```

### 运维人员维护
```bash
1. 启动: start_production.sh
2. 监控: curl http://localhost:8000/stats
3. 日志: 查看终端输出
4. 重启: Ctrl+C 后重新运行
```

## 📊 文件依赖关系

```
模型文件依赖:
Qwen3-8B (基础) + checkpoint-10005 (LoRA)
    ↓ (merge_lora.py)
qwen-news-classifier-merged (完整模型)
    ↓ (deploy_with_limits.py)
运行中的服务

部署脚本依赖:
deploy_with_limits.py
    ↓ (需要)
- Flask, Gunicorn
- Transformers, BitsAndBytes
- qwen-news-classifier-merged/

启动脚本依赖:
start_production.sh
    ↓ (调用)
deploy_with_limits.py
    ↓ (加载)
qwen-news-classifier-merged/
```

## 🗑️ 可删除文件（节省空间）

如果磁盘空间紧张，可以删除以下文件：

### 训练相关（如不再训练）
- `train_qwen.py`
- `prepare_data.py`
- `download_datasets.py`
- `train.log`
- `data/` 目录

### 原始LoRA（已合并）
- `models/qwen-news-classifier/` 目录（保留checkpoint-10005作为备份）

### 基础模型（已合并）
- `models/Qwen/Qwen3-8B/` 目录（如不需要重新合并）

### 未使用的脚本
- `quantize_model.py`
- `inference_qwen.py`

**注意**: 删除前请确保已备份重要文件！

## 💾 磁盘空间使用

```
总计: ~35GB

必需文件:
- qwen-news-classifier-merged/  16GB  (部署必需)
- Python环境                     2GB   (conda环境)
- 系统缓存                       1GB   (临时文件)

可选文件:
- Qwen3-8B/                     16GB  (可删除)
- qwen-news-classifier/         700MB (可删除)
- 训练数据和日志                 若干   (可删除)
```

## 🔄 文件更新记录

| 日期 | 文件 | 变更 |
|------|------|------|
| 2025-12-23 | deploy_with_limits.py | 创建生产部署脚本 |
| 2025-12-23 | start_production.sh | 创建生产启动脚本 |
| 2025-12-23 | API_USAGE.md | 创建API文档 |
| 2025-12-23 | CONCURRENCY_CONFIG.md | 创建并发配置文档 |
| 2025-12-23 | PROJECT_ARCHITECTURE.md | 创建架构文档 |
| 2025-12-23 | merge_lora.py | LoRA合并完成 |

## 📋 检查清单

### 部署前检查
- [ ] 模型文件完整 (`models/qwen-news-classifier-merged/`)
- [ ] 环境已安装 (`conda env list | grep vllm-deploy`)
- [ ] 依赖已安装 (`pip list | grep -E "flask|gunicorn|transformers"`)
- [ ] GPU驱动正常 (`nvidia-smi`)

### 运行时检查
- [ ] 服务正常启动 (`curl http://localhost:8000/health`)
- [ ] API响应正常 (`python test_api.py`)
- [ ] 显存使用合理 (`nvidia-smi`)
- [ ] 无错误日志

### 维护检查
- [ ] 定期监控GPU状态
- [ ] 查看服务统计信息
- [ ] 检查请求队列长度
- [ ] 验证响应时间

---

**文档版本**: v1.0
**更新时间**: 2025-12-23
**维护者**: 部署团队
