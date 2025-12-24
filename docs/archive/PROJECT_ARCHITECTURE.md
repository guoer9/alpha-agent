# Qwen3-8B 金融新闻分类项目架构

## 📁 项目结构

```
/home/zch/qwen_vllm/
│
├── 📂 models/                          # 模型文件目录
│   ├── 📂 Qwen/                        
│   │   └── 📂 Qwen3-8B/                # 基础模型 (16GB)
│   │       ├── config.json
│   │       ├── model-00001-of-00005.safetensors
│   │       ├── model-00002-of-00005.safetensors
│   │       ├── model-00003-of-00005.safetensors
│   │       ├── model-00004-of-00005.safetensors
│   │       ├── model-00005-of-00005.safetensors
│   │       ├── tokenizer.json
│   │       └── ...
│   │
│   ├── 📂 qwen-news-classifier/        # 原始LoRA适配器
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── 📂 checkpoint-6500/         # 训练checkpoint
│   │   └── 📂 checkpoint-10005/        # 最佳checkpoint ⭐
│   │
│   └── 📂 qwen-news-classifier-merged/ # 合并后的完整模型 (16GB) ⭐
│       ├── config.json
│       ├── model-00001-of-00004.safetensors
│       ├── model-00002-of-00004.safetensors
│       ├── model-00003-of-00004.safetensors
│       ├── model-00004-of-00004.safetensors
│       ├── tokenizer.json
│       └── ...
│
├── 📂 data/                            # 训练数据目录
│   └── ...
│
├── 📂 logs/                            # 训练日志
│   └── ...
│
├── 🔧 核心部署脚本
│   ├── deploy_with_transformers.py    # 基础部署脚本
│   ├── deploy_with_limits.py          # 生产部署脚本（带并发限制）⭐
│   ├── start_deploy.sh                # 开发环境启动脚本
│   └── start_production.sh            # 生产环境启动脚本 ⭐
│
├── 🛠️ 工具脚本
│   ├── download_model.py              # 模型下载脚本（原始）
│   ├── download_from_modelscope.py    # 魔塔下载脚本
│   ├── download_base_model.py         # 基础模型下载
│   ├── merge_lora.py                  # LoRA合并脚本 ⭐
│   ├── test_api.py                    # API测试脚本 ⭐
│   └── quantize_model.py              # 量化脚本（未完成）
│
├── ⚙️ 环境配置
│   ├── setup_deploy_env.sh            # 环境安装脚本
│   ├── requirements.txt               # Python依赖
│   └── verify_setup.py                # 环境验证脚本
│
├── 🎓 训练相关
│   ├── train_qwen.py                  # 训练脚本
│   ├── inference_qwen.py              # 推理脚本
│   ├── prepare_data.py                # 数据准备
│   ├── download_datasets.py           # 数据集下载
│   └── train.log                      # 训练日志
│
└── 📚 文档
    ├── README.md                      # 项目说明
    ├── README_DEPLOYMENT.md           # 部署快速指南 ⭐
    ├── PROJECT_ARCHITECTURE.md        # 本文档 ⭐
    ├── DEPLOYMENT_SUMMARY.md          # 完整部署总结 ⭐
    ├── API_USAGE.md                   # API使用文档 ⭐
    ├── CONCURRENCY_CONFIG.md          # 并发配置说明 ⭐
    ├── QUICKSTART.md                  # 快速开始
    ├── INSTALL_5090.md                # 5090安装指南
    └── DEPLOY_5090.md                 # 5090部署指南
```

## 🏗️ 系统架构

### 三层架构

```
┌─────────────────────────────────────────────────────────┐
│                    客户端层 (Client)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Python   │  │  cURL    │  │  Web App │              │
│  │ Client   │  │  Client  │  │          │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
                        ↓ HTTP/JSON
┌─────────────────────────────────────────────────────────┐
│                  API服务层 (API Server)                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Flask/Gunicorn Web Server                      │    │
│  │  - 请求路由                                      │    │
│  │  - 速率限制 (10 req/min)                        │    │
│  │  - 并发控制 (3 concurrent)                      │    │
│  │  - 请求队列 (max 10)                            │    │
│  └─────────────────────────────────────────────────┘    │
│                        ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │  API Endpoints                                   │    │
│  │  - POST /v1/chat/completions (新闻分类)         │    │
│  │  - GET  /health (健康检查)                      │    │
│  │  - GET  /stats (统计信息)                       │    │
│  │  - GET  /v1/models (模型列表)                   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                        ↓ PyTorch
┌─────────────────────────────────────────────────────────┐
│                  模型推理层 (Model)                       │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Qwen3-8B Model (8-bit Quantized)              │    │
│  │  - Transformers + BitsAndBytes                  │    │
│  │  - 显存占用: 7GB                                │    │
│  │  - CPU Offload: 启用                            │    │
│  └─────────────────────────────────────────────────┘    │
│                        ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Tokenizer                                       │    │
│  │  - 文本编码/解码                                 │    │
│  │  - Chat Template应用                            │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                        ↓ CUDA
┌─────────────────────────────────────────────────────────┐
│                  硬件层 (Hardware)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ RTX 3080     │  │ 31GB RAM     │  │ 16-Core CPU  │  │
│  │ 10GB VRAM    │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 🔄 数据流程

### 1. 训练流程（已完成）

```
原始数据
  ↓
数据准备 (prepare_data.py)
  ↓
训练数据集
  ↓
LoRA微调 (train_qwen.py)
  ↓
checkpoint-10005 (最佳模型)
  ↓
LoRA适配器保存
```

### 2. 部署流程

```
基础模型下载 (download_model.py)
  ↓
Qwen3-8B (16GB)
  ↓
LoRA合并 (merge_lora.py)
  ↓
完整模型 (qwen-news-classifier-merged)
  ↓
8-bit量化加载 (deploy_with_limits.py)
  ↓
模型服务 (Flask + Gunicorn)
```

### 3. 推理流程

```
客户端请求
  ↓
速率限制检查 (10 req/min)
  ↓
并发控制 (max 3 concurrent)
  ↓
请求队列 (max 10 waiting)
  ↓
Tokenizer编码
  ↓
模型推理 (GPU)
  ↓
Tokenizer解码
  ↓
返回结果 (JSON)
```

## 🎯 核心组件说明

### 1. 模型组件

**基础模型**: `models/Qwen/Qwen3-8B/`
- 类型: Qwen3ForCausalLM
- 参数: 80亿 (8B)
- 大小: 16GB (FP16)
- 来源: ModelScope魔塔

**LoRA适配器**: `models/qwen-news-classifier/checkpoint-10005/`
- 类型: LoRA (Low-Rank Adaptation)
- 参数: r=64, alpha=128
- 大小: 698MB
- 训练: 金融新闻分类任务

**合并模型**: `models/qwen-news-classifier-merged/`
- 类型: 完整模型 (Base + LoRA)
- 大小: 16GB (FP16)
- 用途: 部署推理

### 2. 部署组件

**主服务**: `deploy_with_limits.py`
- 框架: Flask + Gunicorn
- 量化: 8-bit (BitsAndBytes)
- 并发: 3个请求
- 队列: 10个缓冲
- 速率: 10 req/min

**启动脚本**: `start_production.sh`
- Workers: 1个
- Threads: 4个
- 超时: 120秒
- 日志: 标准输出

### 3. API组件

**端点设计**:
```python
POST /v1/chat/completions      # 新闻分类（主接口）
POST /v1/completions           # 文本补全
GET  /health                   # 健康检查
GET  /stats                    # 统计信息
GET  /v1/models                # 模型列表
```

**请求格式**:
```json
{
  "messages": [
    {"role": "user", "content": "请分析新闻类别：..."}
  ],
  "max_tokens": 100,
  "temperature": 0.3
}
```

**响应格式**:
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "这条新闻属于：货币政策"
    }
  }]
}
```

## ⚙️ 配置参数

### 硬件配置
```yaml
GPU:
  型号: RTX 3080
  显存: 10GB
  驱动: 580.95.05
  CUDA: 13.0

CPU:
  核心: 16
  内存: 31GB
```

### 模型配置
```yaml
模型:
  名称: Qwen3-8B
  参数: 80亿
  量化: 8-bit (INT8)
  显存: 7GB
  
LoRA:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### 服务配置
```yaml
并发控制:
  最大并发: 3
  队列大小: 10
  
速率限制:
  每分钟: 10 requests
  每小时: 100 requests
  
推理参数:
  max_tokens: 512
  temperature: 0.3
  timeout: 120s
```

## 🔐 安全架构

### 1. 速率限制
- IP级别限制: 10 req/min
- 全局限制: 100 req/hour
- 队列满载保护: 503错误

### 2. 资源保护
- 并发控制: Semaphore (3)
- 显存保护: 最大token限制
- CPU Offload: 自动卸载

### 3. 错误处理
- 请求验证
- 异常捕获
- 优雅降级

## 📊 监控体系

### 1. 系统监控
```bash
# GPU监控
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1

# 服务监控
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

### 2. 性能指标
- 请求总数
- 成功/失败数
- 队列长度
- 响应时间
- 显存使用

### 3. 日志系统
- 访问日志: Gunicorn access log
- 错误日志: Gunicorn error log
- 应用日志: Flask logger

## 🚀 部署模式

### 开发模式
```bash
bash start_deploy.sh
# - Flask开发服务器
# - 单线程
# - 调试模式
```

### 生产模式
```bash
bash start_production.sh
# - Gunicorn WSGI服务器
# - 多线程 (4 threads)
# - 并发控制
# - 速率限制
```

## 📈 扩展方案

### 水平扩展
```
负载均衡器 (Nginx)
  ↓
实例1 (GPU 0) - 3并发
实例2 (GPU 1) - 3并发
实例3 (GPU 2) - 3并发
总并发: 9个请求
```

### 垂直扩展
- 升级GPU (更大显存)
- 使用4-bit量化
- 优化模型结构

## 🛠️ 维护流程

### 日常维护
1. 监控GPU状态
2. 检查服务日志
3. 查看统计信息
4. 定期重启服务

### 故障恢复
1. 检查进程状态
2. 查看错误日志
3. 清理GPU显存
4. 重启服务

### 性能优化
1. 调整并发数
2. 优化token限制
3. 调整温度参数
4. 启用缓存

## 📚 文档索引

| 文档 | 用途 | 优先级 |
|------|------|--------|
| `README_DEPLOYMENT.md` | 快速开始 | ⭐⭐⭐ |
| `API_USAGE.md` | API详细说明 | ⭐⭐⭐ |
| `DEPLOYMENT_SUMMARY.md` | 完整部署信息 | ⭐⭐ |
| `CONCURRENCY_CONFIG.md` | 并发配置 | ⭐⭐ |
| `PROJECT_ARCHITECTURE.md` | 本文档 | ⭐ |

## 🎯 使用建议

### 新用户
1. 阅读 `README_DEPLOYMENT.md`
2. 运行 `bash start_production.sh`
3. 测试 `python test_api.py`
4. 查看 `API_USAGE.md`

### 开发者
1. 理解本架构文档
2. 查看 `deploy_with_limits.py`
3. 阅读 `CONCURRENCY_CONFIG.md`
4. 自定义配置参数

### 运维人员
1. 掌握监控命令
2. 了解故障处理
3. 配置自动重启
4. 设置告警机制

---

**架构版本**: v1.0
**更新时间**: 2025-12-23
**状态**: ✅ 生产就绪
