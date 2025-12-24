# Qwen新闻分类微调

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-qwen3--8b--news--classifier-yellow)](https://huggingface.co/guoer9/qwen3-8b-news-classifier)
[![GitHub](https://img.shields.io/badge/GitHub-vllm--branch-blue)](https://github.com/guoer9/money-agent/tree/vllm)

## Qwen3-8B 金融新闻分类服务

基于Qwen3-8B的金融新闻分类模型部署服务，支持8-bit量化，适配RTX 3080 10GB显存。

## 快速开始

### 1. 启动服务

```bash
bash scripts/deployment/start_production.sh
```

服务将在 http://localhost:8000 启动

### 2. 测试API

```bash
python scripts/deployment/test_api.py
```

### 3. 调用API

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
payload = {
    "messages": [
        {"role": "user", "content": "请分析以下新闻的类别：央行宣布降准0.5个百分点"}
    ],
    "max_tokens": 100,
    "temperature": 0.3
}

response = requests.post(url, json=payload)
print(response.json()['choices'][0]['message']['content'])
```

## 项目结构

```
qwen_vllm/
├── models/                              # 模型文件
│   └── qwen-news-classifier-merged/     # 部署模型 (16GB)
├── scripts/
│   ├── deployment/                      # 部署脚本
│   │   ├── deploy_with_limits.py       # 主服务脚本
│   │   ├── start_production.sh         # 启动脚本
│   │   ├── merge_lora.py               # LoRA合并
│   │   └── test_api.py                 # API测试
│   ├── training/                        # 训练脚本
│   └── utils/                           # 工具脚本
├── data/                                # 数据目录
├── logs/                                # 日志目录
├── docs/                                # 文档
└── README.md                            # 本文档
```

## API接口

### 新闻分类
- **端点**: `POST /v1/chat/completions`
- **功能**: 金融新闻分类

### 系统监控
- **健康检查**: `GET /health`
- **统计信息**: `GET /stats`
- **模型列表**: `GET /v1/models`
- **Metrics监控**: `GET /metrics` (JSON格式)
- **Prometheus**: `GET /metrics/prometheus` (Prometheus格式)

## 配置说明

### 硬件要求
- GPU: RTX 3080 10GB (或同等显存)
- 内存: 16GB+
- 磁盘: 20GB+

### 服务配置
- **最大并发**: 3个请求
- **队列大小**: 10个
- **速率限制**: 10请求/分钟
- **最大Token**: 512

### 性能指标
- **响应时间**: 2-5秒
- **吞吐量**: 40-50请求/分钟
- **显存使用**: 7GB (模型) + 3GB (推理)

## 监控命令

### 实时Metrics监控
```bash
# 启动实时监控（推荐）
python scripts/deployment/monitor_metrics.py

# 或使用curl查看
curl http://localhost:8000/metrics | jq
```

### 核心Metrics指标
- **num_waiting_requests**: 队列中等待的请求数
- **num_running_requests**: 正在处理的请求数
- **ttft_p50/p95/p99**: Time to First Token (TTFT)
- **decoding_throughput**: 解码吞吐量 (tokens/秒)
- **total_throughput**: 总吞吐量

### 其他监控
```bash
# GPU状态
watch -n 1 nvidia-smi

# 服务统计
curl http://localhost:8000/stats

# 健康检查
curl http://localhost:8000/health
```

## 维护

### 重启服务
```bash
# 停止: Ctrl+C
# 启动: bash scripts/deployment/start_production.sh
```

### 环境配置
```bash
# 创建环境
bash scripts/utils/setup_deploy_env.sh

# 激活环境
conda activate vllm-deploy
```

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| 服务无响应 | 检查进程和端口，重启服务 |
| 显存不足 | 降低并发数，重启服务 |
| 请求超时 | 检查GPU状态 |

## Kubernetes部署

### 容器化部署

```bash
# 构建镜像
docker build -t qwen-vllm:latest .

# 部署到Kubernetes
kubectl apply -k k8s/base

# 查看状态
kubectl get pods -l app=qwen-vllm
```

### 核心特性

- ✅ **自动扩缩容**: 基于请求队列和TTFT指标的HPA
- ✅ **Prometheus集成**: 通过ServiceMonitor自动采集metrics
- ✅ **健康检查**: Liveness和Readiness探针
- ✅ **GPU调度**: 支持NVIDIA GPU资源管理
- ✅ **滚动更新**: 零停机更新部署

详细文档: [docs/KUBERNETES.md](docs/KUBERNETES.md)

## 技术栈

- **模型**: Qwen3-8B (8-bit量化)
- **框架**: Transformers + BitsAndBytes
- **服务**: Flask + Gunicorn
- **容器**: Docker + Kubernetes
- **监控**: Prometheus + Grafana
- **GPU**: CUDA 13.0

## 文档

- [API使用文档](docs/API.md)
- [Metrics监控](docs/METRICS.md)
- [Kubernetes部署](docs/KUBERNETES.md)
- [项目结构](docs/STRUCTURE.md)

## License

MIT
