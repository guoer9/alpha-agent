#!/bin/bash
# Qwen3-8B 金融新闻分类服务启动脚本

set -e

cd "$(dirname "$0")"

# 检查conda环境
if ! conda env list | grep -q "^vllm-deploy "; then
    echo "错误: conda环境 vllm-deploy 不存在"
    echo "请先运行: bash scripts/utils/setup_deploy_env.sh"
    exit 1
fi

# 检查模型
if [ ! -d "models/qwen-news-classifier-merged" ]; then
    echo "错误: 模型不存在"
    exit 1
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm-deploy

# 显示配置
echo "=========================================="
echo "Qwen3-8B 金融新闻分类服务"
echo "=========================================="
echo "GPU: RTX 3080 10GB"
echo "并发: 3个请求"
echo "速率: 10请求/分钟"
echo "地址: http://0.0.0.0:8000"
echo "=========================================="
echo ""

# 启动服务
python -m gunicorn scripts.deployment.deploy_with_limits:app \
    --bind 0.0.0.0:8000 \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --worker-class gthread \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --preload
