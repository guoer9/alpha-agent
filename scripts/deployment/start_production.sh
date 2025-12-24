#!/bin/bash
# 生产环境启动脚本（带并发限制和性能优化）
# 适配RTX 3080 10GB显存

set -e

echo "=========================================="
echo "启动Qwen金融新闻分类服务（生产模式）"
echo "=========================================="

# 环境配置
ENV_NAME="vllm-deploy"
MODEL_PATH="./models/qwen-news-classifier-merged"
HOST="0.0.0.0"
PORT=8000

# 并发配置（根据机器配置优化）
# RTX 3080 10GB显存，模型占用7GB，剩余3GB用于推理
# 每个请求约需要0.5-1GB显存用于KV cache
MAX_CONCURRENT=3  # 最大并发请求数
WORKERS=1         # Gunicorn worker数（单worker避免显存冲突）
THREADS=4         # 每个worker的线程数

echo ""
echo "机器配置:"
echo "  GPU: RTX 3080 10GB"
echo "  内存: 31GB"
echo "  CPU: 16核"
echo ""
echo "服务配置:"
echo "  最大并发请求: ${MAX_CONCURRENT}"
echo "  Workers: ${WORKERS}"
echo "  Threads: ${THREADS}"
echo "  速率限制: 10请求/分钟 (每IP)"
echo "  最大Token数: 512"
echo "=========================================="

# 检查conda环境
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "错误: conda环境 ${ENV_NAME} 不存在"
    exit 1
fi

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型不存在: $MODEL_PATH"
    exit 1
fi

# 检查GPU
echo ""
echo "GPU状态:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# 安装依赖（如果需要）
pip install flask-limiter gunicorn -q 2>/dev/null || true

echo ""
echo "=========================================="
echo "启动服务..."
echo "=========================================="
echo "地址: http://${HOST}:${PORT}"
echo "健康检查: http://localhost:${PORT}/health"
echo "统计信息: http://localhost:${PORT}/stats"
echo ""
echo "按 Ctrl+C 停止服务"
echo "=========================================="
echo ""

# 使用Gunicorn启动（生产级WSGI服务器）
cd /home/zch/qwen_vllm
gunicorn scripts.deployment.deploy_with_limits:app \
    --bind ${HOST}:${PORT} \
    --workers ${WORKERS} \
    --threads ${THREADS} \
    --timeout 120 \
    --worker-class gthread \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --preload
