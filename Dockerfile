# Qwen3-8B vLLM服务 Dockerfile
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖和Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 配置pip使用阿里云镜像
RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 复制requirements
COPY requirements.txt .

# 安装Python依赖（跳过flash-attn，使用镜像源）
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    transformers accelerate peft trl \
    datasets pandas numpy \
    bitsandbytes tensorboard wandb \
    gunicorn flask-limiter modelscope \
    -i https://mirrors.aliyun.com/pypi/simple/

# 复制应用代码
COPY config.py .
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# 创建模型目录
RUN mkdir -p /app/models

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python3", "-m", "gunicorn", "scripts.deployment.deploy_with_limits:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--worker-class", "gthread", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]
