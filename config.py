"""
服务配置文件
"""

# 服务配置
HOST = "0.0.0.0"
PORT = 8000

# 模型路径
MODEL_PATH = "./models/qwen-news-classifier-merged"

# 并发控制（基于RTX 3080 10GB显存优化）
MAX_CONCURRENT_REQUESTS = 3  # 最大并发请求数
REQUEST_QUEUE_SIZE = 10      # 请求队列大小

# 速率限制 (已禁用)
# RATE_LIMIT_PER_MINUTE = 10   # 每分钟请求数
# RATE_LIMIT_PER_HOUR = 100    # 每小时请求数

# 推理参数
MAX_TOKENS = 512             # 最大token数
DEFAULT_TEMPERATURE = 0.3    # 默认温度参数
TIMEOUT_SECONDS = 120        # 请求超时时间

# Gunicorn配置
WORKERS = 1                  # Worker数量
THREADS = 4                  # 每个worker的线程数

# 日志配置
LOG_LEVEL = "info"
