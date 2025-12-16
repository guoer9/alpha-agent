"""
Celery 配置

注意: 基础配置已整合到 config/settings.py (CeleryConfig)
此文件为 Celery 运行时配置，包含任务路由和定时任务
"""

# 从统一配置中心导入
from alpha_agent.config.settings import celery_config

# 基础配置
broker_url = celery_config.broker_url
result_backend = celery_config.result_backend

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Asia/Shanghai'
enable_utc = True

# 任务路由
task_routes = {
    'alpha_agent.tasks.factor.*': {'queue': 'factor'},
    'alpha_agent.tasks.backtest.*': {'queue': 'backtest'},
    'alpha_agent.tasks.evolution.*': {'queue': 'evolution'},
}

# 并发设置
worker_concurrency = celery_config.worker_concurrency
worker_prefetch_multiplier = celery_config.worker_prefetch_multiplier

# 超时设置
task_time_limit = celery_config.task_time_limit
task_soft_time_limit = celery_config.task_soft_time_limit

# 定时任务
beat_schedule = {
    'daily-factor-update': {
        'task': 'alpha_agent.tasks.factor.update_factors',
        'schedule': 60 * 60 * 24,  # 每天一次
    },
}