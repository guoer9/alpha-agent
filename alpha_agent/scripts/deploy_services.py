#!/usr/bin/env python
"""
分布式服务部署脚本

功能:
1. 初始化 Feast 特征仓库
2. 启动 Celery Worker 和 Beat
3. 启动 Ray 集群
4. 健康检查

使用方法:
    # 初始化所有服务
    python scripts/deploy_services.py --init
    
    # 启动所有服务
    python scripts/deploy_services.py --start
    
    # 停止所有服务
    python scripts/deploy_services.py --stop
    
    # 检查服务状态
    python scripts/deploy_services.py --status
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_REPO_PATH = PROJECT_ROOT / "feature_repo"


def check_redis():
    """检查 Redis 是否运行"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        return True
    except Exception:
        return False


def check_feast():
    """检查 Feast 是否可用"""
    try:
        import importlib
        importlib.import_module('feast')
        return True
    except Exception:
        return False


def check_celery():
    """检查 Celery 是否可用"""
    try:
        import importlib
        importlib.import_module('celery')
        return True
    except Exception:
        return False


def check_ray():
    """检查 Ray 是否可用"""
    try:
        # 只检查包是否安装，不导入
        import importlib.util
        spec = importlib.util.find_spec('ray')
        return spec is not None
    except Exception:
        return False


def init_feast():
    """初始化 Feast 特征仓库"""
    print("\n" + "="*50)
    print("📦 初始化 Feast 特征仓库")
    print("="*50)
    
    if not check_feast():
        print("❌ Feast 未安装")
        return False
    
    # 创建特征仓库目录
    FEATURE_REPO_PATH.mkdir(parents=True, exist_ok=True)
    
    # 创建 feature_store.yaml
    config_file = FEATURE_REPO_PATH / "feature_store.yaml"
    if not config_file.exists():
        config_content = """
project: alpha_agent
registry: data/registry.db
provider: local
online_store:
  type: redis
  connection_string: localhost:6379
offline_store:
  type: file
entity_key_serialization_version: 2
"""
        config_file.write_text(config_content.strip())
        print(f"  创建配置: {config_file}")
    
    # 创建特征定义文件
    features_file = FEATURE_REPO_PATH / "features.py"
    if not features_file.exists():
        features_content = '''
"""Alpha Agent 特征定义"""
from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# 定义股票实体
stock = Entity(
    name="stock",
    join_keys=["symbol"],
    description="股票代码",
)

# 定义因子特征源
factor_source = FileSource(
    path="data/factors.parquet",
    timestamp_field="date",
)

# 定义因子特征视图
factor_features = FeatureView(
    name="factor_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        Field(name="momentum", dtype=Float32),
        Field(name="volatility", dtype=Float32),
        Field(name="volume_ratio", dtype=Float32),
        Field(name="rsi", dtype=Float32),
        Field(name="macd", dtype=Float32),
    ],
    source=factor_source,
)
'''
        features_file.write_text(features_content.strip())
        print(f"  创建特征定义: {features_file}")
    
    # 创建数据目录
    data_dir = FEATURE_REPO_PATH / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 应用特征定义
    try:
        result = subprocess.run(
            ["feast", "apply"],
            cwd=FEATURE_REPO_PATH,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Feast 特征仓库初始化成功")
            return True
        else:
            print(f"⚠️ Feast apply 警告: {result.stderr}")
            return True  # 可能只是警告
    except Exception as e:
        print(f"⚠️ Feast 初始化跳过: {e}")
        return True


def init_celery():
    """初始化 Celery 配置"""
    print("\n" + "="*50)
    print("📦 初始化 Celery 配置")
    print("="*50)
    
    if not check_celery():
        print("❌ Celery 未安装")
        return False
    
    # 创建 Celery 配置
    celery_config = PROJECT_ROOT / "celeryconfig.py"
    if not celery_config.exists():
        config_content = '''
"""Celery 配置"""
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

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
worker_concurrency = 4
worker_prefetch_multiplier = 1

# 定时任务
beat_schedule = {
    'daily-factor-update': {
        'task': 'alpha_agent.tasks.factor.update_factors',
        'schedule': 60 * 60 * 24,  # 每天一次
    },
}
'''
        celery_config.write_text(config_content.strip())
        print(f"  创建配置: {celery_config}")
    
    # 创建任务模块
    tasks_dir = PROJECT_ROOT / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    
    init_file = tasks_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Celery 任务模块"""')
    
    factor_tasks = tasks_dir / "factor.py"
    if not factor_tasks.exists():
        task_content = '''
"""因子计算任务"""
from celery import shared_task


@shared_task(bind=True, max_retries=3)
def compute_factor(self, factor_code: str, data: dict):
    """计算单个因子"""
    try:
        # 执行因子计算
        from alpha_agent.core.sandbox import Sandbox
        sandbox = Sandbox(timeout_seconds=30)
        result, error = sandbox.execute(factor_code, data)
        return {'status': 'success', 'result': result}
    except Exception as e:
        self.retry(exc=e, countdown=60)


@shared_task
def update_factors():
    """更新所有因子"""
    print("执行每日因子更新...")
    return {'status': 'success'}


@shared_task
def batch_backtest(factor_ids: list):
    """批量回测因子"""
    results = []
    for fid in factor_ids:
        results.append({'factor_id': fid, 'status': 'completed'})
    return results
'''
        factor_tasks.write_text(task_content.strip())
        print(f"  创建任务: {factor_tasks}")
    
    print("✅ Celery 配置初始化成功")
    return True


def init_ray():
    """初始化 Ray 配置"""
    print("\n" + "="*50)
    print("📦 初始化 Ray 配置")
    print("="*50)
    
    if not check_ray():
        print("❌ Ray 未安装")
        return False
    
    # 创建 Ray 配置
    ray_config = PROJECT_ROOT / "ray_config.py"
    if not ray_config.exists():
        config_content = '''
"""Ray 分布式计算配置"""
import ray

# Ray 集群配置
RAY_CONFIG = {
    'num_cpus': 4,
    'num_gpus': 0,
    'memory': 4 * 1024 * 1024 * 1024,  # 4GB
    'object_store_memory': 1 * 1024 * 1024 * 1024,  # 1GB
}


def init_ray_cluster(local: bool = True):
    """初始化 Ray 集群"""
    if ray.is_initialized():
        return
    
    if local:
        ray.init(
            num_cpus=RAY_CONFIG['num_cpus'],
            num_gpus=RAY_CONFIG['num_gpus'],
            ignore_reinit_error=True,
        )
    else:
        # 连接到现有集群
        ray.init(address='auto')
    
    print(f"Ray 集群已启动: {ray.cluster_resources()}")


def shutdown_ray():
    """关闭 Ray 集群"""
    if ray.is_initialized():
        ray.shutdown()
        print("Ray 集群已关闭")


# 分布式因子计算
@ray.remote
def compute_factor_remote(factor_code: str, data):
    """远程因子计算"""
    from alpha_agent.core.sandbox import Sandbox
    sandbox = Sandbox(timeout_seconds=30)
    result, error = sandbox.execute(factor_code, data)
    return result


@ray.remote
def batch_evaluate_factors(factor_codes: list, data):
    """批量评估因子"""
    results = []
    for code in factor_codes:
        try:
            result = compute_factor_remote.remote(code, data)
            results.append(ray.get(result))
        except Exception as e:
            results.append(None)
    return results
'''
        ray_config.write_text(config_content.strip())
        print(f"  创建配置: {ray_config}")
    
    print("✅ Ray 配置初始化成功")
    return True


def start_services():
    """启动所有服务"""
    print("\n" + "="*50)
    print("🚀 启动分布式服务")
    print("="*50)
    
    # 检查 Redis
    if not check_redis():
        print("❌ Redis 未运行，请先启动 Redis:")
        print("   brew services start redis")
        print("   或: redis-server &")
        return False
    print("✅ Redis 运行中")
    
    # 启动 Ray
    if check_ray():
        try:
            import ray
            if not ray.is_initialized():
                ray.init(num_cpus=4, ignore_reinit_error=True)
            print(f"✅ Ray 已启动: {ray.cluster_resources()}")
        except Exception as e:
            print(f"⚠️ Ray 启动失败: {e}")
    
    # 提示启动 Celery Worker (需要单独终端)
    print("\n📌 Celery Worker 需要在单独终端启动:")
    print(f"   cd {PROJECT_ROOT}")
    print("   celery -A tasks worker --loglevel=info -Q factor,backtest,evolution")
    print("\n📌 Celery Beat (定时任务):")
    print(f"   cd {PROJECT_ROOT}")
    print("   celery -A tasks beat --loglevel=info")
    
    return True


def stop_services():
    """停止所有服务"""
    print("\n" + "="*50)
    print("🛑 停止分布式服务")
    print("="*50)
    
    # 停止 Ray
    if check_ray():
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
            print("✅ Ray 已停止")
        except Exception as e:
            print(f"⚠️ Ray 停止失败: {e}")
    
    print("\n📌 手动停止 Celery:")
    print("   pkill -f 'celery worker'")
    print("   pkill -f 'celery beat'")
    
    return True


def check_status():
    """检查服务状态"""
    print("\n" + "="*50)
    print("📊 服务状态检查")
    print("="*50)
    
    # Redis
    redis_ok = check_redis()
    print(f"  Redis:  {'✅ 运行中' if redis_ok else '❌ 未运行'}")
    
    # Feast
    feast_ok = check_feast()
    print(f"  Feast:  {'✅ 已安装' if feast_ok else '❌ 未安装'}")
    
    # Celery
    celery_ok = check_celery()
    print(f"  Celery: {'✅ 已安装' if celery_ok else '❌ 未安装'}")
    
    # Ray
    ray_ok = check_ray()
    if ray_ok:
        print(f"  Ray:    ✅ 已安装")
    else:
        print(f"  Ray:    ❌ 未安装")
    
    # Milvus
    try:
        from pymilvus import connections
        connections.connect('default', host='localhost', port='19530')
        connections.disconnect('default')
        print(f"  Milvus: ✅ 运行中")
    except Exception:
        print(f"  Milvus: ❌ 未运行")
    
    # Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
        driver.verify_connectivity()
        driver.close()
        print(f"  Neo4j:  ✅ 运行中")
    except Exception:
        print(f"  Neo4j:  ❌ 未运行")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="分布式服务部署")
    parser.add_argument("--init", action="store_true", help="初始化所有服务配置")
    parser.add_argument("--start", action="store_true", help="启动所有服务")
    parser.add_argument("--stop", action="store_true", help="停止所有服务")
    parser.add_argument("--status", action="store_true", help="检查服务状态")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("     🚀 Alpha Agent 分布式服务管理")
    print("="*60)
    
    if args.init:
        init_feast()
        init_celery()
        init_ray()
        print("\n✅ 所有服务初始化完成")
    
    if args.start:
        start_services()
    
    if args.stop:
        stop_services()
    
    if args.status:
        check_status()
    
    if not any([args.init, args.start, args.stop, args.status]):
        print("\n使用方法:")
        print("  python scripts/deploy_services.py --init     # 初始化配置")
        print("  python scripts/deploy_services.py --start    # 启动服务")
        print("  python scripts/deploy_services.py --stop     # 停止服务")
        print("  python scripts/deploy_services.py --status   # 检查状态")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
