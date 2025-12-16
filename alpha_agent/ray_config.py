"""
Ray 分布式计算配置

注意: 配置已整合到 config/settings.py，此文件保留向后兼容导入和工具函数
"""
import ray

# 从统一配置中心导入
from alpha_agent.config.settings import ray_config, RayConfig

# 向后兼容的字典格式 (deprecated, 推荐使用 ray_config)
RAY_CONFIG = {
    'num_cpus': ray_config.num_cpus,
    'num_gpus': ray_config.num_gpus,
    'memory': ray_config.memory,
    'object_store_memory': ray_config.object_store_memory,
}


def init_ray_cluster(local: bool = None):
    """初始化 Ray 集群
    
    Args:
        local: 是否本地模式，None时使用配置中的local_mode
    """
    if ray.is_initialized():
        return
    
    # 如果未指定，使用配置中的设置
    if local is None:
        local = ray_config.local_mode
    
    if local:
        ray.init(
            num_cpus=ray_config.num_cpus,
            num_gpus=ray_config.num_gpus,
            ignore_reinit_error=True,
        )
    else:
        # 连接到现有集群
        ray.init(address=ray_config.address)
    
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