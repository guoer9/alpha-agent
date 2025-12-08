"""
Modeling模块配置

注意: 配置已整合到 config/settings.py，此文件保留向后兼容导入
"""

# 从统一配置中心导入
from alpha_agent.config.settings import (
    qlib_config,
    QlibConfig,
    gpu_config,
    GPUConfig,
    train_period_config,
    TrainPeriodConfig,
    model_config,
    ModelConfig,
    MODELS_DIR as OUTPUT_DIR,
)

# 向后兼容的字典格式 (deprecated, 推荐使用 dataclass)
QLIB_CONFIG = {
    "provider_uri": qlib_config.provider_uri,
    "region": qlib_config.region,
}

TRAIN_CONFIG = train_period_config.to_dict()

GPU_CONFIG = {
    "device": gpu_config.device,
    "use_gpu": gpu_config.use_gpu,
}

__all__ = [
    'qlib_config', 'QlibConfig',
    'gpu_config', 'GPUConfig',
    'train_period_config', 'TrainPeriodConfig',
    'model_config', 'ModelConfig',
    'OUTPUT_DIR',
    # 向后兼容
    'QLIB_CONFIG', 'TRAIN_CONFIG', 'GPU_CONFIG',
]
