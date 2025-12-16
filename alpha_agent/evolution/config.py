"""
进化引擎配置

注意: 配置已整合到 config/settings.py，此文件保留向后兼容导入
"""

# 从统一配置中心导入
from alpha_agent.config.settings import (
    EvolutionConfig,
    evolution_config,
    EVOLUTION_FAST as FAST_CONFIG,
    EVOLUTION_STANDARD as STANDARD_CONFIG,
    EVOLUTION_THOROUGH as THOROUGH_CONFIG,
)

__all__ = [
    'EvolutionConfig',
    'evolution_config',
    'FAST_CONFIG',
    'STANDARD_CONFIG',
    'THOROUGH_CONFIG',
]
