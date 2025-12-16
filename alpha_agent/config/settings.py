"""
Alpha Agent 全局配置
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
FACTORS_DIR = OUTPUT_DIR / "factors"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

# 确保目录存在
for dir_path in [OUTPUT_DIR, FACTORS_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ==================== Qlib 配置 ====================
@dataclass
class QlibConfig:
    """Qlib框架配置"""
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    region: str = "cn"
    
    # 数据时间范围
    train_start: str = "2010-01-01"
    train_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2023-12-31"
    
    # 股票池
    market: str = "csi300"
    benchmark: str = "SH000300"


# ==================== LLM 配置 ====================
@dataclass
class LLMConfig:
    """大语言模型配置"""
    # 后端: openai, dashscope
    provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "dashscope"))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "qwen-max"))
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", ""))
    
    # 阿里云DashScope (通义千问)
    dashscope_api_key: str = field(default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", ""))
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/api/v1"


# ==================== 因子评估配置 ====================
@dataclass
class FactorConfig:
    """因子评估配置"""
    # IC阈值
    ic_excellent: float = 0.05
    ic_good: float = 0.03
    ic_minimum: float = 0.02
    
    # ICIR阈值
    icir_excellent: float = 0.5
    icir_good: float = 0.3
    
    # 相关性阈值
    max_correlation: float = 0.7


# ==================== 模型配置 ====================
@dataclass
class ModelConfig:
    """模型训练配置"""
    # LightGBM
    lgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "verbose": -1,
    })
    
    # 交叉验证
    n_splits: int = 5
    test_size: float = 0.2


# ==================== 沙箱配置 ====================
@dataclass
class SandboxConfig:
    """沙箱执行配置"""
    timeout: int = 30  # 秒
    max_memory: int = 1024  # MB
    max_retries: int = 3


# ==================== GP搜索配置 ====================
@dataclass
class GPConfig:
    population_size: int = 1000
    generations: int = 20
    tournament_size: int = 20
    p_crossover: float = 0.7
    p_mutation: float = 0.1
    parsimony_coefficient: float = 0.01
    
    # 算子集
    function_set: List[str] = field(default_factory=lambda: [
        'add', 'sub', 'mul', 'div', 
        'sqrt', 'log', 'abs', 'neg',
        'max', 'min'
    ])


# ==================== 向量数据库配置 ====================
@dataclass  
class VectorDBConfig:
    """向量数据库配置"""
    provider: str = "milvus"  # milvus
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "alpha_factors"
    embedding_dim: int = 1536


# ==================== 全局配置实例 ====================
qlib_config = QlibConfig()
llm_config = LLMConfig()
factor_config = FactorConfig()
model_config = ModelConfig()
sandbox_config = SandboxConfig()
gp_config = GPConfig()
vector_db_config = VectorDBConfig()
