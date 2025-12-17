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


# ==================== 因子筛选配置 ====================
@dataclass
class SelectionConfig:
    """因子筛选配置"""

    # 快速筛选
    quick_sample_ratio: float = 0.1
    quick_ic_threshold: float = 0.02

    random_seed: int = 42

    # 语义去重
    enable_dedup: bool = True
    dedup_threshold: float = 0.85

    # 聚类
    enable_cluster: bool = True
    n_clusters: int = 10
    reps_per_cluster: int = 3
    cluster_method: str = "kmeans"  # kmeans | corr_greedy
    cluster_sample_ratio: float = 1.0
    cluster_sample_size: int = 0
    cluster_corr_threshold: float = 0.8

    # 正交化组合
    max_factors: int = 50
    corr_threshold: float = 0.7
    min_marginal_ic: float = 0.002

    # 完整评估
    enable_full_eval: bool = True
    full_ic_threshold: float = 0.02
    full_icir_threshold: float = 0.2
    full_eval_parallel: bool = False
    full_eval_n_workers: int = 4

    # 正交化组合
    enable_orthogonal: bool = True


# ==================== 缓存配置 ====================
@dataclass
class CacheConfig:
    """缓存配置"""

    enabled: bool = True
    redis_url: str = "redis://localhost:6379/1"
    ttl: int = 86400


# ==================== Celery 配置 ====================
@dataclass
class CeleryConfig:
    """Celery任务队列配置"""

    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"
    worker_concurrency: int = 4
    worker_prefetch_multiplier: int = 1
    task_time_limit: int = 3600
    task_soft_time_limit: int = 3300


# ==================== 进化引擎配置 ====================
@dataclass
class EvolutionConfig:
    """进化引擎配置"""

    # 种群参数
    population_size: int = 16
    elite_size: int = 4
    offspring_size: int = 8

    # 迭代参数
    max_generations: int = 10
    min_fitness: float = 0.6
    patience: int = 3

    # 多样性控制
    diversity_threshold: float = 0.3
    random_injection_rate: float = 0.1

    # 适应度权重/目标
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: {
            'ic': 0.25,
            'icir': 0.25,
            'sharpe': 0.25,
            'max_drawdown': 0.15,
            'turnover': 0.10,
        }
    )
    fitness_targets: Dict[str, float] = field(
        default_factory=lambda: {
            'ic': 0.05,
            'icir': 0.5,
            'sharpe': 1.0,
            'max_drawdown': 0.2,
            'turnover': 1.0,
        }
    )

    # 并行配置
    max_workers: int = 8


# 预设进化配置
EVOLUTION_FAST = EvolutionConfig(population_size=8, max_generations=5)
EVOLUTION_STANDARD = EvolutionConfig(population_size=16, max_generations=10)
EVOLUTION_THOROUGH = EvolutionConfig(population_size=32, max_generations=20)


# ==================== Ray 分布式配置 ====================
@dataclass
class RayConfig:
    """Ray分布式计算配置"""

    num_cpus: int = 4
    num_gpus: int = 0
    memory: int = 4 * 1024 * 1024 * 1024
    object_store_memory: int = 1024 * 1024 * 1024
    local_mode: bool = True
    address: str = "auto"


# ==================== GPU 配置 ====================
@dataclass
class GPUConfig:
    """GPU配置"""

    device: int = 0
    use_gpu: bool = True


# ==================== 训练周期配置 ====================
@dataclass
class TrainPeriodConfig:
    """训练周期配置 (Qlib格式)"""

    train_start: str = "2008-01-01"
    train_end: str = "2014-12-31"
    valid_start: str = "2015-01-01"
    valid_end: str = "2016-12-31"
    test_start: str = "2017-01-01"
    test_end: str = "2020-08-01"
    instruments: str = "csi300"

    def to_dict(self) -> Dict:
        return {
            "train_period": (self.train_start, self.train_end),
            "valid_period": (self.valid_start, self.valid_end),
            "test_period": (self.test_start, self.test_end),
            "instruments": self.instruments,
        }


# ==================== 全局配置实例 ====================
qlib_config = QlibConfig()
llm_config = LLMConfig()
factor_config = FactorConfig()
model_config = ModelConfig()
sandbox_config = SandboxConfig()
gp_config = GPConfig()
vector_db_config = VectorDBConfig()

selection_config = SelectionConfig()
cache_config = CacheConfig()
celery_config = CeleryConfig()
evolution_config = EvolutionConfig()
ray_config = RayConfig()
gpu_config = GPUConfig()
train_period_config = TrainPeriodConfig()
