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
    
    # 阿里云DashScope (通义千问) - 使用qwen3-max
    dashscope_api_key: str = field(default_factory=lambda: os.environ.get(
        "DASHSCOPE_API_KEY", "sk-d1359b64bbb94be9b58cde783f1da70a"
    ))
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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


# ==================== 因子筛选配置 ====================
@dataclass
class SelectionConfig:
    """因子筛选Pipeline配置"""
    
    # Stage 1: 快速预筛选
    quick_sample_ratio: float = 0.3      # 采样比例
    quick_ic_threshold: float = 0.005    # 快速IC阈值
    
    # Stage 2: 语义去重
    dedup_threshold: float = 0.92        # 代码相似度阈值
    
    # Stage 3: 聚类
    enable_cluster: bool = True
    n_clusters: int = 20                 # 聚类数
    reps_per_cluster: int = 3            # 每簇代表数
    
    # Stage 4: 完整评估
    full_ic_threshold: float = 0.001     # IC阈值 (测试: 0.001)
    full_icir_threshold: float = 0.01    # ICIR阈值 (测试: 0.01)
    
    # Stage 5: 正交化选择
    max_factors: int = 30                # 最大因子数
    corr_threshold: float = 0.7          # 因子相关性阈值
    min_marginal_ic: float = 0.003       # 最小边际IC
    
    # 并行配置
    use_parallel: bool = True
    n_workers: int = 4                   # 并行工作线程数


# ==================== Redis/Celery配置 ====================
@dataclass
class CacheConfig:
    """Redis缓存配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 2                          # 因子缓存使用db2
    ttl: int = 3600 * 24                 # 缓存24小时
    
    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class CeleryConfig:
    """Celery任务队列配置"""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    
    # 并发
    worker_concurrency: int = 4
    worker_prefetch_multiplier: int = 1
    
    # 超时
    task_time_limit: int = 3600          # 1小时
    task_soft_time_limit: int = 3000


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


# ==================== 进化引擎配置 ====================
@dataclass
class EvolutionConfig:
    """进化引擎配置"""
    
    # 种群参数
    population_size: int = 16          # 初始种群大小
    elite_size: int = 4                # 精英保留数量
    offspring_size: int = 8            # 每代新生成数量
    
    # 迭代参数
    max_generations: int = 10          # 最大迭代代数
    min_fitness: float = 0.6           # 目标适应度
    patience: int = 3                  # 早停耐心（连续N代无改进）
    
    # 多样性控制
    diversity_threshold: float = 0.3   # 代码相似度阈值
    random_injection_rate: float = 0.1 # 随机个体注入比例
    
    # 适应度权重
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'ic': 0.25,
        'icir': 0.25,
        'sharpe': 0.25,
        'max_drawdown': 0.15,
        'turnover': 0.10,
    })
    
    # 适应度目标值（用于归一化）
    fitness_targets: Dict[str, float] = field(default_factory=lambda: {
        'ic': 0.05,           # IC > 0.05 得满分
        'icir': 1.0,          # ICIR > 1.0 得满分
        'sharpe': 2.0,        # Sharpe > 2.0 得满分
        'max_drawdown': 0.30, # MDD < 30% 才有分
        'turnover': 1.0,      # Turnover < 100% 才有分
    })
    
    # 随机探索
    random_accept_rate: float = 0.1    # 随机接受低分因子的概率
    novelty_bonus: float = 0.3         # 新颖度奖励（降低门槛比例）
    
    # 并行配置
    max_workers: int = 8               # 最大并行评估数
    
    def __post_init__(self):
        # 验证权重和为1
        weight_sum = sum(self.fitness_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            for k in self.fitness_weights:
                self.fitness_weights[k] /= weight_sum


# ==================== Ray分布式配置 ====================
@dataclass
class RayConfig:
    """Ray分布式计算配置"""
    num_cpus: int = 4
    num_gpus: int = 0
    memory: int = 4 * 1024 * 1024 * 1024       # 4GB
    object_store_memory: int = 1024 * 1024 * 1024  # 1GB
    local_mode: bool = True                     # 本地模式
    address: str = "auto"                       # 集群地址


# ==================== GPU配置 ====================
@dataclass
class GPUConfig:
    """GPU配置"""
    device: int = 0           # GPU设备ID，-1表示使用CPU
    use_gpu: bool = True      # 是否使用GPU


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
selection_config = SelectionConfig()
cache_config = CacheConfig()
celery_config = CeleryConfig()
model_config = ModelConfig()
sandbox_config = SandboxConfig()
gp_config = GPConfig()
vector_db_config = VectorDBConfig()
evolution_config = EvolutionConfig()
ray_config = RayConfig()
gpu_config = GPUConfig()
train_period_config = TrainPeriodConfig()

# 预设进化配置
EVOLUTION_FAST = EvolutionConfig(
    population_size=8, elite_size=2, offspring_size=4,
    max_generations=5, patience=2
)
EVOLUTION_STANDARD = EvolutionConfig(
    population_size=16, elite_size=4, offspring_size=8,
    max_generations=10, patience=3
)
EVOLUTION_THOROUGH = EvolutionConfig(
    population_size=32, elite_size=8, offspring_size=16,
    max_generations=20, patience=5
)


# ==================== 便捷访问 ====================
__all__ = [
    # 配置类
    'QlibConfig',
    'LLMConfig', 
    'FactorConfig',
    'SelectionConfig',
    'CacheConfig',
    'CeleryConfig',
    'ModelConfig',
    'SandboxConfig',
    'GPConfig',
    'VectorDBConfig',
    'EvolutionConfig',
    'RayConfig',
    'GPUConfig',
    'TrainPeriodConfig',
    # 全局实例
    'qlib_config',
    'llm_config',
    'factor_config',
    'selection_config',
    'cache_config',
    'celery_config',
    'model_config',
    'sandbox_config',
    'gp_config',
    'vector_db_config',
    'evolution_config',
    'ray_config',
    'gpu_config',
    'train_period_config',
    # 预设配置
    'EVOLUTION_FAST',
    'EVOLUTION_STANDARD',
    'EVOLUTION_THOROUGH',
    # 路径
    'BASE_DIR',
    'DATA_DIR',
    'OUTPUT_DIR',
    'FACTORS_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
]
