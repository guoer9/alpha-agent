# Alpha Agent 完整模块API参考文档

> **版本**: 0.7.1  
> **描述**: LLM驱动的因子挖掘系统  
> **导出API数量**: 122个

---

## 目录

1. [包结构概览](#1-包结构概览)
2. [顶层API](#2-顶层api)
3. [agents/ - 多Agent系统](#3-agents---多agent系统)
4. [analysis/ - 风险分析](#4-analysis---风险分析)
5. [config/ - 配置管理](#5-config---配置管理)
6. [core/ - 核心组件](#6-core---核心组件)
7. [evolution/ - 进化式因子生成](#7-evolution---进化式因子生成)
8. [factors/ - 因子库](#8-factors---因子库)
9. [graph/ - 知识图谱](#9-graph---知识图谱)
10. [memory/ - 记忆系统](#10-memory---记忆系统)
11. [mining/ - 因子挖掘](#11-mining---因子挖掘)
12. [modeling/ - 建模模块](#12-modeling---建模模块)
13. [prompt/ - Prompt系统](#13-prompt---prompt系统)
14. [schema/ - 数据字典](#14-schema---数据字典)
15. [selection/ - 因子筛选](#15-selection---因子筛选)
16. [evaluation/ - 评估系统](#16-evaluation---评估系统)
17. [infra/ - 基础设施](#17-infra---基础设施)
18. [raptor/ - RAPTOR层次检索](#18-raptor---raptor层次检索)
19. [tasks/ - Celery任务](#19-tasks---celery任务)
20. [docker/ - 容器配置](#20-docker---容器配置)
21. [scripts/ - 脚本工具](#21-scripts---脚本工具)
22. [配置文件](#22-配置文件)
23. [运行脚本](#23-运行脚本)
24. [依赖列表](#24-依赖列表)
25. [模块依赖关系图](#25-模块依赖关系图)
26. [docs/ - 设计文档](#26-docs---设计文档)
27. [feature_repo/ - Feast特征仓库](#27-feature_repo---feast特征仓库)
28. [modeling/config.py - 模型配置](#28-modelingconfigpy---模型配置)
29. [tests/ - 测试模块](#29-tests---测试模块)
30. [目录结构说明](#30-目录结构说明)
31. [项目开发进度](#31-项目开发进度)
32. [快速使用示例](#32-快速使用示例)

---

## 1. 包结构概览

```
alpha_agent/
├── __init__.py          # 顶层包入口
├── agents/              # 多Agent协作系统
│   ├── mining_agent.py  # 因子挖掘Agent
│   ├── analysis_agent.py # 风险分析Agent
│   ├── orchestrator.py  # 多Agent协调器
│   └── reflexion.py     # 反思Agent
├── analysis/            # 分析模块
│   ├── attribution.py   # 收益归因
│   ├── knowledge_graph.py # Neo4j风险图谱
│   ├── market_regime.py # 市场状态检测
│   └── risk_analysis.py # 风险分析
├── config/              # 配置管理
│   └── settings.py      # 所有配置类
├── core/                # 核心组件
│   ├── base.py          # 基础类和数据结构
│   ├── evaluator.py     # 因子评估器
│   ├── llm.py           # LLM生成器
│   └── sandbox.py       # 安全沙箱
├── evolution/           # 进化式引擎
│   ├── engine.py        # 进化引擎
│   ├── hybrid_engine.py # LLM+GP混合引擎
│   └── individual.py    # 个体表示
├── factors/             # 因子库 (300+因子)
│   ├── alpha158.py      # Qlib Alpha158
│   ├── alpha360.py      # Qlib Alpha360
│   ├── classic_factors.py # Barra/技术因子
│   ├── gtja191.py       # 国泰君安191
│   └── worldquant101.py # WorldQuant 101
├── graph/               # GraphRAG
│   ├── retriever.py     # 图检索器
│   ├── schema.py        # 节点/边定义
│   └── store.py         # 图存储
├── memory/              # 记忆系统
│   ├── experiment_log.py # 实验日志
│   ├── rag.py           # RAG检索
│   └── vector_store.py  # Milvus向量库
├── mining/              # 因子挖掘
│   ├── backtest.py      # 回测模块
│   └── gp_engine.py     # 遗传规划引擎
├── modeling/            # 建模模块
│   ├── ensemble.py      # 集成学习
│   ├── feature_selector.py # 特征选择
│   └── qlib_model_zoo.py # Qlib模型库
├── prompt/              # Prompt系统
│   ├── composer.py      # Prompt组装器
│   └── templates.py     # 模板库
├── schema/              # 数据字典
│   ├── cn_stock_schema.py # A股数据定义
│   └── data_schema.py   # 通用数据Schema
├── selection/           # 因子筛选
│   ├── data_preprocessor.py # 数据预处理 (NEW)
│   ├── factor_cleaner.py # 因子代码清洗
│   ├── factor_wrapper.py # 因子封装
│   └── selector.py      # 筛选器
├── evaluation/          # 评估系统
│   ├── evaluator.py     # 因子评估器
│   └── metrics.py       # 回测指标计算
├── infra/               # 基础设施
│   ├── distributed.py   # Ray分布式计算
│   ├── feature_store.py # Feast特征存储
│   └── task_queue.py    # Celery任务队列
├── raptor/              # RAPTOR层次检索
│   ├── builder.py       # 树构建器
│   ├── retriever.py     # 层次检索器
│   └── tree.py          # 树结构定义
├── tasks/               # Celery任务定义
│   └── factor.py        # 因子计算任务
├── docker/              # Docker配置
│   ├── Dockerfile       # 镜像定义
│   └── docker-compose.yml # 编排文件
├── scripts/             # 脚本工具
│   ├── deploy_services.py # 服务部署
│   └── import_factors.py  # 因子导入
├── celeryconfig.py      # Celery配置
├── ray_config.py        # Ray配置
├── run_factor_mining.py # 因子挖掘运行脚本
├── run_factor_selection.py # 因子筛选运行脚本
├── requirements.txt     # 项目依赖
│
├── docs/                # 📚 设计文档
│   ├── SYSTEM_FLOW.md   # 系统流程
│   ├── RAPTOR_DESIGN.md # RAPTOR设计
│   ├── GRAPHRAG_DESIGN.md # GraphRAG设计
│   ├── EVOLUTION_DESIGN.md # 进化引擎设计
│   ├── PIPELINE.md      # Pipeline流程
│   ├── FACTOR_LIBRARY.md # 因子库设计
│   └── OPTIMIZATION_DESIGN.md # 优化设计
│
├── feature_repo/        # 🏪 Feast特征仓库
│   ├── feature_store.yaml # Feast配置
│   └── features.py      # 特征定义
│
├── tests/               # 🧪 测试模块
│   └── __init__.py
│
├── data/                # 📂 输入数据
├── output/              # 📤 输出结果
│   ├── factors/         # 生成的因子
│   ├── models/          # 训练模型
│   └── logs/            # 运行日志
├── mlruns/              # 📊 MLflow实验
│
├── README.md            # 项目说明
└── PROGRESS.md          # 开发进度
```

---

## 2. 顶层API

### 导入方式

```python
import alpha_agent

# 或选择性导入
from alpha_agent import MiningAgent, AnalysisAgent, Orchestrator
from alpha_agent import LLMGenerator, Sandbox, FactorEvaluator
from alpha_agent import qlib_config, llm_config
```

### 导出列表 (`__all__`) - 共122项

| 类别 | 导出项 |
|------|--------|
| **版本** | `__version__` |
| **配置类 (14)** | `QlibConfig`, `LLMConfig`, `FactorConfig`, `ModelConfig`, `SandboxConfig`, `GPConfig`, `VectorDBConfig`, `SelectionConfig`, `CacheConfig`, `CeleryConfig`, `EvolutionConfig`, `RayConfig`, `GPUConfig`, `TrainPeriodConfig` |
| **配置实例 (14)** | `qlib_config`, `llm_config`, `factor_config`, `model_config`, `sandbox_config`, `gp_config`, `vector_db_config`, `selection_config`, `cache_config`, `celery_config`, `evolution_config`, `ray_config`, `gpu_config`, `train_period_config` |
| **预设配置 (3)** | `EVOLUTION_FAST`, `EVOLUTION_STANDARD`, `EVOLUTION_THOROUGH` |
| **路径 (6)** | `BASE_DIR`, `DATA_DIR`, `OUTPUT_DIR`, `FACTORS_DIR`, `MODELS_DIR`, `LOGS_DIR` |
| **核心 (8)** | `BaseAgent`, `AgentResult`, `FactorResult`, `LLMGenerator`, `Sandbox`, `execute_code`, `FactorEvaluator`, `evaluate_factor` |
| **挖掘 (2)** | `GPEngine`, `run_backtest` |
| **记忆 (3)** | `MilvusStore`, `FactorMemory`, `ExperimentLogger` |
| **分析 (2)** | `RiskKnowledgeGraph`, `RiskAnalyzer` |
| **Agent (3)** | `MiningAgent`, `AnalysisAgent`, `Orchestrator` |
| **评估 (9)** | `BacktestMetrics`, `ICMetrics`, `ReturnMetrics`, `RiskMetrics`, `compute_all_metrics`, `compute_ic_metrics`, `compute_return_metrics`, `compute_risk_metrics`, `EvaluatorConfig` |
| **进化 (3)** | `Individual`, `EvolutionHistory`, `EvolutionaryEngine` |
| **因子库 (15)** | `BARRA_FACTORS`, `TECHNICAL_FACTORS`, `FUNDAMENTAL_FACTORS`, `VOLUME_PRICE_FACTORS`, `ALL_CLASSIC_FACTORS`, `ALPHA158_FACTORS`, `ALPHA360_FACTORS`, `WORLDQUANT_101_FACTORS`, `GTJA191_FACTORS`, `ACADEMIC_PREMIA_FACTORS`, `ALL_FACTORS`, `ClassicFactor`, `FactorCategory`, `FactorLibrary`, `create_factor_library` |
| **GraphRAG (9)** | `NodeType`, `EdgeType`, `FactorNode`, `ReflectionNode`, `RegimeNode`, `ConceptNode`, `GraphEdge`, `GraphStore`, `GraphRetriever` |
| **RAPTOR (6)** | `RaptorTree`, `TreeNode`, `RaptorRetriever`, `RetrievalConfig`, `RaptorBuilder`, `BuildConfig` |
| **Prompt (3)** | `PromptComposer`, `SystemPrompts`, `TaskTemplates` |
| **数据字典 (5)** | `DataSchema`, `FieldSchema`, `DataValidator`, `DataFrequency`, `DataType` |
| **筛选 (10)** | `FactorSelector`, `SelectionResult`, `select_factors`, `quick_filter`, `orthogonal_select`, `FactorWrapper`, `FactorMeta`, `load_factors`, `create_factor_wrapper` |
| **建模 (3)** | `FeatureSelector`, `AlphaEnsemble`, `MODELING_AVAILABLE` |
| **基础设施 (4)** | `FeatureStore`, `RayExecutor`, `distributed_backtest`, `INFRA_AVAILABLE` |

---

## 3. agents/ - 多Agent系统

### 导入

```python
from alpha_agent.agents import (
    MiningAgent, AnalysisAgent, Orchestrator,
    ReflexionAgent, ReflexionMemory, ReflexionEntry
)
```

### 3.1 MiningAgent

因子挖掘Agent，使用LangChain构建，负责生成、执行和评估alpha因子。

```python
class MiningAgent(BaseAgent):
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        enable_memory: bool = True,
    )
    
    def setup(self, df: pd.DataFrame, target: pd.Series) -> None
    def run(self, task: str, max_iterations: int = 5) -> AgentResult
    def step(self, instruction: str) -> Dict[str, Any]
    def chat(self, message: str) -> str
```

**工具列表**:
- `generate_factor` - 生成因子代码
- `execute_factor` - 执行因子代码
- `evaluate_factor` - 评估因子性能
- `search_similar` - 搜索相似因子
- `improve_factor` - 改进现有因子

### 3.2 AnalysisAgent

风险分析Agent，执行组合分析和归因。

```python
class AnalysisAgent(BaseAgent):
    def __init__(self, api_key: str = None, model: str = None)
    
    def setup(self, returns: pd.Series, factor_returns: pd.DataFrame = None)
    def run(self, task: str) -> AgentResult
    def generate_report(self) -> str
```

**工具列表**:
- `analyze_risk` - 分析组合风险
- `detect_market_regime` - 识别市场状态
- `factor_attribution` - 因子归因分析
- `style_analysis` - 风格分析

### 3.3 Orchestrator

多Agent协调器。

```python
class Orchestrator:
    def __init__(self)
    
    def register(self, name: str, agent: BaseAgent) -> None
    def create_task(self, task_type: str, content: str, priority: int = 1) -> Task
    def run_pipeline(self, tasks: List[Task] = None) -> Dict[str, Any]
    def get_status(self) -> Dict[str, Any]
```

### 3.4 ReflexionAgent

反思Agent，从经验中学习。

```python
@dataclass
class ReflexionEntry:
    task: str
    action: str
    result: str
    reflection: str
    lessons: List[str]
    timestamp: str
    success: bool

class ReflexionMemory:
    def add_entry(self, entry: ReflexionEntry) -> None
    def search(self, query: str, top_k: int = 5) -> List[ReflexionEntry]
    def get_lessons(self, task_type: str = None) -> List[str]

class ReflexionAgent:
    def __init__(self, llm: ChatOpenAI = None, memory: ReflexionMemory = None)
    
    def reflect(self, task: str, action: str, result: str, success: bool) -> ReflexionEntry
    def get_advice(self, task: str) -> str
```

---

## 4. analysis/ - 风险分析

### 导入

```python
from alpha_agent.analysis import (
    RiskKnowledgeGraph,
    RiskAnalyzer, RiskReport,
    brinson_attribution, factor_attribution, BrinsonResult,
    MarketRegimeDetector, MarketState,
    detect_style_rotation, detect_sector_rotation,
)
```

### 4.1 RiskKnowledgeGraph

基于Neo4j的风险知识图谱。

```python
class RiskKnowledgeGraph:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    )
    
    def connect(self) -> bool
    def add_factor(self, factor_id: str, name: str, category: str, **props) -> None
    def add_risk(self, risk_id: str, name: str, **props) -> None
    def add_exposure(self, factor_id: str, risk_id: str, weight: float) -> None
    def get_factor_risks(self, factor_id: str) -> List[Dict]
    def get_correlated_factors(self, factor_id: str, min_corr: float = 0.5) -> List[Dict]
```

### 4.2 RiskAnalyzer

风险分析器。

```python
@dataclass
class RiskReport:
    factor_name: str
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    risk_level: str = "medium"
    recommendations: List[str] = field(default_factory=list)

class RiskAnalyzer:
    def analyze(self, factor_returns: pd.Series, risk_factors: pd.DataFrame = None) -> RiskReport
    def compute_var(self, returns: pd.Series, confidence: float = 0.95) -> float
    def compute_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float
    def compute_max_drawdown(self, returns: pd.Series) -> float
```

### 4.3 MarketRegimeDetector

市场状态检测器。

```python
class MarketState(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"

class MarketRegimeDetector:
    def detect(self, returns: pd.Series) -> MarketState
    def get_regime_history(self, returns: pd.Series, window: int = 60) -> pd.Series
```

---

## 5. config/ - 配置管理

### 导入

```python
from alpha_agent.config import (
    # 路径
    BASE_DIR, DATA_DIR, OUTPUT_DIR, FACTORS_DIR, MODELS_DIR, LOGS_DIR,
    # 配置类
    QlibConfig, LLMConfig, FactorConfig, ModelConfig,
    SandboxConfig, GPConfig, VectorDBConfig,
    # 配置实例
    qlib_config, llm_config, factor_config, model_config,
    sandbox_config, gp_config, vector_db_config,
)
```

### 配置类详解 (14个配置类)

#### 核心配置

```python
@dataclass
class QlibConfig:
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    region: str = "cn"
    train_start: str = "2010-01-01"
    train_end: str = "2020-12-31"
    valid_start: str = "2021-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2023-12-31"
    market: str = "csi300"
    benchmark: str = "SH000300"

@dataclass
class LLMConfig:
    provider: str  # "openai" | "dashscope"
    model: str     # "gpt-4" | "qwen-max"
    temperature: float = 0.7
    max_tokens: int = 4096
    openai_api_key: str = ""
    dashscope_api_key: str = ""

@dataclass
class FactorConfig:
    ic_excellent: float = 0.05
    ic_good: float = 0.03
    ic_minimum: float = 0.02
    max_factors: int = 100
```

#### 进化引擎配置 (新增)

```python
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
    
    # 适应度权重
    fitness_weights: Dict[str, float] = {
        'ic': 0.25, 'icir': 0.25, 'sharpe': 0.25,
        'max_drawdown': 0.15, 'turnover': 0.10,
    }
    
    # 并行配置
    max_workers: int = 8

# 预设配置
EVOLUTION_FAST = EvolutionConfig(population_size=8, max_generations=5)
EVOLUTION_STANDARD = EvolutionConfig(population_size=16, max_generations=10)
EVOLUTION_THOROUGH = EvolutionConfig(population_size=32, max_generations=20)
```

#### 分布式配置 (新增)

```python
@dataclass
class RayConfig:
    """Ray分布式计算配置"""
    num_cpus: int = 4
    num_gpus: int = 0
    memory: int = 4 * 1024 * 1024 * 1024  # 4GB
    object_store_memory: int = 1024 * 1024 * 1024  # 1GB
    local_mode: bool = True
    address: str = "auto"

@dataclass
class GPUConfig:
    """GPU配置"""
    device: int = 0     # GPU设备ID，-1表示使用CPU
    use_gpu: bool = True

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
```

#### 其他配置类

```python
@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    redis_url: str = "redis://localhost:6379/1"
    ttl: int = 86400

@dataclass
class CeleryConfig:
    """Celery任务队列配置"""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"
    worker_concurrency: int = 4
    task_time_limit: int = 3600

@dataclass
class VectorDBConfig:
    """向量数据库配置"""
    provider: str = "milvus"
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "alpha_factors"
    embedding_dim: int = 1536
```

---

## 6. core/ - 核心组件

### 导入

```python
from alpha_agent.core import (
    BaseAgent, AgentResult, FactorResult,
    LLMGenerator,
    Sandbox, execute_code,
    FactorEvaluator, evaluate_factor,
)
```

### 6.1 BaseAgent

Agent抽象基类。

```python
class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class AgentResult:
    agent_name: str
    status: AgentStatus = AgentStatus.IDLE
    factors: List[FactorResult] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

class BaseAgent(ABC):
    def __init__(self, name: str = "BaseAgent")
    
    @abstractmethod
    def run(self, *args, **kwargs) -> AgentResult
```

### 6.2 LLMGenerator

LLM因子生成器 (支持OpenAI/DashScope)。

```python
class LLMGenerator:
    def __init__(
        self,
        provider: str = None,    # "openai" | "dashscope"
        model: str = None,
        temperature: float = None,
        api_key: str = None,
        system_prompt: str = None,
    )
    
    def generate(self, instruction: str) -> str
    def fix_error(self, code: str, error: str) -> str
    def improve_factor(self, code: str, ic: float, feedback: str = "") -> str
    def set_system_prompt(self, prompt: str) -> None
    def clear_memory(self) -> None
```

### 6.3 Sandbox

安全代码执行沙箱。

```python
class Sandbox:
    def __init__(self, timeout_seconds: int = None, max_retries: int = None)
    
    def execute(self, code: str, df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]

# 便捷函数
def execute_code(code: str, df: pd.DataFrame, timeout_seconds: int = None) -> Tuple[Optional[pd.Series], Optional[str]]
```

### 6.4 FactorEvaluator

因子评估器。

```python
@dataclass
class EvaluationResult:
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    rank_icir: float = 0.0
    top_return: float = 0.0
    bottom_return: float = 0.0
    long_short_return: float = 0.0
    status: FactorStatus
    recommendation: str = ""

class FactorEvaluator:
    def __init__(self, ic_excellent: float = None, ic_good: float = None, ic_minimum: float = None)
    
    def evaluate(self, factor: pd.Series, target: pd.Series, n_groups: int = 5) -> EvaluationResult

# 便捷函数
def evaluate_factor(factor: pd.Series, target: pd.Series, name: str = "factor") -> FactorResult
```

---

## 7. evolution/ - 进化式因子生成

### 导入

```python
from alpha_agent.evolution import (
    EvolutionConfig,
    Individual, EvolutionHistory,
    EvolutionaryEngine,
)
```

### 7.1 HybridEvolutionEngine

LLM + GP 混合进化引擎 (位于 `hybrid_engine.py`)。

```python
@dataclass
class HybridConfig:
    # Phase 1: LLM探索
    llm_batch_size: int = 10
    llm_rounds: int = 3
    seed_threshold_ic: float = 0.015
    
    # Phase 2: GP精炼
    gp_population: int = 50
    gp_generations: int = 10
    gp_mutation_rate: float = 0.3
    
    # Phase 3: LLM反思
    reflect_top_k: int = 5

class HybridEvolutionEngine:
    def __init__(
        self,
        config: HybridConfig = None,
        llm_generator: Callable = None,
        gp_mutator: Callable = None,
        evaluator: Callable = None,
    )
    
    def evolve(self, initial_factors: List[FactorCandidate] = None) -> List[FactorCandidate]
```

---

## 8. factors/ - 因子库

### 导入

```python
from alpha_agent.factors import (
    # 经典因子
    BARRA_FACTORS, TECHNICAL_FACTORS, FUNDAMENTAL_FACTORS, VOLUME_PRICE_FACTORS,
    ALL_CLASSIC_FACTORS,
    # Qlib因子
    ALPHA158_FACTORS, ALPHA360_FACTORS,
    # WorldQuant
    WORLDQUANT_101_FACTORS,
    # 国泰君安
    GTJA191_FACTORS,
    # 学术溢价
    ACADEMIC_PREMIA_FACTORS,
    # 汇总
    ALL_FACTORS,
    # 类
    ClassicFactor, FactorCategory, FactorLibrary,
    # 函数
    get_alpha158_factors, get_alpha360_factors,
    get_worldquant101_factors, get_gtja191_factors,
)
```

### 因子库统计

| 因子集 | 数量 | 来源 |
|--------|------|------|
| BARRA | ~20 | MSCI Barra CNE5/CNE6 |
| Technical | ~50 | 经典技术指标 |
| Fundamental | ~30 | 财务指标 |
| Alpha158 | 158 | Microsoft Qlib |
| Alpha360 | 360 | Microsoft Qlib (扩展版) |
| WorldQuant101 | 101 | Kakushadze学术论文 |
| GTJA191 | 191 | 国泰君安短周期因子 |
| **总计** | **300+** | |

---

## 9. graph/ - 知识图谱

### 导入

```python
from alpha_agent.graph import (
    NodeType, EdgeType,
    FactorNode, ReflectionNode, RegimeNode, ConceptNode,
    GraphEdge,
    GraphStore,
    GraphRetriever,
)
```

### 9.1 节点类型

```python
class NodeType(Enum):
    FACTOR = "factor"
    REFLECTION = "reflection"
    REGIME = "regime"
    CONCEPT = "concept"

@dataclass
class FactorNode:
    id: str
    name: str
    code: str
    ic: float
    category: str
    embedding: List[float] = None
```

### 9.2 GraphStore

```python
class GraphStore:
    def add_node(self, node: Union[FactorNode, ReflectionNode, ...]) -> str
    def add_edge(self, edge: GraphEdge) -> None
    def query(self, query: str) -> List[Dict]
```

### 9.3 GraphRetriever

```python
class GraphRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]
    def multi_hop_query(self, start_node: str, hops: int = 2) -> List[Dict]
```

---

## 10. memory/ - 记忆系统

### 导入

```python
from alpha_agent.memory import (
    MilvusStore, FactorMemory,
    ExperimentLogger,
    RAGGenerator, FactorDeduplicator,
    create_rag_prompt, check_factor_duplicate,
)
```

### 10.1 MilvusStore

Milvus向量数据库存储。

```python
@dataclass
class FactorRecord:
    factor_id: str
    name: str
    code: str
    description: str
    ic: float
    icir: float
    status: str
    tags: List[str]
    embedding: List[float] = None

class MilvusStore:
    def __init__(self, host: str = None, port: int = None, collection_name: str = None)
    
    def connect(self) -> bool
    def create_collection(self) -> bool
    def insert(self, records: List[FactorRecord]) -> int
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]
    def get_all_factors(self, limit: int = 10000, min_ic: float = None) -> List[Dict]
```

### 10.2 FactorMemory

因子记忆管理器。

```python
class FactorMemory:
    def __init__(self, store: MilvusStore = None)
    
    def add_factor(self, name: str, code: str, ic: float, ...) -> str
    def search_similar(self, code: str, top_k: int = 5) -> List[Dict]
    def is_duplicate(self, code: str, threshold: float = 0.9) -> bool
```

### 10.3 ExperimentLogger

实验日志记录。

```python
class ExperimentLogger:
    def log_factor(self, factor: FactorResult) -> None
    def log_experiment(self, name: str, config: Dict, metrics: Dict) -> None
    def get_history(self, n: int = 100) -> List[Dict]
```

---

## 11. mining/ - 因子挖掘

### 导入

```python
from alpha_agent.mining import (
    GPEngine,
    run_backtest, BacktestResult, format_backtest_report,
    run_qlib_backtest, run_qlib_weight_backtest,
    run_qlib_factor_analysis,
    plot_backtest_result, compute_simple_backtest,
    QLIB_AVAILABLE,
)
```

### 11.1 GPEngine

遗传规划因子搜索引擎。

```python
class GPEngine:
    def __init__(self, config: GPConfig = None)
    
    def run(self, df: pd.DataFrame, target: pd.Series, generations: int = 50) -> List[Dict]
    def mutate(self, code: str) -> str
    def crossover(self, code1: str, code2: str) -> str
```

### 11.2 Backtest

```python
@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    ic: float
    icir: float

def run_backtest(factor: pd.Series, returns: pd.Series, ...) -> BacktestResult
def run_qlib_backtest(factor_expr: str, start: str, end: str, ...) -> Dict
```

---

## 12. modeling/ - 建模模块

### 导入

```python
from alpha_agent.modeling import (
    FeatureSelector, select_features,
    AlphaEnsemble, ensemble_alpha,
    QlibModelZoo, QlibBenchmark,  # 需要Qlib
    QLIB_ZOO_AVAILABLE,
)
```

### 12.1 FeatureSelector

特征选择器 (SHAP/IC/VIF)。

```python
class FeatureSelector:
    def __init__(self, method: str = "shap")  # "shap" | "ic" | "vif" | "rfe"
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None
    def select(self, n_features: int = 30) -> List[str]
    def get_importance(self) -> pd.Series
```

### 12.2 AlphaEnsemble

因子集成学习。

```python
class AlphaEnsemble:
    def __init__(self, method: str = "equal")  # "equal" | "ic_weighted" | "ml"
    
    def fit(self, factors: pd.DataFrame, target: pd.Series) -> None
    def predict(self, factors: pd.DataFrame) -> pd.Series
    def get_weights(self) -> Dict[str, float]
```

### 12.3 QlibModelZoo

Qlib模型库基准测试。

```python
class QlibModelZoo:
    def __init__(self, model_set: str = "standard")  # "fast" | "standard" | "full" | "deep" | "sota"
    
    def run_benchmark(self, dataset: DatasetH) -> Dict[str, Dict]
    def compare_models(self, results: Dict) -> pd.DataFrame
```

**预置模型配置**:
- LGBModel, XGBModel, CatBoostModel
- LinearModel, DNNModelPytorch
- LSTM, GRU, Transformer
- TabNet, Double Ensemble
- TCN, ALSTM, GATs, TRA, HIST, Localformer, SFM

---

## 13. prompt/ - Prompt系统

### 导入

```python
from alpha_agent.prompt import (
    PromptComposer,
    SystemPrompts, SchemaTemplates, RAGTemplates, 
    ReflectionTemplates, TaskTemplates,
    TaskType, RoleType,
    SchemaContext, RAGContext, ReflectionContext, TaskContext,
    ComposedPrompt,
)
```

### 13.1 Prompt架构 (5层分层)

```
┌─────────────────────────────────────────────┐
│ 1. System Prompt (永恒不变)                 │
│    - 角色定义                               │
│    - 能力边界                               │
│    - 输出格式                               │
├─────────────────────────────────────────────┤
│ 2. Schema Context (硬约束)                  │
│    - 数据字典                               │
│    - 可用算子                               │
│    - 禁止事项                               │
├─────────────────────────────────────────────┤
│ 3. RAG Context (软引导) - 动态              │
│    - 相关高分因子                           │
│    - 相似失败案例                           │
│    - 策略概念                               │
├─────────────────────────────────────────────┤
│ 4. History/Feedback (进化压力) - 动态       │
│    - 上轮回测结果                           │
│    - 失败诊断                               │
│    - 改进建议                               │
├─────────────────────────────────────────────┤
│ 5. Task Instruction (任务指令)              │
│    - 具体任务描述                           │
│    - 约束条件                               │
│    - 期望输出                               │
└─────────────────────────────────────────────┘
```

### 13.2 PromptComposer

```python
class TaskType(Enum):
    GENERATE_NEW = "generate_new"
    IMPROVE_FACTOR = "improve_factor"
    MUTATE_FACTOR = "mutate_factor"
    CROSSOVER_FACTORS = "crossover_factors"
    DIAGNOSE_FACTOR = "diagnose_factor"

class RoleType(Enum):
    ALPHA_MINER = "alpha_miner"
    FACTOR_EVALUATOR = "factor_evaluator"
    FACTOR_IMPROVER = "factor_improver"

@dataclass
class ComposedPrompt:
    system: str
    user: str
    token_estimate: int = 0
    schema_included: bool = False
    rag_included: bool = False
    reflection_included: bool = False
    
    def to_messages(self) -> List[Dict]
    def to_langchain_messages(self) -> List[Message]

class PromptComposer:
    def __init__(
        self,
        data_schema = None,      # DataSchema实例
        rag_retriever = None,    # RAG检索器
        graph_retriever = None,  # GraphRAG检索器
    )
    
    def compose(
        self,
        task_type: TaskType,
        task_params: Dict = None,
        role: RoleType = RoleType.ALPHA_MINER,
        schema_context: SchemaContext = None,
        rag_context: RAGContext = None,
        reflection_context: ReflectionContext = None,
        include_schema: bool = True,
        include_rag: bool = True,
        include_reflection: bool = True,
        max_rag_examples: int = 3,
        max_reflections: int = 5,
    ) -> ComposedPrompt
    
    # 便捷方法
    def for_generation(self, theme: str, target_ic: float, ...) -> ComposedPrompt
    def for_improvement(self, original_code: str, current_ic: float, ...) -> ComposedPrompt
    def for_mutation(self, original_code: str, mutation_type: str) -> ComposedPrompt
```

### 13.3 SystemPrompts (角色定义)

```python
class SystemPrompts:
    ALPHA_MINER = """你是一个专业的量化因子挖掘专家...
    ## 你的能力
    - 深入理解金融市场微观结构和量价关系
    - 精通技术分析、基本面分析和另类数据分析
    ## 你的约束
    1. 只能使用提供的数据字段
    2. 代码必须可执行
    3. 避免未来函数
    4. 注意数值稳定性
    """
    
    FACTOR_EVALUATOR = """你是一个量化因子评估专家...
    ## 评估维度
    1. 预测能力: IC、Rank IC、IC_IR
    2. 稳定性: IC时序稳定性
    3. 交易成本: 换手率
    4. 风险特征: 相关性、尾部风险
    """
    
    FACTOR_IMPROVER = """你是一个量化因子改进专家...
    ## 改进策略
    1. 参数调优
    2. 逻辑优化
    3. 正交化
    4. 组合增强
    5. 条件化
    """
```

### 13.4 TaskTemplates (任务指令)

```python
class TaskTemplates:
    GENERATE_NEW = """## 任务: 生成新因子
    请基于以下主题生成一个新的 Alpha 因子:
    - 主题: {theme}
    - 目标IC: >{target_ic}
    - 换手率约束: <{max_turnover}
    """
    
    IMPROVE_FACTOR = """## 任务: 改进因子
    请改进以下因子:
    ```python
    {original_code}
    ```
    当前表现: IC: {current_ic:.4f}, ICIR: {current_icir:.4f}
    问题诊断: {diagnosis}
    改进方向: {improvement_direction}
    """
    
    MUTATE_FACTOR = """## 任务: 因子变异
    请对以下因子进行变异，生成一个变体:
    变异类型: {mutation_type}
    """
    
    CROSSOVER_FACTORS = """## 任务: 因子交叉
    请将以下两个因子的优点结合，生成一个新因子:
    ### 因子A (IC={ic_a:.4f})
    ### 因子B (IC={ic_b:.4f})
    交叉策略: {crossover_strategy}
    """
    
    DIAGNOSE_FACTOR = """## 任务: 因子诊断
    请分析以下因子的问题并给出改进建议...
    """
```

### 13.5 上下文数据类

```python
@dataclass
class SchemaContext:
    fields: List[Dict]                    # 字段列表
    custom_operators: List[str]           # 自定义算子
    forbidden_operations: List[str]       # 禁止操作

@dataclass
class RAGContext:
    similar_factors: List[Dict]           # 相似因子
    related_concepts: List[str]           # 相关概念
    market_regime: str                    # 市场状态

@dataclass
class ReflectionContext:
    successes: List[Dict]                 # 成功经验
    failures: List[Dict]                  # 失败教训
    backtest_summary: Dict                # 回测摘要
```

---

## 14. schema/ - 数据字典

### 导入

```python
from alpha_agent.schema import (
    DataSchema, FieldSchema, DataValidator,
    DataFrequency, DataType,
)
```

### 14.1 DataSchema

```python
class DataFrequency(Enum):
    DAILY = "daily"
    MINUTE = "minute"
    TICK = "tick"

@dataclass
class FieldSchema:
    name: str
    dtype: str
    description: str
    nullable: bool = True

class DataSchema:
    def __init__(self, name: str, fields: List[FieldSchema], frequency: DataFrequency)
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]
    def to_dict(self) -> Dict
```

---

## 15. selection/ - 因子筛选

### 导入

```python
from alpha_agent.selection import (
    # 筛选器
    FactorSelector, SelectionResult,
    select_factors, quick_filter, orthogonal_select,
    # 因子封装
    FactorWrapper, FactorMeta,
    load_factors, create_factor_wrapper,
    # 因子清洗
    FactorCleaner, CleaningStats,
    clean_factors, clean_factor_code,
    adapt_field_references, FIELD_ALIASES, DERIVED_FIELDS,
    # 数据预处理 (NEW)
    add_derived_fields, prepare_train_test_data,
    split_by_date, handle_missing_values,
)
```

### 15.1 FactorSelector

因子筛选器。

```python
@dataclass
class SelectionResult:
    selected_factors: List[str]
    weights: Dict[str, float]
    metrics: Dict[str, float]
    correlation_matrix: pd.DataFrame

class FactorSelector:
    def __init__(self, config: SelectionConfig = None)
    
    def fit(self, factors: pd.DataFrame, target: pd.Series) -> SelectionResult
    def quick_filter(self, factors: pd.DataFrame, target: pd.Series, top_n: int = 100) -> List[str]
    def orthogonal_select(self, factors: pd.DataFrame, max_corr: float = 0.7) -> List[str]
```

### 15.2 FactorWrapper

因子封装为可回测格式。

```python
@dataclass
class FactorMeta:
    name: str
    code: str
    category: str
    ic: float
    source: str

class FactorWrapper:
    def __init__(self, factors: List[FactorMeta])
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame
    def to_qlib_expr(self) -> List[str]
    
    @classmethod
    def from_selection_result(cls, result: SelectionResult) -> "FactorWrapper"
```

### 15.3 FactorCleaner (NEW)

因子代码清洗器，用于处理大模型生成的因子代码。

**设计理念**：清洗而非过滤，保留所有因子。

```python
@dataclass
class CleaningStats:
    """清洗统计信息"""
    total_factors: int = 0
    imports_removed: int = 0
    fields_adapted: int = 0
    code_reformatted: int = 0

class FactorCleaner:
    """因子代码清洗器"""
    def __init__(
        self,
        remove_imports: bool = True,    # 移除预置模块的import
        adapt_fields: bool = True,       # 适配字段别名
        ensure_wrapper: bool = True,     # 确保函数包装
        custom_aliases: Dict[str, str] = None,  # 自定义别名
    )
    
    def clean_code(self, code: str) -> str
    def clean(self, factors: List[Dict]) -> List[Dict]
    def get_stats(self) -> CleaningStats
```

**便捷函数**：

```python
# 清洗因子代码
def clean_factor_code(code: str) -> tuple[str, Dict[str, int]]

# 批量清洗因子
def clean_factors(
    factors: List[Dict],
    available_columns: List[str] = None,
    verbose: bool = True,
) -> List[Dict]

# 移除安全的import语句
def remove_safe_imports(code: str) -> tuple[str, int]

# 适配字段别名
def adapt_field_references(code: str) -> tuple[str, int]

# 添加派生字段到DataFrame
def add_derived_fields(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame
```

**字段别名常量**：

```python
FIELD_ALIASES = {
    'Close': 'close', 'Volume': 'volume', 'Open': 'open',
    'High': 'high', 'Low': 'low', 'VWAP': 'vwap',
    'ret': 'returns', 'Turnover': 'turnover', ...
}

DERIVED_FIELDS = {
    'market_cap', 'market_ret', 'returns', 'amount',
    'amplitude', 'turnover', 'adv5', 'adv10', 'adv20',
}
```

**使用示例**：

```python
from alpha_agent.selection import clean_factors, FactorCleaner

# 函数式用法
cleaned = clean_factors(factors)

# 面向对象用法
cleaner = FactorCleaner(custom_aliases={'MY_FIELD': 'close'})
cleaned = cleaner.clean(factors)
print(cleaner.stats.summary())
# 输出: 清洗统计: 共100个因子, 移除15个import, 适配8个字段, 格式化3个代码
```

### 15.4 DataPreprocessor (NEW)

数据预处理模块，将数据处理逻辑从主pipeline中分离。

**文件**: `selection/data_preprocessor.py`

```python
def add_derived_fields(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    为DataFrame添加派生字段
    
    添加的字段:
    - returns: 日收益率
    - market_cap: 市值估算 (close * volume * 100)
    - market_ret: 市场平均收益
    - amount: 成交额
    - amplitude: 振幅
    - turnover: 换手率别名
    - adv5/10/20: 5/10/20日平均成交量
    """

def prepare_train_test_data(
    data: pd.DataFrame,
    target: pd.Series,
    factor_values: pd.DataFrame,
    train_start: str = "2022-01-01",
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    test_end: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    准备训练和测试数据
    
    Returns:
        X_train, y_train, X_test, y_test
    """

def split_by_date(
    data: pd.DataFrame,
    target: pd.Series,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """按日期分割训练测试集"""

def handle_missing_values(
    X: pd.DataFrame,
    y: pd.Series,
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """处理缺失值：删除target为空的行，因子列填充指定值"""
```

**使用示例**：

```python
from alpha_agent.selection import (
    add_derived_fields, prepare_train_test_data
)

# 添加派生字段
data = add_derived_fields(data)

# 准备训练测试数据
X_train, y_train, X_test, y_test = prepare_train_test_data(
    data, target, factor_values,
    train_start="2022-01-01", train_end="2022-12-31",
    test_start="2023-01-01", test_end="2023-12-31",
)
```

---

## 16. evaluation/ - 评估系统

### 导入

```python
from alpha_agent.evaluation import (
    # 指标数据类
    ICMetrics, ReturnMetrics, RiskMetrics, BacktestMetrics,
    # 计算函数
    compute_ic_metrics, compute_return_metrics,
    compute_risk_metrics, compute_all_metrics,
    # 评估器
    FactorEvaluator, EvaluatorConfig,
)
```

### 16.1 指标数据类

```python
@dataclass
class ICMetrics:
    ic: float = 0.0              # 总IC (Spearman)
    ic_std: float = 0.0          # IC标准差
    icir: float = 0.0            # IC IR
    rank_ic: float = 0.0         # Rank IC
    rank_icir: float = 0.0       # Rank ICIR
    ic_positive_rate: float = 0.0  # IC正比例
    ic_grade: str = "D"          # 等级 (A/B/C/D)
    turnover: float = 0.0        # 换手率

@dataclass
class ReturnMetrics:
    total_return: float = 0.0
    annual_return: float = 0.0
    excess_return: float = 0.0
    top_group_return: float = 0.0
    bottom_group_return: float = 0.0
    long_short_return: float = 0.0
    group_returns: List[float] = None
    monotonicity_score: float = 0.0

@dataclass
class RiskMetrics:
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    downside_volatility: float = 0.0
    sortino_ratio: float = 0.0

@dataclass
class BacktestMetrics:
    ic_metrics: ICMetrics
    return_metrics: ReturnMetrics
    risk_metrics: RiskMetrics
    overall_score: float = 0.0
    pass_threshold: bool = False
```

### 16.2 FactorEvaluator

```python
@dataclass
class EvaluatorConfig:
    # IC阈值
    ic_pass: float = 0.02
    ic_good: float = 0.03
    ic_excellent: float = 0.05
    # 换手率阈值
    max_turnover: float = 0.5
    # 分组数
    n_groups: int = 5
    # 回测天数
    min_periods: int = 100

class FactorEvaluator:
    def __init__(self, config: EvaluatorConfig = None)
    
    def quick_evaluate(self, factor: pd.Series, target: pd.Series) -> ICMetrics
    def full_evaluate(self, factor: pd.Series, target: pd.Series) -> BacktestMetrics
    def generate_report(self, metrics: BacktestMetrics) -> str
```

---

## 17. infra/ - 基础设施

### 导入

```python
from alpha_agent.infra import (
    # 特征存储
    FeatureStore, get_feature_store, FeatureDefinition, FeatureSet,
    # 任务队列
    celery_app, async_task, TaskStatus, TaskResult, TaskManager,
    # 分布式计算
    RayExecutor, distributed_backtest, distributed_factor_compute,
    hyperparameter_search, RAY_AVAILABLE,
)
```

### 17.1 FeatureStore (Feast集成)

```python
@dataclass
class FeatureDefinition:
    name: str
    dtype: str  # "float32" | "int64" | "string"
    description: str = ""
    tags: List[str] = None

@dataclass
class FeatureSet:
    name: str
    entity: str
    features: List[FeatureDefinition]
    ttl_days: int = 1

class FeatureStore:
    def __init__(self, repo_path: str = None)
    
    def connect(self) -> bool
    def register_feature_set(self, feature_set: FeatureSet) -> bool
    def write_features(self, entity_df: pd.DataFrame, feature_df: pd.DataFrame) -> int
    def read_features(self, entity_df: pd.DataFrame, features: List[str]) -> pd.DataFrame
    def list_feature_sets(self) -> List[str]
    def materialize(self, start: datetime, end: datetime) -> None

# 单例获取
def get_feature_store() -> FeatureStore
```

### 17.2 TaskManager (Celery集成)

```python
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Any = None
    error: str = ""
    duration: float = 0.0

class TaskManager:
    def __init__(self, broker_url: str = None)
    
    def register_task(self, name: str, func: Callable) -> None
    def submit(self, task_name: str, *args, **kwargs) -> str
    def get_result(self, task_id: str, timeout: int = 60) -> TaskResult
    def cancel(self, task_id: str) -> bool

# Celery任务装饰器
def async_task(name: str = None, queue: str = "default"):
    """装饰器，注册Celery任务"""
```

### 17.3 RayExecutor (分布式计算)

```python
class RayExecutor:
    def __init__(self, num_cpus: int = None, address: str = None)
    
    def map(self, func: Callable, items: List) -> List
    def submit(self, func: Callable, *args) -> Any
    def shutdown(self) -> None

# 预置分布式函数
def distributed_backtest(
    factors: List[Dict],
    data: pd.DataFrame,
    target: pd.Series,
    n_workers: int = 4,
) -> List[Dict]

def distributed_factor_compute(
    factor_codes: List[str],
    data: pd.DataFrame,
    n_workers: int = 4,
) -> Dict[str, pd.Series]

def hyperparameter_search(
    objective: Callable,
    param_space: Dict,
    n_trials: int = 20,
    n_workers: int = 4,
) -> Dict
```

---

## 18. raptor/ - RAPTOR层次检索

### 导入

```python
from alpha_agent.raptor import (
    RaptorTree, TreeNode,
    RaptorRetriever, RetrievalConfig, RetrievalResult,
    RaptorBuilder, BuildConfig,
)
```

### 18.1 RAPTOR概念

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) 是一种层次化知识组织和检索系统：

```
四层金字塔:
- L0: 原始因子 (叶子节点)
- L1: 因子簇 (相似因子聚合，如"短期动量因子群")
- L2: 策略类型 (如"动量策略"、"价值策略")
- L3: 全局洞察 (跨策略的高级知识)
```

### 18.2 TreeNode

```python
@dataclass
class TreeNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: int = 0                    # 0=因子, 1=簇, 2=策略, 3=全局
    text: str = ""                    # 节点文本/摘要
    summary: str = ""                 # LLM生成的摘要
    parent_id: str = ""               # 父节点ID
    children_ids: List[str] = field(default_factory=list)
    factor_id: str = ""               # 关联的因子ID (L0)
    factor_name: str = ""
    factor_code: str = ""
    cluster_id: int = -1              # 所属聚类
    embedding: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
```

### 18.3 RaptorTree

```python
class RaptorTree:
    def __init__(self, name: str = "factor_tree")
    
    def add_node(self, node: TreeNode) -> str
    def get_node(self, node_id: str) -> Optional[TreeNode]
    def get_children(self, node_id: str) -> List[TreeNode]
    def get_parent(self, node_id: str) -> Optional[TreeNode]
    def get_level_nodes(self, level: int) -> List[TreeNode]
    def get_ancestors(self, node_id: str) -> List[TreeNode]
    def get_descendants(self, node_id: str) -> List[TreeNode]
    def get_leaf_factors(self, node_id: str = None) -> List[TreeNode]
    def link_parent_child(self, parent_id: str, child_id: str) -> None
    def save(self, path: str) -> None
    def load(self, path: str) -> None
    def stats(self) -> Dict
```

### 18.4 RaptorRetriever

```python
@dataclass
class RetrievalConfig:
    strategy: str = "hybrid"      # "top_down" | "traversal" | "hybrid"
    top_k: int = 10
    similarity_threshold: float = 0.5
    include_ancestors: bool = True
    include_siblings: bool = True
    max_depth: int = 3

@dataclass
class RetrievalResult:
    nodes: List[TreeNode]
    scores: List[float]
    paths: List[List[TreeNode]]
    context: str                  # 生成的LLM上下文

class RaptorRetriever:
    def __init__(self, tree: RaptorTree, embedder: Callable = None, config: RetrievalConfig = None)
    
    def retrieve(self, query: str, strategy: str = None, top_k: int = None) -> RetrievalResult
    def retrieve_by_category(self, category: str, top_k: int = 10) -> List[TreeNode]
    def retrieve_by_tags(self, tags: List[str], top_k: int = 10) -> List[TreeNode]
    def retrieve_cluster(self, factor_id: str) -> List[TreeNode]
    def retrieve_strategy_factors(self, strategy_name: str) -> List[TreeNode]
```

---

## 19. tasks/ - Celery任务

### 导入

```python
from alpha_agent.tasks.factor import (
    compute_factor,
    evaluate_factor_ic,
    batch_evaluate_factors,
    update_factors,
    batch_backtest,
    FactorCache,
    get_factor_cache,
)
```

### 19.1 FactorCache

```python
class FactorCache:
    """因子计算结果Redis缓存"""
    
    def __init__(self, host='localhost', port=6379, db=2, ttl=3600*24)
    
    def get(self, factor_code: str, data_hash: str) -> Optional[Dict]
    def set(self, factor_code: str, data_hash: str, result: Dict) -> None
    def clear_all(self) -> None
```

### 19.2 Celery任务

```python
@shared_task(bind=True, max_retries=3, queue='factor')
def compute_factor(self, factor_code: str, data: dict) -> Dict:
    """计算单个因子"""

@shared_task(bind=True, max_retries=2, queue='factor')
def evaluate_factor_ic(
    self,
    factor_id: str,
    factor_code: str,
    data_json: str,
    target_json: str,
    data_hash: str,
) -> Dict:
    """评估单个因子IC"""

@shared_task(queue='factor')
def batch_evaluate_factors(
    factors: List[Dict],
    data_json: str,
    target_json: str,
    data_hash: str,
) -> List[Dict]:
    """批量评估因子 - 并行分发"""

@shared_task
def update_factors() -> Dict:
    """更新所有因子 (定时任务)"""

@shared_task
def batch_backtest(factor_ids: list) -> List[Dict]:
    """批量回测因子"""
```

---

## 20. docker/ - 容器配置

### 20.1 Dockerfile

```dockerfile
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 环境变量
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "run_factor_mining.py", "--mode", "standard"]
```

### 20.2 docker-compose.yml

```yaml
version: '3.8'

services:
  # Alpha Agent主服务
  alpha-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}
      - MILVUS_HOST=milvus
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_HOST=redis
    depends_on:
      - milvus
      - neo4j
      - redis
    volumes:
      - ./data:/app/data
      - ./output:/app/output

  # Milvus向量数据库
  milvus:
    image: milvusdb/milvus:v2.3-latest
    ports:
      - "19530:19530"
      - "9091:9091"
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    depends_on:
      - etcd
      - minio
    volumes:
      - milvus_data:/var/lib/milvus

  # etcd (Milvus元数据)
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
    volumes:
      - etcd_data:/etcd

  # MinIO (Milvus对象存储)
  minio:
    image: minio/minio:latest
    ports:
      - "9001:9001"
    environment:
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    command: minio server /data --console-address ":9001"
    volumes:
      - minio_data:/data

  # Neo4j图数据库
  neo4j:
    image: neo4j:5.11
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data

  # Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Attu (Milvus可视化)
  attu:
    image: zilliz/attu:latest
    ports:
      - "3000:3000"
    environment:
      - MILVUS_URL=milvus:19530

volumes:
  milvus_data:
  etcd_data:
  minio_data:
  neo4j_data:
  redis_data:
```

### 20.3 端口映射表

| 服务 | 端口 | 说明 |
|------|------|------|
| alpha-agent | 8000 | 主服务API |
| Milvus | 19530 | 向量数据库gRPC |
| Milvus | 9091 | Milvus REST |
| MinIO | 9001 | 对象存储控制台 |
| Neo4j | 7474 | HTTP浏览器 |
| Neo4j | 7687 | Bolt协议 |
| Redis | 6379 | 缓存服务 |
| Attu | 3000 | Milvus可视化 |

---

## 21. scripts/ - 脚本工具

### 21.1 deploy_services.py

分布式服务部署脚本，用于初始化和管理基础设施。

```bash
# 使用方法
python scripts/deploy_services.py --init     # 初始化所有服务配置
python scripts/deploy_services.py --start    # 启动所有服务
python scripts/deploy_services.py --stop     # 停止所有服务
python scripts/deploy_services.py --status   # 检查服务状态
```

**功能**:
- `init_feast()` - 初始化Feast特征仓库
- `init_celery()` - 初始化Celery配置
- `init_ray()` - 初始化Ray配置
- `start_services()` - 启动Ray、提示Celery启动命令
- `stop_services()` - 关闭Ray
- `check_status()` - 检查Redis/Feast/Celery/Ray/Milvus/Neo4j状态

### 21.2 import_factors.py

因子导入脚本，将因子库导入到向量数据库。

```bash
python scripts/import_factors.py --source alpha158 --limit 100
python scripts/import_factors.py --source all
```

---

## 22. 配置文件

### 22.1 celeryconfig.py

```python
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
```

### 22.2 ray_config.py

```python
"""Ray 分布式计算配置"""
import ray

RAY_CONFIG = {
    'num_cpus': 4,
    'num_gpus': 0,
    'memory': 4 * 1024 * 1024 * 1024,  # 4GB
    'object_store_memory': 1 * 1024 * 1024 * 1024,  # 1GB
}

def init_ray_cluster(local: bool = True) -> None
def shutdown_ray() -> None

@ray.remote
def compute_factor_remote(factor_code: str, data) -> Any

@ray.remote
def batch_evaluate_factors(factor_codes: list, data) -> List
```

---

## 23. 运行脚本

### 23.1 run_factor_mining.py

因子挖掘主入口脚本。

```bash
# 快速测试 (1轮LLM, 少量因子)
python run_factor_mining.py --mode quick

# 标准运行 (3轮LLM, 完整流程)
python run_factor_mining.py --mode standard

# 深度挖掘 (5轮LLM, 大规模GP)
python run_factor_mining.py --mode deep

# 自定义参数
python run_factor_mining.py --llm-rounds 3 --gp-generations 10 --batch-size 5
```

**RunConfig参数**:

| 参数 | quick | standard | deep | 说明 |
|------|-------|----------|------|------|
| llm_rounds | 1 | 3 | 5 | LLM探索轮数 |
| llm_batch_size | 2 | 3 | 5 | 每轮生成因子数 |
| gp_population | 10 | 30 | 50 | GP种群大小 |
| gp_generations | 2 | 5 | 10 | GP进化代数 |
| seed_threshold_ic | 0.003 | 0.005 | 0.008 | 种子因子IC阈值 |

**环境变量**:
- `DASHSCOPE_API_KEY` - 阿里云DashScope API密钥
- `OPENAI_API_KEY` - OpenAI API密钥 (可选)

### 23.2 run_factor_selection.py

因子筛选脚本。

```bash
python run_factor_selection.py --input factors.parquet --output selected.parquet
python run_factor_selection.py --top-n 50 --max-corr 0.6
```

---

## 24. 依赖列表

### requirements.txt

```
# 核心
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# LLM & Agent
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10
openai>=1.0.0
dashscope>=1.14.0          # 阿里云通义千问

# 向量数据库 (Milvus)
pymilvus>=2.3.0

# 图数据库 (Neo4j)
neo4j>=5.0.0

# 机器学习
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0

# 遗传规划
gplearn>=0.4.2

# Qlib
pyqlib>=0.9.0

# 可视化
matplotlib>=3.5.0
seaborn>=0.11.0

# 可解释性
shap>=0.41.0

# Web服务 (可选)
fastapi>=0.100.0
uvicorn>=0.22.0

# 工具
pydantic>=2.0.0
python-dotenv>=1.0.0
tqdm>=4.64.0
loguru>=0.7.0

# 分布式 (可选)
celery>=5.3.0
redis>=4.5.0
ray>=2.5.0
feast>=0.30.0
```

---

## 25. 模块依赖关系图

```
                    ┌─────────────────────┐
                    │   alpha_agent       │
                    │   (顶层包)          │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   agents/     │    │   analysis/   │    │   modeling/   │
│ MiningAgent   │───▶│ RiskAnalyzer  │    │ QlibModelZoo  │
│ AnalysisAgent │    │ KnowledgeGraph│    │ Ensemble      │
│ Orchestrator  │    └───────────────┘    └───────────────┘
│ Reflexion     │            │                    │
└───────┬───────┘            │                    │
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   core/       │    │   graph/      │    │   selection/  │
│ LLMGenerator  │    │ GraphStore    │    │ FactorSelector│
│ Sandbox       │    │ GraphRetriever│    │ FactorWrapper │
│ Evaluator     │    └───────────────┘    └───────────────┘
│ BaseAgent     │            │                    │
└───────┬───────┘            │                    │
        │                    ▼                    │
        │           ┌───────────────┐             │
        │           │   raptor/     │◀────────────┘
        │           │ RaptorTree    │
        │           │ RaptorRetriever│
        │           └───────────────┘
        │                    │
        ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   memory/     │    │   prompt/     │    │   factors/    │
│ MilvusStore   │    │ PromptComposer│    │ Alpha158      │
│ FactorMemory  │    │ Templates     │    │ WorldQuant101 │
│ RAG           │    └───────────────┘    │ GTJA191       │
└───────────────┘                         └───────────────┘
        │                                         │
        │                                         │
        ▼                                         ▼
┌───────────────┐                        ┌───────────────┐
│   evolution/  │                        │   mining/     │
│ HybridEngine  │                        │ GPEngine      │
│ EvolutionConfig│                       │ Backtest      │
└───────────────┘                        └───────────────┘
        │                                         │
        │                 ┌───────────────┐       │
        │                 │  evaluation/  │◀──────┘
        │                 │ FactorEvaluator│
        │                 │ Metrics       │
        │                 └───────────────┘
        │                         │
        └─────────────────┬───────┴────────────────┐
                          │                        │
                          ▼                        ▼
                 ┌───────────────┐       ┌───────────────┐
                 │   infra/      │       │   tasks/      │
                 │ FeatureStore  │       │ Celery任务    │
                 │ RayExecutor   │       │ FactorCache   │
                 │ TaskManager   │       └───────────────┘
                 └───────────────┘
                          │
                          ▼
                 ┌───────────────┐
                 │   config/     │
                 │ Settings      │
                 │ (所有配置)    │
                 └───────────────┘
                          │
                          ▼
                 ┌───────────────┐
                 │   schema/     │
                 │ DataSchema    │
                 │ (数据定义)    │
                 └───────────────┘
```

---

## 使用示例

### 基础使用

```python
from alpha_agent import MiningAgent, qlib_config
import pandas as pd

# 1. 准备数据
df = pd.read_csv("stock_data.csv")
target = df["future_return"]

# 2. 初始化Agent
agent = MiningAgent(api_key="your-api-key")
agent.setup(df, target)

# 3. 对话模式
response = agent.chat("生成一个20日动量因子")
print(response)

# 4. 批量挖掘
result = agent.run("挖掘5个有效的量价因子", max_iterations=10)
for factor in result.factors:
    print(f"{factor.name}: IC={factor.ic:.4f}")
```

### 高级使用 - 多Agent协作

```python
from alpha_agent import MiningAgent, AnalysisAgent, Orchestrator

# 初始化
orchestrator = Orchestrator()
orchestrator.register("mining", MiningAgent())
orchestrator.register("analysis", AnalysisAgent())

# 创建任务流水线
tasks = [
    orchestrator.create_task("mining", "挖掘动量因子"),
    orchestrator.create_task("analysis", "分析因子风险"),
]

# 运行
results = orchestrator.run_pipeline(tasks)
```

### 进化式因子发现

```python
from alpha_agent.evolution import HybridEvolutionEngine, HybridConfig

config = HybridConfig(
    llm_rounds=5,
    gp_generations=20,
    seed_threshold_ic=0.02,
)

engine = HybridEvolutionEngine(config=config)
best_factors = engine.evolve()

for factor in best_factors[:5]:
    print(f"{factor.name}: IC={factor.ic:.4f}")
```

### 分布式因子评估

```python
from alpha_agent.infra import distributed_backtest, RayExecutor

# 使用Ray并行评估因子
factors = [
    {"id": "f1", "code": "df['close'].pct_change(5)"},
    {"id": "f2", "code": "df['volume'].rolling(10).mean()"},
    # ... 更多因子
]

results = distributed_backtest(
    factors=factors,
    data=df,
    target=target,
    n_workers=4,
)

for r in results:
    print(f"{r['id']}: IC={r['ic']:.4f}")
```

### RAPTOR层次检索

```python
from alpha_agent.raptor import RaptorTree, RaptorRetriever, RetrievalConfig

# 构建因子树
tree = RaptorTree("factor_knowledge")
tree.load("data/raptor_tree.json")

# 配置检索器
config = RetrievalConfig(
    strategy="hybrid",
    top_k=10,
    include_ancestors=True,
)

retriever = RaptorRetriever(tree, config=config)

# 检索相关因子
result = retriever.retrieve("短期反转因子")
print(result.context)  # 生成的LLM上下文
```

### Docker部署

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f alpha-agent

# 检查服务状态
docker-compose ps

# 停止服务
docker-compose down
```

### Celery任务调度

```python
from alpha_agent.tasks.factor import batch_evaluate_factors, evaluate_factor_ic

# 提交异步任务
task = batch_evaluate_factors.delay(
    factors=[{"id": "f1", "code": "..."}],
    data_json=data.to_json(orient='split'),
    target_json=target.to_json(orient='split'),
    data_hash="abc123",
)

# 获取结果
results = task.get(timeout=300)
```

---

## 附录: 快速参考

### 常用导入

```python
# 核心
from alpha_agent import MiningAgent, AnalysisAgent, Orchestrator
from alpha_agent import LLMGenerator, Sandbox, FactorEvaluator

# 配置
from alpha_agent.config import qlib_config, llm_config, factor_config

# 因子库
from alpha_agent.factors import ALPHA158_FACTORS, WORLDQUANT_101_FACTORS

# 评估
from alpha_agent.evaluation import FactorEvaluator, compute_all_metrics

# 分布式
from alpha_agent.infra import RayExecutor, distributed_backtest, FeatureStore

# RAPTOR
from alpha_agent.raptor import RaptorTree, RaptorRetriever
```

### 环境变量

| 变量 | 说明 | 必需 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 阿里云通义千问API密钥 | 是 |
| `OPENAI_API_KEY` | OpenAI API密钥 | 否 |
| `MILVUS_HOST` | Milvus向量数据库地址 | 否 |
| `NEO4J_URI` | Neo4j图数据库地址 | 否 |
| `REDIS_HOST` | Redis缓存地址 | 否 |

### 命令行工具

```bash
# 因子挖掘
python run_factor_mining.py --mode standard

# 服务部署
python scripts/deploy_services.py --init
python scripts/deploy_services.py --status

# Docker
docker-compose up -d
```

---

## 26. docs/ - 设计文档

`docs/` 目录包含详细的系统设计文档，供开发者深入理解系统架构。

| 文档 | 大小 | 内容 |
|------|------|------|
| **SYSTEM_FLOW.md** | 53KB | 完整系统流程、数据流、组件交互 |
| **RAPTOR_DESIGN.md** | 20KB | RAPTOR递归抽象检索设计 |
| **GRAPHRAG_DESIGN.md** | 20KB | GraphRAG知识图谱设计 |
| **EVOLUTION_DESIGN.md** | 20KB | 混合进化引擎(LLM+GP)设计 |
| **PIPELINE.md** | 19KB | 因子挖掘Pipeline流程 |
| **FACTOR_LIBRARY.md** | 17KB | 因子库设计与管理 |
| **OPTIMIZATION_DESIGN.md** | 12KB | 优化模块总体设计 |

```bash
# 查看设计文档
cat docs/SYSTEM_FLOW.md      # 系统整体流程
cat docs/RAPTOR_DESIGN.md    # RAPTOR层次检索设计
cat docs/EVOLUTION_DESIGN.md # 进化引擎设计
```

---

## 27. feature_repo/ - Feast特征仓库

Feast特征存储配置，用于管理因子特征的在线/离线存储。

### 27.1 feature_store.yaml

```yaml
project: alpha_agent
registry: data/registry.db
provider: local
online_store:
  type: redis
  connection_string: localhost:6379
offline_store:
  type: file
entity_key_serialization_version: 2
```

### 27.2 features.py

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String

# 股票实体
stock = Entity(
    name="stock",
    join_keys=["symbol"],
    value_type=String,
    description="股票代码",
)

# 因子特征视图
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
    source=FileSource(path="data/factors.parquet", timestamp_field="date"),
)
```

### 27.3 Feast命令

```bash
# 初始化特征仓库
cd feature_repo
feast apply

# 物化特征到在线存储
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)

# 查看注册的特征
feast feature-views list
```

---

## 28. modeling/config.py - 模型配置

```python
# Qlib配置
QLIB_CONFIG = {
    "provider_uri": "~/.qlib/qlib_data/cn_data",
    "region": "cn",
}

# 训练配置
TRAIN_CONFIG = {
    "train_period": ("2008-01-01", "2014-12-31"),
    "valid_period": ("2015-01-01", "2016-12-31"),
    "test_period": ("2017-01-01", "2020-08-01"),
    "instruments": "csi300",
}

# GPU配置
GPU_CONFIG = {
    "device": 0,      # GPU设备ID
    "use_gpu": True,  # 是否使用GPU
}
```

---

## 29. tests/ - 测试模块

测试目录结构：

```
tests/
├── __init__.py
├── test_core/
│   ├── test_sandbox.py      # 沙箱测试
│   ├── test_evaluator.py    # 评估器测试
│   └── test_llm.py          # LLM测试
├── test_agents/
│   └── test_mining_agent.py # Agent测试
└── conftest.py              # Pytest配置
```

运行测试：
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_core/test_sandbox.py -v

# 覆盖率报告
pytest tests/ --cov=alpha_agent --cov-report=html
```

---

## 30. 目录结构说明

### 数据目录

| 目录 | 用途 |
|------|------|
| **data/** | 输入数据（股票数据、因子数据） |
| **output/** | 输出结果目录 |
| **output/factors/** | 生成的因子代码和评估结果 |
| **output/models/** | 训练好的模型 |
| **output/logs/** | 运行日志 |
| **mlruns/** | MLflow实验跟踪记录 |

### 主要入口文件

| 文件 | 用途 |
|------|------|
| **run_factor_mining.py** | 因子挖掘主入口 |
| **run_factor_selection.py** | 因子筛选入口 |
| **celeryconfig.py** | Celery配置 |
| **ray_config.py** | Ray分布式配置 |
| **requirements.txt** | 项目依赖 |
| **README.md** | 项目说明 |
| **PROGRESS.md** | 开发进度跟踪 |

---

## 31. 项目开发进度

> 来源: PROGRESS.md

### 阶段完成度

| 阶段 | 完成度 | 状态 |
|------|--------|------|
| 0. 环境底座 | 100% | ✅ |
| 1. Mining MVP | 100% | ✅ |
| 2. Memory/RAG | 100% | ✅ |
| 3. Modeling | 100% | ✅ |
| 4. Analysis | 100% | ✅ |
| 5. Multi-Agent | 90% | ✅ |
| 6. 生产化 | 70% | 🟡 |

**总体完成度: ~95%**

### QlibModelZoo 11模型

| 类别 | 模型 | 说明 |
|------|------|------|
| **Boosting** | lgb, lgb_light, xgb, catboost | 主力模型 |
| **Linear** | linear (Ridge) | 基线对比 |
| **Neural** | mlp, lstm, gru, transformer, tabnet | 深度学习 |
| **Ensemble** | double_ensemble | 集成模型 |

### 因子库统计

| 因子库 | 来源 | 数量 |
|--------|------|------|
| 经典因子 | Barra/学术 | 25 |
| Alpha158 | Qlib | 50 |
| Alpha360 | Qlib | 27 |
| WorldQuant 101 | Kakushadze | 29 |
| **总计** | - | **131** |

---

---

## 32. 快速使用示例

### 一站式导入

```python
from alpha_agent import (
    # 配置
    evolution_config, ray_config, qlib_config,
    EVOLUTION_FAST, EVOLUTION_STANDARD,
    
    # Agent
    MiningAgent, AnalysisAgent, Orchestrator,
    
    # 进化引擎
    EvolutionaryEngine, Individual, EvolutionHistory,
    
    # 评估
    FactorEvaluator, BacktestMetrics,
    compute_all_metrics, compute_ic_metrics,
    
    # 因子库
    ALL_FACTORS, FactorLibrary, create_factor_library,
    ALPHA158_FACTORS, WORLDQUANT_101_FACTORS,
    
    # RAPTOR
    RaptorTree, RaptorRetriever, RaptorBuilder,
    
    # Prompt
    PromptComposer, SystemPrompts, TaskTemplates,
    
    # 筛选
    FactorSelector, select_factors, FactorWrapper,
)
```

### 快速挖掘因子

```python
# 初始化Agent
agent = MiningAgent(api_key="your-key")
agent.setup(df, target)

# 对话式挖掘
response = agent.chat("生成一个动量反转混合因子")

# 批量挖掘
result = agent.run("挖掘10个高IC因子")
```

### 进化式因子发现

```python
# 使用进化引擎
engine = EvolutionaryEngine(config=EVOLUTION_STANDARD)
best_factors = engine.evolve(
    df=stock_data,
    target=returns,
    initial_theme="量价因子"
)
```

### 因子评估

```python
evaluator = FactorEvaluator()
metrics = evaluator.evaluate(factor_values, returns)
print(f"IC: {metrics.ic:.4f}, Sharpe: {metrics.sharpe:.2f}")
```

---

*文档生成日期: 2024-12-08*  
*Alpha Agent v0.7.1*  
*总计: 32个章节, 覆盖全部模块*  
*导出API: 122个类/函数/常量*
