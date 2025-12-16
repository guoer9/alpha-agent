"""
Alpha Agent - LLM驱动的因子挖掘系统

技术栈:
- LangChain: 智能体框架
- Milvus: 向量数据库 (因子记忆)
- Neo4j: 图数据库 (风险知识图谱)
- Qlib: 回测框架
- Docker: 容器部署

模块结构:
├── config/       # 配置管理
├── core/         # 核心组件 (LLM, 沙箱, 评估)
├── mining/       # 因子挖掘 (GP搜索, 回测)
├── modeling/     # 建模 (特征选择, 模型Zoo, 集成)
├── memory/       # 记忆系统 (Milvus向量库, RAG)
├── analysis/     # 分析 (Neo4j风险图谱, 归因)
├── agents/       # 多Agent协作
├── evaluation/   # 因子评估 (IC/ICIR/夏普/回撤)
├── evolution/    # 进化引擎 (LLM+GP混合)
├── factors/      # 因子库 (300+经典因子)
├── graph/        # GraphRAG知识图谱
├── raptor/       # RAPTOR层次检索
├── prompt/       # Prompt组装系统
├── schema/       # 数据字典
├── selection/    # 因子筛选
├── infra/        # 基础设施 (Feast, Celery, Ray)
└── docker/       # Docker部署

使用示例:
```python
from alpha_agent import MiningAgent

# 初始化Agent
agent = MiningAgent(api_key="your-key")
agent.setup(df, target)

# 对话模式
response = agent.chat("生成一个动量因子")

# 批量挖掘
result = agent.run("挖掘10个有效因子")
```
"""

__version__ = '0.7.1'

# ==================== 配置 ====================
from .config.settings import (
    # 配置类
    QlibConfig, LLMConfig, FactorConfig, ModelConfig,
    SandboxConfig, GPConfig, VectorDBConfig,
    SelectionConfig, CacheConfig, CeleryConfig,
    EvolutionConfig, RayConfig, GPUConfig, TrainPeriodConfig,
    # 配置实例
    qlib_config, llm_config, factor_config, model_config,
    sandbox_config, gp_config, vector_db_config,
    selection_config, cache_config, celery_config,
    evolution_config, ray_config, gpu_config, train_period_config,
    # 预设配置
    EVOLUTION_FAST, EVOLUTION_STANDARD, EVOLUTION_THOROUGH,
    # 路径
    BASE_DIR, DATA_DIR, OUTPUT_DIR, FACTORS_DIR, MODELS_DIR, LOGS_DIR,
)

# ==================== 核心组件 ====================
from .core import (
    BaseAgent, AgentResult, FactorResult,
    LLMGenerator,
    Sandbox, execute_code,
    FactorEvaluator, evaluate_factor,
)

# ==================== 因子挖掘 ====================
from .mining import GPEngine, run_backtest

# ==================== 记忆系统 ====================
from .memory import MilvusStore, FactorMemory, ExperimentLogger

# ==================== 风险分析 ====================
from .analysis import RiskKnowledgeGraph, RiskAnalyzer

# ==================== 多Agent ====================
from .agents import MiningAgent, Orchestrator, AnalysisAgent

# ==================== 评估系统 ====================
from .evaluation import (
    BacktestMetrics, ICMetrics, ReturnMetrics, RiskMetrics,
    compute_all_metrics, compute_ic_metrics, compute_return_metrics, compute_risk_metrics,
    EvaluatorConfig,
)

# ==================== 进化引擎 ====================
from .evolution import (
    Individual, EvolutionHistory, EvolutionaryEngine,
)

# ==================== 因子库 ====================
from .factors import (
    BARRA_FACTORS, TECHNICAL_FACTORS, FUNDAMENTAL_FACTORS, VOLUME_PRICE_FACTORS,
    ALL_CLASSIC_FACTORS, ALPHA158_FACTORS, ALPHA360_FACTORS, WORLDQUANT_101_FACTORS,
    GTJA191_FACTORS, ACADEMIC_PREMIA_FACTORS, ALL_FACTORS,
    ClassicFactor, FactorCategory, FactorLibrary, create_factor_library,
)

# ==================== GraphRAG ====================
from .graph import (
    NodeType, EdgeType,
    FactorNode, ReflectionNode, RegimeNode, ConceptNode,
    GraphEdge, GraphStore, GraphRetriever,
)

# ==================== RAPTOR ====================
from .raptor import (
    RaptorTree, TreeNode, RaptorRetriever, RetrievalConfig,
    RaptorBuilder, BuildConfig,
)

# ==================== Prompt系统 ====================
from .prompt import PromptComposer, SystemPrompts, TaskTemplates

# ==================== 数据字典 ====================
from .schema import DataSchema, FieldSchema, DataValidator, DataFrequency, DataType

# ==================== 因子筛选 ====================
from .selection import (
    # 筛选器
    FactorSelector, SelectionResult, select_factors, quick_filter, orthogonal_select,
    # 因子封装
    FactorWrapper, FactorMeta, load_factors, create_factor_wrapper,
    # 因子清洗
    FactorCleaner, CleaningStats, clean_factors, clean_factor_code,
    adapt_field_references, FIELD_ALIASES, DERIVED_FIELDS,
    # 数据预处理
    add_derived_fields, prepare_train_test_data, split_by_date, handle_missing_values,
)

# ==================== 建模 ====================
try:
    from .modeling import FeatureSelector, AlphaEnsemble
    MODELING_AVAILABLE = True
except ImportError:
    MODELING_AVAILABLE = False

# ==================== 基础设施 ====================
try:
    from .infra import FeatureStore, RayExecutor, distributed_backtest
    INFRA_AVAILABLE = True
except ImportError:
    INFRA_AVAILABLE = False


# ==================== 导出列表 ====================
__all__ = [
    # 版本
    '__version__',
    
    # ===== 配置 =====
    # 配置类
    'QlibConfig', 'LLMConfig', 'FactorConfig', 'ModelConfig',
    'SandboxConfig', 'GPConfig', 'VectorDBConfig',
    'SelectionConfig', 'CacheConfig', 'CeleryConfig',
    'EvolutionConfig', 'RayConfig', 'GPUConfig', 'TrainPeriodConfig',
    # 配置实例
    'qlib_config', 'llm_config', 'factor_config', 'model_config',
    'sandbox_config', 'gp_config', 'vector_db_config',
    'selection_config', 'cache_config', 'celery_config',
    'evolution_config', 'ray_config', 'gpu_config', 'train_period_config',
    # 预设配置
    'EVOLUTION_FAST', 'EVOLUTION_STANDARD', 'EVOLUTION_THOROUGH',
    # 路径
    'BASE_DIR', 'DATA_DIR', 'OUTPUT_DIR', 'FACTORS_DIR', 'MODELS_DIR', 'LOGS_DIR',
    
    # ===== 核心 =====
    'BaseAgent', 'AgentResult', 'FactorResult',
    'LLMGenerator', 'Sandbox', 'execute_code',
    'FactorEvaluator', 'evaluate_factor',
    
    # ===== 挖掘 =====
    'GPEngine', 'run_backtest',
    
    # ===== 记忆 =====
    'MilvusStore', 'FactorMemory', 'ExperimentLogger',
    
    # ===== 分析 =====
    'RiskKnowledgeGraph', 'RiskAnalyzer',
    
    # ===== Agent =====
    'MiningAgent', 'AnalysisAgent', 'Orchestrator',
    
    # ===== 评估 =====
    'BacktestMetrics', 'ICMetrics', 'ReturnMetrics', 'RiskMetrics',
    'compute_all_metrics', 'compute_ic_metrics', 'compute_return_metrics', 'compute_risk_metrics',
    'EvaluatorConfig',
    
    # ===== 进化 =====
    'Individual', 'EvolutionHistory', 'EvolutionaryEngine',
    
    # ===== 因子库 =====
    'BARRA_FACTORS', 'TECHNICAL_FACTORS', 'FUNDAMENTAL_FACTORS', 'VOLUME_PRICE_FACTORS',
    'ALL_CLASSIC_FACTORS', 'ALPHA158_FACTORS', 'ALPHA360_FACTORS', 'WORLDQUANT_101_FACTORS',
    'GTJA191_FACTORS', 'ACADEMIC_PREMIA_FACTORS', 'ALL_FACTORS',
    'ClassicFactor', 'FactorCategory', 'FactorLibrary', 'create_factor_library',
    
    # ===== GraphRAG =====
    'NodeType', 'EdgeType',
    'FactorNode', 'ReflectionNode', 'RegimeNode', 'ConceptNode',
    'GraphEdge', 'GraphStore', 'GraphRetriever',
    
    # ===== RAPTOR =====
    'RaptorTree', 'TreeNode', 'RaptorRetriever', 'RetrievalConfig',
    'RaptorBuilder', 'BuildConfig',
    
    # ===== Prompt =====
    'PromptComposer', 'SystemPrompts', 'TaskTemplates',
    
    # ===== 数据字典 =====
    'DataSchema', 'FieldSchema', 'DataValidator', 'DataFrequency', 'DataType',
    
    # ===== 筛选 =====
    'FactorSelector', 'SelectionResult', 'select_factors', 'quick_filter', 'orthogonal_select',
    'FactorWrapper', 'FactorMeta', 'load_factors', 'create_factor_wrapper',
    # 数据预处理
    'add_derived_fields', 'prepare_train_test_data', 'split_by_date', 'handle_missing_values',
    
    # ===== 建模 =====
    'FeatureSelector', 'AlphaEnsemble', 'MODELING_AVAILABLE',
    
    # ===== 基础设施 =====
    'FeatureStore', 'RayExecutor', 'distributed_backtest', 'INFRA_AVAILABLE',
]
