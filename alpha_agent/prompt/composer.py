"""
Prompt组装器 - 分层组装LLM提示词

架构:
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
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

from .templates import (
    SystemPrompts,
    SchemaTemplates,
    RAGTemplates,
    ReflectionTemplates,
    TaskTemplates,
)


class TaskType(Enum):
    """任务类型"""
    GENERATE_NEW = "generate_new"
    IMPROVE_FACTOR = "improve_factor"
    MUTATE_FACTOR = "mutate_factor"
    CROSSOVER_FACTORS = "crossover_factors"
    DIAGNOSE_FACTOR = "diagnose_factor"


class RoleType(Enum):
    """角色类型"""
    ALPHA_MINER = "alpha_miner"
    FACTOR_EVALUATOR = "factor_evaluator"
    FACTOR_IMPROVER = "factor_improver"


@dataclass
class SchemaContext:
    """数据Schema上下文"""
    fields: List[Dict] = field(default_factory=list)  # 字段列表
    custom_operators: List[str] = field(default_factory=list)  # 自定义算子
    forbidden_operations: List[str] = field(default_factory=list)  # 禁止操作


@dataclass
class RAGContext:
    """RAG检索上下文"""
    similar_factors: List[Dict] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    market_regime: str = ""


@dataclass
class ReflectionContext:
    """反思/反馈上下文"""
    successes: List[Dict] = field(default_factory=list)
    failures: List[Dict] = field(default_factory=list)
    backtest_summary: Dict = field(default_factory=dict)


@dataclass
class TaskContext:
    """任务上下文"""
    task_type: TaskType = TaskType.GENERATE_NEW
    parameters: Dict = field(default_factory=dict)


@dataclass
class ComposedPrompt:
    """组装后的Prompt"""
    system: str
    user: str
    
    # 元信息
    token_estimate: int = 0
    schema_included: bool = False
    rag_included: bool = False
    reflection_included: bool = False
    
    def to_messages(self) -> List[Dict]:
        """转换为OpenAI消息格式"""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]
    
    def to_langchain_messages(self):
        """转换为LangChain消息格式"""
        from langchain_core.messages import SystemMessage, HumanMessage
        return [
            SystemMessage(content=self.system),
            HumanMessage(content=self.user),
        ]


class PromptComposer:
    """Prompt组装器"""
    
    def __init__(
        self,
        data_schema = None,      # DataSchema实例
        rag_retriever = None,    # RAG检索器
        graph_retriever = None,  # GraphRAG检索器
    ):
        self.data_schema = data_schema
        self.rag_retriever = rag_retriever
        self.graph_retriever = graph_retriever
    
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
    ) -> ComposedPrompt:
        """
        组装完整Prompt
        
        Args:
            task_type: 任务类型
            task_params: 任务参数
            role: 角色类型
            schema_context: 数据Schema上下文
            rag_context: RAG检索上下文
            reflection_context: 反思上下文
            include_schema: 是否包含Schema
            include_rag: 是否包含RAG
            include_reflection: 是否包含反思
            max_rag_examples: RAG示例最大数量
            max_reflections: 反思最大数量
        """
        task_params = task_params or {}
        
        # 1. 组装System Prompt
        system_prompt = self._compose_system(role)
        
        # 2. 组装User Prompt的各部分
        user_parts = []
        
        # 2.1 Schema Context (硬约束)
        if include_schema:
            schema_part = self._compose_schema(schema_context)
            if schema_part:
                user_parts.append(schema_part)
        
        # 2.2 RAG Context (软引导)
        if include_rag:
            rag_part = self._compose_rag(rag_context, max_rag_examples)
            if rag_part:
                user_parts.append(rag_part)
        
        # 2.3 Reflection Context (进化压力)
        if include_reflection:
            reflection_part = self._compose_reflection(reflection_context, max_reflections)
            if reflection_part:
                user_parts.append(reflection_part)
        
        # 2.4 Task Instruction
        task_part = self._compose_task(task_type, task_params)
        user_parts.append(task_part)
        
        # 组合User Prompt
        user_prompt = "\n\n".join(user_parts)
        
        # 估算token数
        token_estimate = self._estimate_tokens(system_prompt + user_prompt)
        
        return ComposedPrompt(
            system=system_prompt,
            user=user_prompt,
            token_estimate=token_estimate,
            schema_included=include_schema and schema_context is not None,
            rag_included=include_rag and rag_context is not None,
            reflection_included=include_reflection and reflection_context is not None,
        )
    
    # ============================================================
    # 各层组装方法
    # ============================================================
    
    def _compose_system(self, role: RoleType) -> str:
        """组装System Prompt"""
        if role == RoleType.ALPHA_MINER:
            return SystemPrompts.ALPHA_MINER
        elif role == RoleType.FACTOR_EVALUATOR:
            return SystemPrompts.FACTOR_EVALUATOR
        elif role == RoleType.FACTOR_IMPROVER:
            return SystemPrompts.FACTOR_IMPROVER
        else:
            return SystemPrompts.ALPHA_MINER
    
    def _compose_schema(self, schema_context: SchemaContext = None) -> str:
        """组装Schema Context"""
        parts = [SchemaTemplates.HEADER]
        
        # 使用传入的schema或默认schema
        if schema_context and schema_context.fields:
            parts.append(self._format_fields(schema_context.fields))
        elif self.data_schema:
            parts.append(self.data_schema.to_llm_prompt())
        else:
            parts.append("(使用标准A股日频数据: open/high/low/close/volume/amount/turnover)")
        
        # 算子列表
        parts.append(SchemaTemplates.OPERATORS)
        
        # 禁止操作
        if schema_context and schema_context.forbidden_operations:
            parts.append("\n### 禁止操作")
            for op in schema_context.forbidden_operations:
                parts.append(f"- ❌ {op}")
        
        return "\n".join(parts)
    
    def _compose_rag(
        self, 
        rag_context: RAGContext = None, 
        max_examples: int = 3
    ) -> str:
        """组装RAG Context"""
        if not rag_context and not self.rag_retriever:
            return ""
        
        parts = [RAGTemplates.HEADER]
        
        # 相似因子
        factors = rag_context.similar_factors if rag_context else []
        for factor in factors[:max_examples]:
            parts.append(RAGTemplates.FACTOR_TEMPLATE.format(
                name=factor.get('name', 'Unknown'),
                ic=factor.get('ic', 0),
                category=factor.get('category', 'unknown'),
                source=factor.get('source', 'unknown'),
                logic=factor.get('logic', factor.get('description', '')),
                code=factor.get('code', '')[:500],
            ))
        
        # 市场状态
        if rag_context and rag_context.market_regime:
            parts.append(f"\n### 当前市场状态\n{rag_context.market_regime}")
        
        # 相关概念
        if rag_context and rag_context.related_concepts:
            parts.append("\n### 相关策略概念")
            for concept in rag_context.related_concepts[:5]:
                parts.append(f"- {concept}")
        
        return "\n".join(parts) if len(parts) > 1 else ""
    
    def _compose_reflection(
        self, 
        reflection_context: ReflectionContext = None,
        max_items: int = 5
    ) -> str:
        """组装Reflection Context"""
        if not reflection_context:
            return ""
        
        parts = [ReflectionTemplates.HEADER]
        has_content = False
        
        # 成功经验
        for success in reflection_context.successes[:max_items // 2]:
            parts.append(ReflectionTemplates.SUCCESS_TEMPLATE.format(
                factor_name=success.get('factor_name', 'Unknown'),
                ic=success.get('ic', 0),
                icir=success.get('icir', 0),
                reason=success.get('reason', ''),
                takeaway=success.get('takeaway', ''),
            ))
            has_content = True
        
        # 失败教训
        for failure in reflection_context.failures[:max_items // 2]:
            parts.append(ReflectionTemplates.FAILURE_TEMPLATE.format(
                factor_name=failure.get('factor_name', 'Unknown'),
                problem=failure.get('problem', ''),
                diagnosis=failure.get('diagnosis', ''),
                suggestion=failure.get('suggestion', ''),
            ))
            has_content = True
        
        # 回测摘要
        if reflection_context.backtest_summary:
            summary = reflection_context.backtest_summary
            parts.append(ReflectionTemplates.BACKTEST_SUMMARY.format(
                total_tested=summary.get('total_tested', 0),
                pass_rate=summary.get('pass_rate', 0),
                best_ic=summary.get('best_ic', 0),
                main_failure_reason=summary.get('main_failure_reason', 'N/A'),
            ))
            has_content = True
        
        return "\n".join(parts) if has_content else ""
    
    def _compose_task(self, task_type: TaskType, params: Dict) -> str:
        """组装Task Instruction"""
        if task_type == TaskType.GENERATE_NEW:
            return TaskTemplates.GENERATE_NEW.format(
                theme=params.get('theme', '量价动量'),
                target_ic=params.get('target_ic', 0.02),
                max_turnover=params.get('max_turnover', '50%'),
            )
        
        elif task_type == TaskType.IMPROVE_FACTOR:
            return TaskTemplates.IMPROVE_FACTOR.format(
                original_code=params.get('original_code', ''),
                current_ic=params.get('current_ic', 0),
                current_icir=params.get('current_icir', 0),
                current_turnover=params.get('current_turnover', 0),
                diagnosis=params.get('diagnosis', ''),
                improvement_direction=params.get('improvement_direction', ''),
            )
        
        elif task_type == TaskType.MUTATE_FACTOR:
            return TaskTemplates.MUTATE_FACTOR.format(
                original_code=params.get('original_code', ''),
                mutation_type=params.get('mutation_type', '参数调整'),
                mutation_instruction=params.get('mutation_instruction', '调整时间窗口'),
            )
        
        elif task_type == TaskType.CROSSOVER_FACTORS:
            return TaskTemplates.CROSSOVER_FACTORS.format(
                code_a=params.get('code_a', ''),
                ic_a=params.get('ic_a', 0),
                code_b=params.get('code_b', ''),
                ic_b=params.get('ic_b', 0),
                crossover_strategy=params.get('crossover_strategy', '特征融合'),
            )
        
        elif task_type == TaskType.DIAGNOSE_FACTOR:
            return TaskTemplates.DIAGNOSE_FACTOR.format(
                code=params.get('code', ''),
                ic=params.get('ic', 0),
                icir=params.get('icir', 0),
                turnover=params.get('turnover', 0),
                ic_series=params.get('ic_series', ''),
            )
        
        else:
            return f"## 任务\n{params.get('instruction', '请生成一个Alpha因子')}"
    
    # ============================================================
    # 辅助方法
    # ============================================================
    
    def _format_fields(self, fields: List[Dict]) -> str:
        """格式化字段列表"""
        lines = ["| 字段 | 类型 | 说明 |", "|------|------|------|"]
        for f in fields:
            lines.append(f"| {f.get('name', '')} | {f.get('type', '')} | {f.get('desc', '')} |")
        return "\n".join(lines)
    
    def _estimate_tokens(self, text: str) -> int:
        """估算token数 (粗略估计: 中文约1.5字/token, 英文约4字符/token)"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
    
    # ============================================================
    # 便捷方法
    # ============================================================
    
    def for_generation(
        self,
        theme: str = "量价动量",
        target_ic: float = 0.02,
        rag_factors: List[Dict] = None,
        failures: List[Dict] = None,
    ) -> ComposedPrompt:
        """生成因子的便捷方法"""
        return self.compose(
            task_type=TaskType.GENERATE_NEW,
            task_params={'theme': theme, 'target_ic': target_ic},
            rag_context=RAGContext(similar_factors=rag_factors or []),
            reflection_context=ReflectionContext(failures=failures or []),
        )
    
    def for_improvement(
        self,
        original_code: str,
        current_ic: float,
        diagnosis: str,
        improvement_direction: str,
    ) -> ComposedPrompt:
        """改进因子的便捷方法"""
        return self.compose(
            task_type=TaskType.IMPROVE_FACTOR,
            task_params={
                'original_code': original_code,
                'current_ic': current_ic,
                'current_icir': 0,
                'current_turnover': 0,
                'diagnosis': diagnosis,
                'improvement_direction': improvement_direction,
            },
            role=RoleType.FACTOR_IMPROVER,
        )
    
    def for_mutation(
        self,
        original_code: str,
        mutation_type: str = "参数调整",
    ) -> ComposedPrompt:
        """变异因子的便捷方法"""
        return self.compose(
            task_type=TaskType.MUTATE_FACTOR,
            task_params={
                'original_code': original_code,
                'mutation_type': mutation_type,
                'mutation_instruction': f"对因子进行{mutation_type}",
            },
        )
