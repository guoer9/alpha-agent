"""
因子挖掘Agent - 基于LangChain构建

使用LangChain框架:
- Tools: 因子生成、执行、评估工具
- Memory: 对话记忆 + 向量记忆
- Agent: ReAct模式
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, TYPE_CHECKING
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# LangChain - 使用TYPE_CHECKING避免运行时依赖
if TYPE_CHECKING:
    from langchain.tools import Tool

# LangChain (支持新旧版本API)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import Tool, StructuredTool
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    # 新版本API (langchain >= 1.0)
    try:
        from langgraph.prebuilt import create_react_agent
        from langgraph.graph import StateGraph
        LANGGRAPH_AVAILABLE = True
    except ImportError:
        LANGGRAPH_AVAILABLE = False
    # 旧版本兼容
    try:
        from langchain.memory import ConversationBufferWindowMemory
    except ImportError:
        ConversationBufferWindowMemory = None
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LANGGRAPH_AVAILABLE = False
    Tool = None
    logger.warning("langchain未安装: pip install langchain langchain-openai langgraph")

from ..core import LLMGenerator, Sandbox, FactorEvaluator
from ..core.base import BaseAgent, AgentResult, FactorResult, FactorStatus, AgentStatus
from ..memory import FactorMemory, ExperimentLogger
from ..config import llm_config


# Agent提示模板
MINING_AGENT_PROMPT = """你是一个专业的量化因子研究员，负责挖掘有效的alpha因子。

你有以下工具可用:
{tools}

使用以下格式:
Question: 你需要回答的问题
Thought: 思考应该做什么
Action: 使用的工具名称，必须是 [{tool_names}] 之一
Action Input: 工具的输入
Observation: 工具返回的结果
... (可以重复多次 Thought/Action/Action Input/Observation)
Thought: 我现在知道最终答案了
Final Answer: 对问题的最终回答

开始!

Question: {input}
Thought: {agent_scratchpad}
"""


class MiningAgent(BaseAgent):
    """因子挖掘Agent"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        enable_memory: bool = True,
    ):
        super().__init__(name="MiningAgent")
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("请安装langchain: pip install langchain langchain-openai")
        
        self.api_key = api_key or llm_config.api_key
        self.model = model or llm_config.model
        self.enable_memory = enable_memory
        
        # 核心组件
        self.llm = None
        self.generator = None
        self.sandbox = Sandbox()
        self.evaluator = FactorEvaluator()
        
        # 记忆
        self.memory = ConversationBufferWindowMemory(
            k=10, memory_key="chat_history", return_messages=True
        )
        self.factor_memory = FactorMemory() if enable_memory else None
        self.experiment_logger = None
        
        # 数据
        self.df: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None
        
        # Agent
        self.agent_executor = None
    
    def setup(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        experiment_name: str = None,
    ):
        """设置数据和初始化"""
        self.df = df
        self.target = target
        self.experiment_logger = ExperimentLogger(experiment_name)
        
        # 初始化LLM
        llm_kwargs = {
            'model': self.model,
            'temperature': 0.7,
        }
        if self.api_key:
            llm_kwargs['api_key'] = self.api_key
        if llm_config.base_url:
            llm_kwargs['base_url'] = llm_config.base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        self.generator = LLMGenerator(api_key=self.api_key, model=self.model)
        
        # 创建工具
        tools = self._create_tools()
        
        # 创建Agent
        prompt = PromptTemplate.from_template(MINING_AGENT_PROMPT)
        agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )
        
        self.log(f"Agent已初始化，数据形状: {df.shape}")
    
    def _create_tools(self) -> List[Tool]:
        """创建Agent工具"""
        tools = [
            Tool(
                name="generate_factor",
                func=self._tool_generate,
                description="生成因子代码。输入是因子类型描述，如'动量因子'、'波动率因子'等。"
            ),
            Tool(
                name="execute_factor",
                func=self._tool_execute,
                description="执行因子代码并返回结果。输入是Python代码字符串。"
            ),
            Tool(
                name="evaluate_factor",
                func=self._tool_evaluate,
                description="评估因子性能。输入是因子名称(之前执行过的)。"
            ),
            Tool(
                name="search_similar",
                func=self._tool_search_similar,
                description="搜索类似的已有因子。输入是因子描述。"
            ),
            Tool(
                name="improve_factor",
                func=self._tool_improve,
                description="改进现有因子。输入格式: '因子代码|||改进方向'"
            ),
        ]
        return tools
    
    def _tool_generate(self, factor_type: str) -> str:
        """生成因子工具"""
        try:
            instruction = f"请生成一个{factor_type}，要求有经济学含义且可能有效。"
            code = self.generator.generate(instruction)
            return f"生成的代码:\n```python\n{code}\n```"
        except Exception as e:
            return f"生成失败: {e}"
    
    def _tool_execute(self, code: str) -> str:
        """执行因子工具"""
        if self.df is None:
            return "错误: 未设置数据，请先调用setup()"
        
        try:
            factor, error = self.sandbox.execute(code, self.df)
            if error:
                return f"执行失败: {error}"
            
            # 保存最近的因子
            self._last_factor = factor
            self._last_code = code
            
            return f"执行成功! 因子形状: {factor.shape}, 非空值: {factor.notna().sum()}"
        except Exception as e:
            return f"执行异常: {e}"
    
    def _tool_evaluate(self, factor_name: str) -> str:
        """评估因子工具"""
        if not hasattr(self, '_last_factor') or self._last_factor is None:
            return "错误: 没有可评估的因子，请先执行因子代码"
        
        if self.target is None:
            return "错误: 未设置目标变量"
        
        try:
            result = self.evaluator.evaluate(self._last_factor, self.target)
            
            # 记录到日志
            if self.experiment_logger:
                self.experiment_logger.log_factor(
                    name=factor_name,
                    code=self._last_code,
                    ic=result.ic,
                    icir=result.icir,
                    status=result.status.value,
                )
            
            # 保存到记忆
            if self.factor_memory and result.status in [FactorStatus.EXCELLENT, FactorStatus.GOOD]:
                self.factor_memory.save_factor(
                    name=factor_name,
                    code=self._last_code,
                    description=f"IC={result.ic:.4f}",
                    ic=result.ic,
                    icir=result.icir,
                    status=result.status.value,
                )
            
            return f"""
评估结果:
- IC: {result.ic:.4f}
- ICIR: {result.icir:.4f}
- 状态: {result.status.value}
- 建议: {result.recommendation}
"""
        except Exception as e:
            return f"评估失败: {e}"
    
    def _tool_search_similar(self, query: str) -> str:
        """搜索相似因子"""
        if self.factor_memory is None:
            return "记忆系统未启用"
        
        try:
            results = self.factor_memory.search_similar(query, top_k=3)
            if not results:
                return "未找到相似因子"
            
            output = "相似因子:\n"
            for r in results:
                output += f"- {r['name']}: IC={r['ic']:.4f}, 相似度={r['score']:.2f}\n"
            return output
        except Exception as e:
            return f"搜索失败: {e}"
    
    def _tool_improve(self, input_str: str) -> str:
        """改进因子"""
        try:
            parts = input_str.split("|||")
            code = parts[0].strip()
            direction = parts[1].strip() if len(parts) > 1 else ""
            
            improved = self.generator.improve_factor(code, 0.0, direction)
            return f"改进后的代码:\n```python\n{improved}\n```"
        except Exception as e:
            return f"改进失败: {e}"
    
    def run(self, task: str) -> AgentResult:
        """运行Agent"""
        if self.agent_executor is None:
            raise ValueError("Agent未初始化，请先调用setup()")
        
        self.status = AgentStatus.RUNNING
        self.result = AgentResult(agent_name=self.name)
        self.log(f"开始任务: {task}")
        
        try:
            response = self.agent_executor.invoke({"input": task})
            self.log(f"任务完成")
            self.status = AgentStatus.SUCCESS
            
        except Exception as e:
            self.log(f"任务失败: {e}")
            self.status = AgentStatus.FAILED
            self.result.error = str(e)
        
        # 保存实验
        if self.experiment_logger:
            self.experiment_logger.save()
        
        return self.result
    
    def step(self, instruction: str) -> FactorResult:
        """执行单步: 生成→执行→评估"""
        # 生成
        code = self.generator.generate(instruction)
        
        # 执行
        factor, error = self.sandbox.execute(code, self.df)
        if error:
            return FactorResult(name="failed", code=code, status=FactorStatus.FAILED, error=error)
        
        # 评估
        eval_result = self.evaluator.evaluate(factor, self.target)
        
        result = FactorResult(
            name=f"factor_{len(self.result.factors)+1}",
            code=code,
            ic=eval_result.ic,
            icir=eval_result.icir,
            status=eval_result.status,
            values=factor,
        )
        
        self.result.add_factor(result)
        return result
    
    def chat(self, message: str) -> str:
        """对话模式"""
        if self.agent_executor is None:
            return "Agent未初始化，请先调用setup()"
        
        response = self.agent_executor.invoke({"input": message})
        return response.get("output", "")
