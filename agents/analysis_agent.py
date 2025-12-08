"""
Analysis Agent - 风险分析和归因Agent

负责:
1. 因子风险分析
2. 收益归因
3. 市场状态识别
4. 生成分析报告
"""

from __future__ import annotations
from typing import Dict, List, Optional, TYPE_CHECKING
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# LangChain - 使用TYPE_CHECKING避免运行时依赖
if TYPE_CHECKING:
    from langchain.tools import Tool

try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_openai import ChatOpenAI
    from langchain.tools import Tool
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Tool = None  # 占位符

from ..core.base import BaseAgent, AgentResult, AgentStatus
from ..analysis import RiskAnalyzer, RiskKnowledgeGraph
from ..analysis.attribution import brinson_attribution, factor_attribution
from ..analysis.market_regime import MarketRegimeDetector, detect_style_rotation
from ..config import llm_config


ANALYSIS_AGENT_PROMPT = """你是一个专业的量化风险分析师。

你有以下工具可用:
{tools}

使用以下格式:
Question: 分析任务
Thought: 思考分析步骤
Action: 工具名称
Action Input: 工具输入
Observation: 工具返回
... (重复)
Thought: 得出结论
Final Answer: 最终分析结果

任务: {input}
{agent_scratchpad}
"""


class AnalysisAgent(BaseAgent):
    """风险分析Agent"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
    ):
        super().__init__(name="AnalysisAgent")
        
        self.api_key = api_key or llm_config.api_key
        self.model = model or llm_config.model
        
        # 分析工具
        self.risk_analyzer = RiskAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.knowledge_graph = None
        
        # 数据
        self.returns: Optional[pd.Series] = None
        self.factor_returns: Optional[pd.DataFrame] = None
        
        # Agent
        self.llm = None
        self.agent_executor = None
    
    def setup(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame = None,
        knowledge_graph: RiskKnowledgeGraph = None,
    ):
        """设置数据"""
        self.returns = returns
        self.factor_returns = factor_returns
        self.knowledge_graph = knowledge_graph
        
        if LANGCHAIN_AVAILABLE:
            self._init_agent()
    
    def _init_agent(self):
        """初始化LangChain Agent"""
        llm_kwargs = {
            'model': self.model,
            'temperature': 0.3,
        }
        if self.api_key:
            llm_kwargs['api_key'] = self.api_key
        if llm_config.base_url:
            llm_kwargs['base_url'] = llm_config.base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        tools = self._create_tools()
        prompt = PromptTemplate.from_template(ANALYSIS_AGENT_PROMPT)
        agent = create_react_agent(self.llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )
    
    def _create_tools(self) -> List[Tool]:
        """创建分析工具"""
        return [
            Tool(
                name="analyze_risk",
                func=self._tool_risk_analysis,
                description="分析组合风险。输入: 无需输入"
            ),
            Tool(
                name="detect_market_regime",
                func=self._tool_market_regime,
                description="识别当前市场状态。输入: 无需输入"
            ),
            Tool(
                name="factor_attribution",
                func=self._tool_factor_attribution,
                description="因子归因分析。输入: 无需输入"
            ),
            Tool(
                name="style_analysis",
                func=self._tool_style_analysis,
                description="风格分析 (价值/成长)。输入: 无需输入"
            ),
        ]
    
    def _tool_risk_analysis(self, _: str = "") -> str:
        """风险分析工具"""
        if self.returns is None:
            return "错误: 未设置收益数据"
        
        report = self.risk_analyzer.analyze(self.returns, self.factor_returns)
        
        return f"""
风险分析结果:
- VaR(95%): {report.var_95:.4f}
- CVaR(95%): {report.cvar_95:.4f}
- 最大回撤: {report.max_drawdown:.2%}
- 风险等级: {report.risk_level}
- 市场Beta: {report.market_beta:.2f}
- 建议: {', '.join(report.recommendations)}
"""
    
    def _tool_market_regime(self, _: str = "") -> str:
        """市场状态识别"""
        if self.returns is None:
            return "错误: 未设置收益数据"
        
        result = self.regime_detector.detect(self.returns)
        
        probs_str = ", ".join([f"{k}:{v:.1%}" for k, v in result.state_probability.items()])
        
        return f"""
市场状态:
- 当前状态: {result.current_state.value}
- 状态概率: {probs_str}
"""
    
    def _tool_factor_attribution(self, _: str = "") -> str:
        """因子归因"""
        if self.returns is None or self.factor_returns is None:
            return "错误: 未设置收益或因子数据"
        
        result = factor_attribution(self.factor_returns, self.returns)
        
        if not result:
            return "归因失败: 数据不足"
        
        exposures = result.get('exposures', {})
        contributions = result.get('contributions', {})
        
        output = "因子归因:\n"
        for factor, exposure in exposures.items():
            contrib = contributions.get(factor, 0)
            output += f"- {factor}: 暴露={exposure:.3f}, 贡献={contrib:.4f}\n"
        
        output += f"- Alpha: {result.get('alpha', 0):.4f}\n"
        output += f"- R²: {result.get('r2', 0):.2%}\n"
        
        return output
    
    def _tool_style_analysis(self, _: str = "") -> str:
        """风格分析"""
        if self.factor_returns is None:
            return "错误: 未设置因子数据"
        
        # 假设有value和growth因子
        if 'value' in self.factor_returns.columns and 'growth' in self.factor_returns.columns:
            result = detect_style_rotation(
                self.factor_returns['value'],
                self.factor_returns['growth'],
            )
            return f"""
风格分析:
- 当前风格: {result['current_style']}
- 相对强度: {result['relative_strength']:.4f}
"""
        
        return "无法进行风格分析: 缺少value/growth因子"
    
    def run(self, task: str) -> AgentResult:
        """运行分析任务"""
        self.status = AgentStatus.RUNNING
        self.result = AgentResult(agent_name=self.name)
        
        if self.agent_executor:
            try:
                response = self.agent_executor.invoke({"input": task})
                self.status = AgentStatus.SUCCESS
            except Exception as e:
                self.result.error = str(e)
                self.status = AgentStatus.FAILED
        else:
            # 无Agent时直接调用工具
            self.log("Agent未初始化，使用简化模式")
            self.result.error = "Agent未初始化"
            self.status = AgentStatus.FAILED
        
        return self.result
    
    def generate_report(self) -> str:
        """生成完整分析报告"""
        if self.returns is None:
            return "错误: 未设置数据"
        
        report_parts = [
            "=" * 60,
            "Alpha Agent 分析报告",
            "=" * 60,
            "",
            self._tool_risk_analysis(),
            "",
            self._tool_market_regime(),
        ]
        
        if self.factor_returns is not None:
            report_parts.extend([
                "",
                self._tool_factor_attribution(),
            ])
        
        return "\n".join(report_parts)
