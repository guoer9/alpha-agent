"""
LLM生成器 - 使用大语言模型生成因子代码
支持: OpenAI, 阿里云DashScope (通义千问)
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

# LangChain
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain.memory import ConversationBufferWindowMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# DashScope
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

from ..config import llm_config


# 系统提示词
SYSTEM_PROMPT = """你是一个专业的量化因子工程师，负责设计alpha因子。

## 输出格式
请使用Python代码，格式如下:
```python
def compute_alpha(df):
    '''
    因子名称: xxx
    因子逻辑: 描述经济学含义
    '''
    # df包含: open, high, low, close, volume, 以及其他特征
    factor = ...  # 你的因子计算逻辑
    return factor  # 返回pd.Series
```

## 注意事项
1. 处理NaN: 使用 fillna() 或 dropna()
2. 避免除零: 使用 np.where() 或添加小常数
3. 标准化: 考虑使用rank或z-score
4. 滞后: 使用 .shift() 避免未来数据泄露

## 常用算子
- 动量: pct_change(), diff()
- 均线: rolling().mean()
- 波动: rolling().std()
- 相关: rolling().corr()
- 排名: rank(pct=True)
"""


class LLMGenerator:
    """LLM因子生成器 - 支持OpenAI和DashScope"""
    
    def __init__(
        self,
        provider: str = None,
        model: str = None,
        temperature: float = None,
        api_key: str = None,
        base_url: str = None,
        system_prompt: str = None,
    ):
        self.provider = provider or llm_config.provider
        self.model = model or llm_config.model
        self.temperature = temperature or llm_config.temperature
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.api_key = api_key
        
        # 根据provider初始化
        if self.provider == "dashscope":
            self._init_dashscope()
        else:
            self._init_openai()
        
        # 对话历史
        self.chat_history: List[Dict] = []
        self.history: List[Dict] = []
    
    def _init_openai(self):
        """初始化OpenAI后端"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("请安装langchain: pip install langchain langchain-openai")
        
        llm_kwargs = {
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': llm_config.max_tokens,
        }
        
        api_key = self.api_key or llm_config.openai_api_key
        base_url = llm_config.openai_base_url
        
        if api_key:
            llm_kwargs['api_key'] = api_key
        if base_url:
            llm_kwargs['base_url'] = base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # 对话记忆
        self.memory = ConversationBufferWindowMemory(
            k=10, return_messages=True, memory_key="chat_history"
        )
        self._build_chain()
    
    def _init_dashscope(self):
        """初始化DashScope (通义千问) 后端"""
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("请安装dashscope: pip install dashscope")
        
        self.api_key = self.api_key or llm_config.dashscope_api_key
        dashscope.base_http_api_url = llm_config.dashscope_base_url
        self.llm = None  # DashScope使用直接调用
        self.memory = None
    
    def _build_chain(self):
        """构建LangChain链"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt
        if self.provider != "dashscope":
            self._build_chain()
    
    def generate(self, instruction: str) -> str:
        """
        生成因子代码
        
        参数:
            instruction: 生成指令
        
        返回:
            生成的代码字符串
        """
        if self.provider == "dashscope":
            response = self._generate_dashscope(instruction)
        else:
            response = self._generate_openai(instruction)
        
        # 提取代码
        code = self._extract_code(response)
        
        # 记录历史
        self.history.append({
            "instruction": instruction,
            "response": response,
            "code": code,
            "timestamp": datetime.now().isoformat()
        })
        
        return code
    
    def _generate_openai(self, instruction: str) -> str:
        """使用OpenAI/LangChain生成"""
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        
        response = self.chain.invoke({
            "input": instruction,
            "chat_history": chat_history
        })
        
        self.memory.save_context(
            {"input": instruction},
            {"output": response}
        )
        return response
    
    def _generate_dashscope(self, instruction: str) -> str:
        """使用DashScope (通义千问) 生成"""
        # 构建消息
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # 添加历史对话
        for h in self.chat_history[-10:]:
            messages.append({"role": "user", "content": h.get("user", "")})
            messages.append({"role": "assistant", "content": h.get("assistant", "")})
        
        messages.append({"role": "user", "content": instruction})
        
        # 调用DashScope
        response = Generation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            result_format="message",
            temperature=self.temperature,
        )
        
        # 提取响应
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            # 保存对话历史
            self.chat_history.append({
                "user": instruction,
                "assistant": content
            })
            return content
        else:
            raise Exception(f"DashScope API错误: {response.code} - {response.message}")
    
    def _extract_code(self, response: str) -> str:
        """从响应中提取代码"""
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        if "def compute_alpha" in response:
            start = response.find("def compute_alpha")
            lines = response[start:].split('\n')
            code_lines = []
            in_function = True
            indent_level = None
            
            for line in lines:
                if indent_level is None and line.strip().startswith('def '):
                    indent_level = len(line) - len(line.lstrip())
                    code_lines.append(line)
                elif indent_level is not None:
                    current_indent = len(line) - len(line.lstrip())
                    if line.strip() and current_indent <= indent_level and not line.strip().startswith('def '):
                        break
                    code_lines.append(line)
            
            return '\n'.join(code_lines).strip()
        
        return response
    
    def fix_error(self, code: str, error: str) -> str:
        """修复代码错误"""
        fix_prompt = f"""
代码执行出错，请修复:

原始代码:
```python
{code}
```

错误信息:
{error}

请输出修复后的完整代码。
"""
        return self.generate(fix_prompt)
    
    def improve_factor(self, code: str, ic: float, feedback: str = "") -> str:
        """改进因子"""
        improve_prompt = f"""
当前因子的IC为 {ic:.4f}，请改进:

当前代码:
```python
{code}
```

{f'反馈: {feedback}' if feedback else ''}

请生成一个改进后的因子，目标是提高IC。
"""
        return self.generate(improve_prompt)
    
    def clear_memory(self):
        """清除对话记忆"""
        self.memory.clear()
