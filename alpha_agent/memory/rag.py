"""
RAG检索增强生成 - 利用历史因子知识增强LLM生成

功能:
1. 检索相似因子代码作为上下文
2. 检索成功/失败案例
3. 构建增强提示词
"""

from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .vector_store import FactorMemory, MilvusStore


# RAG提示模板
RAG_PROMPT_TEMPLATE = """你是一个专业的量化因子研究员。

## 相关历史因子参考
以下是与当前任务相关的历史因子，供你参考：

{context}

## 任务要求
{instruction}

## 注意事项
1. 可以参考上述因子的设计思路，但要避免完全复制
2. 尝试创新和改进
3. 确保代码可执行
4. 考虑因子的经济学含义

请生成新的因子代码：
"""


class RAGGenerator:
    """RAG增强因子生成器"""
    
    def __init__(
        self,
        factor_memory: FactorMemory = None,
        top_k: int = 3,
        include_failed: bool = False,
    ):
        """
        参数:
            factor_memory: 因子记忆库
            top_k: 检索数量
            include_failed: 是否包含失败案例
        """
        self.memory = factor_memory
        self.top_k = top_k
        self.include_failed = include_failed
        self.prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    def retrieve(self, query: str) -> List[Dict]:
        """检索相关因子"""
        if self.memory is None:
            return []
        
        try:
            results = self.memory.search_similar(query, top_k=self.top_k)
            
            # 过滤失败案例
            if not self.include_failed:
                results = [r for r in results if r.get('status') != 'failed']
            
            return results
        except Exception as e:
            logger.warning(f"检索失败: {e}")
            return []
    
    def format_context(self, factors: List[Dict]) -> str:
        """格式化检索结果为上下文"""
        if not factors:
            return "暂无相关历史因子。"
        
        context_parts = []
        for i, factor in enumerate(factors, 1):
            part = f"""
### 因子 {i}: {factor.get('name', 'unknown')}
- IC: {factor.get('ic', 0):.4f}
- 相似度: {factor.get('score', 0):.2f}
```python
{factor.get('code', '# 无代码')}
```
"""
            context_parts.append(part)
        
        return "\n".join(context_parts)
    
    def build_prompt(self, instruction: str) -> str:
        """构建RAG增强提示词"""
        # 检索相关因子
        factors = self.retrieve(instruction)
        
        # 格式化上下文
        context = self.format_context(factors)
        
        # 构建提示词
        prompt = self.prompt_template.format(
            context=context,
            instruction=instruction,
        )
        
        return prompt
    
    def get_similar_factors(self, code: str, threshold: float = 0.9) -> List[Dict]:
        """获取与给定代码相似的因子（用于去重）"""
        if self.memory is None:
            return []
        
        return self.memory.search_similar(code, top_k=5)


class FactorDeduplicator:
    """因子去重器"""
    
    def __init__(
        self,
        factor_memory: FactorMemory = None,
        similarity_threshold: float = 0.95,
    ):
        self.memory = factor_memory
        self.threshold = similarity_threshold
    
    def is_duplicate(self, code: str) -> Tuple[bool, Optional[Dict]]:
        """
        检查因子是否重复
        
        返回:
            (是否重复, 重复的因子信息)
        """
        if self.memory is None:
            return False, None
        
        try:
            duplicate = self.memory.check_duplicate(code, threshold=self.threshold)
            if duplicate:
                return True, duplicate
            return False, None
        except Exception as e:
            logger.warning(f"去重检查失败: {e}")
            return False, None
    
    def find_similar(self, code: str, top_k: int = 5) -> List[Dict]:
        """查找相似因子"""
        if self.memory is None:
            return []
        
        return self.memory.search_similar(code, top_k=top_k)
    
    def deduplicate_batch(self, factors: List[Dict]) -> List[Dict]:
        """批量去重"""
        unique_factors = []
        seen_codes = set()
        
        for factor in factors:
            code = factor.get('code', '')
            
            # 检查是否已在当前批次中
            code_hash = hash(code.strip())
            if code_hash in seen_codes:
                continue
            
            # 检查是否在记忆库中
            is_dup, _ = self.is_duplicate(code)
            if is_dup:
                continue
            
            seen_codes.add(code_hash)
            unique_factors.append(factor)
        
        return unique_factors


# 便捷函数
def create_rag_prompt(instruction: str, memory: FactorMemory = None) -> str:
    """创建RAG增强提示词"""
    rag = RAGGenerator(factor_memory=memory)
    return rag.build_prompt(instruction)


def check_factor_duplicate(
    code: str, 
    memory: FactorMemory = None,
    threshold: float = 0.95,
) -> Tuple[bool, Optional[Dict]]:
    """检查因子是否重复"""
    dedup = FactorDeduplicator(factor_memory=memory, similarity_threshold=threshold)
    return dedup.is_duplicate(code)
