"""
Reflexion机制 - Agent自我反思和改进

参考: Reflexion: Language Agents with Verbal Reinforcement Learning

核心思想:
1. 执行任务并获得反馈
2. 生成反思 (为什么失败/成功)
3. 将反思存入记忆
4. 下次任务时检索相关反思
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@dataclass
class ReflexionEntry:
    """反思记录"""
    task: str
    action: str
    result: str
    success: bool
    reflection: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    lessons: List[str] = field(default_factory=list)


# 反思生成提示
REFLECTION_PROMPT = """你是一个量化因子研究员，需要反思之前的行动。

## 任务
{task}

## 采取的行动
{action}

## 结果
{result}

## 是否成功
{"成功" if {success} else "失败"}

请进行深度反思:
1. 分析成功/失败的根本原因
2. 总结可以学到的经验教训
3. 提出改进建议

以JSON格式输出:
{{
    "analysis": "原因分析",
    "lessons": ["经验1", "经验2"],
    "improvements": ["改进1", "改进2"]
}}
"""


class ReflexionMemory:
    """反思记忆库"""
    
    def __init__(self, max_entries: int = 100):
        self.entries: List[ReflexionEntry] = []
        self.max_entries = max_entries
    
    def add(self, entry: ReflexionEntry):
        """添加反思"""
        self.entries.append(entry)
        
        # 保持最大数量
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def search(self, query: str, top_k: int = 3) -> List[ReflexionEntry]:
        """搜索相关反思 (简单关键词匹配)"""
        query_lower = query.lower()
        scored = []
        
        for entry in self.entries:
            # 简单相关性评分
            score = 0
            if query_lower in entry.task.lower():
                score += 2
            if query_lower in entry.reflection.lower():
                score += 1
            for lesson in entry.lessons:
                if query_lower in lesson.lower():
                    score += 1
            
            if score > 0:
                scored.append((entry, score))
        
        # 按分数排序
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, s in scored[:top_k]]
    
    def get_recent(self, n: int = 5) -> List[ReflexionEntry]:
        """获取最近的反思"""
        return self.entries[-n:]
    
    def get_failures(self) -> List[ReflexionEntry]:
        """获取失败案例"""
        return [e for e in self.entries if not e.success]
    
    def save(self, path: str):
        """保存到文件"""
        data = [
            {
                'task': e.task,
                'action': e.action,
                'result': e.result,
                'success': e.success,
                'reflection': e.reflection,
                'timestamp': e.timestamp,
                'lessons': e.lessons,
            }
            for e in self.entries
        ]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """从文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.entries = [
            ReflexionEntry(
                task=d['task'],
                action=d['action'],
                result=d['result'],
                success=d['success'],
                reflection=d['reflection'],
                timestamp=d.get('timestamp', ''),
                lessons=d.get('lessons', []),
            )
            for d in data
        ]


class ReflexionAgent:
    """带有反思能力的Agent"""
    
    def __init__(
        self,
        llm: "ChatOpenAI" = None,
        memory: ReflexionMemory = None,
    ):
        self.llm = llm
        self.memory = memory or ReflexionMemory()
    
    def reflect(
        self,
        task: str,
        action: str,
        result: str,
        success: bool,
    ) -> ReflexionEntry:
        """生成反思"""
        if self.llm is None:
            # 无LLM时使用简单反思
            reflection = f"任务{'成功' if success else '失败'}。"
            lessons = ["需要改进策略"] if not success else ["继续保持"]
        else:
            # 使用LLM生成反思
            prompt = REFLECTION_PROMPT.format(
                task=task,
                action=action,
                result=result,
                success=success,
            )
            
            try:
                response = self.llm.invoke(prompt)
                content = response.content
                
                # 解析JSON
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    reflection = parsed.get('analysis', content)
                    lessons = parsed.get('lessons', [])
                else:
                    reflection = content
                    lessons = []
            except Exception as e:
                logger.warning(f"反思生成失败: {e}")
                reflection = f"任务{'成功' if success else '失败'}，需要进一步分析。"
                lessons = []
        
        entry = ReflexionEntry(
            task=task,
            action=action,
            result=result,
            success=success,
            reflection=reflection,
            lessons=lessons,
        )
        
        self.memory.add(entry)
        return entry
    
    def get_context(self, task: str) -> str:
        """获取反思上下文"""
        # 搜索相关反思
        relevant = self.memory.search(task, top_k=3)
        
        if not relevant:
            return ""
        
        context_parts = ["## 历史反思参考\n"]
        for entry in relevant:
            context_parts.append(f"""
### 相关任务: {entry.task[:50]}...
- 结果: {"成功" if entry.success else "失败"}
- 反思: {entry.reflection[:200]}...
- 经验: {', '.join(entry.lessons[:3])}
""")
        
        return "\n".join(context_parts)
    
    def should_retry(self, task: str, max_failures: int = 3) -> Tuple[bool, str]:
        """判断是否应该重试"""
        # 获取该任务的失败次数
        failures = [e for e in self.memory.entries 
                   if task.lower() in e.task.lower() and not e.success]
        
        if len(failures) >= max_failures:
            # 汇总失败原因
            reasons = [e.reflection for e in failures[-3:]]
            return False, f"已失败{len(failures)}次。原因: " + "; ".join(reasons)
        
        return True, ""


def create_reflexion_wrapper(agent_class):
    """创建带Reflexion的Agent包装器"""
    
    class ReflexionWrappedAgent(agent_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reflexion = ReflexionAgent()
        
        def run(self, task: str):
            # 获取历史反思
            context = self.reflexion.get_context(task)
            
            # 检查是否应该重试
            should_retry, reason = self.reflexion.should_retry(task)
            if not should_retry:
                logger.warning(f"任务跳过: {reason}")
                return None
            
            # 执行任务
            try:
                result = super().run(task + "\n" + context)
                success = result is not None and not getattr(result, 'error', None)
            except Exception as e:
                result = str(e)
                success = False
            
            # 生成反思
            self.reflexion.reflect(
                task=task,
                action="执行因子生成任务",
                result=str(result)[:500],
                success=success,
            )
            
            return result
    
    return ReflexionWrappedAgent
