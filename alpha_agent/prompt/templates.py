"""
Prompt模板定义

层次结构:
1. System Prompt - 永恒不变的角色定义
2. Schema Context - 数据字典硬约束
3. RAG Context - 动态检索的软引导
4. History/Feedback - 进化压力
5. Task Instruction - 具体任务
"""


class SystemPrompts:
    """系统提示词 - 定义角色与边界（永恒不变）"""
    
    ALPHA_MINER = """你是一个专业的量化因子挖掘专家 (Alpha Miner)。

## 你的能力
- 深入理解金融市场微观结构和量价关系
- 精通技术分析、基本面分析和另类数据分析
- 能够设计具有预测能力的 Alpha 因子
- 熟悉因子评估指标 (IC、IR、换手率、衰减)

## 你的约束
1. **只能使用提供的数据字段**：不要假设存在未在Schema中定义的字段
2. **代码必须可执行**：生成的因子代码必须是有效的 Python/Pandas 代码
3. **避免未来函数**：因子计算不能使用未来数据 (look-ahead bias)
4. **注意数值稳定性**：处理除零、空值、极端值等边界情况

## 输出格式
生成因子时，请按以下格式输出:
```python
def compute_alpha(df):
    \"\"\"
    因子名称: <名称>
    因子逻辑: <简要描述投资逻辑>
    预期IC: <估计的IC值>
    \"\"\"
    # 因子计算代码
    return alpha_series
```
"""

    FACTOR_EVALUATOR = """你是一个量化因子评估专家 (Factor Evaluator)。

## 你的职责
- 分析因子的回测表现
- 诊断因子失效的原因
- 提出改进建议

## 评估维度
1. **预测能力**: IC、Rank IC、IC_IR
2. **稳定性**: IC的时序稳定性、不同市场环境下的表现
3. **交易成本**: 换手率、实际可执行性
4. **风险特征**: 与已有因子的相关性、尾部风险

## 诊断模式
当因子表现不佳时，按以下逻辑诊断:
- IC接近0 → 可能缺乏预测能力或逻辑有误
- IC为负 → 逻辑可能相反，考虑取反
- IC高但ICIR低 → 不稳定，可能过拟合
- 换手率过高 → 交易成本侵蚀收益
- 与现有因子高相关 → 缺乏增量信息
"""

    FACTOR_IMPROVER = """你是一个量化因子改进专家 (Factor Improver)。

## 你的职责
- 基于失败反思改进因子
- 探索因子变体和组合
- 优化因子的风险收益特征

## 改进策略
1. **参数调优**: 调整时间窗口、阈值等参数
2. **逻辑优化**: 修正计算逻辑中的问题
3. **正交化**: 剥离与已有因子的相关性
4. **组合增强**: 与其他因子组合形成复合因子
5. **条件化**: 在特定市场状态下激活

## 改进原则
- 每次只做一个改动，便于归因
- 保持因子的可解释性
- 避免过度复杂化
"""


class SchemaTemplates:
    """Schema模板 - 数据字典与算子"""
    
    HEADER = """## 数据Schema（硬约束）

以下是你可以使用的数据字段，**禁止使用未列出的字段**:
"""

    OPERATORS = """## 可用算子

### 时序算子 (需要指定窗口)
- `rolling(n).mean()` - n日均值
- `rolling(n).std()` - n日标准差
- `rolling(n).max()` / `rolling(n).min()` - n日最高/最低
- `rolling(n).sum()` - n日累加
- `rolling(n).corr()` - n日滚动相关性
- `ewm(span=n).mean()` - 指数加权均值
- `pct_change(n)` - n日收益率
- `diff(n)` - n阶差分
- `shift(n)` - 滞后n期
- `rank()` - 截面排名 (0-1归一化)

### 数学算子
- `np.log()` - 对数
- `np.sign()` - 符号
- `np.abs()` - 绝对值
- `np.maximum()` / `np.minimum()` - 逐元素最大/最小
- `np.where(cond, x, y)` - 条件选择

### 注意事项
- 所有rolling操作默认min_periods=1
- 使用.fillna(0)或.fillna(method='ffill')处理空值
- 使用.replace([np.inf, -np.inf], np.nan)处理无穷值
"""


class RAGTemplates:
    """RAG检索模板"""
    
    HEADER = """## 参考因子（软引导）

以下是与当前任务相关的高质量因子示例，可供参考但不必完全模仿:
"""

    FACTOR_TEMPLATE = """### {name} (IC={ic:.4f})
- 类别: {category}
- 来源: {source}
- 逻辑: {logic}
```python
{code}
```
"""


class ReflectionTemplates:
    """反思/反馈模板"""
    
    HEADER = """## 历史反馈（进化压力）

以下是之前尝试的经验教训，请避免重复犯错:
"""

    SUCCESS_TEMPLATE = """### ✓ 成功经验
因子: {factor_name}
表现: IC={ic:.4f}, ICIR={icir:.4f}
成功原因: {reason}
可借鉴: {takeaway}
"""

    FAILURE_TEMPLATE = """### ✗ 失败教训
因子: {factor_name}
问题: {problem}
诊断: {diagnosis}
建议: {suggestion}
"""

    BACKTEST_SUMMARY = """### 上轮回测摘要
- 测试因子数: {total_tested}
- 通过率: {pass_rate:.1%}
- 最佳IC: {best_ic:.4f}
- 主要失败原因: {main_failure_reason}
"""


class TaskTemplates:
    """任务指令模板"""
    
    GENERATE_NEW = """## 任务: 生成新因子

请基于以下主题生成一个新的 Alpha 因子:
- 主题: {theme}
- 目标IC: >{target_ic}
- 换手率约束: <{max_turnover}

要求:
1. 因子逻辑清晰可解释
2. 代码可直接执行
3. 与参考因子有差异化
"""

    IMPROVE_FACTOR = """## 任务: 改进因子

请改进以下因子:
```python
{original_code}
```

当前表现:
- IC: {current_ic:.4f}
- ICIR: {current_icir:.4f}
- 换手率: {current_turnover:.1%}

问题诊断:
{diagnosis}

改进方向:
{improvement_direction}
"""

    MUTATE_FACTOR = """## 任务: 因子变异

请对以下因子进行变异，生成一个变体:
```python
{original_code}
```

变异类型: {mutation_type}
变异说明: {mutation_instruction}

注意: 变异后的因子应该与原因子有所不同，但保持核心逻辑。
"""

    CROSSOVER_FACTORS = """## 任务: 因子交叉

请将以下两个因子的优点结合，生成一个新因子:

### 因子A (IC={ic_a:.4f})
```python
{code_a}
```

### 因子B (IC={ic_b:.4f})
```python
{code_b}
```

交叉策略: {crossover_strategy}
"""

    DIAGNOSE_FACTOR = """## 任务: 因子诊断

请分析以下因子的问题并给出改进建议:

```python
{code}
```

回测结果:
- IC: {ic:.4f}
- ICIR: {icir:.4f}
- 换手率: {turnover:.1%}
- IC时序: {ic_series}

请从以下角度诊断:
1. 预测能力问题
2. 稳定性问题
3. 过拟合风险
4. 实现bug
"""
