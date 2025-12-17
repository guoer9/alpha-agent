"""
安全沙箱 - 安全执行LLM生成的代码
"""

import sys
import signal
import traceback
from io import StringIO
from typing import Tuple, Optional, Dict, Any
from contextlib import contextmanager
import pandas as pd
import numpy as np

from ..config import sandbox_config


class SandboxError(Exception):
    """沙箱执行错误"""
    pass


class TimeoutError(Exception):
    """超时错误"""
    pass


@contextmanager
def timeout(seconds: int):
    """超时上下文管理器 - 仅在主线程中使用signal"""
    import threading
    
    # 仅在主线程且Unix系统中使用signal
    is_main_thread = threading.current_thread() is threading.main_thread()
    
    if is_main_thread and hasattr(signal, 'SIGALRM'):
        def signal_handler(signum, frame):
            raise TimeoutError(f"执行超时 ({seconds}秒)")
        
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # 非主线程：不使用超时（依赖外部超时控制）
        yield


# 安全的内置函数白名单
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter',
    'float', 'int', 'len', 'list', 'map', 'max', 'min', 'pow',
    'print', 'range', 'round', 'set', 'sorted', 'str', 'sum',
    'tuple', 'zip', 'True', 'False', 'None',
}

# 危险模式
DANGEROUS_PATTERNS = [
    'import os', 'import sys', 'import subprocess',
    '__import__', 'eval(', 'exec(', 'compile(',
    'open(', 'file(', 'input(',
    'globals(', 'locals(', 'vars(',
    '__builtins__', '__code__', '__class__',
    'system(', 'popen(', 'spawn',
]


def create_safe_globals() -> Dict[str, Any]:
    """创建安全的全局命名空间"""
    safe_globals = {
        '__builtins__': {k: __builtins__[k] for k in SAFE_BUILTINS if k in __builtins__}
        if isinstance(__builtins__, dict)
        else {k: getattr(__builtins__, k) for k in SAFE_BUILTINS if hasattr(__builtins__, k)},
        'pd': pd,
        'np': np,
        'pandas': pd,
        'numpy': np,
    }
    
    # 添加常用numpy函数
    for func in ['log', 'exp', 'sqrt', 'abs', 'sign', 'maximum', 'minimum', 'where', 'nan', 'inf']:
        if hasattr(np, func):
            safe_globals[func] = getattr(np, func)
    
    return safe_globals


def validate_code(code: str) -> Tuple[bool, str]:
    """
    验证代码安全性
    
    返回:
        (is_safe, error_message)
    """
    if not code or not code.strip():
        return False, "代码为空"
    
    # 检查危险模式
    code_lower = code.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in code_lower:
            return False, f"检测到危险模式: {pattern}"
    
    # 检查是否包含compute_alpha函数
    if 'def compute_alpha' not in code:
        return False, "代码必须包含 compute_alpha 函数"
    
    return True, ""


class Sandbox:
    """安全沙箱"""
    
    def __init__(
        self,
        timeout_seconds: int = None,
        max_retries: int = None,
    ):
        self.timeout_seconds = timeout_seconds or sandbox_config.timeout
        self.max_retries = max_retries or sandbox_config.max_retries
    
    def execute(
        self,
        code: str,
        df: pd.DataFrame,
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        """
        执行因子代码
        
        参数:
            code: 因子代码
            df: 输入数据
        
        返回:
            (factor_series, error_message)
        """
        # 验证代码
        is_safe, error = validate_code(code)
        if not is_safe:
            return None, error
        
        # 创建安全环境
        safe_globals = create_safe_globals()
        safe_globals['df'] = df.copy()
        
        # 捕获输出
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            with timeout(self.timeout_seconds):
                # 执行代码
                exec(code, safe_globals)
                
                # 检查compute_alpha函数
                if 'compute_alpha' not in safe_globals:
                    return None, "compute_alpha 函数未定义"
                
                # 调用函数
                factor = safe_globals['compute_alpha'](df)
                
                # 验证输出
                if factor is None:
                    return None, "compute_alpha 返回 None"
                
                # 处理numpy scalar (单个值)
                if np.isscalar(factor) or (hasattr(factor, 'ndim') and factor.ndim == 0):
                    # 标量输出：广播为常数因子（与 df.index 对齐），避免静默丢弃
                    try:
                        return pd.Series(factor, index=df.index), None
                    except Exception:
                        return None, None
                
                if not isinstance(factor, (pd.Series, np.ndarray)):
                    return None, None  # 静默跳过
                
                if isinstance(factor, np.ndarray):
                    if factor.ndim == 0:  # 0维数组（标量）
                        try:
                            return pd.Series(factor.item(), index=df.index), None
                        except Exception:
                            return None, None
                    factor = pd.Series(factor, index=df.index)
                
                return factor, None
                
        except TimeoutError as e:
            return None, str(e)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg
        finally:
            sys.stdout = old_stdout


def execute_code(
    code: str,
    df: pd.DataFrame,
    timeout_seconds: int = None,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    便捷函数: 执行因子代码
    
    返回:
        (factor_series, error_message)
    """
    sandbox = Sandbox(timeout_seconds=timeout_seconds)
    return sandbox.execute(code, df)
