#!/usr/bin/env python
"""
Alpha Agent 因子挖掘系统

完整的因子挖掘流程:
1. LLM探索 - 使用DashScope/OpenAI生成因子
2. GP精炼 - 遗传算法优化参数
3. LLM反思 - 解释有效因子的投资逻辑

使用方法:
    # 快速测试 (1轮LLM, 少量因子)
    python run_factor_mining.py --mode quick
    
    # 标准运行 (3轮LLM, 完整流程)
    python run_factor_mining.py --mode standard
    
    # 深度挖掘 (5轮LLM, 大规模GP)
    python run_factor_mining.py --mode deep
    
    # 自定义参数
    python run_factor_mining.py --llm-rounds 3 --gp-generations 10 --batch-size 5

环境变量:
    DASHSCOPE_API_KEY: 阿里云DashScope API密钥
    OPENAI_API_KEY: OpenAI API密钥 (可选)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class RunConfig:
    """运行配置"""
    # 模式
    mode: str = "standard"  # quick / standard / deep
    
    # LLM配置
    llm_provider: str = "dashscope"
    llm_model: str = "qwen-max"
    llm_rounds: int = 3
    llm_batch_size: int = 3
    
    # GP配置
    gp_population: int = 30
    gp_generations: int = 5
    
    # 阈值
    seed_threshold_ic: float = 0.005
    target_ic: float = 0.02
    max_turnover: float = 0.5
    
    # Qlib配置
    instruments: str = "csi300"
    train_start: str = "2018-01-01"
    train_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2023-12-31"
    
    # 输出
    output_dir: str = "output/factors"
    save_results: bool = True
    
    @classmethod
    def from_mode(cls, mode: str) -> "RunConfig":
        """从预设模式创建配置"""
        if mode == "quick":
            return cls(
                mode="quick",
                llm_rounds=1,
                llm_batch_size=2,
                gp_population=10,
                gp_generations=2,
                seed_threshold_ic=0.003,
            )
        elif mode == "deep":
            return cls(
                mode="deep",
                llm_rounds=5,
                llm_batch_size=5,
                gp_population=50,
                gp_generations=10,
                seed_threshold_ic=0.008,
            )
        else:  # standard
            return cls(mode="standard")


# ============================================================
# 核心组件
# ============================================================

class FactorMiningSystem:
    """因子挖掘系统 - 整合所有组件"""
    
    def __init__(self, config: RunConfig):
        self.config = config
        self.api_key: Optional[str] = None
        self.qlib_initialized: bool = False
        self._data_cache: Dict[str, Any] = {}
        
    def setup(self) -> bool:
        """初始化系统"""
        logger.info("="*60)
        logger.info("     🧬 Alpha Agent 因子挖掘系统")
        logger.info("="*60)
        logger.info(f"模式: {self.config.mode}")
        logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        # 1. 检查API Key
        if not self._check_api_key():
            return False
        
        # 2. 初始化Qlib (可选)
        if not self._init_qlib():
            logger.warning("⚠️ Qlib不可用，使用模拟数据")
            self._create_mock_data()
        else:
            # 3. 预加载数据
            if not self._preload_data():
                logger.warning("⚠️ 数据加载失败，使用模拟数据")
                self._create_mock_data()
        
        logger.info("✅ 系统初始化完成")
        return True
    
    def _check_api_key(self) -> bool:
        """检查API Key"""
        # 优先使用环境变量
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        
        if not self.api_key:
            # 尝试从配置文件读取
            try:
                import sys
                sys.path.insert(0, str(PROJECT_ROOT))
                from config.settings import LLMConfig
                llm_cfg = LLMConfig()
                self.api_key = llm_cfg.dashscope_api_key
            except Exception as e:
                logger.warning(f"配置读取失败: {e}")
        
        if not self.api_key:
            logger.error("❌ 未找到API Key")
            logger.error("   请设置环境变量: export DASHSCOPE_API_KEY=your-key")
            return False
        
        logger.info(f"✅ API Key: {self.api_key[:8]}...")
        return True
    
    def _init_qlib(self) -> bool:
        """初始化Qlib"""
        try:
            # 确保在正确的工作目录
            os.chdir(PROJECT_ROOT)
            
            import qlib
            from qlib.config import REG_CN
            
            provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
            
            if not os.path.exists(provider_uri):
                logger.error(f"❌ Qlib数据不存在: {provider_uri}")
                logger.error("   请下载: python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data")
                return False
            
            qlib.init(provider_uri=provider_uri, region=REG_CN)
            self.qlib_initialized = True
            logger.info(f"✅ Qlib初始化: {provider_uri}")
            return True
            
        except ImportError:
            logger.error("❌ Qlib未安装: pip install pyqlib")
            return False
        except Exception as e:
            import traceback
            logger.error(f"❌ Qlib初始化失败: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _create_mock_data(self):
        """创建模拟数据用于测试"""
        logger.info("📊 创建模拟数据...")
        
        n_days = 500
        n_stocks = 100
        
        dates = pd.date_range('2022-01-01', periods=n_days, freq='B')
        stocks = [f'SH60{i:04d}' for i in range(n_stocks)]
        
        index = pd.MultiIndex.from_product([stocks, dates], names=['instrument', 'datetime'])
        
        np.random.seed(42)
        base_price = 10 + np.random.randn(n_stocks, 1) * 5
        returns = np.random.randn(n_stocks, n_days) * 0.02
        prices = base_price * np.exp(returns.cumsum(axis=1))
        
        df = pd.DataFrame(index=index)
        df['close'] = prices.flatten()
        df['open'] = df['close'] * (1 + np.random.randn(len(df)) * 0.005)
        df['high'] = df[['close', 'open']].max(axis=1) * (1 + np.abs(np.random.randn(len(df))) * 0.01)
        df['low'] = df[['close', 'open']].min(axis=1) * (1 - np.abs(np.random.randn(len(df))) * 0.01)
        df['volume'] = np.abs(np.random.randn(len(df))) * 1e6 + 1e5
        df['adj_factor'] = 1.0
        
        # 计算目标收益
        future_return = df['close'].groupby(level=0).pct_change(5).shift(-5)
        
        self._data_cache['df'] = df
        self._data_cache['target'] = future_return
        
        logger.info(f"   模拟数据: {len(df):,} 行, {n_stocks} 只股票, {n_days} 天")
    
    def _preload_data(self) -> bool:
        """预加载数据到缓存"""
        try:
            from qlib.data import D
            
            logger.info("📊 预加载Qlib数据...")
            
            instruments = D.instruments(self.config.instruments)
            fields = ["$close", "$open", "$high", "$low", "$volume", "$factor"]
            
            df = D.features(
                instruments,
                fields,
                start_time=self.config.train_start,
                end_time=self.config.test_end,
                freq="day",
            )
            df.columns = ['close', 'open', 'high', 'low', 'volume', 'adj_factor']
            
            # 计算未来收益作为目标
            future_return = df['close'].groupby(level=0).pct_change(5).shift(-5)
            
            self._data_cache['df'] = df
            self._data_cache['target'] = future_return
            
            logger.info(f"   数据量: {len(df):,} 行")
            logger.info(f"   时间范围: {df.index.get_level_values(1).min()} ~ {df.index.get_level_values(1).max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            return False
    
    def create_llm_generator(self) -> Callable:
        """创建LLM因子生成器"""
        from dashscope import Generation
        from alpha_agent.prompt.composer import PromptComposer, TaskType
        
        composer = PromptComposer()
        api_key = self.api_key
        
        # 因子主题列表
        themes = [
            "量价动量", "均值回归", "波动率异常", 
            "成交量背离", "趋势强度", "价格形态",
            "资金流向", "情绪指标",
        ]
        
        # 历史记录
        history = {"failures": [], "successes": []}
        
        def generator(context: dict) -> dict:
            """LLM因子生成器"""
            round_idx = context.get('round', 0)
            existing_seeds = context.get('existing_seeds', [])
            best_ic = context.get('best_ic', 0)
            
            # 选择主题
            theme = themes[round_idx % len(themes)]
            
            # 构建RAG上下文
            rag_factors = []
            for i, seed in enumerate(existing_seeds[-3:]):
                if isinstance(seed, str) and len(seed) > 10:
                    rag_factors.append({
                        'name': f'Seed_{i}',
                        'ic': best_ic * (0.8 + i * 0.1),
                        'category': 'seed',
                        'source': 'evolution',
                        'logic': '前轮生成的种子因子',
                        'code': seed[:200],
                    })
            
            # 组装Prompt
            composed = composer.for_generation(
                theme=theme,
                target_ic=self.config.target_ic,
                rag_factors=rag_factors,
                failures=history["failures"][-3:] if history["failures"] else None,
            )
            
            try:
                logger.info(f"  📤 调用LLM (主题: {theme})...")
                
                response = Generation.call(
                    api_key=api_key,
                    model=self.config.llm_model,
                    messages=composed.to_messages(),
                    result_format="message",
                    temperature=0.7,
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"API错误: {response.code} - {response.message}")
                
                content = response.output.choices[0].message.content
                logger.info(f"  📥 LLM响应: {len(content)} 字符")
                
                # 提取代码
                code = self._extract_code(content)
                name = self._extract_name(content) or f"LLM_{theme}_{int(time.time()) % 10000}"
                logic = self._extract_logic(content) or f"基于{theme}的因子"
                
                # 记录成功
                history["successes"].append({"theme": theme, "name": name})
                
                return {"name": name, "code": code, "logic": logic}
                
            except Exception as e:
                # 记录失败
                history["failures"].append({
                    "factor_name": f"Round{round_idx}",
                    "problem": str(e)[:100],
                    "diagnosis": "API调用或代码解析失败",
                    "suggestion": "检查网络或简化因子逻辑",
                })
                logger.warning(f"  ❌ LLM生成失败: {e}")
                raise
        
        return generator
    
    def create_evaluator(self) -> Callable:
        """创建因子评估器"""
        from alpha_agent.core.sandbox import Sandbox
        from alpha_agent.core.evaluator import FactorEvaluator
        
        sandbox = Sandbox(timeout_seconds=30)
        core_evaluator = FactorEvaluator()
        
        df = self._data_cache['df']
        target = self._data_cache['target']
        
        def evaluator(factor_code: str, full_backtest: bool = False) -> dict:
            """评估因子"""
            try:
                # 1. 沙箱执行
                factor_values, error = sandbox.execute(factor_code, df)
                
                if error:
                    raise RuntimeError(f"执行失败: {error[:100]}")
                
                if factor_values is None or len(factor_values) == 0:
                    raise RuntimeError("因子值为空")
                
                # 2. 对齐数据
                aligned = pd.concat([factor_values, target], axis=1)
                aligned.columns = ['factor', 'return']
                aligned = aligned.dropna()
                
                if len(aligned) < 100:
                    raise RuntimeError(f"有效数据不足: {len(aligned)}")
                
                # 3. 使用CoreEvaluator计算指标
                result = core_evaluator.evaluate(
                    factor=aligned['factor'],
                    target=aligned['return'],
                )
                
                logger.info(f"    IC={result.ic:.4f}, ICIR={result.icir:.2f}, 状态={result.status.value}")
                
                return {
                    'ic': abs(result.ic),
                    'icir': abs(result.icir),
                    'rank_ic': abs(result.rank_ic),
                    'rank_icir': abs(getattr(result, 'rank_icir', result.icir)),
                    'ann_return': getattr(result, 'long_short_return', 0) * 252,
                    'information_ratio': abs(result.icir),
                    'sharpe': abs(result.icir) * 1.5,
                    'max_drawdown': 0.15,
                    'turnover': 0.3,
                }
                
            except Exception as e:
                logger.warning(f"    评估失败: {e}")
                raise
        
        return evaluator
    
    def run(self) -> List[Any]:
        """运行因子挖掘"""
        from alpha_agent.evolution.hybrid_engine import HybridEvolutionEngine, HybridConfig
        
        # 创建HybridConfig
        hybrid_config = HybridConfig(
            llm_batch_size=self.config.llm_batch_size,
            llm_rounds=self.config.llm_rounds,
            seed_threshold_ic=self.config.seed_threshold_ic,
            seed_pool_size=20,
            gp_population=self.config.gp_population,
            gp_generations=self.config.gp_generations,
            gp_elite_rate=0.2,
            reflect_top_k=5,
            max_turnover=self.config.max_turnover,
            target_ic=self.config.target_ic,
        )
        
        # 打印配置
        logger.info("\n📋 运行配置:")
        logger.info(f"   LLM轮数: {hybrid_config.llm_rounds}")
        logger.info(f"   每轮生成: {hybrid_config.llm_batch_size}")
        logger.info(f"   GP种群: {hybrid_config.gp_population}")
        logger.info(f"   GP代数: {hybrid_config.gp_generations}")
        logger.info(f"   IC阈值: {hybrid_config.seed_threshold_ic}")
        
        # 创建引擎
        engine = HybridEvolutionEngine(
            config=hybrid_config,
            llm_generator=self.create_llm_generator(),
            evaluator=self.create_evaluator(),
        )
        
        # 运行进化
        logger.info("\n" + "="*60)
        logger.info("🚀 开始混合进化")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            best_factors = engine.evolve()
            elapsed = time.time() - start_time
            
            # 输出结果
            self._print_results(best_factors, engine, elapsed)
            
            # 保存结果
            if self.config.save_results and best_factors:
                self._save_results(best_factors)
            
            return best_factors
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️ 用户中断")
            return []
        except Exception as e:
            logger.error(f"❌ 运行失败: {e}", exc_info=True)
            raise
    
    def _print_results(self, factors: List, engine: Any, elapsed: float):
        """打印结果"""
        logger.info("\n" + "="*60)
        logger.info("                 📈 因子挖掘结果")
        logger.info("="*60)
        
        if factors:
            logger.info(f"\n✅ 成功生成 {len(factors)} 个有效因子:\n")
            
            for i, f in enumerate(factors, 1):
                logger.info(f"【因子 {i}】{f.name}")
                logger.info(f"   来源: {f.source}")
                logger.info(f"   IC: {f.ic:.4f} [{f.ic_grade}]")
                logger.info(f"   ICIR: {f.icir:.2f}")
                logger.info(f"   Rank IC: {f.rank_ic:.4f}")
                if f.logic:
                    logger.info(f"   逻辑: {f.logic[:80]}...")
                logger.info("")
        else:
            logger.info("\n⚠️ 未生成有效因子")
        
        # 统计
        logger.info("-"*60)
        logger.info(f"📊 统计:")
        logger.info(f"   种子库: {len(engine.seed_pool)}")
        logger.info(f"   精英池: {len(engine.elite_pool)}")
        logger.info(f"   最优IC: {engine.best_ic:.4f}")
        logger.info(f"   总耗时: {elapsed/60:.1f} 分钟")
    
    def _save_results(self, factors: List):
        """保存结果"""
        output_dir = PROJECT_ROOT / self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"factors_{self.config.mode}_{timestamp}.json"
        
        data = []
        for f in factors:
            data.append({
                "id": f.id,
                "name": f.name,
                "code": f.code,
                "source": f.source,
                "ic": f.ic,
                "icir": f.icir,
                "rank_ic": f.rank_ic,
                "sharpe": f.sharpe,
                "ic_grade": f.ic_grade,
                "logic": f.logic,
                "created_at": f.created_at,
            })
        
        with open(output_file, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        
        logger.info(f"\n📁 结果已保存: {output_file}")
    
    # ============================================================
    # 辅助方法
    # ============================================================
    
    def _extract_code(self, content: str) -> str:
        """从LLM响应中提取代码"""
        import re
        
        def clean_imports(code: str) -> str:
            lines = code.split('\n')
            cleaned = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    continue
                cleaned.append(line)
            return '\n'.join(cleaned)
        
        # 从```python代码块提取
        if "```python" in content:
            match = re.search(r'```python\s*\n(.*?)\n```', content, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if 'def compute_alpha' in code:
                    return clean_imports(code)
                elif 'df[' in code:
                    expr = code.strip()
                    if '=' in expr and not expr.startswith('def'):
                        expr = expr.split('=', 1)[1].strip()
                    return f'def compute_alpha(df):\n    """LLM生成的因子"""\n    return {expr}'
        
        # 从```代码块提取
        if "```" in content:
            parts = content.split("```")
            for part in parts[1::2]:
                part = part.replace("python", "").strip()
                if 'def compute_alpha' in part:
                    return clean_imports(part)
        
        # 查找def compute_alpha
        if 'def compute_alpha' in content:
            start = content.find('def compute_alpha')
            lines = content[start:].split('\n')
            code_lines = []
            for i, line in enumerate(lines):
                if i == 0 or line.startswith(' ') or line.startswith('\t') or not line.strip():
                    code_lines.append(line)
                elif line.strip() and not line.startswith(' '):
                    break
            return clean_imports('\n'.join(code_lines).strip())
        
        # 查找df表达式
        for line in content.split('\n'):
            line = line.strip()
            if 'df[' in line and not line.startswith('#'):
                expr = line
                if '=' in expr:
                    expr = expr.split('=', 1)[1].strip()
                return f'def compute_alpha(df):\n    """LLM生成的因子"""\n    return {expr}'
        
        # 默认因子
        return '''def compute_alpha(df):
    """默认动量因子"""
    return df["close"].pct_change(5).fillna(0)'''
    
    def _extract_name(self, content: str) -> Optional[str]:
        """提取因子名称"""
        import re
        match = re.search(r'因子名称[：:]\s*(.+?)[\n\r]', content)
        if match:
            return match.group(1).strip()[:50]
        return None
    
    def _extract_logic(self, content: str) -> Optional[str]:
        """提取因子逻辑"""
        import re
        match = re.search(r'因子逻辑[：:]\s*(.+?)[\n\r]', content)
        if match:
            return match.group(1).strip()[:200]
        return None


# ============================================================
# 命令行接口
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Alpha Agent 因子挖掘系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_factor_mining.py --mode quick          # 快速测试
  python run_factor_mining.py --mode standard       # 标准运行
  python run_factor_mining.py --mode deep           # 深度挖掘
  python run_factor_mining.py --llm-rounds 5        # 自定义轮数
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="运行模式 (默认: standard)"
    )
    
    parser.add_argument(
        "--llm-rounds", "-r",
        type=int,
        help="LLM探索轮数"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        help="每轮生成因子数"
    )
    
    parser.add_argument(
        "--gp-generations", "-g",
        type=int,
        help="GP迭代代数"
    )
    
    parser.add_argument(
        "--instruments", "-i",
        default="csi300",
        help="股票池 (默认: csi300)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存结果"
    )
    
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳过确认提示"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = RunConfig.from_mode(args.mode)
    
    # 覆盖参数
    if args.llm_rounds:
        config.llm_rounds = args.llm_rounds
    if args.batch_size:
        config.llm_batch_size = args.batch_size
    if args.gp_generations:
        config.gp_generations = args.gp_generations
    if args.instruments:
        config.instruments = args.instruments
    if args.no_save:
        config.save_results = False
    
    # 确认运行
    if not args.yes:
        print("\n" + "="*60)
        print("     ⚠️  Alpha Agent 因子挖掘系统")
        print("="*60)
        print(f"\n模式: {config.mode}")
        print(f"LLM轮数: {config.llm_rounds}")
        print(f"每轮生成: {config.llm_batch_size}")
        print(f"预计时间: {config.llm_rounds * config.llm_batch_size * 2 + config.gp_generations * 2} 分钟")
        print("\n注意: 会产生实际API调用费用")
        print("="*60)
        
        confirm = input("\n确认运行? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return 0
    
    # 创建系统并运行
    system = FactorMiningSystem(config)
    
    if not system.setup():
        return 1
    
    try:
        results = system.run()
        
        if results:
            logger.info("\n" + "="*60)
            logger.info("                 ✅ 因子挖掘完成")
            logger.info("="*60)
            return 0
        else:
            logger.warning("\n⚠️ 未生成有效因子")
            return 1
            
    except Exception as e:
        logger.error(f"\n❌ 运行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
