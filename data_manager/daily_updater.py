"""
工业级每日数据更新器
- 并行获取 + 并行处理
- 增量更新
- 数据校验
- 自动重试
- 完整日志

性能目标:
- 沪深300: < 60秒
- 中证500: < 90秒
- 全市场: < 10分钟
"""

import os
import sys
import json
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

from .data_fetcher import DataFetcher
from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_update.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class DailyUpdater:
    """工业级每日更新器"""
    
    # 股票池配置
    POOL_CONFIG = {
        "csi300": {"index": "000300", "topk": 50},
        "csi500": {"index": "000905", "topk": 80},
        "csi1000": {"index": "000852", "topk": 100},
        "csi800": {"index": "000906", "topk": 80},
        "sse50": {"index": "000016", "topk": 20},
    }
    
    def __init__(self,
                 qlib_data_dir: str = "~/.qlib/qlib_data/cn_data",
                 cache_dir: str = "./data/cache",
                 config_file: str = "./data/update_config.json",
                 max_workers: int = 15):
        """
        初始化
        
        Args:
            qlib_data_dir: Qlib 数据目录
            cache_dir: 缓存目录
            config_file: 配置文件
            max_workers: 最大并行数
        """
        self.qlib_data_dir = os.path.expanduser(qlib_data_dir)
        self.cache_dir = cache_dir
        self.config_file = config_file
        self.max_workers = max_workers
        
        # 初始化组件
        self.fetcher = DataFetcher(
            cache_dir=cache_dir,
            max_workers=max_workers
        )
        self.processor = DataProcessor(
            qlib_data_dir=qlib_data_dir,
            max_workers=max_workers
        )
        self.engineer = FeatureEngineer()
        
        # 加载配置
        self.config = self._load_config()
        
        # 创建目录
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config_file) if os.path.dirname(config_file) else ".", exist_ok=True)
    
    def _load_config(self) -> Dict:
        """加载配置"""
        default = {
            "last_update": None,
            "update_days": 5,
            "pools": ["csi300"],
            "compute_factors": False,
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    default.update(config)
            except:
                pass
        
        return default
    
    def _save_config(self):
        """保存配置"""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    # ============ 主更新流程 ============
    
    def run_update(self,
                   pools: List[str] = None,
                   days: int = None,
                   full_update: bool = False,
                   compute_factors: bool = False) -> Dict:
        """
        运行数据更新
        
        Args:
            pools: 股票池列表，如 ["csi300", "csi500"]
            days: 更新最近N天，默认5天
            full_update: 是否全量更新
            compute_factors: 是否计算高级因子
            
        Returns:
            更新结果
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("开始数据更新")
        logger.info("=" * 60)
        
        result = {
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "pools": [],
            "total_stocks": 0,
            "fetched": 0,
            "processed": 0,
            "failed": 0,
            "elapsed": 0,
        }
        
        try:
            # 1. 确定参数
            if pools is None:
                pools = self.config.get("pools", ["csi300"])
            if days is None:
                days = self.config.get("update_days", 5)
            
            # 2. 确定日期范围
            if full_update:
                start_date = "20080101"
                logger.info("模式: 全量更新")
            else:
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
                logger.info(f"模式: 增量更新 (最近 {days} 天)")
            
            end_date = datetime.now().strftime("%Y%m%d")
            logger.info(f"日期范围: {start_date} ~ {end_date}")
            
            # 3. 更新交易日历
            logger.info("\n[1/4] 更新交易日历...")
            self._update_calendar()
            
            # 4. 获取所有股票
            logger.info("\n[2/4] 获取股票列表...")
            all_stocks = set()
            
            for pool in pools:
                if pool in self.POOL_CONFIG:
                    index_code = self.POOL_CONFIG[pool]["index"]
                else:
                    index_code = pool
                
                stocks = self.fetcher.fetch_index_stocks(index_code)
                logger.info(f"  {pool}: {len(stocks)} 只")
                all_stocks.update(stocks)
                result["pools"].append({"name": pool, "count": len(stocks)})
            
            all_stocks = list(all_stocks)
            result["total_stocks"] = len(all_stocks)
            logger.info(f"  总计: {len(all_stocks)} 只 (去重后)")
            
            # 5. 并行获取数据
            logger.info("\n[3/4] 获取股票数据...")
            data = self.fetcher.fetch_stocks_parallel(
                all_stocks, start_date, end_date,
                show_progress=True
            )
            result["fetched"] = len(data)
            
            # 6. 并行处理数据
            logger.info("\n[4/4] 处理数据...")
            process_result = self.processor.process_stocks_parallel(
                data, show_progress=True
            )
            result["processed"] = process_result["success"]
            result["failed"] = process_result["failed"]
            
            # 7. 更新股票池文件
            logger.info("\n更新股票池文件...")
            for pool in pools:
                if pool in self.POOL_CONFIG:
                    index_code = self.POOL_CONFIG[pool]["index"]
                    stocks = self.fetcher.fetch_index_stocks(index_code)
                    self.processor.update_instruments(pool, stocks)
            
            # 8. 计算高级因子（可选）
            if compute_factors:
                logger.info("\n计算高级因子...")
                self._compute_factors(data)
            
            # 9. 更新配置
            self.config["last_update"] = datetime.now().isoformat()
            self._save_config()
            
            result["status"] = "success"
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"更新失败: {e}")
        
        result["elapsed"] = time.time() - start_time
        result["end_time"] = datetime.now().isoformat()
        
        # 打印摘要
        self._print_summary(result)
        
        return result
    
    def _update_calendar(self):
        """更新交易日历"""
        try:
            dates = self.fetcher.fetch_trade_dates("20080101")
            if dates:
                self.processor.update_calendar(dates)
                logger.info(f"  交易日历: {len(dates)} 天")
        except Exception as e:
            logger.error(f"  更新日历失败: {e}")
    
    def _compute_factors(self, data: Dict[str, pd.DataFrame]):
        """计算高级因子"""
        import pandas as pd
        
        for symbol, df in data.items():
            try:
                df_factors = self.engineer.compute_all_factors(df)
                # 保存因子（可选）
            except Exception as e:
                logger.warning(f"  {symbol} 因子计算失败: {e}")
    
    def _print_summary(self, result: Dict):
        """打印更新摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("更新完成!")
        logger.info("=" * 60)
        logger.info(f"状态: {result['status']}")
        logger.info(f"耗时: {result['elapsed']:.1f} 秒")
        logger.info(f"股票总数: {result['total_stocks']}")
        logger.info(f"获取成功: {result['fetched']}")
        logger.info(f"处理成功: {result['processed']}")
        logger.info(f"失败: {result['failed']}")
        
        if result["pools"]:
            logger.info("\n股票池:")
            for pool in result["pools"]:
                logger.info(f"  {pool['name']}: {pool['count']} 只")
        
        # 性能统计
        if result["elapsed"] > 0 and result["total_stocks"] > 0:
            speed = result["total_stocks"] / result["elapsed"]
            logger.info(f"\n处理速度: {speed:.1f} 只/秒")
        
        logger.info("=" * 60)
    
    # ============ 快捷方法 ============
    
    def update_csi300(self, days: int = 5) -> Dict:
        """更新沪深300"""
        return self.run_update(pools=["csi300"], days=days)
    
    def update_csi500(self, days: int = 5) -> Dict:
        """更新中证500"""
        return self.run_update(pools=["csi500"], days=days)
    
    def update_all(self, days: int = 5) -> Dict:
        """更新所有主要指数"""
        return self.run_update(pools=["csi300", "csi500", "csi1000"], days=days)
    
    def full_update(self, pools: List[str] = None) -> Dict:
        """全量更新"""
        if pools is None:
            pools = ["csi300", "csi500"]
        return self.run_update(pools=pools, full_update=True)
    
    # ============ 数据检查 ============
    
    def check_data(self) -> Dict:
        """检查数据状态"""
        result = self.processor.check_data_integrity()
        
        logger.info("\n数据检查结果:")
        logger.info(f"  总股票数: {result['total']}")
        logger.info(f"  完整: {result['complete']}")
        logger.info(f"  不完整: {len(result['incomplete'])}")
        
        if result["issues"]:
            logger.info(f"  问题数: {len(result['issues'])}")
        
        return result
    
    def get_status(self) -> Dict:
        """获取更新状态"""
        return {
            "last_update": self.config.get("last_update"),
            "pools": self.config.get("pools"),
            "update_days": self.config.get("update_days"),
        }


# ============ 命令行入口 ============

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="工业级数据更新工具")
    parser.add_argument("--pools", nargs="+", default=["csi300"], 
                       help="股票池: csi300, csi500, csi1000")
    parser.add_argument("--days", type=int, default=5, help="更新天数")
    parser.add_argument("--full", action="store_true", help="全量更新")
    parser.add_argument("--factors", action="store_true", help="计算高级因子")
    parser.add_argument("--check", action="store_true", help="检查数据")
    parser.add_argument("--status", action="store_true", help="查看状态")
    parser.add_argument("--workers", type=int, default=15, help="并行数")
    
    args = parser.parse_args()
    
    updater = DailyUpdaterPro(max_workers=args.workers)
    
    if args.status:
        status = updater.get_status()
        print(f"上次更新: {status['last_update']}")
        print(f"股票池: {status['pools']}")
        return
    
    if args.check:
        updater.check_data()
        return
    
    updater.run_update(
        pools=args.pools,
        days=args.days,
        full_update=args.full,
        compute_factors=args.factors
    )


if __name__ == "__main__":
    main()
