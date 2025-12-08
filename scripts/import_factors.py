#!/usr/bin/env python
"""
因子库导入与验证脚本

功能:
1. 加载所有预定义因子 (300+)
   - 经典因子 (Barra/技术/基本面/量价): 25个
   - Alpha158 (Qlib): 50个
   - Alpha360 (Qlib): 27个
   - WorldQuant 101: 29个
   - 国泰君安 191: 30个
   - Academic Premia: 18个
2. 导入到Milvus向量数据库 (用于RAG检索)
3. 导入到Neo4j知识图谱 (用于GraphRAG)
4. 导入到Redis缓存 (用于快速访问)
5. 使用Qlib数据验证因子IC

使用方法:
    # 导入所有因子到数据库
    python scripts/import_factors.py --import-all
    
    # 只验证因子IC (不导入)
    python scripts/import_factors.py --validate-only
    
    # 导入并验证
    python scripts/import_factors.py --import-all --validate
    
    # 只导入前N个因子 (测试)
    python scripts/import_factors.py --import-all --limit 10
    
    # 按来源导入
    python scripts/import_factors.py --import-milvus
    python scripts/import_factors.py --import-neo4j
    python scripts/import_factors.py --import-redis
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# 因子加载
# ============================================================

def load_all_factors() -> List[Dict]:
    """加载所有预定义因子"""
    # 直接从factors模块导入，避免触发alpha_agent顶层的可选依赖检查
    from alpha_agent.factors.classic_factors import ALL_CLASSIC_FACTORS, ClassicFactor
    from alpha_agent.factors.alpha158 import ALPHA158_FACTORS
    from alpha_agent.factors.alpha360 import ALPHA360_FACTORS
    from alpha_agent.factors.worldquant101 import WORLDQUANT_101_FACTORS
    from alpha_agent.factors.gtja191 import GTJA191_FACTORS
    from alpha_agent.factors.academic_premia import ACADEMIC_PREMIA_FACTORS
    
    def factor_to_dict(f: ClassicFactor, source: str = 'classic') -> Dict:
        """将ClassicFactor转换为字典"""
        return {
            'id': f.id,
            'name': f.name,
            'name_en': getattr(f, 'name_en', f.name),
            'code': f.code,
            'category': f.category.value if hasattr(f.category, 'value') else str(f.category),
            'source': source,
            'description': f.description,
            'tags': f.tags,
            'ic': f.historical_ic,
            'icir': f.historical_icir,
            'reference': f.reference,
            'logic': f.logic,
            'author': getattr(f, 'author', ''),
            'year': getattr(f, 'year', 0),
        }
    
    all_factors = []
    
    # 1. 经典因子 (Barra/技术/基本面/量价)
    logger.info("加载经典因子...")
    for f in ALL_CLASSIC_FACTORS:
        all_factors.append(factor_to_dict(f, 'classic'))
    logger.info(f"  经典因子: {len(ALL_CLASSIC_FACTORS)}")
    
    # 2. Alpha158 (Qlib)
    logger.info("加载Alpha158因子...")
    for f in ALPHA158_FACTORS:
        all_factors.append(factor_to_dict(f, 'qlib_alpha158'))
    logger.info(f"  Alpha158: {len(ALPHA158_FACTORS)}")
    
    # 3. Alpha360 (Qlib)
    logger.info("加载Alpha360因子...")
    for f in ALPHA360_FACTORS:
        all_factors.append(factor_to_dict(f, 'qlib_alpha360'))
    logger.info(f"  Alpha360: {len(ALPHA360_FACTORS)}")
    
    # 4. WorldQuant 101
    logger.info("加载WorldQuant101因子...")
    for f in WORLDQUANT_101_FACTORS:
        all_factors.append(factor_to_dict(f, 'worldquant101'))
    logger.info(f"  WorldQuant101: {len(WORLDQUANT_101_FACTORS)}")
    
    # 5. 国泰君安 191
    logger.info("加载国泰君安191因子...")
    for f in GTJA191_FACTORS:
        all_factors.append(factor_to_dict(f, 'gtja191'))
    logger.info(f"  国泰君安191: {len(GTJA191_FACTORS)}")
    
    # 6. Academic Premia 学术溢价
    logger.info("加载Academic Premia因子...")
    for f in ACADEMIC_PREMIA_FACTORS:
        all_factors.append(factor_to_dict(f, 'academic_premia'))
    logger.info(f"  Academic Premia: {len(ACADEMIC_PREMIA_FACTORS)}")
    
    logger.info(f"✅ 总计加载 {len(all_factors)} 个因子")
    return all_factors


# ============================================================
# Milvus导入
# ============================================================

def import_to_milvus(factors: List[Dict], batch_size: int = 50) -> Tuple[int, int]:
    """导入因子到Milvus向量数据库"""
    from alpha_agent.memory.vector_store import MilvusStore, FactorRecord, MILVUS_AVAILABLE
    
    if not MILVUS_AVAILABLE:
        logger.error("❌ Milvus SDK未安装: pip install pymilvus")
        return 0, len(factors)
    
    logger.info("\n" + "="*50)
    logger.info("导入到Milvus向量数据库")
    logger.info("="*50)
    
    try:
        store = MilvusStore(collection_name="alpha_factors")
        if not store.connect():
            logger.error("❌ Milvus连接失败")
            return 0, len(factors)
        
        # 确保集合存在
        store.create_collection()
        
        success_count = 0
        fail_count = 0
        
        # 批量导入
        for i in range(0, len(factors), batch_size):
            batch = factors[i:i+batch_size]
            
            for f in batch:
                try:
                    # 创建FactorRecord
                    record = FactorRecord(
                        factor_id=f.get('id', ''),
                        name=f.get('name', ''),
                        code=f.get('code', ''),
                        description=f.get('description', ''),
                        ic=f.get('ic', 0),
                        icir=f.get('icir', 0),
                        status='active',
                        tags=f.get('tags', []),
                    )
                    
                    # 插入 (会自动生成embedding)
                    store.insert(record)
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"  导入失败 {f.get('id')}: {e}")
                    fail_count += 1
            
            logger.info(f"  进度: {min(i+batch_size, len(factors))}/{len(factors)}")
        
        logger.info(f"✅ Milvus导入完成: 成功 {success_count}, 失败 {fail_count}")
        return success_count, fail_count
        
    except Exception as e:
        logger.error(f"❌ Milvus导入失败: {e}")
        return 0, len(factors)


def import_to_milvus_simple(factors: List[Dict]) -> Tuple[int, int]:
    """简化版Milvus导入 (不使用embedding)"""
    try:
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        
        logger.info("\n" + "="*50)
        logger.info("导入到Milvus (简化模式)")
        logger.info("="*50)
        
        # 连接
        connections.connect('default', host='localhost', port='19530')
        logger.info("✅ Milvus连接成功")
        
        collection_name = "alpha_factors"
        
        # 删除旧集合
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"  删除旧集合: {collection_name}")
        
        # 创建Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="factor_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="name_en", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64),  # 因子来源
            FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="ic", dtype=DataType.FLOAT),
            FieldSchema(name="icir", dtype=DataType.FLOAT),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),  # 简化embedding
        ]
        
        schema = CollectionSchema(fields, description="Alpha因子库")
        collection = Collection(collection_name, schema)
        logger.info(f"  创建集合: {collection_name}")
        
        # 准备数据
        factor_ids = []
        names = []
        names_en = []
        categories = []
        sources = []
        codes = []
        descriptions = []
        ics = []
        icirs = []
        embeddings = []
        
        for f in factors:
            factor_ids.append(f.get('id', '')[:128])
            names.append(f.get('name', '')[:256])
            names_en.append(f.get('name_en', f.get('name', ''))[:256])
            categories.append(f.get('category', '')[:64])
            sources.append(f.get('source', 'unknown')[:64])
            
            # 截断code
            code = f.get('code', '')
            if len(code) > 4096:
                code = code[:4090] + "..."
            codes.append(code)
            
            # 截断description
            desc = f.get('description', '')
            if len(desc) > 1024:
                desc = desc[:1020] + "..."
            descriptions.append(desc)
            
            ics.append(float(f.get('ic', 0) or 0))
            icirs.append(float(f.get('icir', 0) or 0))
            
            # 简单hash作为embedding (后续可替换为真正的向量)
            code_hash = hash(f.get('code', '') + f.get('description', ''))
            embedding = [float((code_hash >> i) & 1) for i in range(128)]
            embeddings.append(embedding)
        
        # 批量插入
        data = [factor_ids, names, names_en, categories, sources, codes, descriptions, ics, icirs, embeddings]
        collection.insert(data)
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        collection.load()
        
        logger.info(f"✅ Milvus导入完成: {len(factors)} 个因子")
        
        connections.disconnect('default')
        return len(factors), 0
        
    except Exception as e:
        logger.error(f"❌ Milvus导入失败: {e}")
        import traceback
        traceback.print_exc()
        return 0, len(factors)


# ============================================================
# Neo4j导入
# ============================================================

def import_to_neo4j(factors: List[Dict]) -> Tuple[int, int]:
    """导入因子到Neo4j知识图谱"""
    try:
        from neo4j import GraphDatabase
        
        logger.info("\n" + "="*50)
        logger.info("导入到Neo4j知识图谱")
        logger.info("="*50)
        
        driver = GraphDatabase.driver(
            'bolt://localhost:7687',
            auth=('neo4j', 'password')
        )
        
        success_count = 0
        fail_count = 0
        
        with driver.session() as session:
            # 清空旧数据 (可选)
            session.run("MATCH (n:Factor) DETACH DELETE n")
            session.run("MATCH (n:Category) DETACH DELETE n")
            logger.info("  清空旧数据")
            
            # 创建分类节点
            categories = set(f.get('category', 'unknown') for f in factors)
            for cat in categories:
                session.run(
                    "MERGE (c:Category {name: $name})",
                    name=cat
                )
            logger.info(f"  创建分类节点: {len(categories)}")
            
            # 创建因子节点和关系
            for f in factors:
                try:
                    session.run("""
                        MERGE (f:Factor {factor_id: $factor_id})
                        SET f.name = $name,
                            f.code = $code,
                            f.ic = $ic,
                            f.icir = $icir,
                            f.source = $source,
                            f.description = $description
                        WITH f
                        MATCH (c:Category {name: $category})
                        MERGE (f)-[:BELONGS_TO]->(c)
                    """,
                        factor_id=f.get('id', ''),
                        name=f.get('name', ''),
                        code=f.get('code', '')[:2000],  # 限制长度
                        ic=float(f.get('ic', 0) or 0),
                        icir=float(f.get('icir', 0) or 0),
                        source=f.get('source', ''),
                        description=f.get('description', '')[:500],
                        category=f.get('category', 'unknown'),
                    )
                    success_count += 1
                except Exception as e:
                    logger.warning(f"  Neo4j导入失败 {f.get('id')}: {e}")
                    fail_count += 1
            
            # 创建索引
            session.run("CREATE INDEX factor_id_index IF NOT EXISTS FOR (f:Factor) ON (f.factor_id)")
            session.run("CREATE INDEX factor_name_index IF NOT EXISTS FOR (f:Factor) ON (f.name)")
        
        driver.close()
        logger.info(f"✅ Neo4j导入完成: 成功 {success_count}, 失败 {fail_count}")
        return success_count, fail_count
        
    except Exception as e:
        logger.error(f"❌ Neo4j导入失败: {e}")
        return 0, len(factors)


# ============================================================
# Redis缓存
# ============================================================

def import_to_redis(factors: List[Dict]) -> Tuple[int, int]:
    """导入因子到Redis缓存"""
    try:
        import redis
        
        logger.info("\n" + "="*50)
        logger.info("导入到Redis缓存")
        logger.info("="*50)
        
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        
        # 清空旧数据
        keys = r.keys("factor:*")
        if keys:
            r.delete(*keys)
        r.delete("factor_index")
        
        success_count = 0
        
        for f in factors:
            factor_id = f.get('id', '')
            key = f"factor:{factor_id}"
            
            # 存储因子数据
            r.hset(key, mapping={
                'name': f.get('name', ''),
                'category': f.get('category', ''),
                'code': f.get('code', '')[:4000],
                'ic': str(f.get('ic', 0)),
                'icir': str(f.get('icir', 0)),
                'source': f.get('source', ''),
            })
            
            # 添加到索引
            r.sadd("factor_index", factor_id)
            
            # 按分类索引
            r.sadd(f"category:{f.get('category', 'unknown')}", factor_id)
            
            success_count += 1
        
        logger.info(f"✅ Redis导入完成: {success_count} 个因子")
        return success_count, 0
        
    except Exception as e:
        logger.error(f"❌ Redis导入失败: {e}")
        return 0, len(factors)


# ============================================================
# 因子验证
# ============================================================

def validate_factors(factors: List[Dict], limit: int = None) -> pd.DataFrame:
    """验证因子IC"""
    logger.info("\n" + "="*50)
    logger.info("因子IC验证")
    logger.info("="*50)
    
    # 初始化Qlib
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        
        provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        logger.info("✅ Qlib初始化成功")
        
    except Exception as e:
        logger.error(f"❌ Qlib初始化失败: {e}")
        return pd.DataFrame()
    
    # 加载数据
    logger.info("📊 加载市场数据...")
    instruments = D.instruments("csi300")
    fields = ["$close", "$open", "$high", "$low", "$volume"]
    
    df = D.features(
        instruments,
        fields,
        start_time="2022-01-01",
        end_time="2023-12-31",
        freq="day",
    )
    df.columns = ['close', 'open', 'high', 'low', 'volume']
    
    # 计算目标收益
    target = df['close'].groupby(level=0).pct_change(5).shift(-5)
    logger.info(f"  数据量: {len(df):,} 行")
    
    # 验证因子
    from alpha_agent.core.sandbox import Sandbox
    from alpha_agent.core.evaluator import FactorEvaluator
    
    sandbox = Sandbox(timeout_seconds=10)
    evaluator = FactorEvaluator()
    
    results = []
    factors_to_test = factors[:limit] if limit else factors
    
    logger.info(f"开始验证 {len(factors_to_test)} 个因子...")
    
    for i, f in enumerate(factors_to_test):
        factor_id = f.get('id', f'factor_{i}')
        code = f.get('code', '')
        
        if not code or 'def compute_alpha' not in code:
            continue
        
        try:
            # 执行因子
            factor_values, error = sandbox.execute(code, df)
            
            if error or factor_values is None:
                results.append({
                    'id': factor_id,
                    'name': f.get('name', ''),
                    'category': f.get('category', ''),
                    'status': 'error',
                    'ic': None,
                    'icir': None,
                    'error': str(error)[:100] if error else 'empty result',
                })
                continue
            
            # 计算IC
            aligned = pd.concat([factor_values, target], axis=1)
            aligned.columns = ['factor', 'return']
            aligned = aligned.dropna()
            
            if len(aligned) < 100:
                results.append({
                    'id': factor_id,
                    'name': f.get('name', ''),
                    'category': f.get('category', ''),
                    'status': 'insufficient_data',
                    'ic': None,
                    'icir': None,
                })
                continue
            
            eval_result = evaluator.evaluate(aligned['factor'], aligned['return'])
            
            results.append({
                'id': factor_id,
                'name': f.get('name', ''),
                'category': f.get('category', ''),
                'status': eval_result.status.value,
                'ic': eval_result.ic,
                'icir': eval_result.icir,
                'rank_ic': eval_result.rank_ic,
            })
            
            if (i + 1) % 20 == 0:
                logger.info(f"  进度: {i+1}/{len(factors_to_test)}")
                
        except Exception as e:
            results.append({
                'id': factor_id,
                'name': f.get('name', ''),
                'category': f.get('category', ''),
                'status': 'exception',
                'ic': None,
                'icir': None,
                'error': str(e)[:100],
            })
    
    # 生成报告
    results_df = pd.DataFrame(results)
    
    logger.info("\n" + "="*50)
    logger.info("验证结果汇总")
    logger.info("="*50)
    
    if len(results_df) > 0:
        valid = results_df[results_df['ic'].notna()]
        logger.info(f"  总因子数: {len(results_df)}")
        logger.info(f"  有效因子: {len(valid)}")
        
        if len(valid) > 0:
            logger.info(f"  平均IC: {valid['ic'].mean():.4f}")
            logger.info(f"  IC>2%: {(valid['ic'].abs() > 0.02).sum()}")
            logger.info(f"  IC>3%: {(valid['ic'].abs() > 0.03).sum()}")
            
            # 按分类统计
            logger.info("\n按分类统计:")
            for cat in valid['category'].unique():
                cat_df = valid[valid['category'] == cat]
                logger.info(f"  {cat}: {len(cat_df)}个, 平均IC={cat_df['ic'].mean():.4f}")
            
            # Top因子
            logger.info("\nTop 10 因子:")
            top10 = valid.nlargest(10, 'ic')
            for _, row in top10.iterrows():
                logger.info(f"  {row['id']}: IC={row['ic']:.4f}, ICIR={row['icir']:.2f}")
    
    # 保存结果
    output_dir = PROJECT_ROOT / "output" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"factor_validation_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n📁 结果已保存: {output_file}")
    
    return results_df


# ============================================================
# 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="因子库导入与验证")
    
    parser.add_argument("--import-all", action="store_true", help="导入所有因子到数据库")
    parser.add_argument("--import-milvus", action="store_true", help="只导入到Milvus")
    parser.add_argument("--import-neo4j", action="store_true", help="只导入到Neo4j")
    parser.add_argument("--import-redis", action="store_true", help="只导入到Redis")
    parser.add_argument("--validate", action="store_true", help="验证因子IC")
    parser.add_argument("--validate-only", action="store_true", help="只验证不导入")
    parser.add_argument("--limit", type=int, help="限制处理的因子数量")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("     📦 因子库导入与验证工具")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 加载因子
    factors = load_all_factors()
    
    if args.limit:
        factors = factors[:args.limit]
        logger.info(f"限制处理前 {args.limit} 个因子")
    
    # 导入到数据库
    if args.import_all or args.import_milvus:
        import_to_milvus_simple(factors)
    
    if args.import_all or args.import_neo4j:
        import_to_neo4j(factors)
    
    if args.import_all or args.import_redis:
        import_to_redis(factors)
    
    # 验证
    if args.validate or args.validate_only:
        validate_factors(factors, limit=args.limit or 50)
    
    # 如果没有指定任何操作，显示帮助
    if not any([args.import_all, args.import_milvus, args.import_neo4j, 
                args.import_redis, args.validate, args.validate_only]):
        print("\n使用方法:")
        print("  python scripts/import_factors.py --import-all        # 导入所有数据库")
        print("  python scripts/import_factors.py --validate-only     # 只验证IC")
        print("  python scripts/import_factors.py --import-all --validate  # 导入并验证")
        print("  python scripts/import_factors.py --limit 10          # 限制数量")
    
    print("\n" + "="*60)
    print("✅ 完成")
    print("="*60)


if __name__ == "__main__":
    main()
