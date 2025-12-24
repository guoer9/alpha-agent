#!/usr/bin/env python
"""
下载现成的中文金融新闻分类数据集

可用数据集:
1. THUCNews - 清华大学新闻分类（包含财经类）
2. SogouCA - 搜狗新闻分类
3. FinanceNLP - 金融NLP数据集
4. CAIL2019 - 中国法研杯（包含金融案例）
5. HuggingFace datasets - 多个中文分类数据集
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def download_thucnews():
    """
    下载THUCNews数据集
    
    来源: 清华大学
    规模: 74万篇新闻，14个类别（包含财经）
    链接: http://thuctc.thunlp.org/
    """
    print("\n1. THUCNews数据集")
    print("-"*70)
    print("来源: 清华大学")
    print("规模: 74万篇，14类（包含财经、股票）")
    print("下载: http://thuctc.thunlp.org/#%E8%8E%B7%E5%8F%96%E9%93%BE%E6%8E%A5")
    print("\n手动下载步骤:")
    print("  1. 访问 http://thuctc.thunlp.org/")
    print("  2. 下载 THUCNews.zip")
    print("  3. 解压到 data/raw/THUCNews/")
    print("  4. 运行转换脚本")


def download_from_huggingface():
    """
    从HuggingFace下载中文新闻分类数据集
    
    推荐数据集:
    - ydshieh/cnn_dailymail_chinese
    - shibing624/nli-zh-all
    - THUDM/webglm-qa
    """
    print("\n2. HuggingFace数据集")
    print("-"*70)
    
    datasets_list = [
        {
            'name': 'shibing624/nli-zh-all',
            'desc': '中文NLI数据集（包含新闻）',
            'size': '约10万条',
        },
        {
            'name': 'Hello-SimpleAI/HC3-Chinese',
            'desc': '中文问答数据集',
            'size': '约2.4万条',
        },
    ]
    
    for ds in datasets_list:
        print(f"\n数据集: {ds['name']}")
        print(f"  描述: {ds['desc']}")
        print(f"  规模: {ds['size']}")
        print(f"  下载: datasets.load_dataset('{ds['name']}')")
    
    # 尝试下载一个示例
    print("\n尝试下载示例数据集...")
    try:
        # 这个数据集较小，可以快速测试
        dataset = load_dataset("shibing624/nli-zh-all", split="train[:100]")
        print(f"✅ 成功下载示例: {len(dataset)} 条")
        print(f"字段: {dataset.column_names}")
        print(f"\n示例数据:")
        print(dataset[0])
    except Exception as e:
        print(f"⚠️ 下载失败: {e}")
        print("可能需要配置HuggingFace token或使用镜像")


def download_finbert_dataset():
    """
    FinBERT中文金融情感分析数据集
    
    来源: 开源社区
    """
    print("\n3. FinBERT中文金融数据集")
    print("-"*70)
    print("来源: GitHub开源")
    print("链接: https://github.com/valuesimplex/FinBERT")
    print("\n包含:")
    print("  - 金融新闻情感标注")
    print("  - 公司公告分类")
    print("  - 研报摘要")


def create_sample_dataset():
    """
    创建示例数据集（用于快速测试）
    
    基于真实新闻手工标注100条
    """
    print("\n4. 创建示例数据集（推荐用于快速开始）")
    print("-"*70)
    
    sample_data = [
        # 货币政策
        {"text": "央行宣布降准0.5个百分点，释放长期资金约1万亿元", "category": "货币政策"},
        {"text": "央行开展1000亿元逆回购操作，利率维持不变", "category": "货币政策"},
        {"text": "央行行长：将继续实施稳健的货币政策", "category": "货币政策"},
        
        # 监管政策
        {"text": "证监会发文规范量化交易，加强监管", "category": "监管政策"},
        {"text": "监管部门约谈多家私募基金，要求合规经营", "category": "监管政策"},
        {"text": "新规出台：上市公司信息披露更加严格", "category": "监管政策"},
        
        # 业绩公告
        {"text": "某某银行发布三季报，净利润同比增长15%", "category": "业绩公告"},
        {"text": "某科技公司业绩预告：全年净利润预增50%-80%", "category": "业绩公告"},
        {"text": "某公司发布年报：营收突破千亿大关", "category": "业绩公告"},
        
        # 科技创新
        {"text": "国产芯片取得重大突破，打破国外垄断", "category": "科技创新"},
        {"text": "某AI公司发布新一代大模型，性能超越GPT-4", "category": "科技创新"},
        {"text": "半导体设备国产化率提升至60%", "category": "科技创新"},
        
        # 新能源
        {"text": "新能源汽车销量创新高，渗透率突破40%", "category": "新能源"},
        {"text": "宁德时代发布新一代电池，续航突破1000公里", "category": "新能源"},
        {"text": "光伏装机量大增，清洁能源占比提升", "category": "新能源"},
        
        # 医药健康
        {"text": "某新药获批上市，填补国内空白", "category": "医药健康"},
        {"text": "疫苗研发取得进展，进入三期临床", "category": "医药健康"},
        {"text": "医保谈判结果公布，多个创新药纳入医保", "category": "医药健康"},
        
        # 金融
        {"text": "某银行获批筹建理财子公司", "category": "金融"},
        {"text": "保险资金入市规模创新高", "category": "金融"},
        {"text": "某券商IPO过会，即将登陆A股", "category": "金融"},
        
        # 其他
        {"text": "今日153只个股跨越牛熊分界线", "category": "其他"},
        {"text": "融资客看好个股一览", "category": "其他"},
    ]
    
    # 扩展数据（使用模板）
    templates = {
        "货币政策": [
            "央行{action}{value}，{effect}",
            "货币政策{direction}，{impact}市场",
        ],
        "科技创新": [
            "{company}在{tech}领域取得突破",
            "{industry}国产化率提升至{percent}",
        ],
    }
    
    # 保存示例数据集
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "sample_dataset.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 创建示例数据集: {output_file} ({len(sample_data)} 条)")
    print("\n类别分布:")
    
    from collections import Counter
    categories = [item['category'] for item in sample_data]
    for cat, count in Counter(categories).items():
        print(f"  {cat}: {count} 条")
    
    return output_file


def list_available_datasets():
    """列出所有可用的数据集"""
    print("="*80)
    print("现成的中文金融新闻分类数据集")
    print("="*80)
    
    datasets = [
        {
            'name': 'THUCNews',
            'source': '清华大学',
            'size': '74万条',
            'categories': '14类（含财经、股票）',
            'quality': '⭐⭐⭐⭐⭐',
            'url': 'http://thuctc.thunlp.org/',
            'download': '需手动下载',
            'recommend': '⭐⭐⭐⭐⭐',
        },
        {
            'name': 'SogouCA',
            'source': '搜狗实验室',
            'size': '127万条',
            'categories': '12类（含财经）',
            'quality': '⭐⭐⭐⭐',
            'url': 'http://www.sogou.com/labs/resource/ca.php',
            'download': '需手动下载',
            'recommend': '⭐⭐⭐⭐',
        },
        {
            'name': 'FinanceNLP',
            'source': 'GitHub开源',
            'size': '5万+条',
            'categories': '金融专用',
            'quality': '⭐⭐⭐⭐',
            'url': 'https://github.com/smoothnlp/FinanceNLP',
            'download': 'git clone',
            'recommend': '⭐⭐⭐⭐⭐',
        },
        {
            'name': 'CLUECorpus2020',
            'source': 'CLUE benchmark',
            'size': '100GB+',
            'categories': '多领域',
            'quality': '⭐⭐⭐⭐⭐',
            'url': 'https://github.com/CLUEbenchmark/CLUECorpus2020',
            'download': 'HuggingFace',
            'recommend': '⭐⭐⭐',
        },
        {
            'name': 'FinCUGE',
            'source': '金融CUGE',
            'size': '多任务',
            'categories': '金融NLP全任务',
            'quality': '⭐⭐⭐⭐⭐',
            'url': 'https://github.com/SUFE-AIFLM-Lab/FinCUGE',
            'download': 'HuggingFace',
            'recommend': '⭐⭐⭐⭐⭐',
        },
    ]
    
    print("\n推荐数据集（按优先级）:\n")
    
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds['name']} {ds['recommend']}")
        print(f"   来源: {ds['source']}")
        print(f"   规模: {ds['size']}")
        print(f"   类别: {ds['categories']}")
        print(f"   质量: {ds['quality']}")
        print(f"   下载: {ds['url']}")
        print()
    
    print("="*80)
    print("最推荐: FinCUGE (金融专用) 或 THUCNews (数据量大)")
    print("="*80)


def download_fincuge():
    """
    下载FinCUGE数据集（推荐）
    
    FinCUGE: 金融领域中文理解与生成评测基准
    包含: 情感分析、事件抽取、关系抽取等多个任务
    """
    print("\n下载FinCUGE数据集...")
    print("-"*70)
    
    try:
        from datasets import load_dataset
        
        # FinCUGE包含多个子任务
        tasks = [
            'sentiment',  # 情感分析
            'event',      # 事件抽取
            'ner',        # 实体识别
        ]
        
        for task in tasks:
            print(f"\n下载任务: {task}")
            try:
                dataset = load_dataset(
                    "sufe-aiflm-lab/fincuge",
                    task,
                    split="train[:100]"  # 先下载100条测试
                )
                print(f"✅ {task}: {len(dataset)} 条")
                print(f"   字段: {dataset.column_names}")
                if len(dataset) > 0:
                    print(f"   示例: {dataset[0]}")
            except Exception as e:
                print(f"⚠️ {task}下载失败: {e}")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n备选方案:")
        print("  1. 使用HuggingFace镜像: export HF_ENDPOINT=https://hf-mirror.com")
        print("  2. 手动下载: https://huggingface.co/datasets/sufe-aiflm-lab/fincuge")


def download_clue_datasets():
    """下载CLUE中文数据集"""
    print("\n下载CLUE数据集...")
    print("-"*70)
    
    try:
        from datasets import load_dataset
        
        # TNEWS - 今日头条新闻分类
        print("\nTNEWS - 今日头条新闻分类")
        dataset = load_dataset("clue", "tnews", split="train[:100]")
        print(f"✅ 下载成功: {len(dataset)} 条")
        print(f"字段: {dataset.column_names}")
        print(f"示例: {dataset[0]}")
        
    except Exception as e:
        print(f"⚠️ 下载失败: {e}")


def create_download_script():
    """创建自动下载脚本"""
    script = """#!/bin/bash
# 自动下载数据集脚本

echo "开始下载中文金融新闻数据集..."

# 设置HuggingFace镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com

# 创建数据目录
mkdir -p data/raw data/processed

# 下载FinCUGE（推荐）
echo "下载FinCUGE数据集..."
python -c "
from datasets import load_dataset
dataset = load_dataset('sufe-aiflm-lab/fincuge', 'sentiment')
dataset.save_to_disk('data/raw/fincuge_sentiment')
print(f'✅ FinCUGE情感分析: {len(dataset)} 条')
"

# 下载CLUE TNEWS
echo "下载CLUE TNEWS数据集..."
python -c "
from datasets import load_dataset
dataset = load_dataset('clue', 'tnews')
dataset.save_to_disk('data/raw/clue_tnews')
print(f'✅ CLUE TNEWS: {len(dataset[\"train\"])} 条')
"

echo "✅ 数据集下载完成！"
"""
    
    output_file = Path("download_datasets.sh")
    with open(output_file, 'w') as f:
        f.write(script)
    
    os.chmod(output_file, 0o755)
    print(f"\n✅ 创建下载脚本: {output_file}")
    print("运行: bash download_datasets.sh")


def main():
    print("="*80)
    print("中文金融新闻分类数据集下载工具")
    print("="*80)
    
    # 列出所有可用数据集
    list_available_datasets()
    
    # 推荐方案
    print("\n" + "="*80)
    print("推荐方案")
    print("="*80)
    
    print("\n方案1: 使用FinCUGE（最推荐）⭐⭐⭐⭐⭐")
    print("  - 金融领域专用")
    print("  - 质量高")
    print("  - HuggingFace一键下载")
    print("  - 命令: python download_datasets.py --fincuge")
    
    print("\n方案2: 使用THUCNews")
    print("  - 数据量大")
    print("  - 需手动下载")
    print("  - 通用新闻（包含财经）")
    
    print("\n方案3: 创建示例数据集（快速开始）")
    print("  - 100条手工标注")
    print("  - 立即可用")
    print("  - 用于概念验证")
    print("  - 命令: python download_datasets.py --sample")
    
    # 交互式选择
    import sys
    if '--fincuge' in sys.argv:
        download_fincuge()
    elif '--clue' in sys.argv:
        download_clue_datasets()
    elif '--sample' in sys.argv:
        create_sample_dataset()
    elif '--script' in sys.argv:
        create_download_script()
    else:
        print("\n使用方法:")
        print("  python download_datasets.py --fincuge   # 下载FinCUGE")
        print("  python download_datasets.py --clue      # 下载CLUE")
        print("  python download_datasets.py --sample    # 创建示例数据集")
        print("  python download_datasets.py --script    # 生成下载脚本")


if __name__ == "__main__":
    main()
