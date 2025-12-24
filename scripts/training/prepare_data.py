#!/usr/bin/env python
"""
数据准备脚本 - 新闻分类数据集

功能:
1. 收集新闻数据（AkShare）
2. 标注数据（半自动）
3. 转换为训练格式（使用chat template）
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import train_test_split


def collect_news_data(days: int = 30, output_dir: str = "data/raw"):
    """收集新闻数据"""
    print(f"收集最近{days}天的新闻数据...")
    
    try:
        import akshare as ak
    except ImportError:
        print("请安装: pip install akshare")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_news = []
    
    # 获取新闻
    try:
        news = ak.stock_news_em(symbol="全部")
        print(f"✅ 获取新闻: {len(news)} 条")
        
        for _, row in news.iterrows():
            all_news.append({
                'text': row['新闻标题'],
                'content': row.get('新闻内容', ''),
                'date': str(row['发布时间']),
                'source': row.get('文章来源', ''),
                'type': 'news',
            })
    except Exception as e:
        print(f"获取新闻失败: {e}")
    
    # 获取公告
    try:
        notices = ak.stock_notice_report()
        print(f"✅ 获取公告: {len(notices)} 条")
        
        for _, row in notices.iterrows():
            all_news.append({
                'text': row['公告标题'],
                'content': '',
                'date': str(row.get('公告日期', '')),
                'source': row.get('证券简称', ''),
                'type': 'announcement',
            })
    except Exception as e:
        print(f"获取公告失败: {e}")
    
    # 保存
    output_file = output_path / f"news_raw_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_news:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 保存原始数据: {output_file} ({len(all_news)} 条)")
    return output_file


def auto_annotate(input_file: str, output_file: str = None):
    """自动标注（基于关键词）"""
    print(f"\n自动标注数据...")
    
    if output_file is None:
        output_file = str(Path(input_file).parent / "annotated.jsonl")
    
    # 类别关键词
    category_keywords = {
        "货币政策": ["降准", "降息", "加息", "流动性", "货币政策", "央行", "利率"],
        "监管政策": ["监管", "规范", "约谈", "整顿", "合规", "证监会", "处罚"],
        "业绩公告": ["净利润", "营收", "业绩", "财报", "季报", "年报", "预告"],
        "科技创新": ["芯片", "半导体", "AI", "人工智能", "科技", "研发", "专利"],
        "新能源": ["新能源", "光伏", "锂电", "电池", "充电桩", "储能"],
        "医药健康": ["医药", "疫苗", "生物", "医疗", "药品", "临床"],
        "金融": ["银行", "保险", "证券", "信托", "基金", "理财"],
        "其他": [],
    }
    
    annotated = []
    stats = {cat: 0 for cat in category_keywords.keys()}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            
            # 匹配类别
            matched_categories = []
            for category, keywords in category_keywords.items():
                if category == "其他":
                    continue
                if any(kw in text for kw in keywords):
                    matched_categories.append(category)
            
            # 选择第一个匹配的类别
            category = matched_categories[0] if matched_categories else "其他"
            
            annotated.append({
                'text': text,
                'category': category,
                'date': item.get('date', ''),
                'source': item.get('source', ''),
            })
            
            stats[category] += 1
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in annotated:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 标注完成: {output_file}")
    print(f"\n类别分布:")
    for cat, count in stats.items():
        print(f"  {cat}: {count} 条")
    
    return output_file


def generate_training_dataset(
    annotated_file: str,
    output_dir: str = "data/processed",
    test_size: float = 0.2,
):
    """
    生成训练数据集（使用chat template格式）
    
    格式: 对话式分类
    User: 请分析以下新闻的类别：{新闻文本}
    Assistant: 这条新闻属于：{类别}
    """
    print(f"\n生成训练数据集...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 读取标注数据
    data = []
    with open(annotated_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"总数据量: {len(data)} 条")
    
    # 划分训练/验证/测试集
    train_data, temp_data = train_test_split(data, test_size=test_size*2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
    # 转换为chat格式
    def to_chat_format(item):
        """转换为chat template格式"""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"请分析以下财经新闻的类别：\n\n{item['text']}\n\n从以下类别中选择一个：货币政策、监管政策、业绩公告、科技创新、新能源、医药健康、金融、其他"
                },
                {
                    "role": "assistant",
                    "content": f"这条新闻属于：{item['category']}"
                }
            ]
        }
    
    # 保存
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        output_file = output_path / f"{split_name}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                chat_item = to_chat_format(item)
                f.write(json.dumps(chat_item, ensure_ascii=False) + '\n')
        
        print(f"✅ 保存{split_name}集: {output_file}")
    
    print("\n数据集生成完成！")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true', help='收集新闻数据')
    parser.add_argument('--annotate', action='store_true', help='自动标注')
    parser.add_argument('--generate-dataset', action='store_true', help='生成训练数据集')
    parser.add_argument('--days', type=int, default=30, help='收集天数')
    
    args = parser.parse_args()
    
    if args.collect:
        raw_file = collect_news_data(days=args.days)
        
        if args.annotate and raw_file:
            annotated_file = auto_annotate(raw_file)
            
            if args.generate_dataset:
                generate_training_dataset(annotated_file)
    
    elif args.generate_dataset:
        # 使用已有的标注文件
        annotated_file = "data/raw/annotated.jsonl"
        if Path(annotated_file).exists():
            generate_training_dataset(annotated_file)
        else:
            print(f"❌ 标注文件不存在: {annotated_file}")
            print("请先运行: python prepare_data.py --collect --annotate")


if __name__ == "__main__":
    main()
