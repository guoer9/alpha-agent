#!/usr/bin/env python
"""
使用魔塔（ModelScope）国内镜像下载Qwen模型权重

魔塔社区: https://modelscope.cn/
Qwen2.5-7B-Instruct: https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct
"""

import os
import argparse


def download_from_modelscope(model_name: str, cache_dir: str = None):
    """
    从魔塔下载模型
    
    Args:
        model_name: 模型名称，如 "Qwen/Qwen2.5-7B-Instruct"
        cache_dir: 缓存目录，默认为 ./models/
    """
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("正在安装 modelscope...")
        os.system("pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")
        from modelscope import snapshot_download
    
    if cache_dir is None:
        cache_dir = "./models"
    
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"从魔塔（ModelScope）下载模型")
    print(f"=" * 60)
    print(f"模型: {model_name}")
    print(f"保存目录: {cache_dir}")
    print(f"=" * 60)
    
    # 下载模型
    model_dir = snapshot_download(
        model_id=model_name,
        cache_dir=cache_dir,
        revision="master",
    )
    
    print(f"\n✅ 模型下载完成!")
    print(f"   模型路径: {model_dir}")
    
    return model_dir


def main():
    parser = argparse.ArgumentParser(description="从魔塔下载Qwen模型")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen3-8B-Instruct",
        help="模型名称 (默认: Qwen/Qwen3-8B-Instruct)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="模型缓存目录 (默认: ./models)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["0.6b", "1.7b", "4b", "8b", "14b", "32b"],
        default=None,
        help="快速选择模型大小"
    )
    
    args = parser.parse_args()
    
    # 根据size快速选择模型 (Qwen3系列)
    if args.model_size:
        size_map = {
            "0.6b": "Qwen/Qwen3-0.6B",
            "1.7b": "Qwen/Qwen3-1.7B",
            "4b": "Qwen/Qwen3-4B",
            "8b": "Qwen/Qwen3-8B",
            "14b": "Qwen/Qwen3-14B",
            "32b": "Qwen/Qwen3-32B",
        }
        args.model = size_map[args.model_size]
    
    download_from_modelscope(args.model, args.cache_dir)


if __name__ == "__main__":
    main()
