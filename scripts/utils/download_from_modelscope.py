#!/usr/bin/env python3
"""
从魔塔ModelScope下载Qwen模型
国内下载速度快，无需代理
"""

import os
import sys
import argparse
from pathlib import Path

def download_from_modelscope(
    model_id: str,
    output_dir: str,
    revision: str = "master",
):
    """从ModelScope下载模型"""
    
    print("=" * 60)
    print("从魔塔ModelScope下载模型")
    print("=" * 60)
    print(f"模型ID: {model_id}")
    print(f"输出目录: {output_dir}")
    print(f"版本: {revision}")
    print("=" * 60 + "\n")
    
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("✗ modelscope未安装")
        print("\n安装命令:")
        print("  pip install modelscope")
        return False
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载 {model_id}...")
    print("魔塔下载速度快，无需代理\n")
    
    try:
        model_dir = snapshot_download(
            model_id=model_id,
            cache_dir=str(output_path.parent),
            revision=revision,
        )
        
        print("\n" + "=" * 60)
        print("✓ 下载完成!")
        print("=" * 60)
        print(f"模型位于: {model_dir}")
        
        # 如果下载到cache目录，创建软链接
        if str(output_path) != model_dir:
            print(f"\n创建软链接: {output_path} -> {model_dir}")
            if output_path.exists() and output_path.is_symlink():
                output_path.unlink()
            if not output_path.exists():
                output_path.symlink_to(model_dir)
        
        print("\n下一步:")
        print("1. 合并LoRA适配器:")
        print(f"   python merge_lora.py --base-model {output_path}")
        print("\n2. 或直接使用vLLM部署:")
        print(f"   python deploy_vllm.py --base-model-path {output_path} --enable-lora")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. 模型ID不正确")
        print("3. 磁盘空间不足")
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 确认模型ID: https://modelscope.cn/models")
        print("3. 检查磁盘空间: df -h")
        return False

def main():
    parser = argparse.ArgumentParser(description="从魔塔ModelScope下载模型")
    parser.add_argument(
        "--model-id",
        type=str,
        default="qwen/Qwen3-8B",
        help="ModelScope模型ID (默认: qwen/Qwen3-8B)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/Qwen/Qwen3-8B",
        help="输出目录 (默认: ./models/Qwen/Qwen3-8B)"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="master",
        help="模型版本 (默认: master)"
    )
    
    args = parser.parse_args()
    
    success = download_from_modelscope(
        model_id=args.model_id,
        output_dir=args.output_dir,
        revision=args.revision,
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
