#!/usr/bin/env python3
"""
合并LoRA适配器到基础模型
将LoRA适配器与基础模型合并为完整模型，用于vLLM部署
"""

import os
import sys
import argparse
from pathlib import Path

def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
):
    """合并LoRA适配器到基础模型"""
    
    print("=" * 60)
    print("LoRA适配器合并")
    print("=" * 60)
    print(f"基础模型: {base_model_path}")
    print(f"适配器路径: {adapter_path}")
    print(f"输出路径: {output_path}")
    print("=" * 60 + "\n")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError as e:
        print(f"导入错误: {e}")
        print("\n请先安装依赖:")
        print("  pip install transformers peft torch")
        return False
    
    # 检查路径
    base_model_path = Path(base_model_path)
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    
    if not base_model_path.exists():
        print(f"✗ 基础模型不存在: {base_model_path}")
        return False
    
    if not adapter_path.exists():
        print(f"✗ 适配器不存在: {adapter_path}")
        return False
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("步骤1: 加载基础模型...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ 基础模型加载成功")
    except Exception as e:
        print(f"✗ 基础模型加载失败: {e}")
        return False
    
    print("\n步骤2: 加载LoRA适配器...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
        )
        print("✓ LoRA适配器加载成功")
    except Exception as e:
        print(f"✗ LoRA适配器加载失败: {e}")
        return False
    
    print("\n步骤3: 合并模型...")
    try:
        merged_model = model.merge_and_unload()
        print("✓ 模型合并成功")
    except Exception as e:
        print(f"✗ 模型合并失败: {e}")
        return False
    
    print("\n步骤4: 保存合并后的模型...")
    try:
        merged_model.save_pretrained(str(output_path))
        print(f"✓ 模型保存成功: {output_path}")
    except Exception as e:
        print(f"✗ 模型保存失败: {e}")
        return False
    
    print("\n步骤5: 保存tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path),
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(str(output_path))
        print(f"✓ Tokenizer保存成功: {output_path}")
    except Exception as e:
        print(f"✗ Tokenizer保存失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("合并完成!")
    print("=" * 60)
    print(f"合并后的模型位于: {output_path}")
    print("\n现在可以使用vLLM部署:")
    print(f"  python deploy_vllm.py --model-path {output_path}")
    print("=" * 60)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="合并LoRA适配器")
    parser.add_argument(
        "--base-model",
        type=str,
        default="./models/Qwen/Qwen3-8B",
        help="基础模型路径"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./models/qwen-news-classifier",
        help="LoRA适配器路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/qwen-news-classifier-merged",
        help="输出路径"
    )
    
    args = parser.parse_args()
    
    success = merge_lora_adapter(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        output_path=args.output,
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
