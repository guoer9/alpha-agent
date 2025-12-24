#!/usr/bin/env python
"""
Qwen新闻分类微调训练脚本（参考HuggingFace官方方案）

参考: https://huggingface.co/docs/trl/sft_trainer
模型: Qwen3-8B-Instruct
方法: LoRA + SFTTrainer
任务: 新闻分类（对话格式）
GPU: RTX 5090 (24GB)

官方推荐配置:
- SFTTrainer (专门用于chat模型)
- apply_chat_template (自动处理特殊token)
- Flash Attention 2 (加速)
- bf16混合精度
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import json


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="./models/Qwen/Qwen3-8B")
    use_flash_attention: bool = field(default=True)


@dataclass
class DataArguments:
    train_file: str = field(default="data/processed/train.jsonl")
    val_file: str = field(default="data/processed/val.jsonl")
    max_length: int = field(default=512)


@dataclass
class LoraArguments:
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="all")  # all表示所有linear层


def load_and_prepare_model(model_args):
    """加载模型和tokenizer（官方推荐方式）"""
    print(f"加载模型: {model_args.model_name_or_path}")
    
    # 加载tokenizer（官方推荐配置）
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,  # 使用fast tokenizer
    )
    
    # 设置pad_token（官方推荐）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型（官方推荐配置）
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,  # 5090原生支持bf16
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention else "sdpa",  # sdpa是默认优化
        use_cache=False,  # 训练时禁用cache
    )
    
    # 启用gradient checkpointing（节省显存）
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    print(f"✅ 模型加载完成")
    print(f"   参数量: {model.num_parameters() / 1e9:.2f}B")
    print(f"   设备: {next(model.parameters()).device}")
    print(f"   dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer


def get_lora_config(lora_args):
    """获取LoRA配置（官方推荐）"""
    print("\n配置LoRA...")
    
    # Qwen2.5官方推荐的target modules
    if lora_args.lora_target_modules == "all":
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    else:
        target_modules = lora_args.lora_target_modules.split(",")
    
    # 官方推荐的LoRA配置
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    
    print(f"✅ LoRA配置:")
    print(f"   r={lora_args.lora_r}, alpha={lora_args.lora_alpha}")
    print(f"   target_modules={len(target_modules)}个")
    
    return lora_config


def formatting_func(example, tokenizer):
    """
    格式化函数（官方推荐方式）
    
    SFTTrainer会自动调用apply_chat_template
    我们只需要返回messages格式
    """
    return example["messages"]


def load_and_process_dataset(data_args):
    """
    加载数据集（官方推荐方式）
    
    SFTTrainer会自动处理chat template
    """
    print(f"\n加载数据集...")
    
    # 加载数据
    dataset = load_dataset(
        'json',
        data_files={
            'train': data_args.train_file,
            'validation': data_args.val_file,
        }
    )
    
    print(f"✅ 数据加载完成")
    print(f"   训练集: {len(dataset['train'])} 条")
    print(f"   验证集: {len(dataset['validation'])} 条")
    
    # 检查数据格式
    if len(dataset['train']) > 0:
        print(f"\n数据示例:")
        print(f"   {dataset['train'][0]}")
    
    return dataset


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    training_args: TrainingArguments,
):
    """训练主函数（使用SFTTrainer - 官方推荐）"""
    
    # 1. 加载模型和tokenizer
    model, tokenizer = load_and_prepare_model(model_args)
    
    # 2. 获取LoRA配置
    lora_config = get_lora_config(lora_args)
    
    # 3. 加载数据集
    dataset = load_and_process_dataset(data_args)
    
    # 4. 创建SFTTrainer（官方推荐）
    print("\n创建SFTTrainer...")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,  # 新版trl使用processing_class
        peft_config=lora_config,  # SFTTrainer自动应用LoRA
        formatting_func=lambda x: formatting_func(x, tokenizer),
    )
    
    print(f"✅ SFTTrainer创建完成")
    print(f"   可训练参数: {trainer.model.num_parameters(only_trainable=True) / 1e6:.2f}M")
    
    # 5. 开始训练
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)
    
    trainer.train()
    
    # 6. 保存模型
    print("\n保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"✅ 模型已保存到: {training_args.output_dir}")


def main():
    # 模型配置
    model_args = ModelArguments(
        model_name_or_path="./models/Qwen/Qwen3-8B",  # 使用本地下载的Qwen3模型
        use_flash_attention=False,  # 暂时禁用，未安装flash-attn
    )
    
    # 数据配置
    data_args = DataArguments(
        train_file="data/processed/train.jsonl",
        val_file="data/processed/val.jsonl",
        max_length=512,
    )
    
    # LoRA配置
    lora_args = LoraArguments(
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules="all",
    )
    
    # 训练配置（参考官方推荐，针对RTX 5090优化）
    training_args = TrainingArguments(
        output_dir="./models/qwen-news-classifier",
        
        # 训练参数（官方推荐）
        num_train_epochs=3,
        per_device_train_batch_size=8,  # RTX 5090 32GB显存最大化
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  # 有效batch_size=16
        
        # 学习率（官方推荐范围）
        learning_rate=5e-5,  # 官方推荐1e-5到1e-4
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,  # 官方推荐0.03
        
        # 优化器（官方推荐）
        optim="adamw_torch",  # 或"paged_adamw_8bit"节省显存
        weight_decay=0.01,
        max_grad_norm=1.0,  # 梯度裁剪
        
        # 精度（5090优化）
        bf16=True,  # 5090原生支持
        tf32=True,  # 启用TF32加速
        fp16=False,
        
        # 保存策略
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,  # 官方推荐
        metric_for_best_model="eval_loss",
        
        # 评估策略
        eval_strategy="steps",
        eval_steps=100,
        
        # 日志
        logging_steps=10,
        logging_dir="./logs",
        logging_first_step=True,
        report_to=["tensorboard"],
        
        # 其他（官方推荐）
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 新版推荐
        seed=42,
        data_seed=42,
        dataloader_num_workers=8,  # 25核CPU加速数据加载
        dataloader_pin_memory=True,  # 加速数据传输
        group_by_length=False,  # 分类任务不需要
        ddp_find_unused_parameters=False,
    )
    
    # 开始训练
    train(model_args, data_args, lora_args, training_args)


if __name__ == "__main__":
    main()
