# RTX 5090 部署指南

## 完整方案总结

### ✅ 参考HuggingFace官方方案

**关键改进**:
1. ✅ 使用 `SFTTrainer` (专门用于chat模型)
2. ✅ 使用 `tokenizer.apply_chat_template()` (自动处理特殊token)
3. ✅ 官方推荐的超参数配置
4. ✅ Flash Attention 2 + bf16 (5090优化)

---

## 5090机器部署流程

### Step 1: 打包文件（当前机器）

```bash
cd /Volumes/2tb/mydata/code/Quantitative_trading/qlib_trading

# 打包训练代码
tar -czf qwen_training.tar.gz \
    alpha_agent/training/news_classifier/ \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='models/*' \
    --exclude='logs/*'

# 查看大小
ls -lh qwen_training.tar.gz
```

### Step 2: 传输到5090

```bash
# 使用scp
scp qwen_training.tar.gz user@5090-ip:/home/user/

# 或使用rsync（支持断点续传）
rsync -avz --progress qwen_training.tar.gz user@5090-ip:/home/user/
```

### Step 3: 5090机器环境配置

```bash
# SSH登录5090
ssh user@5090-ip

# 解压
tar -xzf qwen_training.tar.gz
cd alpha_agent/training/news_classifier

# 创建conda环境
conda create -n qwen-train python=3.10 -y
conda activate qwen-train

# 安装PyTorch（CUDA 12.1，5090专用）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install transformers>=4.37.0 accelerate peft trl datasets

# 安装Flash Attention 2（重要！加速30%）
pip install flash-attn --no-build-isolation

# 安装其他工具
pip install tensorboard wandb akshare jieba snownlp
```

### Step 4: 准备数据

```bash
# 方案A: 使用示例数据集（快速测试）
python download_datasets.py --sample
python prepare_data.py --generate-dataset

# 方案B: 下载FinCUGE（推荐）
export HF_ENDPOINT=https://hf-mirror.com  # 国内加速
python download_datasets.py --fincuge

# 方案C: 使用AkShare收集真实数据
python prepare_data.py --collect --annotate --generate-dataset
```

### Step 5: 开始训练

```bash
# 检查GPU
nvidia-smi

# 开始训练
python train_qwen.py

# 后台运行（推荐）
nohup python train_qwen.py > train.log 2>&1 &

# 查看日志
tail -f train.log

# 查看TensorBoard
tensorboard --logdir=./logs --port=6006
```

---

## 训练配置说明（官方推荐）

### 模型选择

```python
# 推荐: Qwen2.5-7B-Instruct
model_name = "Qwen/Qwen2.5-7B-Instruct"
# 显存占用: ~18GB (LoRA)
# 训练时间: 2-4小时 (1万条数据)

# 备选: Qwen2.5-3B-Instruct (显存不够时)
model_name = "Qwen/Qwen2.5-3B-Instruct"
# 显存占用: ~10GB
# 训练时间: 1-2小时
```

### LoRA配置（官方推荐）

```python
lora_r = 64          # 官方推荐: 8-64
lora_alpha = 128     # 通常是r的2倍
lora_dropout = 0.05  # 官方推荐: 0.05-0.1

# Target modules: 所有attention和FFN层
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # FFN
]
```

### 训练超参数（官方推荐）

```python
# 批大小
per_device_train_batch_size = 4  # 官方推荐从小开始
gradient_accumulation_steps = 4  # 有效batch_size = 16

# 学习率
learning_rate = 5e-5  # 官方推荐范围: 1e-5 到 1e-4
warmup_ratio = 0.03   # 官方推荐: 0.03

# 精度（5090优化）
bf16 = True   # 5090原生支持
tf32 = True   # 启用TF32加速

# Flash Attention 2
attn_implementation = "flash_attention_2"  # 加速30%
```

---

## 关键技术点（官方方案）

### 1. 使用SFTTrainer（不是普通Trainer）

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    peft_config=lora_config,  # 自动应用LoRA
    formatting_func=lambda x: x["messages"],  # 返回messages
    max_seq_length=512,
    packing=False,  # 分类任务不需要packing
)
```

**优势**:
- 自动处理chat template
- 自动应用LoRA
- 优化的数据处理

### 2. Chat Template自动处理（不手搓）

```python
# ✅ 正确方式（官方推荐）
messages = [
    {"role": "user", "content": "分析新闻类别：..."},
    {"role": "assistant", "content": "这条新闻属于：货币政策"}
]

# SFTTrainer自动调用
text = tokenizer.apply_chat_template(messages, tokenize=False)

# ❌ 错误方式（不要手搓）
text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
```

### 3. Gradient Checkpointing（节省显存）

```python
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 配置
gradient_checkpointing_kwargs = {"use_reentrant": False}  # 新版推荐
```

---

## 性能预估（RTX 5090）

### 7B模型 + LoRA

| 配置 | 批大小 | 显存 | 速度 | 训练时间 |
|------|--------|------|------|---------|
| bf16 + Flash Attn 2 | 4 | ~16GB | ~1200 tok/s | 2-3小时 (1万条) |
| bf16 + SDPA | 4 | ~18GB | ~900 tok/s | 3-4小时 |
| fp16 | 4 | ~18GB | ~800 tok/s | 4-5小时 |

### 3B模型 + LoRA

| 配置 | 批大小 | 显存 | 速度 | 训练时间 |
|------|--------|------|------|---------|
| bf16 + Flash Attn 2 | 8 | ~10GB | ~2000 tok/s | 1-2小时 (1万条) |

---

## 完整命令清单

### 在5090机器上执行

```bash
# 1. 环境准备
conda create -n qwen-train python=3.10 -y
conda activate qwen-train

# 2. 安装PyTorch（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装依赖
pip install transformers accelerate peft trl datasets
pip install flash-attn --no-build-isolation
pip install tensorboard wandb akshare jieba snownlp

# 4. 准备数据（选择一种）
# 方案A: 示例数据集（快速测试）
python download_datasets.py --sample
python prepare_data.py --generate-dataset

# 方案B: FinCUGE数据集（推荐）
export HF_ENDPOINT=https://hf-mirror.com
python download_datasets.py --fincuge

# 5. 开始训练
python train_qwen.py

# 6. 监控训练
tensorboard --logdir=./logs --port=6006
# 浏览器访问: http://5090-ip:6006

# 7. 训练完成后测试
python inference_qwen.py
```

---

## 故障排查

### Q1: Flash Attention安装失败

```bash
# 方案1: 使用预编译wheel
pip install flash-attn --no-build-isolation

# 方案2: 从源码编译（需要时间）
pip install flash-attn --no-build-isolation --no-cache-dir

# 方案3: 不使用Flash Attention（降级）
# 修改train_qwen.py: use_flash_attention=False
```

### Q2: 显存不足

```bash
# 方案1: 减小batch size
per_device_train_batch_size = 2
gradient_accumulation_steps = 8

# 方案2: 使用3B模型
model_name = "Qwen/Qwen2.5-3B-Instruct"

# 方案3: 使用4bit量化
load_in_4bit = True
```

### Q3: 数据集下载失败

```bash
# 使用HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后加载
dataset = load_dataset('json', data_files='local_file.jsonl')
```

---

## 训练后集成

### 在原系统中使用微调模型

```python
# alpha_agent/data_sources/news_processor.py

from alpha_agent.training.news_classifier.inference_qwen import QwenNewsClassifier

class NewsProcessor:
    def __init__(self, use_qwen: bool = True):
        if use_qwen:
            self.classifier = QwenNewsClassifier(
                model_path="./models/qwen-news-classifier",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
    
    def _extract_themes(self, text: str):
        """使用Qwen模型提取主题"""
        result = self.classifier.classify(text)
        return [result['category']]
```

---

## 总结

### ✅ 已完成

1. ✅ 参考HuggingFace官方方案优化训练脚本
2. ✅ 使用SFTTrainer + apply_chat_template
3. ✅ 5090优化配置（bf16 + Flash Attn 2）
4. ✅ 完整的requirements.txt
5. ✅ 示例数据集创建成功（23条）
6. ✅ 部署文档

### 🚀 下一步

**在5090机器上**:
```bash
# 1. 解压代码
tar -xzf qwen_training.tar.gz

# 2. 安装环境
conda create -n qwen-train python=3.10 -y
conda activate qwen-train
pip install -r requirements.txt

# 3. 准备数据
python download_datasets.py --sample

# 4. 开始训练
python train_qwen.py
```

**预计**: 2-4小时完成训练！

所有代码已就绪，可以直接迁移到5090运行！
