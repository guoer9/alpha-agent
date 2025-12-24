# Qwen新闻分类微调快速开始

## 现成数据集推荐

### ⭐⭐⭐⭐⭐ 最推荐：FinCUGE

```bash
# 安装
pip install datasets

# 下载（自动）
python -c "
from datasets import load_dataset
dataset = load_dataset('sufe-aiflm-lab/fincuge', 'sentiment')
dataset.save_to_disk('data/raw/fincuge')
print(f'下载完成: {len(dataset)} 条')
"
```

**优势**:
- 金融领域专用
- 质量高（上海财经大学）
- HuggingFace一键下载
- 包含情感分析、事件抽取等

### ⭐⭐⭐⭐ 备选：THUCNews

```bash
# 手动下载
wget http://thuctc.thunlp.org/message
# 解压到 data/raw/THUCNews/
```

**优势**:
- 数据量大（74万条）
- 清华大学出品
- 包含财经、股票类别

---

## 完整训练流程（5090机器）

### Step 1: 环境准备

```bash
# 在5090机器上
conda create -n qwen-train python=3.10 -y
conda activate qwen-train

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.37.0 accelerate peft datasets bitsandbytes trl
pip install flash-attn --no-build-isolation  # Flash Attention 2
pip install tensorboard wandb
```

### Step 2: 准备数据

```bash
# 方案A: 使用FinCUGE（推荐）
python download_datasets.py --fincuge

# 方案B: 使用AkShare收集+自动标注
python prepare_data.py --collect --annotate --generate-dataset
```

### Step 3: 训练

```bash
# 单卡训练（RTX 5090）
python train_qwen.py

# 预计时间: 2-4小时（取决于数据量）
# 显存占用: ~18GB（7B模型+LoRA）
```

### Step 4: 推理测试

```bash
python inference_qwen.py
```

---

## 数据格式说明

### 训练数据格式（JSONL）

```json
{
  "messages": [
    {
      "role": "user",
      "content": "请分析以下财经新闻的类别：\n\n央行降准0.5%\n\n从以下类别中选择一个：货币政策、监管政策、业绩公告、科技创新、新能源、医药健康、金融、其他"
    },
    {
      "role": "assistant",
      "content": "这条新闻属于：货币政策"
    }
  ]
}
```

**关键点**:
- ✅ 使用标准chat格式
- ✅ tokenizer.apply_chat_template()自动处理
- ✅ 不手搓特殊token

---

## 迁移到5090机器

### 打包文件

```bash
# 在当前机器
cd /Volumes/2tb/mydata/code/Quantitative_trading/qlib_trading

# 打包训练代码
tar -czf qwen_training.tar.gz \
    alpha_agent/training/news_classifier/ \
    --exclude='*.pyc' \
    --exclude='__pycache__'

# 查看大小
ls -lh qwen_training.tar.gz
```

### 传输到5090

```bash
# 方式1: scp
scp qwen_training.tar.gz user@5090-ip:/path/to/

# 方式2: rsync
rsync -avz qwen_training.tar.gz user@5090-ip:/path/to/
```

### 5090机器上运行

```bash
# 解压
tar -xzf qwen_training.tar.gz
cd alpha_agent/training/news_classifier

# 创建环境
conda create -n qwen-train python=3.10 -y
conda activate qwen-train

# 安装依赖（5090专用）
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft datasets
pip install flash-attn --no-build-isolation

# 下载数据集
python download_datasets.py --fincuge

# 开始训练
python train_qwen.py
```

---

## 性能优化（5090专用）

### GPU配置

```python
# train_qwen.py 中的配置

# 5090优化配置
training_args = TrainingArguments(
    per_device_train_batch_size=16,  # 5090显存24GB，可以开大
    gradient_accumulation_steps=1,   # 不需要梯度累积
    bf16=True,                        # 5090支持bf16
    tf32=True,                        # 启用TF32加速
    gradient_checkpointing=True,      # 节省显存
    dataloader_num_workers=8,         # 5090 CPU也强
)

# Flash Attention 2（5090支持）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # 加速30%
)
```

### 预期性能

| 配置 | 批大小 | 显存占用 | 速度 | 训练时间(1万条) |
|------|--------|---------|------|----------------|
| 7B+LoRA | 16 | ~18GB | ~1000 tokens/s | 2-3小时 |
| 3B+LoRA | 32 | ~12GB | ~2000 tokens/s | 1-2小时 |

---

## 使用微调模型处理新闻

### 集成到现有系统

```python
# alpha_agent/data_sources/news_processor.py 修改

from alpha_agent.training.news_classifier.inference_qwen import QwenNewsClassifier

class NewsProcessor:
    def __init__(self, use_qwen_model: bool = True):
        if use_qwen_model:
            self.qwen_classifier = QwenNewsClassifier(
                model_path="./models/qwen-news-classifier"
            )
        else:
            self.qwen_classifier = None
    
    def _extract_themes(self, text: str) -> List[str]:
        """使用Qwen模型提取主题"""
        if self.qwen_classifier:
            result = self.qwen_classifier.classify(text)
            return [result['category']]
        else:
            # 降级到关键词匹配
            return self._extract_themes_by_keywords(text)
```

---

## 常见问题

### Q1: 显存不够怎么办？

```python
# 使用3B模型
model_name = "Qwen/Qwen2.5-3B-Instruct"  # 只需8GB显存

# 或使用4bit量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 显存减半
)
```

### Q2: 没有5090怎么办？

```bash
# 使用Google Colab（免费T4 GPU）
# 或使用AutoDL租GPU（RTX 4090，2元/小时）
```

### Q3: 数据集下载失败？

```bash
# 使用HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后加载
dataset = load_dataset('json', data_files='local_file.jsonl')
```

---

## 总结

### 已创建文件

1. ✅ `download_datasets.py` - 数据集下载
2. ✅ `prepare_data.py` - 数据准备
3. ✅ `train_qwen.py` - 训练脚本（使用chat template）
4. ✅ `inference_qwen.py` - 推理脚本
5. ✅ `QUICKSTART.md` - 快速开始指南

### 推荐数据集

**首选**: FinCUGE (金融专用，HuggingFace一键下载)  
**备选**: THUCNews (数据量大，需手动下载)

### 关键技术

- ✅ 使用`tokenizer.apply_chat_template()` - 不手搓token
- ✅ LoRA微调 - 节省显存和时间
- ✅ Flash Attention 2 - 加速30%
- ✅ bf16训练 - 5090原生支持

### 下一步

```bash
# 在5090机器上运行
python download_datasets.py --fincuge
python train_qwen.py
```

预计2-4小时完成训练！
