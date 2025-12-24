# vLLM部署指南 - Qwen微调模型

## 机器配置

根据检测，您的机器配置如下:

- **GPU**: NVIDIA GeForce RTX 3080
- **内存**: 31GB (可用约14GB)
- **CPU**: 16核
- **微调模型**: Qwen3-8B + LoRA适配器

## 当前状态

✓ 微调模型已完成: `./models/qwen-news-classifier`
- LoRA适配器 (r=64, alpha=128)
- 两个checkpoint: checkpoint-6500 和 checkpoint-10005 (使用最新的checkpoint-10005)

⚠️ 需要解决的问题:
1. NVIDIA驱动问题 (nvidia-smi无法工作)
2. 基础模型Qwen3-8B缺失
3. PyTorch和vLLM未安装

## 部署步骤

### 步骤1: 修复NVIDIA驱动 (重要!)

```bash
# 检查驱动状态
nvidia-smi

# 如果失败，重新安装驱动
sudo apt update
sudo apt install nvidia-driver-535  # 或更新版本

# 重启系统
sudo reboot

# 重启后验证
nvidia-smi
```

### 步骤2: 安装依赖

```bash
# 激活conda环境 (如果有)
conda activate qwen-train

# 或创建新环境
conda create -n vllm-deploy python=3.10 -y
conda activate vllm-deploy

# 安装PyTorch (CUDA 12.1，适配RTX 3080)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装vLLM
pip install vllm

# 安装其他依赖
pip install transformers peft accelerate huggingface_hub
```

### 步骤3: 下载基础模型

由于您的LoRA适配器基于Qwen3-8B，需要先下载基础模型:

```bash
# 方案A: 使用提供的脚本 (推荐)
python download_base_model.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --output-dir ./models/Qwen/Qwen3-8B \
    --use-mirror

# 方案B: 手动下载
# 如果Qwen3-8B不可用，可以使用Qwen2.5-7B-Instruct作为替代
```

**注意**: Qwen3-8B可能是内部版本，如果无法下载，建议:
1. 使用Qwen2.5-7B-Instruct替代
2. 或重新训练LoRA适配器基于公开可用的模型

### 步骤4: 合并LoRA适配器

将LoRA适配器与基础模型合并为完整模型:

```bash
# 使用checkpoint-10005 (最新的checkpoint)
python merge_lora.py \
    --base-model ./models/Qwen/Qwen3-8B \
    --adapter ./models/qwen-news-classifier/checkpoint-10005 \
    --output ./models/qwen-news-classifier-merged
```

**预计时间**: 5-10分钟
**磁盘空间**: 需要约16GB (8B模型的完整权重)

### 步骤5: 使用vLLM部署

```bash
# 部署合并后的模型
python deploy_vllm.py \
    --model-path ./models/qwen-news-classifier-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048

# 或使用后台运行
nohup python deploy_vllm.py \
    --model-path ./models/qwen-news-classifier-merged \
    --port 8000 > vllm.log 2>&1 &
```

### 步骤6: 测试部署

```bash
# 测试API
curl http://localhost:8000/v1/models

# 测试生成
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen-news-classifier-merged",
        "prompt": "分析以下新闻的类别：央行宣布降准0.5个百分点",
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

## 性能优化

### RTX 3080配置建议

RTX 3080有10GB显存，建议配置:

```bash
python deploy_vllm.py \
    --model-path ./models/qwen-news-classifier-merged \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    --tensor-parallel-size 1
```

### 如果显存不足

```bash
# 方案1: 降低显存利用率
--gpu-memory-utilization 0.75

# 方案2: 减少最大序列长度
--max-model-len 1024

# 方案3: 使用量化 (需要重新合并模型时指定)
# 在merge_lora.py中添加量化参数
```

## 替代方案: 直接使用LoRA (不推荐)

如果无法下载基础模型或合并失败，可以使用transformers直接加载:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型 (如果有)
base_model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen/Qwen3-8B",
    device_map="auto",
    trust_remote_code=True,
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(
    base_model,
    "./models/qwen-news-classifier/checkpoint-10005",
)

tokenizer = AutoTokenizer.from_pretrained(
    "./models/qwen-news-classifier/checkpoint-10005",
    trust_remote_code=True,
)

# 推理
messages = [{"role": "user", "content": "分析新闻类别：..."}]
text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## 故障排查

### Q1: nvidia-smi失败

```bash
# 检查驱动
lspci | grep -i nvidia  # 应该能看到GPU

# 重新安装驱动
sudo apt install nvidia-driver-535
sudo reboot
```

### Q2: 基础模型下载失败

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python download_base_model.py --use-mirror

# 或手动指定代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### Q3: vLLM显存不足

```bash
# 降低配置
python deploy_vllm.py \
    --gpu-memory-utilization 0.7 \
    --max-model-len 1024
```

### Q4: Qwen3-8B不存在

Qwen3-8B可能是内部版本，建议:

```bash
# 使用Qwen2.5-7B-Instruct替代
python download_base_model.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --output-dir ./models/Qwen/Qwen2.5-7B-Instruct

# 然后合并
python merge_lora.py \
    --base-model ./models/Qwen/Qwen2.5-7B-Instruct \
    --adapter ./models/qwen-news-classifier/checkpoint-10005 \
    --output ./models/qwen-news-classifier-merged
```

## API使用示例

### Python客户端

```python
import requests

url = "http://localhost:8000/v1/completions"
data = {
    "model": "qwen-news-classifier-merged",
    "prompt": "分析以下新闻的类别：央行宣布降准0.5个百分点",
    "max_tokens": 100,
    "temperature": 0.7
}

response = requests.post(url, json=data)
print(response.json())
```

### Chat API

```python
url = "http://localhost:8000/v1/chat/completions"
data = {
    "model": "qwen-news-classifier-merged",
    "messages": [
        {"role": "user", "content": "分析以下新闻的类别：央行宣布降准0.5个百分点"}
    ],
    "max_tokens": 100
}

response = requests.post(url, json=data)
print(response.json())
```

## 总结

### 快速开始 (假设驱动正常)

```bash
# 1. 安装依赖
pip install torch vllm transformers peft huggingface_hub

# 2. 下载基础模型
python download_base_model.py --use-mirror

# 3. 合并LoRA
python merge_lora.py

# 4. 部署vLLM
python deploy_vllm.py

# 5. 测试
curl http://localhost:8000/v1/models
```

### 预计时间

- 修复驱动: 10-30分钟 (需要重启)
- 安装依赖: 5-10分钟
- 下载基础模型: 30-60分钟 (取决于网络)
- 合并LoRA: 5-10分钟
- 启动vLLM: 1-2分钟

**总计**: 约1-2小时

### 关键文件

- `deploy_vllm.py`: vLLM部署脚本
- `merge_lora.py`: LoRA合并脚本
- `download_base_model.py`: 模型下载脚本
- `./models/qwen-news-classifier/checkpoint-10005`: 最佳微调模型

现在可以开始部署了！
