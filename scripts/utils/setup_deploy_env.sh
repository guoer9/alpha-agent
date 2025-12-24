#!/bin/bash
# 创建vLLM部署环境
# 与训练环境隔离，避免依赖冲突

set -e

echo "=========================================="
echo "创建vLLM部署环境"
echo "=========================================="

# 环境名称
ENV_NAME="vllm-deploy"

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: conda未安装或不在PATH中"
    exit 1
fi

# 删除已存在的环境（如果有）
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "检测到已存在的环境 ${ENV_NAME}，正在删除..."
    conda env remove -n ${ENV_NAME} -y
fi

# 创建新环境
echo ""
echo "步骤1: 创建conda环境 (Python 3.10)"
conda create -n ${ENV_NAME} python=3.10 -y

# 激活环境
echo ""
echo "步骤2: 激活环境"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# 安装PyTorch (CUDA 11.8)
echo ""
echo "步骤3: 安装PyTorch 2.3.1 + CUDA 11.8"
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 安装transformers生态
echo ""
echo "步骤4: 安装transformers、peft、accelerate"
pip install transformers>=4.46.0 peft>=0.13.0 accelerate>=0.34.0

# 安装vLLM
echo ""
echo "步骤5: 安装vLLM"
pip install vllm

# 安装其他工具
echo ""
echo "步骤6: 安装其他依赖"
pip install modelscope huggingface_hub

# 验证安装
echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

echo ""
echo "=========================================="
echo "环境创建完成!"
echo "=========================================="
echo "激活环境: conda activate ${ENV_NAME}"
echo ""
echo "下一步:"
echo "1. 合并LoRA适配器:"
echo "   python merge_lora.py"
echo ""
echo "2. 部署vLLM服务:"
echo "   python deploy_vllm.py"
echo "=========================================="
