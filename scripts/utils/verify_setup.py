#!/usr/bin/env python
"""
验证5090环境配置和代码完整性

检查项:
1. CUDA版本 (需要12.8+)
2. PyTorch版本和CUDA支持
3. Transformers库版本
4. 数据文件存在性
5. 训练脚本语法检查
6. 内存和显存估算
"""

import sys
import subprocess
from pathlib import Path


def check_cuda():
    """检查CUDA版本"""
    print("="*70)
    print("1. 检查CUDA版本")
    print("="*70)
    
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        output = result.stdout
        
        if 'release 12.8' in output or 'release 12.9' in output or 'release 13' in output:
            print("✅ CUDA版本满足要求 (12.8+)")
            print(output)
        else:
            print("❌ CUDA版本过低")
            print(output)
            print("\n⚠️ RTX 5090需要CUDA 12.8+")
            return False
    except FileNotFoundError:
        print("❌ nvcc未找到，CUDA可能未安装")
        return False
    
    return True


def check_pytorch():
    """检查PyTorch"""
    print("\n" + "="*70)
    print("2. 检查PyTorch")
    print("="*70)
    
    try:
        import torch
        
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            
            # 检查版本
            torch_version = float('.'.join(torch.__version__.split('.')[:2]))
            cuda_version = float(torch.version.cuda)
            
            if torch_version >= 2.5 and cuda_version >= 12.8:
                print("✅ 版本满足5090要求")
                return True
            else:
                print(f"⚠️ 版本可能不满足要求")
                print(f"   需要: PyTorch>=2.5, CUDA>=12.8")
                print(f"   当前: PyTorch={torch_version}, CUDA={cuda_version}")
                return False
        else:
            print("❌ CUDA不可用")
            return False
            
    except ImportError:
        print("❌ PyTorch未安装")
        print("安装: pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu128")
        return False


def check_transformers():
    """检查Transformers库"""
    print("\n" + "="*70)
    print("3. 检查Transformers生态")
    print("="*70)
    
    required_packages = {
        'transformers': '4.46.0',
        'accelerate': '0.34.0',
        'peft': '0.13.0',
        'trl': '0.11.0',
        'datasets': '2.16.0',
    }
    
    all_ok = True
    
    for package, min_version in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
            
            if version == 'unknown':
                print(f"   ⚠️ 无法获取版本号")
                all_ok = False
        except ImportError:
            print(f"❌ {package}: 未安装")
            all_ok = False
    
    return all_ok


def check_flash_attention():
    """检查Flash Attention"""
    print("\n" + "="*70)
    print("4. 检查Flash Attention 2")
    print("="*70)
    
    try:
        import flash_attn
        print(f"✅ Flash Attention: {flash_attn.__version__}")
        return True
    except ImportError:
        print("⚠️ Flash Attention未安装（可选，但推荐）")
        print("   安装: MAX_JOBS=8 pip install flash-attn --no-build-isolation")
        print("   如果编译失败，训练脚本会自动降级到SDPA")
        return False


def check_data_files():
    """检查数据文件"""
    print("\n" + "="*70)
    print("5. 检查数据文件")
    print("="*70)
    
    files_to_check = [
        'data/raw/sample_dataset.jsonl',
        'data/processed/train.jsonl',
        'data/processed/val.jsonl',
    ]
    
    all_exist = True
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"⚠️ {file_path} 不存在")
            all_exist = False
    
    if not all_exist:
        print("\n生成数据:")
        print("  python download_datasets.py --sample")
        print("  python prepare_data.py --generate-dataset")
    
    return all_exist


def check_training_script():
    """检查训练脚本语法"""
    print("\n" + "="*70)
    print("6. 检查训练脚本")
    print("="*70)
    
    try:
        import py_compile
        
        scripts = [
            'train_qwen.py',
            'inference_qwen.py',
            'prepare_data.py',
        ]
        
        for script in scripts:
            try:
                py_compile.compile(script, doraise=True)
                print(f"✅ {script}: 语法正确")
            except py_compile.PyCompileError as e:
                print(f"❌ {script}: 语法错误")
                print(f"   {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"⚠️ 检查失败: {e}")
        return False


def estimate_resources():
    """估算资源需求"""
    print("\n" + "="*70)
    print("7. 资源需求估算")
    print("="*70)
    
    configs = [
        {
            'model': 'Qwen2.5-7B',
            'batch_size': 4,
            'grad_accum': 4,
            'vram': '16-18GB',
            'speed': '~1200 tok/s',
            'time': '2-4小时 (1万条)',
        },
        {
            'model': 'Qwen2.5-3B',
            'batch_size': 8,
            'grad_accum': 2,
            'vram': '10-12GB',
            'speed': '~2000 tok/s',
            'time': '1-2小时 (1万条)',
        },
    ]
    
    print("\nRTX 5090 (24GB显存) 配置建议:\n")
    
    for cfg in configs:
        print(f"{cfg['model']}:")
        print(f"  批大小: {cfg['batch_size']}")
        print(f"  梯度累积: {cfg['grad_accum']}")
        print(f"  显存占用: {cfg['vram']}")
        print(f"  训练速度: {cfg['speed']}")
        print(f"  预计时间: {cfg['time']}")
        print()
    
    print("推荐: Qwen2.5-7B (效果更好，5090显存足够)")


def main():
    print("="*70)
    print("RTX 5090 环境验证")
    print("="*70)
    print()
    
    results = {
        'CUDA': check_cuda(),
        'PyTorch': check_pytorch(),
        'Transformers': check_transformers(),
        'Flash Attention': check_flash_attention(),
        'Data Files': check_data_files(),
        'Scripts': check_training_script(),
    }
    
    estimate_resources()
    
    # 总结
    print("\n" + "="*70)
    print("验证结果")
    print("="*70)
    
    for item, status in results.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {item}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ 所有检查通过，可以开始训练！")
        print("\n运行命令:")
        print("  python train_qwen.py")
    else:
        print("\n⚠️ 部分检查未通过，请先解决问题")
        
        if not results['PyTorch']:
            print("\n安装PyTorch:")
            print("  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu128")
        
        if not results['Transformers']:
            print("\n安装Transformers:")
            print("  pip install transformers==4.46.0 accelerate peft trl datasets")
        
        if not results['Data Files']:
            print("\n准备数据:")
            print("  python download_datasets.py --sample")
            print("  python prepare_data.py --generate-dataset")


if __name__ == "__main__":
    main()
