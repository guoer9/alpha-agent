#!/usr/bin/env python
"""
Qwen新闻分类推理脚本

使用微调后的模型进行新闻分类
关键: 使用tokenizer.apply_chat_template()
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from typing import List, Dict


class QwenNewsClassifier:
    """Qwen新闻分类器"""
    
    def __init__(
        self,
        model_path: str = "./models/qwen-news-classifier",
        base_model: str = "./models/Qwen/Qwen3-8B",
        device: str = "cuda",
    ):
        self.device = device
        print(f"加载模型: {model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # 加载LoRA权重
        try:
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print("✅ LoRA权重加载成功")
        except:
            print("⚠️ LoRA权重未找到，使用基础模型")
        
        self.model.eval()
        
        # 类别列表
        self.categories = [
            "货币政策", "监管政策", "业绩公告", "科技创新",
            "新能源", "医药健康", "金融", "其他"
        ]
    
    def classify(self, text: str) -> Dict:
        """
        分类单条新闻
        
        使用chat template，不手搓prompt
        """
        # 构建对话消息
        messages = [
            {
                "role": "user",
                "content": f"请分析以下财经新闻的类别：\n\n{text}\n\n从以下类别中选择一个：{', '.join(self.categories)}"
            }
        ]
        
        # 使用apply_chat_template（自动处理特殊token）
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # 推理时添加生成提示
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # 低温度，更确定性
                top_p=0.9,
                do_sample=False,  # 分类任务不采样
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],  # 只取生成的部分
            skip_special_tokens=True,
        )
        
        # 提取类别
        category = self._extract_category(response)
        
        return {
            'text': text,
            'category': category,
            'raw_response': response,
        }
    
    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """批量分类"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            for text in batch:
                result = self.classify(text)
                results.append(result)
            
            if (i + batch_size) % 100 == 0:
                print(f"  进度: {i + batch_size}/{len(texts)}")
        
        return results
    
    def _extract_category(self, response: str) -> str:
        """从回复中提取类别"""
        response = response.strip()
        
        # 尝试匹配类别
        for cat in self.categories:
            if cat in response:
                return cat
        
        return "其他"


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    # 测试数据
    test_cases = [
        "央行宣布降准0.5个百分点，释放长期资金约1万亿元",
        "证监会发文规范量化交易，多家私募被约谈",
        "某某银行发布三季报，净利润同比增长15%",
        "国产芯片取得重大突破，7nm工艺实现量产",
        "新能源汽车销量创新高，宁德时代股价大涨",
    ]
    
    print("="*70)
    print("Qwen新闻分类器测试")
    print("="*70)
    
    # 加载模型
    classifier = QwenNewsClassifier(
        model_path="./models/qwen-news-classifier",
        base_model="./models/Qwen/Qwen3-8B",
    )
    
    # 测试分类
    print("\n分类结果:\n")
    for i, text in enumerate(test_cases, 1):
        result = classifier.classify(text)
        print(f"{i}. {text[:40]}...")
        print(f"   类别: {result['category']}")
        print(f"   原始回复: {result['raw_response']}")
        print()
