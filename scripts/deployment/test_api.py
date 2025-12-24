#!/usr/bin/env python3
"""
金融新闻分类API测试脚本
快速测试部署的模型服务
"""

import requests
import json

# 服务地址
BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查接口"""
    print("=" * 60)
    print("测试1: 健康检查")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        result = response.json()
        print(f"✓ 服务状态: {result['status']}")
        return True
    except Exception as e:
        print(f"✗ 服务异常: {e}")
        return False

def test_classify_single():
    """测试单条新闻分类"""
    print("\n" + "=" * 60)
    print("测试2: 单条新闻分类")
    print("=" * 60)
    
    news = "央行宣布降准0.5个百分点，释放长期资金约1万亿元"
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"请分析以下新闻的类别：{news}"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
        result = response.json()
        category = result['choices'][0]['message']['content']
        
        print(f"新闻: {news}")
        print(f"分类: {category}")
        print("✓ 分类成功")
        return True
    except Exception as e:
        print(f"✗ 分类失败: {e}")
        return False

def test_classify_batch():
    """测试批量新闻分类"""
    print("\n" + "=" * 60)
    print("测试3: 批量新闻分类")
    print("=" * 60)
    
    news_list = [
        "央行宣布降准0.5个百分点",
        "A股三大指数集体收涨，沪指涨1.2%",
        "证监会发布新规，加强上市公司监管",
        "特斯拉宣布在华建设新工厂",
        "茅台股价创历史新高，市值突破3万亿"
    ]
    
    print(f"测试 {len(news_list)} 条新闻...\n")
    
    success_count = 0
    for i, news in enumerate(news_list, 1):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"请分析以下新闻的类别：{news}"
                }
            ],
            "max_tokens": 100,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
            result = response.json()
            category = result['choices'][0]['message']['content']
            
            print(f"{i}. ✓ {news[:30]}...")
            print(f"   分类: {category}\n")
            success_count += 1
        except Exception as e:
            print(f"{i}. ✗ {news[:30]}...")
            print(f"   错误: {e}\n")
    
    print(f"成功: {success_count}/{len(news_list)}")
    return success_count == len(news_list)

def test_models():
    """测试模型列表接口"""
    print("\n" + "=" * 60)
    print("测试4: 模型列表")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        result = response.json()
        
        print(f"可用模型:")
        for model in result['data']:
            print(f"  - {model['id']}")
        
        print("✓ 获取模型列表成功")
        return True
    except Exception as e:
        print(f"✗ 获取失败: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("金融新闻分类API测试")
    print("=" * 60)
    print(f"服务地址: {BASE_URL}")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("健康检查", test_health),
        ("单条分类", test_classify_single),
        ("批量分类", test_classify_batch),
        ("模型列表", test_models),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n测试异常: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！服务运行正常。")
    else:
        print("\n⚠️  部分测试失败，请检查服务状态。")

if __name__ == "__main__":
    main()
