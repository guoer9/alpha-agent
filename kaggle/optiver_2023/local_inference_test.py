"""
本地推理测试脚本
用于在本地 (Mac) 验证特征工程和模型推理流程
不依赖 optiver2023 模块
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings
warnings.filterwarnings('ignore')

# ============ 1. 定义特征工程 (必须与提交代码一致) ============

def create_features(df):
    """特征工程"""
    df = df.copy()
    
    # 基础特征
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["spread_pct"] = df["spread"] / (df["wap"] + 1e-8)
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2
    df["liquidity_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"] + 1)
    
    # 市场紧迫度
    df["market_urgency"] = df["spread"] * df["liquidity_imbalance"]
    df["market_urgency_v2"] = (
        df["mid_price"] - 
        (df["bid_price"] * df["bid_size"] + df["ask_price"] * df["ask_size"]) / 
        (df["bid_size"] + df["ask_size"] + 1)
    )
    
    # 不平衡特征
    df["price_imbalance"] = (df["ask_price"] - df["bid_price"]) / (df["ask_price"] + df["bid_price"] + 1e-8)
    df["size_imbalance"] = (df["ask_size"] - df["bid_size"]) / (df["ask_size"] + df["bid_size"] + 1)
    df["matched_ratio"] = df["matched_size"] / (df["imbalance_size"] + df["matched_size"] + 1)
    df["imbalance_intensity"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
    
    # 价格关系
    df["wap_ref_diff"] = df["wap"] - df["reference_price"]
    df["wap_ref_pct"] = df["wap_ref_diff"] / (df["reference_price"] + 1e-8)
    
    # 价格位置
    for col in ["reference_price", "far_price", "near_price", "bid_price", "ask_price", "wap"]:
        if col in df.columns:
            df[f"{col}_diff_mid"] = df[col] - df["mid_price"]
    
    # 时间特征
    df["seconds_bucket"] = df["seconds_in_bucket"] // 60
    df["is_last_minute"] = (df["seconds_in_bucket"] >= 540).astype(int)
    
    # 交叉特征
    df["spread_x_imbalance"] = df["spread"] * df["imbalance_size"]
    df["urgency_x_flag"] = df["market_urgency"] * df["imbalance_buy_sell_flag"]
    
    return df

FEATURE_COLS = [
    'imbalance_size', 'imbalance_buy_sell_flag', 'reference_price',
    'matched_size', 'far_price', 'near_price', 'bid_price', 'bid_size',
    'ask_price', 'ask_size', 'wap',
    'spread', 'spread_pct', 'mid_price', 'liquidity_imbalance',
    'market_urgency', 'market_urgency_v2',
    'price_imbalance', 'size_imbalance', 'matched_ratio', 'imbalance_intensity',
    'wap_ref_diff', 'wap_ref_pct',
    'reference_price_diff_mid', 'far_price_diff_mid', 'near_price_diff_mid',
    'bid_price_diff_mid', 'ask_price_diff_mid', 'wap_diff_mid',
    'seconds_bucket', 'is_last_minute',
    'spread_x_imbalance', 'urgency_x_flag',
]

# ============ 2. 主流程 ============

def main():
    print("="*50)
    print("🚀 开始本地推理测试")
    print("="*50)
    
    # 路径设置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    model_path = os.path.join(base_dir, "model.txt")
    test_file = os.path.join(data_dir, "example_test_files/test.csv")
    
    # 1. 检查模型
    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件: {model_path}")
        print("请先运行: python kaggle/optiver_baseline.py")
        return
    
    print(f"✅ 加载模型: {model_path}")
    model = lgb.Booster(model_file=model_path)
    
    # 2. 加载测试数据
    if not os.path.exists(test_file):
        print(f"❌ 未找到测试文件: {test_file}")
        return
        
    print(f"✅ 加载测试数据: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"   数据形状: {test_df.shape}")
    
    # 3. 模拟推理循环
    print("\n🔄 开始模拟推理...")
    
    # 这里的 example_test_files 是批量数据，但在比赛中是一行一行或者是小批量给的
    # 我们直接处理整个 dataframe 来验证逻辑
    
    # 特征工程
    print("   执行特征工程...")
    processed_df = create_features(test_df)
    
    # 填充缺失值
    print("   处理缺失值...")
    for col in FEATURE_COLS:
        if col not in processed_df.columns:
            print(f"   ⚠️ 警告: 缺少列 {col}, 填充 0")
            processed_df[col] = 0
    
    processed_df[FEATURE_COLS] = processed_df[FEATURE_COLS].fillna(0)
    processed_df[FEATURE_COLS] = processed_df[FEATURE_COLS].replace([np.inf, -np.inf], 0)
    
    # 预测
    print("   执行预测...")
    predictions = model.predict(processed_df[FEATURE_COLS])
    
    # 4. 结果展示
    test_df['predicted_target'] = predictions
    
    print("\n📊 预测结果预览:")
    print(test_df[['stock_id', 'seconds_in_bucket', 'predicted_target']].head(10))
    
    print("\n📈 统计信息:")
    print(test_df['predicted_target'].describe())
    
    print("\n✅ 测试完成！代码逻辑正常。")

if __name__ == "__main__":
    main()
