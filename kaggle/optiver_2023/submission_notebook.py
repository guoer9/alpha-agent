"""
Kaggle 提交 Notebook 模板

这是用于 Kaggle 提交的代码模板
需要在 Kaggle Notebook 环境中运行

比赛要求使用 API 进行实时推理
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optiver2023  # Kaggle 比赛 API
from typing import List


# ============ 特征工程 (与训练一致) ============

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程"""
    df = df.copy()
    
    # 基础特征
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["spread_pct"] = df["spread"] / (df["wap"] + 1e-8)
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2
    df["liquidity_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"] + 1)
    
    # 市场紧迫度 (最强特征)
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
    
    # 时间特征
    df["seconds_bucket"] = df["seconds_in_bucket"] // 60
    df["is_last_minute"] = (df["seconds_in_bucket"] >= 540).astype(int)
    
    # 交叉特征
    df["spread_x_imbalance"] = df["spread"] * df["imbalance_size"]
    df["urgency_x_flag"] = df["market_urgency"] * df["imbalance_buy_sell_flag"]
    
    return df


# ============ 特征列表 ============

FEATURE_COLS = [
    "imbalance_size", "imbalance_buy_sell_flag", "reference_price",
    "matched_size", "far_price", "near_price", "bid_price", "bid_size",
    "ask_price", "ask_size", "wap",
    "spread", "spread_pct", "mid_price", "liquidity_imbalance",
    "market_urgency", "market_urgency_v2",
    "price_imbalance", "size_imbalance", "matched_ratio", "imbalance_intensity",
    "wap_ref_diff", "wap_ref_pct",
    "seconds_bucket", "is_last_minute",
    "spread_x_imbalance", "urgency_x_flag",
]


# ============ 主推理循环 ============

def main():
    """Kaggle 提交主函数"""
    
    # 加载预训练模型
    model = lgb.Booster(model_file="/kaggle/input/your-model/model.txt")
    
    # Kaggle API 环境
    env = optiver2023.make_env()
    iter_test = env.iter_test()
    
    # 推理循环
    for test_df, revealed_targets, sample_prediction in iter_test:
        # 特征工程
        test_df = create_features(test_df)
        
        # 处理缺失值
        test_df[FEATURE_COLS] = test_df[FEATURE_COLS].fillna(0)
        test_df[FEATURE_COLS] = test_df[FEATURE_COLS].replace([np.inf, -np.inf], 0)
        
        # 预测
        predictions = model.predict(test_df[FEATURE_COLS])
        
        # 提交
        sample_prediction["target"] = predictions
        env.predict(sample_prediction)


if __name__ == "__main__":
    main()
