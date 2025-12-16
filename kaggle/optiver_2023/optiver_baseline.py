"""
Kaggle: Optiver Trading at the Close - 基线方案

比赛目标: 预测 NASDAQ 股票收盘价相对于合成指数的变动
评估指标: MAE (Mean Absolute Error)

基于银牌方案的特征工程 + LightGBM

使用方法:
1. 下载比赛数据: https://www.kaggle.com/competitions/optiver-trading-at-the-close/data
2. 将 train.csv 放到 kaggle/data/ 目录
3. 运行: python kaggle/optiver_baseline.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple
import warnings
import os

warnings.filterwarnings('ignore')


# ============ 特征工程 ============

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    核心特征工程 - 基于银牌方案
    
    特征类别:
    1. 基础特征: 价差、流动性
    2. 不平衡特征: 买卖压力
    3. 时序特征: 滞后、变化率
    4. 统计特征: 均值、标准差
    """
    df = df.copy()
    
    # ===== 1. 基础特征 =====
    # 价差 (Spread)
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["spread_pct"] = df["spread"] / df["wap"]
    
    # 中间价
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2
    
    # 流动性不平衡
    df["liquidity_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"] + 1)
    
    # ===== 2. 市场紧迫度 (最强特征!) =====
    df["market_urgency"] = df["spread"] * df["liquidity_imbalance"]
    
    # 市场紧迫度 V2
    df["market_urgency_v2"] = (
        df["mid_price"] - 
        (df["bid_price"] * df["bid_size"] + df["ask_price"] * df["ask_size"]) / 
        (df["bid_size"] + df["ask_size"] + 1)
    )
    
    # ===== 3. 不平衡特征 =====
    # 价格不平衡
    df["price_imbalance"] = (df["ask_price"] - df["bid_price"]) / (df["ask_price"] + df["bid_price"] + 1e-8)
    
    # 数量不平衡
    df["size_imbalance"] = (df["ask_size"] - df["bid_size"]) / (df["ask_size"] + df["bid_size"] + 1)
    
    # 匹配比例
    df["matched_ratio"] = df["matched_size"] / (df["imbalance_size"] + df["matched_size"] + 1)
    
    # 不平衡强度
    df["imbalance_intensity"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
    
    # ===== 4. 价格关系特征 =====
    # WAP 与参考价格的偏离
    df["wap_ref_diff"] = df["wap"] - df["reference_price"]
    df["wap_ref_pct"] = df["wap_ref_diff"] / (df["reference_price"] + 1e-8)
    
    # 价格位置
    price_cols = ["reference_price", "far_price", "near_price", "bid_price", "ask_price", "wap"]
    for col in price_cols:
        if col in df.columns:
            df[f"{col}_diff_mid"] = df[col] - df["mid_price"]
    
    # ===== 5. 时间特征 =====
    df["seconds_bucket"] = df["seconds_in_bucket"] // 60  # 分钟桶
    df["is_last_minute"] = (df["seconds_in_bucket"] >= 540).astype(int)  # 最后一分钟
    
    # ===== 6. 滞后特征 (按 stock_id 分组) =====
    for col in ["wap", "spread", "imbalance_size", "matched_size"]:
        if col in df.columns:
            df[f"{col}_diff_1"] = df.groupby("stock_id")[col].diff(1)
            df[f"{col}_pct_1"] = df.groupby("stock_id")[col].pct_change(1)
    
    # ===== 7. 统计特征 (按 stock_id 分组) =====
    for col in ["wap", "spread", "bid_size", "ask_size"]:
        if col in df.columns:
            df[f"{col}_mean"] = df.groupby("stock_id")[col].transform("mean")
            df[f"{col}_std"] = df.groupby("stock_id")[col].transform("std")
    
    # ===== 8. 交叉特征 =====
    df["spread_x_imbalance"] = df["spread"] * df["imbalance_size"]
    df["urgency_x_flag"] = df["market_urgency"] * df["imbalance_buy_sell_flag"]
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """获取特征列名"""
    exclude_cols = [
        "row_id", "target", "date_id", "time_id", 
        "stock_id", "seconds_in_bucket"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


# ============ 模型训练 ============

def train_lightgbm(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    params: Dict = None
) -> Tuple[lgb.Booster, Dict]:
    """
    训练 LightGBM 模型
    
    使用时间序列分割避免数据泄露
    """
    if params is None:
        params = {
            "objective": "mae",
            "metric": "mae",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 128,
            "max_depth": 8,
            "min_child_samples": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "n_jobs": -1,
            "seed": 42,
        }
    
    # 时间序列分割
    split_day = int(train_df["date_id"].max() * 0.8)
    
    train_mask = train_df["date_id"] <= split_day
    valid_mask = train_df["date_id"] > split_day
    
    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, "target"]
    X_valid = train_df.loc[valid_mask, feature_cols]
    y_valid = train_df.loc[valid_mask, "target"]
    
    print(f"训练集: {len(X_train):,} 样本")
    print(f"验证集: {len(X_valid):,} 样本")
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # 训练
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )
    
    # 验证集预测
    y_pred = model.predict(X_valid)
    mae = np.mean(np.abs(y_valid - y_pred))
    
    results = {
        "mae": mae,
        "best_iteration": model.best_iteration,
        "feature_importance": dict(zip(feature_cols, model.feature_importance())),
    }
    
    print(f"\n验证集 MAE: {mae:.6f}")
    
    return model, results


def get_top_features(importance_dict: Dict, top_n: int = 20) -> List[str]:
    """获取最重要的特征"""
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_features[:top_n]]


# ============ 推理 ============

def predict(model: lgb.Booster, test_df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """生成预测"""
    X_test = test_df[feature_cols]
    predictions = model.predict(X_test)
    return predictions


# ============ 主流程 ============

def main():
    """主函数"""
    print("=" * 60)
    print("Optiver Trading at the Close - 基线方案")
    print("=" * 60)
    
    # 数据路径
    data_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(data_dir, "data", "train.csv")
    
    # 检查数据
    if not os.path.exists(train_path):
        print(f"\n❌ 未找到训练数据: {train_path}")
        print("\n请按以下步骤操作:")
        print("1. 访问 https://www.kaggle.com/competitions/optiver-trading-at-the-close/data")
        print("2. 下载 train.csv")
        print(f"3. 将文件放到 {os.path.join(data_dir, 'data')} 目录")
        return
    
    # 1. 加载数据
    print("\n【1】加载数据...")
    df = pd.read_csv(train_path)
    print(f"  数据形状: {df.shape}")
    print(f"  日期范围: {df['date_id'].min()} ~ {df['date_id'].max()}")
    print(f"  股票数量: {df['stock_id'].nunique()}")
    
    # 2. 特征工程
    print("\n【2】特征工程...")
    df = create_features(df)
    feature_cols = get_feature_columns(df)
    print(f"  特征数量: {len(feature_cols)}")
    
    # 3. 处理缺失值
    print("\n【3】处理缺失值...")
    df[feature_cols] = df[feature_cols].fillna(0)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    
    # 4. 训练模型
    print("\n【4】训练 LightGBM...")
    model, results = train_lightgbm(df, feature_cols)
    
    # 5. 特征重要性
    print("\n【5】Top 15 重要特征:")
    top_features = get_top_features(results["feature_importance"], 15)
    for i, feat in enumerate(top_features, 1):
        importance = results["feature_importance"][feat]
        print(f"  {i:2d}. {feat:30s}: {importance:,}")
    
    # 6. 保存模型
    model_path = os.path.join(data_dir, "model.txt")
    model.save_model(model_path)
    print(f"\n✅ 模型已保存: {model_path}")
    
    print("\n" + "=" * 60)
    print(f"验证集 MAE: {results['mae']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
