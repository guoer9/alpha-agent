"""
Hull Tactical 数据预处理脚本
将 Kaggle CSV 转换为 Qlib 二进制格式
"""
import os
import sys
import shutil
from pathlib import Path

import pandas as pd
import numpy as np


# ============ 配置 ============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
QLIB_DATA_DIR = Path.home() / ".qlib/qlib_data/hull_data"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# 特征列前缀 (排除 date_id 和 target 列)
FEATURE_PREFIXES = ["D", "E", "I", "M", "P", "S", "V"]
TARGET_COL = "market_forward_excess_returns"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """获取所有特征列"""
    feature_cols = []
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in FEATURE_PREFIXES):
            feature_cols.append(col)
    return sorted(feature_cols)


def show_data_info() -> None:
    """显示数据信息"""
    if not TRAIN_CSV.exists():
        print("❌ 训练数据不存在，请先下载数据")
        return

    df = pd.read_csv(TRAIN_CSV)
    feature_cols = get_feature_columns(df)

    print("=" * 60)
    print("Hull Tactical 数据概览")
    print("=" * 60)
    print(f"训练集形状: {df.shape}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"目标变量: {TARGET_COL}")
    print(f"\n特征列表:")
    for prefix in FEATURE_PREFIXES:
        cols = [c for c in feature_cols if c.startswith(prefix)]
        print(f"  {prefix}: {len(cols)} 个 ({cols[:3]}...)")

    print(f"\n目标变量统计:")
    print(df[TARGET_COL].describe())


def prepare_qlib_data() -> None:
    """将 CSV 转换为 Qlib 格式"""
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"训练数据不存在: {TRAIN_CSV}")

    print(f"📂 读取训练数据: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"   形状: {train_df.shape}")

    # 获取特征列
    feature_cols = get_feature_columns(train_df)
    print(f"   特征数量: {len(feature_cols)}")

    # Qlib 需要 datetime 和 instrument 列
    # date_id 是整数索引，转换为日期 (假设从 1988-01-01 开始)
    base_date = pd.Timestamp("1988-01-01")
    train_df["datetime"] = base_date + pd.to_timedelta(train_df["date_id"], unit="D")
    train_df["instrument"] = "SPY"  # 单资产，伪造一个代码

    # 重命名目标列为 Qlib 标准的 label
    if TARGET_COL in train_df.columns:
        train_df["label"] = train_df[TARGET_COL]

    # 选择需要的列
    qlib_cols = ["datetime", "instrument"] + feature_cols + ["label"]
    qlib_df = train_df[qlib_cols].copy()

    # 处理缺失值
    qlib_df = qlib_df.fillna(0)

    # 清理旧数据
    if QLIB_DATA_DIR.exists():
        shutil.rmtree(QLIB_DATA_DIR)
    QLIB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 临时目录保存 CSV
    csv_temp_dir = DATA_DIR / "qlib_csv"
    if csv_temp_dir.exists():
        shutil.rmtree(csv_temp_dir)
    csv_temp_dir.mkdir()

    # 设置索引并保存
    qlib_df = qlib_df.set_index(["instrument", "datetime"])
    for inst, group in qlib_df.groupby(level="instrument"):
        group = group.droplevel("instrument")
        group.index.name = "date"
        output_path = csv_temp_dir / f"{inst}.csv"
        group.to_csv(output_path)
        print(f"   保存: {output_path}")

    # 调用 Qlib dump 命令
    include_fields = ",".join(feature_cols + ["label"])
    cmd = (
        f"python -m qlib.run.dump_data "
        f"--csv_path {csv_temp_dir} "
        f"--qlib_dir {QLIB_DATA_DIR} "
        f"--include_fields {include_fields} "
        f"--date_field_name date"
    )

    print(f"\n🔄 执行 Qlib 数据转换...")
    print(f"   命令: {cmd}")
    exit_code = os.system(cmd)

    if exit_code == 0:
        print(f"\n✅ 数据转换完成！")
        print(f"   Qlib 数据目录: {QLIB_DATA_DIR}")
    else:
        print(f"\n❌ 数据转换失败，退出码: {exit_code}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "info":
        show_data_info()
    else:
        show_data_info()
        print("\n")
        prepare_qlib_data()
