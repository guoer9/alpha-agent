"""
验证目标变量 market_forward_excess_returns 的计算方式

结论: official ≈ forward_returns - risk_free_rate
相关性: 0.999978
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """加载训练数据"""
    train = pd.read_csv(DATA_DIR / "train.csv")
    base_date = pd.Timestamp('1988-01-01')
    train['date'] = base_date + pd.to_timedelta(train['date_id'], unit='D')
    return train


def mad_winsorize(series: pd.Series, threshold: float = 4) -> pd.Series:
    """
    MAD Winsorization (缩尾处理)
    
    将超出边界的极端值拉回到边界值
    边界 = median ± threshold × MAD
    """
    valid = series.dropna()
    median = valid.median()
    mad = (valid - median).abs().median()
    lower = median - threshold * mad
    upper = median + threshold * mad
    return series.clip(lower=lower, upper=upper), lower, upper


def verify_formula(train: pd.DataFrame):
    """验证不同公式的匹配度"""
    forward_returns = train['forward_returns']
    official = train['market_forward_excess_returns']
    
    print("=" * 70)
    print("验证目标变量计算方式")
    print("=" * 70)
    
    # ============ 测试不同公式 ============
    formulas = {
        # 公式1: 减去expanding mean (累积均值)
        "forward_returns - expanding_mean": 
            forward_returns - forward_returns.expanding(min_periods=1).mean(),
        
        # 公式2: 减去5年滚动均值 (min_periods=1)
        "forward_returns - rolling_1260(min=1)":
            forward_returns - forward_returns.rolling(1260, min_periods=1).mean(),
        
        # 公式3: 减去5年滚动均值 (固定窗口)
        "forward_returns - rolling_1260(min=1260)":
            forward_returns - forward_returns.rolling(1260, min_periods=1260).mean(),
        
        # 公式4: 减去risk_free
        "forward_returns - risk_free":
            forward_returns - train['risk_free_rate'],
    }
    
    print("\n【公式匹配度测试】")
    print("-" * 70)
    
    results = []
    for name, calculated in formulas.items():
        valid_mask = ~calculated.isna()
        corr = calculated[valid_mask].corr(official[valid_mask])
        mae = (calculated[valid_mask] - official[valid_mask]).abs().mean()
        results.append((name, corr, mae))
        print(f"{name}:")
        print(f"  相关性: {corr:.6f}, MAE: {mae:.8f}")
    
    # 找出最佳公式
    best = max(results, key=lambda x: x[1])
    print(f"\n最佳公式: {best[0]} (相关性: {best[1]:.6f})")
    
    return best[0]


def detailed_verification(train: pd.DataFrame):
    """详细验证最佳公式"""
    forward_returns = train['forward_returns']
    official = train['market_forward_excess_returns']
    risk_free = train['risk_free_rate']
    
    # 最佳公式: forward_returns - risk_free
    calculated = forward_returns - risk_free
    
    print("\n" + "=" * 70)
    print("【详细验证: forward_returns - risk_free】")
    print("=" * 70)
    
    # 基本统计
    corr = calculated.corr(official)
    mae = (calculated - official).abs().mean()
    max_diff = (calculated - official).abs().max()
    
    print(f"\n验证结果:")
    print(f"  相关性: {corr:.6f}")
    print(f"  平均绝对误差: {mae:.8f}")
    print(f"  最大误差: {max_diff:.8f}")
    
    # 残差分析
    residual = official - calculated
    print(f"\n残差分析:")
    print(f"  残差均值: {residual.mean():.8f}")
    print(f"  残差标准差: {residual.std():.8f}")
    
    # 检查MAD Winsorization的影响
    print(f"\n【Winsorization检验】")
    winsorized, lower, upper = mad_winsorize(calculated, threshold=4)
    corr_win = winsorized.corr(official)
    print(f"  MAD(4)边界: [{lower:.6f}, {upper:.6f}]")
    print(f"  官方边界: [{official.min():.6f}, {official.max():.6f}]")
    print(f"  Winsorize后相关性: {corr_win:.6f}")
    
    # 官方边界更宽，说明可能没有严格的Winsorization
    print(f"\n结论: 官方边界更宽，Winsorization影响较小")
    
    return calculated, official


def plot_verification(calculated: pd.Series, official: pd.Series, train: pd.DataFrame):
    """绘制验证图表"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 时序对比 (前500天)
    ax = axes[0, 0]
    ax.plot(calculated.values[:500], alpha=0.7, label='计算值', linewidth=0.8)
    ax.plot(official.values[:500], alpha=0.7, label='官方值', linewidth=0.8)
    ax.set_title('前500天对比')
    ax.legend()
    ax.set_xlabel('日期ID')
    ax.set_ylabel('超额收益')
    
    # 2. 散点图
    ax = axes[0, 1]
    corr = calculated.corr(official)
    ax.scatter(calculated, official, alpha=0.2, s=1)
    lim = max(abs(official.min()), abs(official.max()))
    ax.plot([-lim, lim], [-lim, lim], 'r--', label='y=x')
    ax.set_xlabel('计算值')
    ax.set_ylabel('官方值')
    ax.set_title(f'散点图 (相关性={corr:.4f})')
    ax.legend()
    
    # 3. 残差分布
    ax = axes[1, 0]
    residual = official - calculated
    ax.hist(residual, bins=50, alpha=0.7, edgecolor='white', color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--')
    ax.axvline(x=residual.mean(), color='orange', linestyle='--', label=f'均值={residual.mean():.6f}')
    ax.set_title('残差分布')
    ax.set_xlabel('残差 (官方 - 计算)')
    ax.legend()
    
    # 4. expanding mean可视化
    ax = axes[1, 1]
    forward_returns = train['forward_returns']
    expanding_mean = forward_returns.expanding(min_periods=1).mean()
    ax.plot(train['date'], forward_returns, alpha=0.3, linewidth=0.5, label='forward_returns')
    ax.plot(train['date'], expanding_mean, color='red', linewidth=1, label='expanding_mean')
    ax.set_title('Expanding Mean (预期收益)')
    ax.set_xlabel('日期')
    ax.set_ylabel('收益率')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_formula_verification.png', dpi=150)
    plt.close()
    print(f"\n图表已保存: target_formula_verification.png")


def print_conclusion():
    """打印最终结论"""
    print("\n" + "=" * 70)
    print("【最终结论】")
    print("=" * 70)
    print("""
官方计算方式 (相关性 0.999978):

  market_forward_excess_returns = forward_returns - risk_free_rate

  其中:
  - forward_returns: 次日市场收益
  - risk_free_rate: 无风险利率 (日化)
  
  含义:
  - 超额收益 = 市场收益 - 无风险收益
  - 标准的超额收益定义
  
  注意:
  - 残差极小(~0.0003)，可能是精度或额外处理
  - 对建模影响可忽略
""")


def main():
    train = load_data()
    print(f"数据加载完成: {train.shape}")
    
    # 验证公式
    verify_formula(train)
    
    # 详细验证
    calculated, official = detailed_verification(train)
    
    # 绘图
    plot_verification(calculated, official, train)
    
    # 结论
    print_conclusion()


if __name__ == "__main__":
    main()
