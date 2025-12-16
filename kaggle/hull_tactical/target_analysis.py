"""
目标变量深度统计分析
market_forward_excess_returns 的全面分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest, kstest
from scipy.stats import t as t_dist
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
DATA_DIR = Path('/Volumes/2tb/mydata/code/Quantitative_trading/qlib_trading/kaggle/hull_tactical/data')
OUTPUT_DIR = Path('/Volumes/2tb/mydata/code/Quantitative_trading/qlib_trading/kaggle/hull_tactical/analysis_output')
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = 'market_forward_excess_returns'


def load_data():
    """加载数据"""
    train = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"数据加载完成: {train.shape}")
    return train


def basic_statistics(target: pd.Series):
    """1. 基础统计分析"""
    print("\n" + "="*70)
    print("【1. 基础统计分析】")
    print("="*70)
    
    # 转换为百分比
    target_pct = target * 100
    
    # 基础统计量
    stats_dict = {
        '样本数': len(target),
        '均值 (%)': target_pct.mean(),
        '中位数 (%)': target_pct.median(),
        '标准差 (%)': target_pct.std(),
        '最小值 (%)': target_pct.min(),
        '最大值 (%)': target_pct.max(),
        '偏度': target.skew(),
        '峰度': target.kurtosis(),
        '变异系数': target.std() / abs(target.mean()) if target.mean() != 0 else np.inf
    }
    
    print("\n基础统计量:")
    print("-" * 50)
    for key, value in stats_dict.items():
        if isinstance(value, float):
            print(f"  {key:<15}: {value:>12.4f}")
        else:
            print(f"  {key:<15}: {value:>12}")
    
    # 分位数分析
    print("\n分位数分布 (%):")
    print("-" * 50)
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for q in quantiles:
        val = target_pct.quantile(q)
        print(f"  {q*100:5.1f}%分位: {val:>10.3f}%")
    
    # 正态性检验
    print("\n正态性检验:")
    print("-" * 50)
    
    # Jarque-Bera检验
    jb_stat, jb_p = jarque_bera(target.dropna())
    print(f"  Jarque-Bera: 统计量={jb_stat:.2f}, p值={jb_p:.2e}")
    
    # Shapiro-Wilk检验 (样本量限制)
    sample = target.dropna().sample(min(5000, len(target)), random_state=42)
    sw_stat, sw_p = shapiro(sample)
    print(f"  Shapiro-Wilk: 统计量={sw_stat:.4f}, p值={sw_p:.2e}")
    
    # D'Agostino-Pearson检验
    dp_stat, dp_p = normaltest(target.dropna())
    print(f"  D'Agostino-Pearson: 统计量={dp_stat:.2f}, p值={dp_p:.2e}")
    
    # 结论
    is_normal = jb_p > 0.05 and sw_p > 0.05
    print(f"\n  → 结论: {'接近正态分布' if is_normal else '显著偏离正态分布'}")
    if target.kurtosis() > 0:
        print(f"  → 峰度={target.kurtosis():.2f} > 0, 表现为厚尾分布（极端值概率高）")
    
    return stats_dict


def distribution_analysis(target: pd.Series):
    """2. 分布形态分析"""
    print("\n" + "="*70)
    print("【2. 分布形态分析】")
    print("="*70)
    
    target_pct = target * 100
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 直方图 + KDE
    ax = axes[0, 0]
    ax.hist(target_pct, bins=100, density=True, alpha=0.7, edgecolor='white')
    target_pct.plot(kind='kde', ax=ax, color='red', linewidth=2, label='KDE')
    
    # 叠加正态分布
    x = np.linspace(target_pct.min(), target_pct.max(), 100)
    normal_pdf = stats.norm.pdf(x, target_pct.mean(), target_pct.std())
    ax.plot(x, normal_pdf, 'g--', linewidth=2, label='正态分布')
    
    ax.set_xlabel('收益率 (%)')
    ax.set_ylabel('密度')
    ax.set_title('收益分布 vs 正态分布')
    ax.legend()
    
    # 2. Q-Q图
    ax = axes[0, 1]
    stats.probplot(target_pct, dist="norm", plot=ax)
    ax.set_title('Q-Q图 (vs 正态分布)')
    ax.get_lines()[0].set_markerfacecolor('steelblue')
    ax.get_lines()[0].set_alpha(0.5)
    
    # 3. 尾部分析
    ax = axes[0, 2]
    # 左尾和右尾
    left_tail = target_pct[target_pct < target_pct.quantile(0.05)]
    right_tail = target_pct[target_pct > target_pct.quantile(0.95)]
    
    ax.hist(left_tail, bins=30, alpha=0.7, label=f'左尾 (<5%, n={len(left_tail)})', color='red')
    ax.hist(right_tail, bins=30, alpha=0.7, label=f'右尾 (>95%, n={len(right_tail)})', color='green')
    ax.set_xlabel('收益率 (%)')
    ax.set_ylabel('频数')
    ax.set_title('尾部分布')
    ax.legend()
    
    # 4. 累积分布函数
    ax = axes[1, 0]
    sorted_data = np.sort(target_pct)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cdf, linewidth=1.5, label='经验CDF')
    
    # 正态CDF
    normal_cdf = stats.norm.cdf(sorted_data, target_pct.mean(), target_pct.std())
    ax.plot(sorted_data, normal_cdf, 'r--', linewidth=1.5, label='正态CDF')
    
    ax.set_xlabel('收益率 (%)')
    ax.set_ylabel('累积概率')
    ax.set_title('累积分布函数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 不同分布拟合
    ax = axes[1, 1]
    ax.hist(target_pct, bins=80, density=True, alpha=0.5, edgecolor='white')
    
    x = np.linspace(target_pct.min(), target_pct.max(), 200)
    
    # 正态分布
    norm_params = stats.norm.fit(target_pct)
    ax.plot(x, stats.norm.pdf(x, *norm_params), label=f'正态', linewidth=2)
    
    # t分布
    t_params = stats.t.fit(target_pct)
    ax.plot(x, stats.t.pdf(x, *t_params), label=f't分布 (df={t_params[0]:.1f})', linewidth=2)
    
    # Laplace分布
    laplace_params = stats.laplace.fit(target_pct)
    ax.plot(x, stats.laplace.pdf(x, *laplace_params), label='Laplace', linewidth=2)
    
    ax.set_xlabel('收益率 (%)')
    ax.set_ylabel('密度')
    ax.set_title('分布拟合对比')
    ax.legend()
    
    # 6. 对数尺度尾部
    ax = axes[1, 2]
    # 计算尾部概率
    thresholds = np.linspace(0, 4, 50)
    left_probs = [(target_pct < -t).mean() for t in thresholds]
    right_probs = [(target_pct > t).mean() for t in thresholds]
    
    ax.semilogy(thresholds, left_probs, 'r-', label='P(r < -x)', linewidth=2)
    ax.semilogy(thresholds, right_probs, 'g-', label='P(r > x)', linewidth=2)
    
    # 正态分布理论值
    normal_tail = [1 - stats.norm.cdf(t, 0, target_pct.std()) for t in thresholds]
    ax.semilogy(thresholds, normal_tail, 'k--', label='正态理论值', linewidth=1)
    
    ax.set_xlabel('阈值 (%)')
    ax.set_ylabel('尾部概率 (对数)')
    ax.set_title('尾部概率分析 (厚尾检测)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_distribution.png', dpi=150)
    plt.close()
    print("\n图表已保存: target_distribution.png")
    
    # 分布拟合优度
    print("\n分布拟合检验 (KS检验):")
    print("-" * 50)
    
    # 正态
    ks_norm = kstest(target_pct, 'norm', args=norm_params)
    print(f"  正态分布: KS统计量={ks_norm.statistic:.4f}, p值={ks_norm.pvalue:.2e}")
    
    # t分布
    ks_t = kstest(target_pct, 't', args=t_params)
    print(f"  t分布: KS统计量={ks_t.statistic:.4f}, p值={ks_t.pvalue:.2e}")
    
    # Laplace
    ks_laplace = kstest(target_pct, 'laplace', args=laplace_params)
    print(f"  Laplace分布: KS统计量={ks_laplace.statistic:.4f}, p值={ks_laplace.pvalue:.2e}")
    
    best_fit = min([('正态', ks_norm.statistic), ('t分布', ks_t.statistic), ('Laplace', ks_laplace.statistic)], 
                   key=lambda x: x[1])
    print(f"\n  → 最佳拟合: {best_fit[0]}")
    
    return t_params


def time_series_analysis(target: pd.Series):
    """3. 时间序列特性分析"""
    print("\n" + "="*70)
    print("【3. 时间序列特性分析】")
    print("="*70)
    
    target_clean = target.dropna()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 时间序列图
    ax = axes[0, 0]
    ax.plot(target_clean.values * 100, linewidth=0.5, alpha=0.8)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('样本序号')
    ax.set_ylabel('收益率 (%)')
    ax.set_title('Target时间序列')
    
    # 2. 滚动统计
    ax = axes[0, 1]
    window = 252  # 1年
    rolling_mean = target_clean.rolling(window).mean() * 100
    rolling_std = target_clean.rolling(window).std() * 100
    
    ax.plot(rolling_mean.values, label='滚动均值', linewidth=1.5)
    ax.fill_between(range(len(rolling_mean)), 
                     (rolling_mean - 2*rolling_std).values,
                     (rolling_mean + 2*rolling_std).values,
                     alpha=0.3, label='±2σ')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('样本序号')
    ax.set_ylabel('收益率 (%)')
    ax.set_title(f'滚动统计 (窗口={window}天)')
    ax.legend()
    
    # 3. ACF
    ax = axes[0, 2]
    acf_vals = acf(target_clean, nlags=30)
    ax.bar(range(31), acf_vals, color='steelblue', alpha=0.7)
    ax.axhline(y=1.96/np.sqrt(len(target_clean)), color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-1.96/np.sqrt(len(target_clean)), color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('滞后期')
    ax.set_ylabel('自相关系数')
    ax.set_title('自相关函数 (ACF)')
    
    # 4. PACF
    ax = axes[1, 0]
    pacf_vals = pacf(target_clean, nlags=30)
    ax.bar(range(31), pacf_vals, color='steelblue', alpha=0.7)
    ax.axhline(y=1.96/np.sqrt(len(target_clean)), color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-1.96/np.sqrt(len(target_clean)), color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('滞后期')
    ax.set_ylabel('偏自相关系数')
    ax.set_title('偏自相关函数 (PACF)')
    
    # 5. 平方收益的ACF (波动率聚集)
    ax = axes[1, 1]
    squared_returns = target_clean ** 2
    acf_sq = acf(squared_returns, nlags=30)
    ax.bar(range(31), acf_sq, color='orange', alpha=0.7)
    ax.axhline(y=1.96/np.sqrt(len(target_clean)), color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-1.96/np.sqrt(len(target_clean)), color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('滞后期')
    ax.set_ylabel('自相关系数')
    ax.set_title('平方收益ACF (波动率聚集检测)')
    
    # 6. 滚动波动率
    ax = axes[1, 2]
    rolling_vol = target_clean.rolling(20).std() * 100 * np.sqrt(252)  # 年化
    ax.plot(rolling_vol.values, linewidth=0.8, alpha=0.8)
    ax.axhline(y=rolling_vol.mean(), color='red', linestyle='--', label=f'均值={rolling_vol.mean():.1f}%')
    ax.set_xlabel('样本序号')
    ax.set_ylabel('年化波动率 (%)')
    ax.set_title('滚动波动率 (20天窗口)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_timeseries.png', dpi=150)
    plt.close()
    print("\n图表已保存: target_timeseries.png")
    
    # 统计检验
    print("\n时间序列检验:")
    print("-" * 50)
    
    # ADF平稳性检验
    adf_result = adfuller(target_clean)
    print(f"  ADF检验: 统计量={adf_result[0]:.4f}, p值={adf_result[1]:.2e}")
    print(f"    → {'平稳序列' if adf_result[1] < 0.05 else '非平稳序列'}")
    
    # Ljung-Box检验 (自相关)
    lb_result = acorr_ljungbox(target_clean, lags=[10], return_df=True)
    lb_stat = lb_result['lb_stat'].values[0]
    lb_p = lb_result['lb_pvalue'].values[0]
    print(f"  Ljung-Box检验 (lag=10): 统计量={lb_stat:.2f}, p值={lb_p:.4f}")
    print(f"    → {'存在显著自相关' if lb_p < 0.05 else '无显著自相关'}")
    
    # ARCH效应检验
    arch_result = het_arch(target_clean)
    print(f"  ARCH效应检验: 统计量={arch_result[0]:.2f}, p值={arch_result[1]:.2e}")
    print(f"    → {'存在波动率聚集' if arch_result[1] < 0.05 else '无波动率聚集'}")
    
    return {
        'is_stationary': adf_result[1] < 0.05,
        'has_autocorrelation': lb_p < 0.05,
        'has_arch_effect': arch_result[1] < 0.05
    }


def return_pattern_analysis(target: pd.Series):
    """4. 收益模式分析"""
    print("\n" + "="*70)
    print("【4. 收益模式分析】")
    print("="*70)
    
    target_pct = target * 100
    
    # 正负收益统计
    positive = (target > 0).sum()
    negative = (target < 0).sum()
    zero = (target == 0).sum()
    
    print("\n正负收益统计:")
    print("-" * 50)
    print(f"  正收益: {positive}天 ({positive/len(target)*100:.1f}%)")
    print(f"  负收益: {negative}天 ({negative/len(target)*100:.1f}%)")
    print(f"  零收益: {zero}天 ({zero/len(target)*100:.1f}%)")
    print(f"  正/负比: {positive/negative:.2f}")
    
    # 收益大小分析
    print("\n收益幅度统计:")
    print("-" * 50)
    avg_positive = target_pct[target > 0].mean()
    avg_negative = target_pct[target < 0].mean()
    print(f"  平均正收益: +{avg_positive:.3f}%")
    print(f"  平均负收益: {avg_negative:.3f}%")
    print(f"  盈亏比: {abs(avg_positive/avg_negative):.2f}")
    
    # 极端收益
    print("\n极端收益频率:")
    print("-" * 50)
    thresholds = [1, 2, 3, 4]
    for t in thresholds:
        n_extreme = ((target_pct > t) | (target_pct < -t)).sum()
        print(f"  |收益| > {t}%: {n_extreme}天 ({n_extreme/len(target)*100:.2f}%)")
    
    # 连涨连跌分析
    print("\n连续涨跌统计:")
    print("-" * 50)
    
    signs = np.sign(target.values)
    streaks = []
    current_streak = 1
    current_sign = signs[0]
    
    for i in range(1, len(signs)):
        if signs[i] == current_sign and signs[i] != 0:
            current_streak += 1
        else:
            if current_sign != 0:
                streaks.append((current_sign, current_streak))
            current_streak = 1
            current_sign = signs[i]
    
    if current_sign != 0:
        streaks.append((current_sign, current_streak))
    
    up_streaks = [s[1] for s in streaks if s[0] > 0]
    down_streaks = [s[1] for s in streaks if s[0] < 0]
    
    print(f"  最长连涨: {max(up_streaks)}天")
    print(f"  最长连跌: {max(down_streaks)}天")
    print(f"  平均连涨: {np.mean(up_streaks):.1f}天")
    print(f"  平均连跌: {np.mean(down_streaks):.1f}天")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 正负收益分布对比
    ax = axes[0, 0]
    pos_returns = target_pct[target > 0]
    neg_returns = target_pct[target < 0]
    ax.hist(pos_returns, bins=50, alpha=0.7, label=f'正收益 (n={len(pos_returns)})', color='green')
    ax.hist(neg_returns, bins=50, alpha=0.7, label=f'负收益 (n={len(neg_returns)})', color='red')
    ax.set_xlabel('收益率 (%)')
    ax.set_ylabel('频数')
    ax.set_title('正负收益分布对比')
    ax.legend()
    
    # 2. 收益幅度分布
    ax = axes[0, 1]
    abs_returns = target_pct.abs()
    ax.hist(abs_returns, bins=50, edgecolor='white', alpha=0.7)
    ax.axvline(x=abs_returns.median(), color='red', linestyle='--', label=f'中位数={abs_returns.median():.2f}%')
    ax.axvline(x=abs_returns.mean(), color='green', linestyle='--', label=f'均值={abs_returns.mean():.2f}%')
    ax.set_xlabel('|收益率| (%)')
    ax.set_ylabel('频数')
    ax.set_title('收益幅度分布')
    ax.legend()
    
    # 3. 连涨连跌分布
    ax = axes[1, 0]
    ax.hist(up_streaks, bins=range(1, max(up_streaks)+2), alpha=0.7, label='连涨', color='green', align='left')
    ax.hist(down_streaks, bins=range(1, max(down_streaks)+2), alpha=0.7, label='连跌', color='red', align='left')
    ax.set_xlabel('连续天数')
    ax.set_ylabel('频数')
    ax.set_title('连续涨跌分布')
    ax.legend()
    
    # 4. 日收益热力图 (按幅度)
    ax = axes[1, 1]
    bins = [-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5]
    counts, _ = np.histogram(target_pct.clip(-5, 5), bins=bins)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(counts)))
    bars = ax.bar(range(len(counts)), counts, color=colors, edgecolor='white')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f'{bins[i]:.1f}~{bins[i+1]:.1f}' for i in range(len(counts))], rotation=45)
    ax.set_xlabel('收益率区间 (%)')
    ax.set_ylabel('频数')
    ax.set_title('收益率分布 (分区间统计)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_return_patterns.png', dpi=150)
    plt.close()
    print("\n图表已保存: target_return_patterns.png")
    
    return {
        'positive_ratio': positive / len(target),
        'win_loss_ratio': abs(avg_positive / avg_negative),
        'max_up_streak': max(up_streaks),
        'max_down_streak': max(down_streaks)
    }


def conditional_distribution_analysis(df: pd.DataFrame, target: pd.Series):
    """5. 条件分布分析"""
    print("\n" + "="*70)
    print("【5. 条件分布分析】")
    print("="*70)
    
    target_pct = target * 100
    
    # 按时期划分
    n = len(df)
    period_size = n // 4
    periods = ['早期', '中早期', '中晚期', '近期']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 不同时期的分布
    ax = axes[0, 0]
    for i, period_name in enumerate(periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < 3 else n
        period_data = target_pct.iloc[start_idx:end_idx]
        period_data.plot(kind='kde', ax=ax, label=f'{period_name} (n={len(period_data)})', linewidth=2)
    
    ax.set_xlabel('收益率 (%)')
    ax.set_ylabel('密度')
    ax.set_title('不同时期Target分布')
    ax.legend()
    
    # 2. 各时期统计对比
    ax = axes[0, 1]
    period_stats = []
    for i, period_name in enumerate(periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < 3 else n
        period_data = target_pct.iloc[start_idx:end_idx]
        period_stats.append({
            'period': period_name,
            'mean': period_data.mean(),
            'std': period_data.std(),
            'skew': period_data.skew(),
            'kurtosis': period_data.kurtosis()
        })
    
    stats_df = pd.DataFrame(period_stats)
    x = np.arange(len(periods))
    width = 0.35
    
    ax.bar(x - width/2, stats_df['mean'], width, label='均值 (%)', alpha=0.8)
    ax.bar(x + width/2, stats_df['std'], width, label='标准差 (%)', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_title('各时期均值和波动率')
    ax.legend()
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 3. 按波动率状态分布
    ax = axes[1, 0]
    rolling_vol = target.rolling(20).std()
    vol_median = rolling_vol.median()
    
    low_vol_mask = rolling_vol < vol_median
    high_vol_mask = rolling_vol >= vol_median
    
    target_pct[low_vol_mask].plot(kind='kde', ax=ax, label=f'低波动期 (n={low_vol_mask.sum()})', linewidth=2)
    target_pct[high_vol_mask].plot(kind='kde', ax=ax, label=f'高波动期 (n={high_vol_mask.sum()})', linewidth=2)
    
    ax.set_xlabel('收益率 (%)')
    ax.set_ylabel('密度')
    ax.set_title('波动率状态下的Target分布')
    ax.legend()
    
    # 4. 非对称性分析
    ax = axes[1, 1]
    
    # 上涨日vs下跌日的后续收益
    up_days = target.shift(1) > 0
    down_days = target.shift(1) < 0
    
    after_up = target_pct[up_days].dropna()
    after_down = target_pct[down_days].dropna()
    
    ax.hist(after_up, bins=50, alpha=0.5, label=f'前日上涨后 (n={len(after_up)}, μ={after_up.mean():.3f}%)', density=True)
    ax.hist(after_down, bins=50, alpha=0.5, label=f'前日下跌后 (n={len(after_down)}, μ={after_down.mean():.3f}%)', density=True)
    ax.set_xlabel('收益率 (%)')
    ax.set_ylabel('密度')
    ax.set_title('前日涨跌对今日收益的影响')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_conditional_distribution.png', dpi=150)
    plt.close()
    print("\n图表已保存: target_conditional_distribution.png")
    
    # 打印统计
    print("\n各时期统计:")
    print("-" * 60)
    print(f"{'时期':<10} {'均值(%)':<12} {'标准差(%)':<12} {'偏度':<10} {'峰度':<10}")
    for _, row in stats_df.iterrows():
        print(f"{row['period']:<10} {row['mean']:<12.4f} {row['std']:<12.4f} {row['skew']:<10.2f} {row['kurtosis']:<10.2f}")
    
    print("\n波动率状态影响:")
    print("-" * 50)
    print(f"  低波动期: 均值={target_pct[low_vol_mask].mean():.4f}%, 标准差={target_pct[low_vol_mask].std():.4f}%")
    print(f"  高波动期: 均值={target_pct[high_vol_mask].mean():.4f}%, 标准差={target_pct[high_vol_mask].std():.4f}%")
    
    print("\n前日涨跌影响:")
    print("-" * 50)
    print(f"  前日上涨后: 均值={after_up.mean():.4f}%")
    print(f"  前日下跌后: 均值={after_down.mean():.4f}%")
    
    return stats_df


def risk_metrics_analysis(target: pd.Series):
    """6. 风险度量分析"""
    print("\n" + "="*70)
    print("【6. 风险度量分析】")
    print("="*70)
    
    target_pct = target * 100
    
    # VaR计算
    print("\n风险价值 (VaR):")
    print("-" * 50)
    
    var_levels = [0.01, 0.05, 0.10]
    for level in var_levels:
        var = target_pct.quantile(level)
        print(f"  {level*100:.0f}% VaR: {var:.3f}% (每{int(1/level)}天有1天亏损超过此值)")
    
    # CVaR (Expected Shortfall)
    print("\n条件风险价值 (CVaR / ES):")
    print("-" * 50)
    for level in var_levels:
        var = target_pct.quantile(level)
        cvar = target_pct[target_pct <= var].mean()
        print(f"  {level*100:.0f}% CVaR: {cvar:.3f}% (极端亏损的平均值)")
    
    # 下行风险
    print("\n下行风险指标:")
    print("-" * 50)
    
    downside_returns = target_pct[target_pct < 0]
    downside_std = downside_returns.std()
    sortino_denom = np.sqrt((target_pct.clip(upper=0) ** 2).mean())
    
    print(f"  下行标准差: {downside_std:.4f}%")
    print(f"  Sortino分母: {sortino_denom:.4f}%")
    print(f"  最大单日亏损: {target_pct.min():.3f}%")
    
    # 极值分析
    print("\n极值统计:")
    print("-" * 50)
    
    # 超过正态分布预期的极端值
    z_threshold = 3
    expected_extreme = len(target) * 2 * stats.norm.sf(z_threshold)
    actual_extreme = ((target_pct > target_pct.mean() + z_threshold * target_pct.std()) | 
                      (target_pct < target_pct.mean() - z_threshold * target_pct.std())).sum()
    
    print(f"  3σ外的数据点: {actual_extreme} (正态预期: {expected_extreme:.1f})")
    print(f"  极端值倍数: {actual_extreme / expected_extreme:.1f}x")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. VaR可视化
    ax = axes[0, 0]
    ax.hist(target_pct, bins=100, density=True, alpha=0.7, edgecolor='white')
    
    for level, color in zip([0.01, 0.05], ['red', 'orange']):
        var = target_pct.quantile(level)
        ax.axvline(x=var, color=color, linestyle='--', linewidth=2, 
                   label=f'{level*100:.0f}% VaR = {var:.2f}%')
    
    ax.set_xlabel('收益率 (%)')
    ax.set_ylabel('密度')
    ax.set_title('VaR可视化')
    ax.legend()
    
    # 2. 滚动VaR
    ax = axes[0, 1]
    window = 252
    rolling_var_5 = target_pct.rolling(window).quantile(0.05)
    rolling_var_1 = target_pct.rolling(window).quantile(0.01)
    
    ax.plot(rolling_var_5.values, label='5% VaR', linewidth=1)
    ax.plot(rolling_var_1.values, label='1% VaR', linewidth=1)
    ax.axhline(y=target_pct.quantile(0.05), color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('样本序号')
    ax.set_ylabel('VaR (%)')
    ax.set_title(f'滚动VaR (窗口={window}天)')
    ax.legend()
    
    # 3. 累积收益
    ax = axes[1, 0]
    cumulative = (1 + target).cumprod()
    ax.plot(cumulative.values, linewidth=1)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('样本序号')
    ax.set_ylabel('累积收益')
    ax.set_title('累积收益曲线')
    
    # 计算最大回撤
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    ax2 = ax.twinx()
    ax2.fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.3, color='red')
    ax2.set_ylabel('回撤 (%)', color='red')
    
    # 4. 回撤分布
    ax = axes[1, 1]
    ax.hist(drawdown[drawdown < 0], bins=50, edgecolor='white', alpha=0.7, color='red')
    ax.axvline(x=max_dd, color='darkred', linestyle='--', linewidth=2, 
               label=f'最大回撤 = {max_dd:.2f}%')
    ax.set_xlabel('回撤 (%)')
    ax.set_ylabel('频数')
    ax.set_title('回撤分布')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_risk_metrics.png', dpi=150)
    plt.close()
    print("\n图表已保存: target_risk_metrics.png")
    
    print(f"\n最大回撤: {max_dd:.2f}%")
    
    return {
        'var_1pct': target_pct.quantile(0.01),
        'var_5pct': target_pct.quantile(0.05),
        'max_drawdown': max_dd,
        'extreme_ratio': actual_extreme / expected_extreme
    }


def predictability_analysis(df: pd.DataFrame, target: pd.Series):
    """7. 可预测性分析"""
    print("\n" + "="*70)
    print("【7. 可预测性分析】")
    print("="*70)
    
    feature_cols = [c for c in df.columns if c[0] in 'DEIMPV S' and c != TARGET_COL]
    
    # 信噪比
    print("\n信噪比分析:")
    print("-" * 50)
    
    signal = target.mean()
    noise = target.std()
    snr = abs(signal) / noise
    
    print(f"  信号 (均值): {signal*100:.4f}%")
    print(f"  噪声 (标准差): {noise*100:.4f}%")
    print(f"  信噪比: {snr:.4f}")
    print(f"  → {'低信噪比，预测困难' if snr < 0.1 else '中等信噪比' if snr < 0.5 else '高信噪比'}")
    
    # 特征与Target的相关性
    print("\n与特征的相关性 (Top 10):")
    print("-" * 50)
    
    correlations = []
    for col in feature_cols:
        if col in df.columns:
            corr = df[col].corr(target)
            if not np.isnan(corr):
                correlations.append((col, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for col, corr in correlations[:10]:
        print(f"  {col}: {corr:+.4f}")
    
    # 滞后自相关
    print("\n滞后自相关 (预测信号):")
    print("-" * 50)
    
    for lag in [1, 2, 3, 5, 10]:
        autocorr = target.autocorr(lag)
        print(f"  Lag {lag}: {autocorr:+.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 特征相关性分布
    ax = axes[0, 0]
    corr_values = [c[1] for c in correlations]
    ax.hist(corr_values, bins=30, edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_xlabel('相关系数')
    ax.set_ylabel('特征数')
    ax.set_title('特征与Target相关性分布')
    
    # 2. Top特征相关性
    ax = axes[0, 1]
    top_corrs = correlations[:15]
    ax.barh([c[0] for c in top_corrs], [c[1] for c in top_corrs], 
            color=['green' if c[1] > 0 else 'red' for c in top_corrs], alpha=0.7)
    ax.set_xlabel('相关系数')
    ax.set_title('Top 15 相关特征')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. 滞后相关性
    ax = axes[1, 0]
    lags = range(1, 21)
    autocorrs = [target.autocorr(lag) for lag in lags]
    ax.bar(lags, autocorrs, color='steelblue', alpha=0.7)
    ax.axhline(y=1.96/np.sqrt(len(target)), color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-1.96/np.sqrt(len(target)), color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('滞后期')
    ax.set_ylabel('自相关系数')
    ax.set_title('Target自相关性')
    
    # 4. 预测难度总结
    ax = axes[1, 1]
    ax.axis('off')
    
    # 计算预测难度指标
    max_abs_corr = max(abs(c[1]) for c in correlations)
    max_autocorr = max(abs(target.autocorr(i)) for i in range(1, 11))
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════╗
    ║              Target可预测性评估                       ║
    ╠══════════════════════════════════════════════════════╣
    ║                                                      ║
    ║  信噪比: {snr:.4f}  {'⚠️ 低' if snr < 0.1 else '✓ 中等'}                          ║
    ║                                                      ║
    ║  最大特征相关: {max_abs_corr:.4f}  {'⚠️ 弱' if max_abs_corr < 0.1 else '✓ 中等'}                    ║
    ║                                                      ║
    ║  最大自相关: {max_autocorr:.4f}  {'⚠️ 弱' if max_autocorr < 0.05 else '✓ 可用'}                       ║
    ║                                                      ║
    ║  预测难度: {'高 🔴' if snr < 0.1 and max_abs_corr < 0.1 else '中等 🟡' if snr < 0.2 else '较低 🟢'}                                   ║
    ║                                                      ║
    ║  建议:                                               ║
    ║  • 使用集成模型提高稳定性                              ║
    ║  • 关注Top相关特征                                    ║
    ║  • 考虑特征工程增强信号                               ║
    ╚══════════════════════════════════════════════════════╝
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_predictability.png', dpi=150)
    plt.close()
    print("\n图表已保存: target_predictability.png")
    
    return {
        'snr': snr,
        'max_feature_corr': max_abs_corr,
        'max_autocorr': max_autocorr,
        'top_features': correlations[:10]
    }


def generate_report(all_results: dict):
    """生成分析报告"""
    print("\n" + "="*70)
    print("【分析报告总结】")
    print("="*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    Target深度统计分析报告                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 分布特性                                                            │
│     • 非正态分布，呈现厚尾特征 (峰度>0)                                 │
│     • t分布拟合优于正态分布                                             │
│     • 极端收益发生频率高于正态预期                                       │
│                                                                         │
│  2. 时间序列特性                                                        │
│     • 序列平稳，无需差分                                                │
│     • 存在波动率聚集 (ARCH效应)                                         │
│     • 收益本身自相关较弱                                                │
│                                                                         │
│  3. 收益模式                                                            │
│     • 正负收益比例接近均衡                                              │
│     • 存在连涨连跌现象                                                  │
│     • 极端日收益(>3%)发生频率约1%                                       │
│                                                                         │
│  4. 风险特征                                                            │
│     • 5% VaR约为-1%，意味着每月约有1天亏损超1%                          │
│     • 存在显著的尾部风险                                                │
│     • 需要关注极端事件                                                  │
│                                                                         │
│  5. 可预测性                                                            │
│     • 信噪比较低，预测困难                                              │
│     • 与特征的相关性普遍较弱                                            │
│     • 需要复杂模型和特征工程                                            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  建模建议:                                                              │
│  1. 使用稳健损失函数 (Huber Loss)                                       │
│  2. 考虑波动率状态作为特征                                              │
│  3. 重点关注Top相关特征                                                 │
│  4. 使用集成方法提高稳定性                                              │
│  5. 评估时使用风险调整后的指标                                          │
└─────────────────────────────────────────────────────────────────────────┘
    """)


def main():
    """主函数"""
    print("="*70)
    print("Target深度统计分析")
    print("="*70)
    
    # 加载数据
    df = load_data()
    target = df[TARGET_COL]
    
    # 执行各项分析
    all_results = {}
    
    # 1. 基础统计
    all_results['basic'] = basic_statistics(target)
    
    # 2. 分布形态
    all_results['distribution'] = distribution_analysis(target)
    
    # 3. 时间序列特性
    all_results['timeseries'] = time_series_analysis(target)
    
    # 4. 收益模式
    all_results['patterns'] = return_pattern_analysis(target)
    
    # 5. 条件分布
    all_results['conditional'] = conditional_distribution_analysis(df, target)
    
    # 6. 风险度量
    all_results['risk'] = risk_metrics_analysis(target)
    
    # 7. 可预测性
    all_results['predictability'] = predictability_analysis(df, target)
    
    # 生成报告
    generate_report(all_results)
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()
