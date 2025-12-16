"""
Hull Tactical 高级数据分析
包含: 时序分析、市场状态、特征IC、降维聚类、风险分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_PREFIXES = ["D", "E", "I", "M", "P", "S", "V"]


def load_data():
    """加载数据"""
    train = pd.read_csv(DATA_DIR / "train.csv")
    base_date = pd.Timestamp("1988-01-01")
    train["date"] = base_date + pd.to_timedelta(train["date_id"], unit="D")
    train["year"] = train["date"].dt.year
    train["month"] = train["date"].dt.month
    train["weekday"] = train["date"].dt.weekday
    return train


def get_feature_cols(df):
    """获取所有特征列"""
    cols = []
    for col in df.columns:
        for p in FEATURE_PREFIXES:
            if col.startswith(p) and len(col) > 1:
                suffix = col[1:]
                if suffix.isdigit():
                    cols.append(col)
                    break
    return cols


# ============================================================
# 1. 时序分析
# ============================================================
def analyze_time_series_properties(train: pd.DataFrame):
    """时序特性分析: ACF、平稳性、季节性"""
    print("\n" + "=" * 70)
    print("1. 时序特性分析")
    print("=" * 70)
    
    target = train["market_forward_excess_returns"].dropna()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1.1 自相关函数 (ACF)
    ax = axes[0, 0]
    lags = 60
    acf_values = [target.autocorr(lag=i) for i in range(lags)]
    ax.bar(range(lags), acf_values, color='steelblue', alpha=0.7)
    ax.axhline(y=1.96/np.sqrt(len(target)), color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-1.96/np.sqrt(len(target)), color='r', linestyle='--', alpha=0.5)
    ax.set_title("自相关函数 (ACF)", fontsize=12)
    ax.set_xlabel("滞后期")
    ax.set_ylabel("相关系数")
    
    # 1.2 偏自相关函数 (PACF) - 简化计算
    ax = axes[0, 1]
    pacf_values = []
    for lag in range(1, min(21, lags)):
        if lag == 1:
            pacf_values.append(target.autocorr(lag=1))
        else:
            # 简化的PACF估计
            pacf_values.append(target.autocorr(lag=lag))
    ax.bar(range(1, len(pacf_values)+1), pacf_values, color='teal', alpha=0.7)
    ax.axhline(y=1.96/np.sqrt(len(target)), color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-1.96/np.sqrt(len(target)), color='r', linestyle='--', alpha=0.5)
    ax.set_title("滞后相关性 (前20期)", fontsize=12)
    ax.set_xlabel("滞后期")
    ax.set_ylabel("相关系数")
    
    # 1.3 滚动均值和标准差 (平稳性检验)
    ax = axes[0, 2]
    window = 252
    rolling_mean = target.rolling(window).mean()
    rolling_std = target.rolling(window).std()
    ax.plot(train["date"], rolling_mean, label="滚动均值", alpha=0.8)
    ax.plot(train["date"], rolling_std, label="滚动标准差", alpha=0.8)
    ax.set_title(f"滚动统计量 ({window}日)", fontsize=12)
    ax.legend()
    ax.set_xlabel("日期")
    
    # 1.4 月度季节性
    ax = axes[1, 0]
    monthly_returns = train.groupby("month")["market_forward_excess_returns"].mean()
    colors = ['green' if x > 0 else 'red' for x in monthly_returns]
    monthly_returns.plot(kind='bar', ax=ax, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title("月度季节性效应", fontsize=12)
    ax.set_xlabel("月份")
    ax.set_ylabel("平均超额收益")
    ax.set_xticklabels(range(1, 13), rotation=0)
    
    # 1.5 星期效应
    ax = axes[1, 1]
    weekday_returns = train.groupby("weekday")["market_forward_excess_returns"].mean()
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    available_weekdays = [weekday_names[i] for i in weekday_returns.index]
    colors = ['green' if x > 0 else 'red' for x in weekday_returns]
    ax.bar(available_weekdays, weekday_returns.values, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title("星期效应", fontsize=12)
    ax.set_ylabel("平均超额收益")
    
    # 1.6 年度效应
    ax = axes[1, 2]
    yearly_returns = train.groupby("year")["market_forward_excess_returns"].mean()
    colors = ['green' if x > 0 else 'red' for x in yearly_returns]
    yearly_returns.plot(kind='bar', ax=ax, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title("年度效应", fontsize=12)
    ax.set_xlabel("年份")
    ax.set_ylabel("平均超额收益")
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_series_properties.png", dpi=150)
    plt.close()
    print(f"图表已保存: time_series_properties.png")
    
    # 打印统计检验结果
    print("\n季节性效应统计:")
    print(f"  最佳月份: {monthly_returns.idxmax()}月 ({monthly_returns.max():.6f})")
    print(f"  最差月份: {monthly_returns.idxmin()}月 ({monthly_returns.min():.6f})")
    print(f"  最佳星期: 周{weekday_returns.idxmax()+1} ({weekday_returns.max():.6f})")
    print(f"  最差星期: 周{weekday_returns.idxmin()+1} ({weekday_returns.min():.6f})")


# ============================================================
# 2. 市场状态分析
# ============================================================
def analyze_market_regimes(train: pd.DataFrame):
    """市场状态分析: 牛熊市、波动率聚类、危机时期"""
    print("\n" + "=" * 70)
    print("2. 市场状态分析")
    print("=" * 70)
    
    target = "market_forward_excess_returns"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 2.1 识别牛熊市 (基于滚动收益)
    ax = axes[0, 0]
    train["rolling_return_60"] = train[target].rolling(60).mean()
    train["regime"] = np.where(train["rolling_return_60"] > 0, "牛市", "熊市")
    
    bull_pct = (train["regime"] == "牛市").mean() * 100
    bear_pct = 100 - bull_pct
    
    ax.fill_between(train["date"], 0, train[target], 
                    where=train["regime"]=="牛市", color='green', alpha=0.3, label=f'牛市 ({bull_pct:.1f}%)')
    ax.fill_between(train["date"], 0, train[target],
                    where=train["regime"]=="熊市", color='red', alpha=0.3, label=f'熊市 ({bear_pct:.1f}%)')
    ax.set_title("牛熊市识别 (60日滚动均值)", fontsize=12)
    ax.legend()
    ax.set_xlabel("日期")
    ax.set_ylabel("超额收益")
    
    # 2.2 波动率聚类 (GARCH效应)
    ax = axes[0, 1]
    train["abs_return"] = train[target].abs()
    train["rolling_vol"] = train[target].rolling(20).std()
    
    # 波动率分位数
    vol_quantiles = train["rolling_vol"].quantile([0.25, 0.5, 0.75])
    train["vol_regime"] = pd.cut(train["rolling_vol"], 
                                  bins=[0, vol_quantiles[0.25], vol_quantiles[0.5], vol_quantiles[0.75], np.inf],
                                  labels=["低波动", "中低波动", "中高波动", "高波动"])
    
    ax.plot(train["date"], train["rolling_vol"], color='orange', alpha=0.7)
    ax.axhline(y=vol_quantiles[0.5], color='r', linestyle='--', label='中位数')
    ax.set_title("波动率聚类 (20日滚动标准差)", fontsize=12)
    ax.legend()
    ax.set_xlabel("日期")
    ax.set_ylabel("波动率")
    
    # 2.3 不同波动率环境下的收益分布
    ax = axes[1, 0]
    vol_groups = train.groupby("vol_regime")[target]
    vol_stats = vol_groups.agg(['mean', 'std', 'count'])
    
    x = range(len(vol_stats))
    colors = ['green', 'lightgreen', 'lightsalmon', 'red']
    ax.bar(x, vol_stats['mean'], color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(vol_stats.index)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title("不同波动率环境下的平均收益", fontsize=12)
    ax.set_ylabel("平均超额收益")
    
    # 添加标准差标注
    for i, (idx, row) in enumerate(vol_stats.iterrows()):
        ax.annotate(f'std={row["std"]:.4f}', (i, row['mean']), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # 2.4 历史危机时期标注
    ax = axes[1, 1]
    cumulative = (1 + train[target]).cumprod()
    ax.plot(train["date"], cumulative, color='blue', alpha=0.8)
    ax.set_yscale('log')
    
    # 标注重大事件
    events = {
        "1990-08-02": "海湾战争",
        "1997-10-27": "亚洲金融危机",
        "2000-03-10": "互联网泡沫",
        "2001-09-11": "911事件",
        "2008-09-15": "雷曼破产",
    }
    
    for date_str, event in events.items():
        event_date = pd.Timestamp(date_str)
        if event_date >= train["date"].min() and event_date <= train["date"].max():
            ax.axvline(x=event_date, color='red', linestyle='--', alpha=0.5)
            ax.text(event_date, cumulative.max(), event, rotation=90, 
                    verticalalignment='top', fontsize=8)
    
    ax.set_title("累计收益与重大事件", fontsize=12)
    ax.set_xlabel("日期")
    ax.set_ylabel("累计收益 (对数)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "market_regimes.png", dpi=150)
    plt.close()
    print(f"图表已保存: market_regimes.png")
    
    print("\n市场状态统计:")
    print(f"  牛市占比: {bull_pct:.1f}%")
    print(f"  熊市占比: {bear_pct:.1f}%")
    print(f"\n波动率环境下的收益:")
    print(vol_stats.round(6))


# ============================================================
# 3. 特征IC分析
# ============================================================
def analyze_feature_ic(train: pd.DataFrame):
    """特征信息系数 (IC) 分析"""
    print("\n" + "=" * 70)
    print("3. 特征IC分析")
    print("=" * 70)
    
    feature_cols = get_feature_cols(train)
    target = "market_forward_excess_returns"
    
    # 计算每个特征的滚动IC
    ic_results = {}
    for col in feature_cols:
        valid_data = train[[col, target]].dropna()
        if len(valid_data) > 100:
            ic = valid_data[col].corr(valid_data[target])
            ic_results[col] = ic
    
    ic_series = pd.Series(ic_results).sort_values(key=abs, ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 3.1 IC排名
    ax = axes[0, 0]
    top_20 = ic_series.head(20)
    colors = ['green' if x > 0 else 'red' for x in top_20]
    top_20.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
    ax.set_title("Top 20 特征IC值", fontsize=12)
    ax.set_xlabel("IC值")
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 3.2 IC按类别分布
    ax = axes[0, 1]
    ic_by_group = {}
    for prefix in FEATURE_PREFIXES:
        group_cols = [c for c in ic_results.keys() if c.startswith(prefix)]
        if group_cols:
            ic_by_group[prefix] = [ic_results[c] for c in group_cols]
    
    ax.boxplot(ic_by_group.values(), labels=ic_by_group.keys())
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title("各类别特征IC分布", fontsize=12)
    ax.set_ylabel("IC值")
    
    # 3.3 滚动IC (Top 3特征)
    ax = axes[1, 0]
    top_3 = ic_series.head(3).index.tolist()
    window = 252
    
    for col in top_3:
        rolling_ic = train[[col, target]].rolling(window).corr().unstack()[col][target]
        ax.plot(train["date"], rolling_ic, label=col, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_title(f"Top 3 特征滚动IC ({window}日)", fontsize=12)
    ax.legend()
    ax.set_xlabel("日期")
    ax.set_ylabel("IC值")
    
    # 3.4 IC衰减 (滞后相关性)
    ax = axes[1, 1]
    top_feature = ic_series.index[0]
    lags = range(-10, 21)
    lag_corrs = []
    
    for lag in lags:
        if lag >= 0:
            corr = train[top_feature].corr(train[target].shift(-lag))
        else:
            corr = train[top_feature].shift(lag).corr(train[target])
        lag_corrs.append(corr)
    
    ax.bar(lags, lag_corrs, color='steelblue', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title(f"{top_feature} 的滞后相关性", fontsize=12)
    ax.set_xlabel("滞后期 (负=特征领先)")
    ax.set_ylabel("相关系数")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_ic_analysis.png", dpi=150)
    plt.close()
    print(f"图表已保存: feature_ic_analysis.png")
    
    print("\nTop 10 IC值:")
    for feat, ic in ic_series.head(10).items():
        print(f"  {feat}: {ic:.4f}")


# ============================================================
# 4. PCA降维分析
# ============================================================
def analyze_pca(train: pd.DataFrame):
    """PCA降维分析"""
    print("\n" + "=" * 70)
    print("4. PCA降维分析")
    print("=" * 70)
    
    feature_cols = get_feature_cols(train)
    target = "market_forward_excess_returns"
    
    # 准备数据 (填充缺失值)
    X = train[feature_cols].fillna(train[feature_cols].median())
    y = train[target]
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 4.1 解释方差比例
    ax = axes[0, 0]
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax.bar(range(1, len(pca.explained_variance_ratio_)+1), 
           pca.explained_variance_ratio_, alpha=0.7, label='单个')
    ax.plot(range(1, len(cumsum)+1), cumsum, 'r-o', markersize=3, label='累计')
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90%阈值')
    ax.set_xlabel("主成分")
    ax.set_ylabel("解释方差比例")
    ax.set_title("PCA解释方差", fontsize=12)
    ax.legend()
    ax.set_xlim(0, 30)
    
    # 找到90%方差的成分数
    n_90 = np.argmax(cumsum >= 0.9) + 1
    print(f"解释90%方差需要 {n_90} 个主成分")
    
    # 4.2 PC1 vs PC2 散点图
    ax = axes[0, 1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn', 
                         alpha=0.3, s=5, vmin=-0.03, vmax=0.03)
    plt.colorbar(scatter, ax=ax, label='超额收益')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA投影 (颜色=收益)", fontsize=12)
    
    # 4.3 主成分载荷 (PC1)
    ax = axes[1, 0]
    loadings_pc1 = pd.Series(pca.components_[0], index=feature_cols)
    top_loadings = loadings_pc1.abs().sort_values(ascending=False).head(15)
    loadings_pc1[top_loadings.index].plot(kind='barh', ax=ax, color='steelblue', alpha=0.7)
    ax.set_title("PC1 主要载荷", fontsize=12)
    ax.set_xlabel("载荷系数")
    
    # 4.4 主成分与目标的相关性
    ax = axes[1, 1]
    pc_corrs = []
    for i in range(min(20, X_pca.shape[1])):
        corr = np.corrcoef(X_pca[:, i], y)[0, 1]
        pc_corrs.append(corr)
    
    colors = ['green' if x > 0 else 'red' for x in pc_corrs]
    ax.bar(range(1, len(pc_corrs)+1), pc_corrs, color=colors, alpha=0.7)
    ax.set_xlabel("主成分")
    ax.set_ylabel("与目标的相关系数")
    ax.set_title("主成分与超额收益的相关性", fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_analysis.png", dpi=150)
    plt.close()
    print(f"图表已保存: pca_analysis.png")


# ============================================================
# 5. 风险分析
# ============================================================
def analyze_risk(train: pd.DataFrame):
    """风险分析: 尾部风险、VaR、最大回撤"""
    print("\n" + "=" * 70)
    print("5. 风险分析")
    print("=" * 70)
    
    target = "market_forward_excess_returns"
    returns = train[target].dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 5.1 收益分布与正态对比
    ax = axes[0, 0]
    returns.hist(bins=100, ax=ax, density=True, alpha=0.7, color='steelblue', label='实际分布')
    
    # 正态分布曲线
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_pdf = stats.norm.pdf(x, returns.mean(), returns.std())
    ax.plot(x, normal_pdf, 'r-', linewidth=2, label='正态分布')
    ax.set_title("收益分布 vs 正态分布", fontsize=12)
    ax.legend()
    ax.set_xlabel("超额收益")
    ax.set_ylabel("密度")
    
    # 5.2 QQ图
    ax = axes[0, 1]
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title("Q-Q图 (正态性检验)", fontsize=12)
    
    # 5.3 VaR分析
    ax = axes[1, 0]
    var_levels = [0.01, 0.05, 0.10]
    var_values = [returns.quantile(level) for level in var_levels]
    
    returns.hist(bins=100, ax=ax, alpha=0.7, color='steelblue')
    for i, (level, var) in enumerate(zip(var_levels, var_values)):
        ax.axvline(x=var, color=['red', 'orange', 'yellow'][i], 
                   linestyle='--', linewidth=2, label=f'VaR {int(level*100)}%: {var:.4f}')
    ax.set_title("Value at Risk (VaR)", fontsize=12)
    ax.legend()
    ax.set_xlabel("超额收益")
    
    # 5.4 最大回撤
    ax = axes[1, 1]
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    
    ax.fill_between(train["date"][:len(drawdown)], drawdown, 0, color='red', alpha=0.3)
    ax.set_title("最大回撤", fontsize=12)
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤比例")
    
    # 找到最大回撤
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    ax.annotate(f'最大回撤: {max_dd:.2%}', 
                xy=(train["date"].iloc[max_dd_idx], max_dd),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_analysis.png", dpi=150)
    plt.close()
    print(f"图表已保存: risk_analysis.png")
    
    # 打印风险统计
    print("\n风险统计:")
    print(f"  偏度: {returns.skew():.4f}")
    print(f"  峰度: {returns.kurtosis():.4f}")
    print(f"  VaR 1%: {var_values[0]:.4f}")
    print(f"  VaR 5%: {var_values[1]:.4f}")
    print(f"  最大回撤: {max_dd:.2%}")
    print(f"  夏普比率: {returns.mean() / returns.std() * np.sqrt(252):.4f}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("Hull Tactical 高级数据分析")
    print("=" * 70)
    
    train = load_data()
    print(f"数据加载完成: {train.shape}")
    
    # 执行所有分析
    analyze_time_series_properties(train)
    analyze_market_regimes(train)
    analyze_feature_ic(train)
    analyze_pca(train)
    analyze_risk(train)
    
    print("\n" + "=" * 70)
    print("高级分析完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
