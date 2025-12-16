"""
异常值分析

分析内容:
1. 各特征异常值统计 (IQR & Z-score)
2. 极端值时间分布（黑天鹅事件）
3. 异常值与Target的关系
4. 特征稳定性分析
5. 异常值处理建议
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_INFO = {
    'D': '债券/信用',
    'E': '经济指标',
    'I': '通胀相关',
    'M': '货币/利率',
    'P': '价格/估值',
    'S': '情绪/调查',
    'V': '波动率/风险',
}


def load_data() -> pd.DataFrame:
    """加载数据"""
    train = pd.read_csv(DATA_DIR / "train.csv")
    base_date = pd.Timestamp('1988-01-01')
    train['date'] = base_date + pd.to_timedelta(train['date_id'], unit='D')
    train['year'] = train['date'].dt.year
    return train


def get_feature_cols(df: pd.DataFrame) -> list:
    """获取特征列"""
    exclude = ['date_id', 'forward_returns', 'risk_free_rate', 
               'market_forward_excess_returns', 'date', 'year', 'month']
    return [c for c in df.columns if c not in exclude]


def detect_outliers_iqr(series: pd.Series, k: float = 1.5):
    """IQR方法检测异常值"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper), lower, upper


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0):
    """Z-score方法检测异常值"""
    valid_data = series.dropna()
    z = np.abs((valid_data - valid_data.mean()) / valid_data.std())
    mask = pd.Series(False, index=series.index)
    mask[valid_data.index] = z > threshold
    return mask, -threshold, threshold


def detect_outliers_mad(series: pd.Series, threshold: float = 4.0):
    """MAD方法检测异常值 (本比赛官方使用的方法)"""
    valid_data = series.dropna()
    median = valid_data.median()
    mad = (valid_data - median).abs().median()
    # 使用缩放因子使MAD与标准差可比
    scaled_mad = mad * 1.4826
    lower = median - threshold * scaled_mad
    upper = median + threshold * scaled_mad
    mask = (series < lower) | (series > upper)
    return mask, lower, upper


def detect_outliers_quantile(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99):
    """分位数方法检测异常值"""
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return (series < lower) | (series > upper), lower, upper


def detect_outliers_modified_zscore(series: pd.Series, threshold: float = 3.5):
    """Modified Z-score (基于MAD的Z分数，更稳健)"""
    valid_data = series.dropna()
    median = valid_data.median()
    mad = (valid_data - median).abs().median()
    if mad == 0:
        return pd.Series(False, index=series.index), median, median
    modified_z = 0.6745 * (valid_data - median) / mad
    mask = pd.Series(False, index=series.index)
    mask[valid_data.index] = np.abs(modified_z) > threshold
    return mask, median - threshold * mad / 0.6745, median + threshold * mad / 0.6745


def detect_outliers_rolling(series: pd.Series, window: int = 252, threshold: float = 3.0):
    """滚动窗口方法 (时序感知)"""
    rolling_median = series.rolling(window, min_periods=20).median()
    rolling_mad = (series - rolling_median).abs().rolling(window, min_periods=20).median()
    lower = rolling_median - threshold * rolling_mad * 1.4826
    upper = rolling_median + threshold * rolling_mad * 1.4826
    return (series < lower) | (series > upper), lower, upper


def detect_outliers_ensemble(df: pd.DataFrame, feature_cols: list, min_votes: int = 3):
    """
    多方法综合检测异常值
    使用投票机制: 当>=min_votes种方法同时判定为异常时，才认定为异常
    
    返回: 
        outlier_mask: 每个样本每个特征是否为异常的DataFrame
        outlier_details: 异常值的详细信息
    """
    print("\n" + "="*70)
    print(f"【1. 多方法综合异常值检测 (投票阈值≥{min_votes})】")
    print("="*70)
    
    # 存储每个特征的异常检测结果
    all_outliers = {}
    feature_stats = []
    
    for col in feature_cols:
        data = df[col]
        valid_mask = ~data.isna()
        
        if valid_mask.sum() < 100:
            continue
        
        # 6种方法检测
        outliers_iqr, _, _ = detect_outliers_iqr(data.dropna(), k=1.5)
        outliers_z, _, _ = detect_outliers_zscore(data.dropna(), threshold=3.0)
        outliers_mad, _, _ = detect_outliers_mad(data.dropna(), threshold=4.0)
        outliers_q, _, _ = detect_outliers_quantile(data.dropna(), 0.01, 0.99)
        outliers_mz, _, _ = detect_outliers_modified_zscore(data.dropna(), threshold=3.5)
        outliers_roll, _, _ = detect_outliers_rolling(data, window=252, threshold=3.0)
        
        # 投票矩阵
        vote_df = pd.DataFrame(index=data.dropna().index)
        vote_df['iqr'] = outliers_iqr.astype(int)
        vote_df['zscore'] = outliers_z.astype(int)
        vote_df['mad'] = outliers_mad.astype(int)
        vote_df['quantile'] = outliers_q.astype(int)
        vote_df['mod_zscore'] = outliers_mz.astype(int)
        
        # 计算投票数
        vote_df['votes'] = vote_df.sum(axis=1)
        vote_df['is_outlier'] = vote_df['votes'] >= min_votes
        
        # 记录结果
        all_outliers[col] = vote_df['is_outlier']
        
        # 统计
        n_outliers = vote_df['is_outlier'].sum()
        outlier_pct = n_outliers / len(vote_df) * 100
        
        feature_stats.append({
            'feature': col,
            'prefix': col[0],
            'n_outliers': n_outliers,
            'outlier_pct': outlier_pct,
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        })
    
    stats_df = pd.DataFrame(feature_stats)
    
    # 输出统计
    print(f"\n检测到的异常值统计:")
    print("-" * 70)
    
    total_outliers = stats_df['n_outliers'].sum()
    print(f"  总异常数据点: {total_outliers:,}")
    print(f"  平均每特征异常率: {stats_df['outlier_pct'].mean():.2f}%")
    
    # 按类别统计
    print("\n各类别异常值:")
    for prefix, name in FEATURE_INFO.items():
        subset = stats_df[stats_df['prefix'] == prefix]
        if len(subset) > 0:
            total = subset['n_outliers'].sum()
            avg_pct = subset['outlier_pct'].mean()
            print(f"  {prefix} ({name}): {total:,}个异常点, 平均{avg_pct:.2f}%")
    
    # 异常最多的特征
    print("\n异常值最多的10个特征:")
    for _, row in stats_df.nlargest(10, 'outlier_pct').iterrows():
        print(f"  {row['feature']}: {row['n_outliers']}个 ({row['outlier_pct']:.2f}%)")
    
    return all_outliers, stats_df


def analyze_outlier_samples(df: pd.DataFrame, all_outliers: dict, feature_cols: list):
    """
    分析异常样本的具体信息
    """
    print("\n" + "="*70)
    print("【2. 异常样本详细分析】")
    print("="*70)
    
    # 统计每个样本有多少特征异常
    outlier_count_per_sample = pd.Series(0, index=df.index)
    
    for col, outliers in all_outliers.items():
        # 将异常标记映射回原始索引
        for idx in outliers[outliers].index:
            if idx in outlier_count_per_sample.index:
                outlier_count_per_sample[idx] += 1
    
    df['outlier_feature_count'] = outlier_count_per_sample
    
    # 统计
    print("\n每样本异常特征数分布:")
    print("-" * 70)
    
    bins = [0, 1, 3, 5, 10, 100]
    labels = ['0', '1-2', '3-4', '5-9', '10+']
    df['outlier_level'] = pd.cut(df['outlier_feature_count'], bins=bins, labels=labels, right=False)
    
    level_counts = df['outlier_level'].value_counts().sort_index()
    for level, count in level_counts.items():
        pct = count / len(df) * 100
        print(f"  {level}个异常特征: {count:,}样本 ({pct:.1f}%)")
    
    # 严重异常样本 (≥5个特征异常)
    severe_outliers = df[df['outlier_feature_count'] >= 5].copy()
    
    print(f"\n严重异常样本 (≥5个特征异常): {len(severe_outliers)}个")
    
    if len(severe_outliers) > 0:
        print("\n最严重的20个异常样本:")
        print("-" * 70)
        severe_outliers = severe_outliers.sort_values('outlier_feature_count', ascending=False)
        
        for _, row in severe_outliers.head(20).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            target = row['market_forward_excess_returns'] * 100
            print(f"  {date_str}: {row['outlier_feature_count']}个异常特征, Target={target:+.2f}%")
        
        # 按年份分布
        print("\n严重异常样本的年份分布:")
        yearly = severe_outliers.groupby('year').size()
        for year, count in yearly.items():
            if count > 0:
                print(f"  {year}年: {count}个")
    
    return df


def visualize_outliers(df: pd.DataFrame, all_outliers: dict, stats_df: pd.DataFrame):
    """
    可视化异常值
    """
    print("\n" + "="*70)
    print("【3. 异常值可视化】")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 各特征异常率
    ax = axes[0, 0]
    top_features = stats_df.nlargest(20, 'outlier_pct')
    ax.barh(top_features['feature'], top_features['outlier_pct'], color='steelblue', alpha=0.7)
    ax.set_xlabel('异常率 (%)')
    ax.set_title('异常率最高的20个特征')
    ax.invert_yaxis()
    
    # 2. 每样本异常特征数分布
    ax = axes[0, 1]
    outlier_counts = df['outlier_feature_count']
    ax.hist(outlier_counts[outlier_counts > 0], bins=30, edgecolor='white', alpha=0.7)
    ax.axvline(x=5, color='red', linestyle='--', label='严重异常阈值')
    ax.set_xlabel('异常特征数')
    ax.set_ylabel('样本数')
    ax.set_title('每样本异常特征数分布')
    ax.legend()
    
    # 3. 异常值时间分布
    ax = axes[1, 0]
    ax.plot(df['date'], df['outlier_feature_count'], alpha=0.7, linewidth=0.5)
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='严重异常阈值')
    ax.set_xlabel('日期')
    ax.set_ylabel('异常特征数')
    ax.set_title('异常值时间分布')
    ax.legend()
    
    # 4. 异常样本的Target分布
    ax = axes[1, 1]
    normal = df[df['outlier_feature_count'] < 5]['market_forward_excess_returns'] * 100
    severe = df[df['outlier_feature_count'] >= 5]['market_forward_excess_returns'] * 100
    
    ax.hist(normal, bins=50, alpha=0.5, label=f'正常样本 (n={len(normal)})', density=True)
    ax.hist(severe, bins=30, alpha=0.5, label=f'严重异常样本 (n={len(severe)})', density=True)
    ax.set_xlabel('Target (%)')
    ax.set_ylabel('密度')
    ax.set_title('正常 vs 异常样本的Target分布')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'outlier_detection_results.png', dpi=150)
    plt.close()
    print("图表已保存: outlier_detection_results.png")


def analyze_outliers_by_feature(df: pd.DataFrame, feature_cols: list):
    """
    综合异常值检测主函数
    """
    # 1. 多方法检测
    all_outliers, stats_df = detect_outliers_ensemble(df, feature_cols, min_votes=3)
    
    # 2. 分析异常样本
    df = analyze_outlier_samples(df, all_outliers, feature_cols)
    
    # 3. 可视化
    visualize_outliers(df, all_outliers, stats_df)
    
    return stats_df


def analyze_extreme_events(df: pd.DataFrame, feature_cols: list):
    """
    2. 极端值时间分布 (黑天鹅事件分析)
    """
    print("\n" + "="*70)
    print("【2. 极端值时间分布 (黑天鹅事件)】")
    print("="*70)
    
    # 统计每天有多少特征出现极端值
    extreme_counts = pd.Series(0, index=df.index)
    
    for col in feature_cols:
        data = df[col]
        if data.notna().sum() < 10:
            continue
        outliers, _, _ = detect_outliers_iqr(data, k=3)  # 使用更严格的阈值
        extreme_counts += outliers.astype(int)
    
    df['extreme_count'] = extreme_counts
    
    # 找出极端事件日
    extreme_days = df[df['extreme_count'] >= 5].copy()  # 5个以上特征同时异常
    
    print(f"\n极端事件日 (≥5个特征同时异常): {len(extreme_days)}天")
    
    if len(extreme_days) > 0:
        # 按年统计
        yearly_extreme = extreme_days.groupby('year').size()
        print("\n各年极端事件天数:")
        for year, count in yearly_extreme.items():
            if count > 0:
                print(f"  {year}年: {count}天")
        
        # 最极端的日子
        print("\n最极端的10个交易日:")
        top_extreme = extreme_days.nlargest(10, 'extreme_count')
        for _, row in top_extreme.iterrows():
            target = row['market_forward_excess_returns']
            print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['extreme_count']}个异常特征, "
                  f"Target={target*100:+.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 极端事件时间分布
    ax = axes[0, 0]
    ax.plot(df['date'], df['extreme_count'], alpha=0.7, linewidth=0.5)
    ax.axhline(y=5, color='red', linestyle='--', label='极端阈值')
    ax.set_title('每日异常特征数量')
    ax.set_xlabel('日期')
    ax.set_ylabel('异常特征数')
    ax.legend()
    
    # 年度极端事件
    ax = axes[0, 1]
    yearly = df.groupby('year')['extreme_count'].agg(['sum', 'mean'])
    ax.bar(yearly.index, yearly['sum'], alpha=0.7)
    ax.set_title('年度异常特征累计数')
    ax.set_xlabel('年份')
    ax.set_ylabel('异常数累计')
    
    # 极端日的Target分布
    ax = axes[1, 0]
    normal_target = df[df['extreme_count'] < 5]['market_forward_excess_returns']
    extreme_target = df[df['extreme_count'] >= 5]['market_forward_excess_returns']
    ax.hist(normal_target, bins=50, alpha=0.5, label=f'正常日 (n={len(normal_target)})', density=True)
    if len(extreme_target) > 0:
        ax.hist(extreme_target, bins=30, alpha=0.5, label=f'极端日 (n={len(extreme_target)})', density=True)
    ax.set_title('极端日 vs 正常日的Target分布')
    ax.set_xlabel('Target (超额收益)')
    ax.legend()
    
    # Target本身的极端值
    ax = axes[1, 1]
    target = df['market_forward_excess_returns']
    ax.hist(target, bins=100, alpha=0.7, edgecolor='white')
    
    # 标记±3%以外的区域
    ax.axvline(x=0.03, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=-0.03, color='red', linestyle='--', alpha=0.7)
    extreme_target_count = ((target > 0.03) | (target < -0.03)).sum()
    ax.set_title(f'Target分布 (|r|>3%: {extreme_target_count}天, {extreme_target_count/len(target)*100:.1f}%)')
    ax.set_xlabel('Target (超额收益)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'extreme_events_analysis.png', dpi=150)
    plt.close()
    print("\n图表已保存: extreme_events_analysis.png")
    
    return extreme_days


def analyze_outlier_target_relationship(df: pd.DataFrame, feature_cols: list):
    """
    3. 异常值与Target的关系
    """
    print("\n" + "="*70)
    print("【3. 异常值与Target的关系】")
    print("="*70)
    
    results = []
    
    for col in feature_cols:
        data = df[col]
        if data.notna().sum() < 100:
            continue
            
        outliers, _, _ = detect_outliers_iqr(data)
        
        if outliers.sum() > 10:
            target = df['market_forward_excess_returns']
            
            # 异常组 vs 正常组的Target均值
            outlier_target = target[outliers].dropna()
            normal_target = target[~outliers].dropna()
            
            if len(outlier_target) > 5 and len(normal_target) > 5:
                mean_diff = outlier_target.mean() - normal_target.mean()
                std_diff = outlier_target.std() - normal_target.std()
                
                # T检验
                t_stat, p_value = stats.ttest_ind(outlier_target, normal_target)
                
                results.append({
                    'feature': col,
                    'outlier_count': outliers.sum(),
                    'outlier_target_mean': outlier_target.mean() * 100,
                    'normal_target_mean': normal_target.mean() * 100,
                    'mean_diff': mean_diff * 100,
                    'std_diff': std_diff * 100,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
    
    results_df = pd.DataFrame(results)
    
    significant = results_df[results_df['significant']]
    print(f"\n总分析特征数: {len(results_df)}")
    print(f"异常值与Target显著相关的特征: {len(significant)}")
    
    if len(significant) > 0:
        print("\n显著相关的特征:")
        print("-" * 70)
        for _, row in significant.sort_values('p_value').head(10).iterrows():
            direction = "↑" if row['mean_diff'] > 0 else "↓"
            print(f"  {row['feature']}: 异常日Target={row['outlier_target_mean']:+.3f}%, "
                  f"正常日={row['normal_target_mean']:+.3f}%, "
                  f"差异={row['mean_diff']:+.3f}% {direction}")
    
    # 可视化
    if len(results_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 散点图
        ax = axes[0]
        colors = ['red' if s else 'gray' for s in results_df['significant']]
        ax.scatter(results_df['outlier_count'], results_df['mean_diff'], 
                   c=colors, alpha=0.6, s=50)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('异常值数量')
        ax.set_ylabel('Target均值差异 (%)')
        ax.set_title('特征异常值与Target的关系')
        
        # P值分布
        ax = axes[1]
        ax.hist(results_df['p_value'], bins=20, edgecolor='white', alpha=0.7)
        ax.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        ax.set_xlabel('P值')
        ax.set_ylabel('特征数')
        ax.set_title('显著性检验P值分布')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'outlier_target_relationship.png', dpi=150)
        plt.close()
        print("\n图表已保存: outlier_target_relationship.png")
    
    return results_df


def analyze_feature_distributions(df: pd.DataFrame, feature_cols: list):
    """
    4. 特征分布形态分析
    """
    print("\n" + "="*70)
    print("【4. 特征分布形态分析】")
    print("="*70)
    
    # 选择代表性特征进行可视化
    sample_features = []
    for prefix in FEATURE_INFO.keys():
        prefix_cols = [c for c in feature_cols if c.startswith(prefix)]
        if prefix_cols:
            sample_features.append(prefix_cols[0])
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    distribution_stats = []
    
    for i, col in enumerate(sample_features[:9]):
        ax = axes[i] if i < 9 else None
        data = df[col].dropna()
        
        if len(data) < 10:
            continue
        
        # 统计
        skew = data.skew()
        kurt = data.kurtosis()
        
        # Shapiro-Wilk正态性检验 (采样)
        sample = data.sample(min(5000, len(data)), random_state=42)
        _, p_normal = stats.shapiro(sample) if len(sample) < 5000 else (0, 0)
        
        distribution_stats.append({
            'feature': col,
            'skewness': skew,
            'kurtosis': kurt,
            'is_normal': p_normal > 0.05,
            'distribution_type': 'Normal' if abs(skew) < 0.5 and abs(kurt) < 1 
                                 else 'Skewed' if abs(skew) > 1 
                                 else 'Heavy-tailed' if kurt > 3 
                                 else 'Other'
        })
        
        if ax is not None:
            ax.hist(data, bins=50, alpha=0.7, density=True, edgecolor='white')
            
            # 拟合正态分布
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, stats.norm.pdf(x, data.mean(), data.std()), 
                   'r-', linewidth=2, label='正态分布')
            
            ax.set_title(f'{col}\n偏度={skew:.2f}, 峰度={kurt:.1f}')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=150)
    plt.close()
    print("图表已保存: feature_distributions.png")
    
    # 分布类型统计
    stats_df = pd.DataFrame(distribution_stats)
    print("\n特征分布类型统计:")
    print(stats_df['distribution_type'].value_counts())
    
    return stats_df


def provide_recommendations(outlier_df: pd.DataFrame, extreme_days: pd.DataFrame):
    """
    5. 异常值处理建议
    """
    print("\n" + "="*70)
    print("【5. 异常值处理建议】")
    print("="*70)
    
    # 高异常率特征
    high_outlier_features = outlier_df[outlier_df['outlier_pct'] > 10]['feature'].tolist()
    heavy_tail_features = outlier_df[outlier_df['kurtosis'] > 3]['feature'].tolist()
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                        异常值处理建议                                │
├─────────────────────────────────────────────────────────────────────┤
│ 1. 高异常率特征 ({len(high_outlier_features)}个, >10%异常):                            │
│    {', '.join(high_outlier_features[:10])}{'...' if len(high_outlier_features) > 10 else ''}
│    建议: Winsorization (1%-99% 分位数截断)                           │
│                                                                     │
│ 2. 厚尾分布特征 ({len(heavy_tail_features)}个):                                     │
│    峰度>3，极端值概率高于正态分布                                     │
│    建议: 使用稳健标准化 (RobustScaler) 而非 Z-score                   │
│                                                                     │
│ 3. 极端事件日 ({len(extreme_days)}天):                                              │
│    多特征同时异常，可能是市场危机                                     │
│    建议: 保留!这些是模型需要学习的重要样本                            │
│                                                                     │
│ 4. Target已做Winsorization (±4%):                                   │
│    官方已处理目标变量的极端值                                         │
│    建议: 无需额外处理Target                                          │
│                                                                     │
│ 5. 推荐预处理流程:                                                  │
│    ① 缺失值: 前向填充 → 均值填充                                     │
│    ② 异常值: Winsorization (1%-99%)                                 │
│    ③ 标准化: RobustScaler (中位数+IQR)                              │
│    ④ 保留极端事件日，不删除样本                                      │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # 代码示例
    print("\n推荐预处理代码:")
    print("-" * 70)
    print("""
from sklearn.preprocessing import RobustScaler

def preprocess_features(df, feature_cols):
    '''特征预处理'''
    df = df.copy()
    
    # 1. 缺失值填充
    df[feature_cols] = df[feature_cols].fillna(method='ffill')
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    
    # 2. Winsorization (1%-99%)
    for col in feature_cols:
        q1, q99 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower=q1, upper=q99)
    
    # 3. 稳健标准化
    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df, scaler
""")


def main():
    print("="*70)
    print("异常值分析")
    print("="*70)
    
    # 加载数据
    train = load_data()
    feature_cols = get_feature_cols(train)
    print(f"\n数据加载完成: {train.shape}")
    
    # 1. 各特征异常值统计
    outlier_df = analyze_outliers_by_feature(train, feature_cols)
    
    # 2. 极端事件分析
    extreme_days = analyze_extreme_events(train, feature_cols)
    
    # 3. 异常值与Target关系
    outlier_target_df = analyze_outlier_target_relationship(train, feature_cols)
    
    # 4. 特征分布分析
    dist_df = analyze_feature_distributions(train, feature_cols)
    
    # 5. 处理建议
    provide_recommendations(outlier_df, extreme_days)
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()
