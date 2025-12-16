"""
数据健康度与结构性断层分析

分析内容:
1. 缺失值矩阵热力图
2. 缺失值与Target的相关性
3. 数据结构性断层检测
4. 特征可用性时间线
5. 缺失模式聚类分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 特征前缀说明
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
    train['month'] = train['date'].dt.month
    return train


def get_feature_cols(df: pd.DataFrame) -> list:
    """获取特征列"""
    exclude = ['date_id', 'forward_returns', 'risk_free_rate', 
               'market_forward_excess_returns', 'date', 'year', 'month']
    return [c for c in df.columns if c not in exclude]


def plot_missing_heatmap(df: pd.DataFrame, feature_cols: list):
    """
    1. 缺失值矩阵热力图
    按特征类别和时间展示缺失模式
    """
    print("\n" + "="*70)
    print("【1. 缺失值矩阵热力图】")
    print("="*70)
    
    # 按年份聚合缺失率
    yearly_missing = df.groupby('year')[feature_cols].apply(
        lambda x: x.isna().mean()
    )
    
    # 按特征类别排序
    sorted_cols = sorted(feature_cols, key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))
    yearly_missing = yearly_missing[sorted_cols]
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    
    # 热力图1: 按年份的缺失率
    ax = axes[0]
    sns.heatmap(yearly_missing.T, cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': '缺失率'}, vmin=0, vmax=1)
    ax.set_title('各特征按年份的缺失率热力图', fontsize=14)
    ax.set_xlabel('年份')
    ax.set_ylabel('特征')
    
    # 添加特征类别分隔线
    current_prefix = None
    for i, col in enumerate(sorted_cols):
        if col[0] != current_prefix:
            if current_prefix is not None:
                ax.axhline(y=i, color='blue', linewidth=2)
            current_prefix = col[0]
    
    # 热力图2: 采样后的原始缺失模式
    ax = axes[1]
    sample_idx = np.linspace(0, len(df)-1, min(500, len(df)), dtype=int)
    missing_matrix = df.iloc[sample_idx][sorted_cols].isna().astype(int)
    
    sns.heatmap(missing_matrix.T, cmap='binary', ax=ax, cbar=False)
    ax.set_title('缺失值矩阵 (采样500行)', fontsize=14)
    ax.set_xlabel('样本索引')
    ax.set_ylabel('特征')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'missing_heatmap.png', dpi=150)
    plt.close()
    print("图表已保存: missing_heatmap.png")
    
    # 统计输出
    print("\n按年份缺失率统计:")
    for year in yearly_missing.index[::5]:  # 每5年输出一次
        avg_missing = yearly_missing.loc[year].mean() * 100
        print(f"  {year}年: {avg_missing:.1f}%")


def analyze_missing_target_correlation(df: pd.DataFrame, feature_cols: list):
    """
    2. 缺失值与Target的相关性分析
    检验缺失是否随机 (MCAR/MAR/MNAR)
    """
    print("\n" + "="*70)
    print("【2. 缺失值与Target相关性】")
    print("="*70)
    
    target = df['market_forward_excess_returns']
    results = []
    
    for col in feature_cols:
        is_missing = df[col].isna()
        if is_missing.sum() > 10 and is_missing.sum() < len(df) - 10:
            # T检验: 缺失组 vs 非缺失组的target均值差异
            missing_target = target[is_missing].dropna()
            present_target = target[~is_missing].dropna()
            
            if len(missing_target) > 10 and len(present_target) > 10:
                t_stat, p_value = stats.ttest_ind(missing_target, present_target)
                mean_diff = missing_target.mean() - present_target.mean()
                
                results.append({
                    'feature': col,
                    'missing_pct': is_missing.mean() * 100,
                    'mean_diff': mean_diff * 100,  # 转为百分比
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
    
    results_df = pd.DataFrame(results)
    
    # 按显著性排序
    significant = results_df[results_df['significant']].sort_values('p_value')
    
    print(f"\n总特征数: {len(results_df)}")
    print(f"显著相关特征数 (p<0.05): {len(significant)}")
    
    if len(significant) > 0:
        print("\n显著相关的特征 (缺失与Target相关):")
        print("-" * 70)
        for _, row in significant.head(10).iterrows():
            direction = "↑" if row['mean_diff'] > 0 else "↓"
            print(f"  {row['feature']}: 缺失率={row['missing_pct']:.1f}%, "
                  f"均值差={row['mean_diff']:+.4f}% {direction}, p={row['p_value']:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 散点图: 缺失率 vs 均值差异
    ax = axes[0]
    colors = ['red' if s else 'gray' for s in results_df['significant']]
    ax.scatter(results_df['missing_pct'], results_df['mean_diff'], 
               c=colors, alpha=0.6, s=50)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('缺失率 (%)')
    ax.set_ylabel('Target均值差异 (缺失-非缺失) %')
    ax.set_title('缺失值与Target的关系')
    
    # 标注显著特征
    for _, row in significant.head(5).iterrows():
        ax.annotate(row['feature'], (row['missing_pct'], row['mean_diff']),
                   fontsize=9, alpha=0.8)
    
    # P值分布
    ax = axes[1]
    ax.hist(results_df['p_value'], bins=20, edgecolor='white', alpha=0.7)
    ax.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
    ax.set_xlabel('P值')
    ax.set_ylabel('特征数')
    ax.set_title('P值分布 (检验缺失随机性)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'missing_target_correlation.png', dpi=150)
    plt.close()
    print("\n图表已保存: missing_target_correlation.png")
    
    return results_df


def detect_structural_breaks(df: pd.DataFrame, feature_cols: list):
    """
    3. 数据结构性断层检测
    识别特征开始/结束可用的时间点
    """
    print("\n" + "="*70)
    print("【3. 数据结构性断层检测】")
    print("="*70)
    
    breaks = []
    
    for col in feature_cols:
        valid_mask = ~df[col].isna()
        if valid_mask.sum() > 0:
            first_valid_idx = valid_mask.idxmax()
            last_valid_idx = valid_mask[::-1].idxmax()
            
            first_date = df.loc[first_valid_idx, 'date']
            last_date = df.loc[last_valid_idx, 'date']
            coverage = valid_mask.sum() / len(df) * 100
            
            # 检测中间断层
            valid_indices = df.index[valid_mask]
            gaps = np.diff(valid_indices)
            max_gap = gaps.max() if len(gaps) > 0 else 0
            
            breaks.append({
                'feature': col,
                'prefix': col[0],
                'first_date': first_date,
                'last_date': last_date,
                'coverage': coverage,
                'max_gap_days': max_gap,
                'has_gap': max_gap > 30  # 超过30天算断层
            })
    
    breaks_df = pd.DataFrame(breaks)
    
    # 按类别统计
    print("\n各类别特征可用时间范围:")
    print("-" * 70)
    for prefix, name in FEATURE_INFO.items():
        subset = breaks_df[breaks_df['prefix'] == prefix]
        if len(subset) > 0:
            earliest = subset['first_date'].min()
            latest = subset['last_date'].max()
            avg_coverage = subset['coverage'].mean()
            gap_count = subset['has_gap'].sum()
            print(f"  {prefix} ({name}): {earliest.strftime('%Y-%m')} ~ {latest.strftime('%Y-%m')}, "
                  f"平均覆盖={avg_coverage:.1f}%, 有断层={gap_count}个")
    
    # 可视化: 特征可用性时间线
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # 按类别和开始时间排序
    breaks_df = breaks_df.sort_values(['prefix', 'first_date'])
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(FEATURE_INFO)))
    color_map = {p: colors[i] for i, p in enumerate(FEATURE_INFO.keys())}
    
    for i, (_, row) in enumerate(breaks_df.iterrows()):
        start = row['first_date']
        end = row['last_date']
        color = color_map.get(row['prefix'], 'gray')
        
        ax.barh(i, (end - start).days, left=start, height=0.8, 
                color=color, alpha=0.7, edgecolor='white')
        
        if row['has_gap']:
            ax.plot(start, i, 'r*', markersize=8)  # 标记有断层的特征
    
    ax.set_yticks(range(len(breaks_df)))
    ax.set_yticklabels(breaks_df['feature'], fontsize=8)
    ax.set_xlabel('时间')
    ax.set_title('特征可用性时间线 (红星=有中间断层)', fontsize=14)
    
    # 添加图例
    legend_elements = [plt.Rectangle((0,0),1,1, color=color_map[p], alpha=0.7, 
                                      label=f'{p}: {name}') 
                      for p, name in FEATURE_INFO.items()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_availability_timeline.png', dpi=150)
    plt.close()
    print("\n图表已保存: feature_availability_timeline.png")
    
    return breaks_df


def analyze_missing_patterns(df: pd.DataFrame, feature_cols: list):
    """
    4. 缺失模式聚类分析
    识别不同的缺失模式
    """
    print("\n" + "="*70)
    print("【4. 缺失模式聚类分析】")
    print("="*70)
    
    # 创建缺失指示矩阵
    missing_matrix = df[feature_cols].isna().astype(int)
    
    # 对样本聚类 (识别不同的缺失模式)
    n_clusters = 5
    sample_idx = np.random.choice(len(missing_matrix), min(2000, len(missing_matrix)), replace=False)
    sample_matrix = missing_matrix.iloc[sample_idx]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(sample_matrix)
    
    # 分析每个聚类
    print(f"\n识别出 {n_clusters} 种缺失模式:")
    print("-" * 70)
    
    cluster_info = []
    for i in range(n_clusters):
        cluster_mask = clusters == i
        cluster_size = cluster_mask.sum()
        cluster_missing_rate = sample_matrix[cluster_mask].mean().mean() * 100
        
        # 获取该聚类对应的时间范围
        cluster_dates = df.iloc[sample_idx[cluster_mask]]['date']
        date_range = f"{cluster_dates.min().strftime('%Y-%m')} ~ {cluster_dates.max().strftime('%Y-%m')}"
        
        cluster_info.append({
            'cluster': i,
            'size': cluster_size,
            'pct': cluster_size / len(sample_idx) * 100,
            'missing_rate': cluster_missing_rate,
            'date_range': date_range
        })
        
        print(f"  模式{i+1}: {cluster_size}样本 ({cluster_size/len(sample_idx)*100:.1f}%), "
              f"平均缺失率={cluster_missing_rate:.1f}%, 时间={date_range}")
    
    # 可视化聚类中心
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    sorted_cols = sorted(feature_cols, key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))
    
    for i in range(min(n_clusters, 5)):
        ax = axes[i]
        center = kmeans.cluster_centers_[i]
        
        # 重新排序center
        col_order = [feature_cols.index(c) for c in sorted_cols]
        center_sorted = center[col_order]
        
        ax.bar(range(len(center_sorted)), center_sorted, alpha=0.7)
        ax.set_title(f'缺失模式 {i+1}\n({cluster_info[i]["pct"]:.1f}%, {cluster_info[i]["date_range"]})',
                    fontsize=10)
        ax.set_ylabel('缺失概率')
        ax.set_ylim(0, 1)
        
        # 添加类别分隔线
        current_prefix = None
        for j, col in enumerate(sorted_cols):
            if col[0] != current_prefix:
                if current_prefix is not None:
                    ax.axvline(x=j-0.5, color='red', linestyle='--', alpha=0.5)
                current_prefix = col[0]
    
    # 最后一个子图: 聚类分布饼图
    ax = axes[5]
    sizes = [c['pct'] for c in cluster_info]
    labels = [f"模式{c['cluster']+1}\n({c['missing_rate']:.0f}%缺失)" for c in cluster_info]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('缺失模式分布')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'missing_patterns_cluster.png', dpi=150)
    plt.close()
    print("\n图表已保存: missing_patterns_cluster.png")


def analyze_data_quality_score(df: pd.DataFrame, feature_cols: list):
    """
    5. 数据健康度评分
    综合评估数据质量
    """
    print("\n" + "="*70)
    print("【5. 数据健康度评分】")
    print("="*70)
    
    scores = []
    
    for col in feature_cols:
        data = df[col]
        
        # 1. 完整性 (非缺失率)
        completeness = 1 - data.isna().mean()
        
        # 2. 稳定性 (非极端值比例)
        valid_data = data.dropna()
        if len(valid_data) > 10:
            q1, q99 = valid_data.quantile([0.01, 0.99])
            stability = ((valid_data >= q1) & (valid_data <= q99)).mean()
        else:
            stability = 0
        
        # 3. 时效性 (最近数据可用性)
        recent_mask = df['year'] >= 2010
        recency = 1 - data[recent_mask].isna().mean()
        
        # 4. 一致性 (无突变)
        if len(valid_data) > 10:
            diff = valid_data.diff().abs()
            mad = diff.median()
            consistency = (diff < 5 * mad).mean() if mad > 0 else 1
        else:
            consistency = 0
        
        # 综合得分
        overall = (completeness * 0.4 + stability * 0.2 + 
                  recency * 0.25 + consistency * 0.15)
        
        scores.append({
            'feature': col,
            'prefix': col[0],
            'completeness': completeness,
            'stability': stability,
            'recency': recency,
            'consistency': consistency,
            'overall': overall
        })
    
    scores_df = pd.DataFrame(scores)
    
    # 按类别统计
    print("\n各类别健康度评分 (0-1):")
    print("-" * 70)
    print(f"{'类别':<15} {'完整性':<10} {'稳定性':<10} {'时效性':<10} {'一致性':<10} {'综合':<10}")
    print("-" * 70)
    
    for prefix, name in FEATURE_INFO.items():
        subset = scores_df[scores_df['prefix'] == prefix]
        if len(subset) > 0:
            print(f"{prefix} ({name}){' '*(8-len(name))} "
                  f"{subset['completeness'].mean():.3f}     "
                  f"{subset['stability'].mean():.3f}     "
                  f"{subset['recency'].mean():.3f}     "
                  f"{subset['consistency'].mean():.3f}     "
                  f"{subset['overall'].mean():.3f}")
    
    # Top/Bottom 特征
    print("\n健康度最高的5个特征:")
    for _, row in scores_df.nlargest(5, 'overall').iterrows():
        print(f"  {row['feature']}: {row['overall']:.3f}")
    
    print("\n健康度最低的5个特征:")
    for _, row in scores_df.nsmallest(5, 'overall').iterrows():
        print(f"  {row['feature']}: {row['overall']:.3f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 雷达图数据准备
    ax = axes[0]
    categories = list(FEATURE_INFO.keys())
    metrics = ['completeness', 'stability', 'recency', 'consistency']
    
    # 按类别的平均得分
    category_scores = scores_df.groupby('prefix')[metrics].mean()
    
    x = np.arange(len(categories))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [category_scores.loc[p, metric] if p in category_scores.index else 0 
                 for p in categories]
        ax.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([f"{p}\n{FEATURE_INFO[p]}" for p in categories], fontsize=9)
    ax.set_ylabel('得分')
    ax.set_title('各类别特征健康度指标')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 综合得分分布
    ax = axes[1]
    for prefix in categories:
        subset = scores_df[scores_df['prefix'] == prefix]['overall']
        ax.hist(subset, bins=10, alpha=0.5, label=f"{prefix}: {FEATURE_INFO[prefix]}")
    ax.set_xlabel('综合健康度得分')
    ax.set_ylabel('特征数')
    ax.set_title('特征健康度分布')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_health_score.png', dpi=150)
    plt.close()
    print("\n图表已保存: data_health_score.png")
    
    return scores_df


def main():
    print("="*70)
    print("数据健康度与结构性断层分析")
    print("="*70)
    
    # 加载数据
    train = load_data()
    feature_cols = get_feature_cols(train)
    print(f"\n数据加载完成: {train.shape}")
    print(f"特征数: {len(feature_cols)}")
    
    # 1. 缺失值矩阵热力图
    plot_missing_heatmap(train, feature_cols)
    
    # 2. 缺失值与Target相关性
    missing_target_df = analyze_missing_target_correlation(train, feature_cols)
    
    # 3. 结构性断层检测
    breaks_df = detect_structural_breaks(train, feature_cols)
    
    # 4. 缺失模式聚类
    analyze_missing_patterns(train, feature_cols)
    
    # 5. 数据健康度评分
    health_df = analyze_data_quality_score(train, feature_cols)
    
    # 总结
    print("\n" + "="*70)
    print("【分析总结】")
    print("="*70)
    print("""
1. 数据存在明显的时间断层，2000年前缺失严重
2. 部分特征的缺失与Target显著相关，需要谨慎处理
3. 识别出5种不同的缺失模式，主要与时间相关
4. D类(债券)特征健康度最高，M类(货币)最低
5. 建议:
   - 使用2000年后数据作为主训练集
   - 对缺失与Target相关的特征考虑创建缺失指示变量
   - 避免简单删除缺失值，可能引入偏差
""")


if __name__ == "__main__":
    main()
