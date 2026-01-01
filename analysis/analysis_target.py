import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import linregress

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def safe_linregress(x, y):
    """安全的线性回归计算，处理x值相同的情况"""
    if len(x) < 2 or np.var(x) == 0:
        return 0, 0, 0, 0, 0  # 斜率, 截距, r值, p值, 标准误
    
    try:
        return linregress(x, y)
    except:
        return 0, 0, 0, 0, 0

def analyze_gate_data(df_order):
    """分析开孔数数据分布并创建分桶方案"""
    # 统计开孔数的分布
    gate_counts = df_order['开闸孔数'].dropna()
    
    print("开孔数统计分析:")
    print(f"总记录数: {len(gate_counts)}")
    print(f"唯一值: {gate_counts.unique()}")
    print(f"最小值: {gate_counts.min()}, 最大值: {gate_counts.max()}")
    print(f"均值: {gate_counts.mean():.2f}, 中位数: {gate_counts.median()}")
    print(f"标准差: {gate_counts.std():.2f}")
    
    # 计算分位数
    quantiles = [0, 0.25, 0.5, 0.75, 1.0]
    quantile_values = gate_counts.quantile(quantiles)
    print("分位数:")
    for q, val in zip(quantiles, quantile_values):
        print(f"  {q*100}%: {val}")
    
    # 创建分桶方案
    # 基于分位数创建分桶边界
    bins = list(quantile_values.values)
    # 确保边界是整数
    bins = [int(round(b)) for b in bins]
    # 确保边界唯一
    bins = sorted(list(set(bins)))
    
    print(f"分桶边界: {bins}")
    
    return bins

def create_gate_bins(df_order, bins):
    """为开孔数创建分桶标签"""
    # 创建分桶
    labels = [f'bin_{i}' for i in range(len(bins)-1)]
    df_order['gate_bin'] = pd.cut(df_order['开闸孔数'], bins=bins, labels=labels, include_lowest=True)
    
    # 统计每个分桶的样本数量
    bin_counts = df_order['gate_bin'].value_counts().sort_index()
    print("分桶样本分布:")
    for bin_label, count in bin_counts.items():
        print(f"  {bin_label}: {count} 样本 ({count/len(df_order)*100:.1f}%)")
    
    return df_order

def analyze_target_variables(df_order):
    """分析目标变量：开闸时间、开闸时长、目标水位"""
    print("\n=== 目标变量分析 ===")
    
    # 分析开闸时间
    df_order['开闸时间'] = pd.to_datetime(df_order['开闸时间'])
    df_order['开闸小时'] = df_order['开闸时间'].dt.hour + df_order['开闸时间'].dt.minute/60
    
    print("\n1. 开闸时间分析:")
    print(f"时间范围: {df_order['开闸时间'].min()} 到 {df_order['开闸时间'].max()}")
    print(f"开闸小时分布:")
    print(df_order['开闸小时'].describe())
    
    # 分析开闸时长 - 添加优化处理后的分析
    df_order['开闸时长'] = pd.to_numeric(df_order['开闸时长'], errors='coerce')
    df_order = df_order[df_order['开闸时长'] > 0]  # 过滤无效值
    
    # 开闸时长优化处理
    df_order['处理后的开闸时长'] = df_order['开闸时长'].apply(
        lambda x: 1 if x <= 2 else 
                  2 if x <= 3 else 
                  3 if x <= 4 else 
                  4 if x <= 5 else 
                  5 if x <= 6 else 6
    )
    
    print("\n2. 开闸时长分析 (原始值):")
    print(df_order['开闸时长'].describe())
    
    print("\n2.1 开闸时长分析 (优化后):")
    print(df_order['处理后的开闸时长'].value_counts().sort_index())
    
    # 分析目标水位 - 添加优化处理后的分析
    df_order['目标水位'] = pd.to_numeric(df_order['目标水位'], errors='coerce')
    df_order = df_order[df_order['目标水位'] > 0]  # 过滤无效值
    
    # 目标水位优化处理
    df_order['处理后的目标水位'] = df_order['目标水位'].clip(lower=0.5, upper=3.6)
    
    print("\n3. 目标水位分析 (原始值):")
    print(df_order['目标水位'].describe())
    
    print("\n3.1 目标水位分析 (优化后):")
    print(df_order['处理后的目标水位'].describe())
    
    # 分析开闸孔数 - 添加优化处理后的分析
    print("\n4. 开闸孔数分析 (原始值):")
    print(df_order['开闸孔数'].describe())
    
    # 开闸孔数优化处理
    gate_bins = [2, 8, 10, 12, 24, 28]
    labels = ['bin_1', 'bin_2', 'bin_3', 'bin_4', 'bin_5']
    
    def process_gate_count(x):
        if x == 28:
            return 5  # 28单独作为一类
        else:
            bin_idx = pd.cut([x], bins=gate_bins, labels=range(len(labels)), right=False)[0]
            return int(bin_idx) + 1  # 从1开始编号
    
    df_order['处理后的开闸孔数'] = df_order['开闸孔数'].apply(process_gate_count)
    
    print("\n4.1 开闸孔数分析 (优化后):")
    print(df_order['处理后的开闸孔数'].value_counts().sort_index())
    
    return df_order

def visualize_target_variables(df_order, save_path):
    """可视化目标变量分布"""
    # 设置绘图风格
    plt.style.use('seaborn-v0_8')
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 开闸时间分布（小时）
    axes[0, 0].hist(df_order['开闸小时'].dropna(), bins=24, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('开闸时间 (小时)')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('开闸时间分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 开闸时长分布
    axes[0, 1].hist(df_order['开闸时长'].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('开闸时长')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('开闸时长分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 目标水位分布
    axes[1, 0].hist(df_order['目标水位'].dropna(), bins=30, alpha=0.7, color='salmon', edgecolor='black')
    axes[1, 0].set_xlabel('目标水位')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('目标水位分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 开闸孔数分布
    axes[1, 1].hist(df_order['开闸孔数'].dropna(), bins=range(1, int(df_order['开闸孔数'].max())+2), 
                   alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_xlabel('开闸孔数')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('开闸孔数分布')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/target_variables_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建关系图
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 开闸时间 vs 开闸时长
    axes[0, 0].scatter(df_order['开闸小时'], df_order['开闸时长'], alpha=0.6, color='steelblue')
    axes[0, 0].set_xlabel('开闸时间 (小时)')
    axes[0, 0].set_ylabel('开闸时长')
    axes[0, 0].set_title('开闸时间 vs 开闸时长')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 开闸时长 vs 目标水位
    axes[0, 1].scatter(df_order['开闸时长'], df_order['目标水位'], alpha=0.6, color='forestgreen')
    axes[0, 1].set_xlabel('开闸时长')
    axes[0, 1].set_ylabel('目标水位')
    axes[0, 1].set_title('开闸时长 vs 目标水位')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 开闸孔数 vs 开闸时长
    axes[1, 0].scatter(df_order['开闸孔数'], df_order['开闸时长'], alpha=0.6, color='coral')
    axes[1, 0].set_xlabel('开闸孔数')
    axes[1, 0].set_ylabel('开闸时长')
    axes[1, 0].set_title('开闸孔数 vs 开闸时长')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 开闸孔数 vs 目标水位
    axes[1, 1].scatter(df_order['开闸孔数'], df_order['目标水位'], alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('开闸孔数')
    axes[1, 1].set_ylabel('目标水位')
    axes[1, 1].set_title('开闸孔数 vs 目标水位')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/target_variables_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存至: {save_path}/")

def process_and_save_order_data(prefix="00", output_dir="features_0822"):
    """处理调令数据并保存优化后的CSV文件"""
    # 加载调令信息
    df_order = pd.read_csv(
        f"imports/{prefix}调令信息.csv",
        parse_dates=['SIGNTM', '开闸时间']
    )
    
    # 分析开孔数数据并创建分桶
    gate_bins = analyze_gate_data(df_order)
    df_order = create_gate_bins(df_order, gate_bins)
    
    # 分析目标变量
    df_order = analyze_target_variables(df_order)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存处理后的调令数据
    output_path = f"{output_dir}/{prefix}_processed_orders.csv"
    df_order.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"优化后的调令数据已保存至: {output_path}")
    
    # 创建可视化目录并保存图表
    vis_dir = f"{output_dir}/visualizations/{prefix}"
    os.makedirs(vis_dir, exist_ok=True)
    visualize_target_variables(df_order, vis_dir)
    
    return df_order

if __name__ == '__main__':
    # 处理不同前缀的调令数据
    process_and_save_order_data(prefix="00")
    process_and_save_order_data(prefix="07")
    process_and_save_order_data(prefix="11")