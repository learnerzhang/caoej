
import collections
import gzip
import glob
import os
import json
import warnings
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from scipy.signal import find_peaks, savgol_filter
from matplotlib.dates import DateFormatter, HourLocator
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from datetime import timedelta, datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import linregress

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


warnings.filterwarnings("ignore")
data_path = 'exports'


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

def analyze_gate_duration_holes(df_order, save_path):
    """分析开闸时长_孔洞和开闸时长_孔洞_log数据"""
    print("\n=== 开闸时长_孔洞分析 ===")
    
    # 分析开闸时长_孔洞
    print("开闸时长_孔洞分析:")
    print(df_order['开闸时长_孔洞'].describe())
    
    # 分析开闸时长_孔洞_log
    print("\n开闸时长_孔洞_log分析:")
    print(df_order['开闸时长_孔洞_log'].describe())
    
    # 可视化这两个新变量的分布
    plt.style.use('seaborn-v0_8')
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 开闸时长_孔洞分布
    axes[0, 0].hist(df_order['开闸时长_孔洞'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('开闸时长_孔洞')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('开闸时长_孔洞分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 开闸时长_孔洞_log分布
    axes[0, 1].hist(df_order['开闸时长_孔洞_log'].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('开闸时长_孔洞_log')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('开闸时长_孔洞_log分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 开闸时长_孔洞 vs 开闸孔数
    axes[1, 0].scatter(df_order['开闸孔数'], df_order['开闸时长_孔洞'], alpha=0.6, color='coral')
    axes[1, 0].set_xlabel('开闸孔数')
    axes[1, 0].set_ylabel('开闸时长_孔洞')
    axes[1, 0].set_title('开闸孔数 vs 开闸时长_孔洞')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 开闸时长_孔洞_log vs 开闸孔数
    axes[1, 1].scatter(df_order['开闸孔数'], df_order['开闸时长_孔洞_log'], alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('开闸孔数')
    axes[1, 1].set_ylabel('开闸时长_孔洞_log')
    axes[1, 1].set_title('开闸孔数 vs 开闸时长_孔洞_log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/gate_duration_holes_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建与目标水位的关系图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 开闸时长_孔洞 vs 目标水位
    axes[0].scatter(df_order['开闸时长_孔洞'], df_order['目标水位'], alpha=0.6, color='steelblue')
    axes[0].set_xlabel('开闸时长_孔洞')
    axes[0].set_ylabel('目标水位')
    axes[0].set_title('开闸时长_孔洞 vs 目标水位')
    axes[0].grid(True, alpha=0.3)
    
    # 开闸时长_孔洞_log vs 目标水位
    axes[1].scatter(df_order['开闸时长_孔洞_log'], df_order['目标水位'], alpha=0.6, color='forestgreen')
    axes[1].set_xlabel('开闸时长_孔洞_log')
    axes[1].set_ylabel('目标水位')
    axes[1].set_title('开闸时长_孔洞_log vs 目标水位')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/gate_duration_holes_vs_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"开闸时长_孔洞分析图表已保存至: {save_path}/")

def analyze_target_variables(df_order):
    """分析目标变量：开闸时间、开闸时长、目标水位"""
    print("\n=== 目标变量分析 ===")
    
    # 分析开闸时间
    df_order['开闸时间'] = pd.to_datetime(df_order['开闸时间'])
    df_order['开闸小时'] = df_order['开闸时间'].dt.hour + df_order['开闸时间'].dt.minute/60
    
    print(f"原始数据中28孔的数量: {len(df_order[df_order['开闸孔数'] == 28])}")
    print("\n开闸时间分析:")
    print(f"时间范围: {df_order['开闸时间'].min()} 到 {df_order['开闸时间'].max()}")
    print(f"开闸小时分布:")
    print(df_order['开闸小时'].describe())
    
    # 分析开闸时长 - 添加优化处理后的分析
    print(f"开始开闸时长后28孔的数量: {len(df_order[df_order['开闸孔数'] == 28])}")
    df_order['开闸时长'] = pd.to_numeric(df_order['开闸时长'], errors='coerce')
    df_order = df_order[df_order['开闸时长'] > 0]  # 过滤无效值
    print(f"过滤开闸时长后28孔的数量: {len(df_order[df_order['开闸孔数'] == 28])}")
    # 开闸时长优化处理
    df_order['处理后的开闸时长'] = df_order['开闸时长'].apply(
        lambda x: 1 if x < 2 else 
                  2 if x < 3 else 
                  3 if x < 4 else 
                  4 if x < 5 else 
                  5 if x < 6 else
                  6 if x < 7 else 7
    )
    
    print("\n开闸时长分析 (原始值):")
    print(df_order['开闸时长'].describe())
    
    print("\n开闸时长分析 (优化后):")
    print(df_order['处理后的开闸时长'].value_counts().sort_index())
    print("\n开闸时长分析 (优化后):")
    print(df_order['处理后的开闸时长'].describe())
    # 目标水位优化处理
    df_order['处理后的目标水位'] = df_order['目标水位'].clip(lower=1.0, upper=3.6)
    
    print("\n目标水位分析 (原始值):")
    print(df_order['目标水位'].describe())
    
    print("\n目标水位分析 (优化后):")
    print(df_order['处理后的目标水位'].describe())
    
    # 分析开闸孔数 - 添加优化处理后的分析
    print("\n开闸孔数分析 (原始值):")
    print(df_order['开闸孔数'].describe())
    
    # 开闸孔数优化处理 - 根据实际数据分布重新设计分类
    def process_gate_count(x):
        if x in [24, 28]:  # 将24和28孔作为同一类
            return 6
        elif x in [20, 18, 16]:  # 中等偏大的孔数
            return 5
        elif x in [14, 12]:  # 中等孔数
            return 4
        elif x in [10]:  # 中等偏小的孔数
            return 3
        elif x in [8]:  # 中等偏小的孔数
            return 2
        else:  # 小孔数 (6, 4, 2)
            return 1
    print("\n================================开闸孔数分析 (优化后):")
    df_order['处理后的开闸孔数'] = df_order['开闸孔数'].apply(process_gate_count)
    print("\n开闸孔数分析 (优化后):")
    print(df_order['处理后的开闸孔数'].value_counts().sort_index())
    
    # 添加类别标签映射
    category_labels = {
        1: "小孔数(2-6)",
        2: "中小孔数(8)",
        3: "中小孔数(10)",
        4: "中等孔数(12-14)",
        5: "中大孔数(16-20)",
        6: "大孔数(24-28)"
    }
    
    df_order['处理后的开闸孔数标签'] = df_order['处理后的开闸孔数'].map(category_labels)
    print("\n开闸孔数类别分布:")
    print(df_order['处理后的开闸孔数标签'].value_counts().sort_index())
    
    print("\n开闸孔数分析 (优化后):")
    print(df_order['处理后的开闸孔数'].describe())
    
    # 分析类别分布并调整
    class_distribution = df_order['处理后的开闸孔数'].value_counts().sort_index()
    print("开闸孔数类别分布:")
    print(class_distribution)
    # 如果样本不均衡严重，进行过采样或欠采样
    min_samples = class_distribution.min()
    max_samples = class_distribution.max()
    
    if max_samples / min_samples > 5:  # 如果最大类别是最小类别的5倍以上
        print("检测到严重样本不均衡，建议进行采样处理")

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

def process_and_save_order_data(prefix="00", output_dir="imports"):
    vis_dir = f"{output_dir}/visualizations/{prefix}"
    os.makedirs(vis_dir, exist_ok=True)

    """处理调令数据并保存优化后的CSV文件"""
    # 加载调令信息
    df_order = pd.read_csv(
        f"imports/{prefix}调令信息.csv",
        parse_dates=['SIGNTM', '开闸时间']
    )
    
    # 原有的数据处理代码...
    df_order['目标水位'] = pd.to_numeric(df_order['目标水位'], errors='coerce')
    df_order['目标水位'] = df_order['目标水位'].fillna(1)
    df_order['目标水位'] = df_order['目标水位'].astype(float)
    df_order.loc[df_order['目标水位'] < 1, '目标水位'] = 1.0

    df_order['开闸时长'] = pd.to_numeric(df_order['开闸时长'], errors='coerce')
    df_order['开闸时长'] = df_order['开闸时长'].fillna(7.0)
    df_order['开闸时长'] = df_order['开闸时长'].astype(float)

    # 新增：计算开闸时长_孔洞和开闸时长_孔洞_log
    df_order['开闸时长_孔洞'] = (df_order['开闸时长'] * df_order['开闸孔数']).round(3)
    df_order['开闸时长_孔洞_log'] = np.log(df_order['开闸时长_孔洞'] + 1).round(3)
    df_order.loc[df_order['开闸时长_孔洞_log'] < 1, '开闸时长_孔洞_log'] = 1.0

    # 原有的可视化代码...
    df_order.groupby('开闸孔数').size().plot(kind='bar', figsize=(15, 6))
    plt.title('开闸孔数分布')
    plt.xlabel('开闸孔数')
    plt.ylabel('频次')
    plt.savefig(f'{vis_dir}/开闸孔数分布.png', dpi=300, bbox_inches='tight')
    plt.close()

    df_order.groupby('开闸时长').size().plot(kind='bar', figsize=(15, 6))
    plt.title('开闸时长分布')
    plt.xlabel('开闸时长')
    plt.ylabel('频次')
    plt.savefig(f'{vis_dir}/开闸时长分布.png', dpi=300, bbox_inches='tight')
    plt.close()

    df_order.groupby('开闸时长_孔洞').size().plot(kind='bar', figsize=(15, 6))
    plt.title('开闸时长_孔洞分布')
    plt.xlabel('开闸时长_孔洞')
    plt.ylabel('频次')
    plt.savefig(f'{vis_dir}/开闸时长_孔洞分布.png', dpi=300, bbox_inches='tight')
    plt.close()

    df_order.groupby('开闸时长_孔洞_log').size().plot(kind='bar', figsize=(15, 6))
    plt.title('开闸时长_孔洞_log分布')
    plt.xlabel('开闸时长_孔洞_log')
    plt.ylabel('频次')
    plt.savefig(f'{vis_dir}/开闸时长_孔洞_log分布.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 分析开孔数数据并创建分桶
    gate_bins = analyze_gate_data(df_order)
    df_order = create_gate_bins(df_order, gate_bins)
    
    # 分析目标变量
    df_order = analyze_target_variables(df_order)
    
    # 新增：分析处理后的开闸时长_孔洞和开闸时长_孔洞_log
    # 计算处理后的开闸时长_孔洞和开闸时长_孔洞_log
    df_order['处理后的开闸时长_孔洞'] = df_order['处理后的开闸时长'] * df_order['处理后的开闸孔数']
    df_order['处理后的开闸时长_孔洞_log'] = np.log1p(df_order['处理后的开闸时长_孔洞'])
    
    # 新增：分析开闸时长_孔洞和开闸时长_孔洞_log
    analyze_gate_duration_holes(df_order, vis_dir)
    
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

def analyze_dealed_target_variables(prefix="00", output_dir="imports"):
    save_path = f"{output_dir}/visualizations/{prefix}"
    # 加载调令信息
    df_order = pd.read_csv(
        f"imports/{prefix}_processed_orders.csv",
        parse_dates=['SIGNTM', '开闸时间']
    )

    # 原有的可视化代码...
    plt.style.use('seaborn-v0_8')
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 开闸时间分布（小时）
    axes[0, 0].hist(df_order['开闸小时'].dropna(), bins=24, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('开闸时间 (小时)')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('开闸时间分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 开闸时长分布
    axes[0, 1].hist(df_order['处理后的开闸时长'].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('开闸时长')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('开闸时长分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 目标水位分布
    axes[1, 0].hist(df_order['处理后的目标水位'].dropna(), bins=30, alpha=0.7, color='salmon', edgecolor='black')
    axes[1, 0].set_xlabel('目标水位')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('目标水位分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 开闸孔数分布
    axes[1, 1].hist(df_order['处理后的开闸孔数'].dropna(), bins=range(1, int(df_order['处理后的开闸孔数'].max())+2), 
                   alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_xlabel('开闸孔数')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('开闸孔数分布')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/dealed_target_variables_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
    # 可视化处理后的新变量
    plt.style.use('seaborn-v0_8')
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 处理后的开闸时长_孔洞分布
    axes[0].hist(df_order['处理后的开闸时长_孔洞'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('处理后的开闸时长_孔洞')
    axes[0].set_ylabel('频次')
    axes[0].set_title('处理后的开闸时长_孔洞分布')
    axes[0].grid(True, alpha=0.3)
    
    # 处理后的开闸时长_孔洞_log分布
    axes[1].hist(df_order['处理后的开闸时长_孔洞_log'].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('处理后的开闸时长_孔洞_log')
    axes[1].set_ylabel('频次')
    axes[1].set_title('处理后的开闸时长_孔洞_log分布')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/dealed_gate_duration_holes_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    
def main():
    # 数据分析和可视化
    process_and_save_order_data(prefix="00")
    process_and_save_order_data(prefix="07")
    process_and_save_order_data(prefix="11")

    analyze_dealed_target_variables(prefix="00")
    analyze_dealed_target_variables(prefix="07")
    analyze_dealed_target_variables(prefix="11")


if __name__ == "__main__":
    main()
    pass

