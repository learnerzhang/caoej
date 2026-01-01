import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import find_peaks, savgol_filter
from matplotlib.dates import DateFormatter, HourLocator
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from datetime import timedelta, datetime

# 设置字体，确保中文正常显示
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
USE_CHINESE = True

# 1. 数据加载与预处理 (优化版)
def load_and_preprocess(data_path, file_pattern):
    """加载并预处理多个潮位数据文件"""
    files = glob.glob(f"{data_path}/**/{file_pattern}", recursive=True)
    if not files:
        raise FileNotFoundError(f"未找到匹配的文件: {file_pattern}")
    
    df_list = []
    for f in files:
        # 读取CSV，无表头
        temp_df = pd.read_csv(f, header=None, names=['station_id', 'time', 'water_level'])
        df_list.append(temp_df)
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    # 转换时间格式并排序
    full_df['time'] = pd.to_datetime(full_df['time'])
    full_df.sort_values(['station_id', 'time'], inplace=True)
    
    # 数据去重（关键修复）
    full_df = full_df.drop_duplicates(subset=['station_id', 'time'])
    
    # 添加时间特征
    full_df['hour'] = full_df['time'].dt.hour
    full_df['day'] = full_df['time'].dt.day
    full_df['month'] = full_df['time'].dt.month
    full_df['date'] = full_df['time'].dt.date  # 添加日期列
    
    # 计算变化率（处理时间不连续情况）
    full_df['time_diff'] = full_df.groupby('station_id')['time'].diff().dt.total_seconds() / 3600
    full_df['level_diff'] = full_df.groupby('station_id')['water_level'].diff()
    full_df['change_rate'] = full_df['level_diff'] / full_df['time_diff']
    
    # 处理可能的NaN值
    full_df['change_rate'] = full_df['change_rate'].fillna(0)
    
    return full_df

# 2. 潮汐拐点检测 (优化版)
def detect_tidal_features(df, station_id=3018, prominence=1.0, time_range=None):
    """检测潮汐特征：高潮、低潮和拐点"""
    station_df = df[df['station_id'] == station_id].copy()
    if station_df.empty:
        raise ValueError(f"站点 {station_id} 无数据")
    
    # 时间范围筛选（解决数据量过大问题）
    if time_range:
        start_date, end_date = time_range
        station_df = station_df[(station_df['time'] >= start_date) & (station_df['time'] <= end_date)]
    
    if len(station_df) < 10:
        raise ValueError(f"站点 {station_id} 在选定时间段内数据不足")
    
    # 平滑处理（动态窗口大小）
    window_size = min(21, max(5, len(station_df)//10))
    if window_size % 2 == 0:  # 确保窗口大小为奇数
        window_size += 1
    
    station_df['smoothed'] = savgol_filter(
        station_df['water_level'], 
        window_length=window_size, 
        polyorder=2
    )
    
    # 动态设置distance参数（基于数据密度）
    avg_interval = station_df['time'].diff().mean().total_seconds() / 3600
    min_distance = max(4, int(4 / avg_interval))  # 至少间隔4小时
    
    # 检测高潮位 (峰值)
    high_tide_idx, _ = find_peaks(
        station_df['smoothed'], 
        prominence=prominence,
        distance=min_distance
    )
    high_tides = station_df.iloc[high_tide_idx]
    
    # 检测低潮位 (波谷)
    low_tide_idx, _ = find_peaks(
        -station_df['smoothed'], 
        prominence=prominence,
        distance=min_distance
    )
    low_tides = station_df.iloc[low_tide_idx]
    
    # 检测关键拐点 (变化率转折)
    station_df['change_rate_diff'] = station_df['change_rate'].diff().abs()
    turning_points = station_df[
        (station_df['change_rate_diff'] > 0.5) & 
        (station_df['change_rate'].abs() > 0.3)
    ]
    
    return {
        'data': station_df,
        'high_tides': high_tides,
        'low_tides': low_tides,
        'turning_points': turning_points
    }

# 3. 潮汐周期分析 (优化版)
def analyze_tidal_periods(tidal_features):
    """分析潮汐周期特征"""
    ht = tidal_features['high_tides']
    lt = tidal_features['low_tides']
    
    if len(ht) < 2 or len(lt) < 2:
        return {
            'high_tide_periods': None,
            'low_tide_periods': None,
            'rise_fall_times': []
        }
    
    # 计算高潮间隔（过滤异常值）
    high_tide_periods = ht['time'].diff().dt.total_seconds() / 3600
    valid_high_periods = high_tide_periods[(high_tide_periods > 10) & (high_tide_periods < 15)]
    
    # 计算低潮间隔（过滤异常值）
    low_tide_periods = lt['time'].diff().dt.total_seconds() / 3600
    valid_low_periods = low_tide_periods[(low_tide_periods > 10) & (low_tide_periods < 15)]
    
    # 计算涨落潮持续时间
    rise_fall_times = []
    
    # 确保高低潮时间序列对齐
    all_events = pd.concat([
        ht[['time', 'water_level']].assign(event_type='high'),
        lt[['time', 'water_level']].assign(event_type='low')
    ]).sort_values('time')
    
    # 找到连续的涨落周期
    for i in range(1, len(all_events)):
        prev = all_events.iloc[i-1]
        curr = all_events.iloc[i]
        
        if prev['event_type'] == 'low' and curr['event_type'] == 'high':
            # 涨潮周期
            duration = (curr['time'] - prev['time']).total_seconds() / 3600
            rise_fall_times.append(('rise', duration, prev['time'], curr['time']))
        elif prev['event_type'] == 'high' and curr['event_type'] == 'low':
            # 落潮周期
            duration = (curr['time'] - prev['time']).total_seconds() / 3600
            rise_fall_times.append(('fall', duration, prev['time'], curr['time']))
    
    return {
        'high_tide_periods': valid_high_periods.mean() if not valid_high_periods.empty else None,
        'low_tide_periods': valid_low_periods.mean() if not valid_low_periods.empty else None,
        'rise_fall_times': rise_fall_times
    }

# 4. 可视化分析 (优化版)
def plot_tidal_analysis(tidal_features, tidal_periods, save_dir="", daily_plot=False, date_range=None, weekly_plot=False, week_num=None):
    """绘制潮汐分析图表
    daily_plot: 是否按天生成多张图
    weekly_plot: 是否绘制周图
    week_num: 周编号
    """
    df = tidal_features['data']
    ht = tidal_features['high_tides']
    lt = tidal_features['low_tides']
    tp = tidal_features['turning_points']
    
    if len(df) == 0:
        print("无数据可绘制")
        return
    
    # 确保保存目录存在
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 按天生成多张图
    if daily_plot:
        # 获取所有日期
        dates = df['time'].dt.date.unique()
        
        for day in dates:
            start = pd.Timestamp(day)
            end = start + timedelta(days=1)
            
            # 过滤当天数据
            day_df = df[(df['time'] >= start) & (df['time'] < end)]
            if day_df.empty:
                continue
                
            day_ht = ht[(ht['time'] >= start) & (ht['time'] < end)]
            day_lt = lt[(lt['time'] >= start) & (lt['time'] < end)]
            day_tp = tp[(tp['time'] >= start) & (tp['time'] < end)]
            
            # 创建子特征集
            day_features = {
                'data': day_df,
                'high_tides': day_ht,
                'low_tides': day_lt,
                'turning_points': day_tp
            }
            
            # 绘制单日图表
            _plot_single_day(day_features, tidal_periods, save_dir, day)
        
        return
    
    # 绘制整个时间范围的图表
    fig = plt.figure(figsize=(14, 10))
    
    # 水位曲线
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(df['time'], df['water_level'], 'b-', label='实际水位', alpha=0.7)
    plt.plot(df['time'], df['smoothed'], 'g-', label='平滑曲线', linewidth=1.5)
    
    # 标记高低潮
    if not ht.empty:
        plt.scatter(ht['time'], ht['water_level'], c='red', s=80, marker='^', label='高潮位', zorder=5)
        for i, row in ht.iterrows():
            label = f"H: {row['water_level']:.2f}m" if USE_CHINESE else f"H: {row['water_level']:.2f}m"
            plt.annotate(label, 
                        (row['time'], row['water_level']),
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', fontsize=9, color='darkred')
    
    if not lt.empty:
        plt.scatter(lt['time'], lt['water_level'], c='blue', s=80, marker='v', label='低潮位', zorder=5)
        for i, row in lt.iterrows():
            label = f"L: {row['water_level']:.2f}m" if USE_CHINESE else f"L: {row['water_level']:.2f}m"
            plt.annotate(label, 
                        (row['time'], row['water_level']),
                        xytext=(0, -25), textcoords='offset points',
                        ha='center', fontsize=9, color='darkblue')
    
    # 标记拐点
    if not tp.empty:
        plt.scatter(tp['time'], tp['water_level'], c='purple', s=100, marker='*', 
                   label='变化拐点' if USE_CHINESE else 'Turning Points', edgecolors='gold', zorder=6)
    
    # 添加周期信息
    period_info = ""
    if tidal_periods['high_tide_periods'] is not None:
        period_info += f"平均高潮间隔: {tidal_periods['high_tide_periods']:.1f}小时\n"
    if tidal_periods['low_tide_periods'] is not None:
        period_info += f"平均低潮间隔: {tidal_periods['low_tide_periods']:.1f}小时"
    
    if period_info:
        plt.text(0.02, 0.95, period_info, transform=ax1.transAxes,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # 根据绘图类型设置标题
    if weekly_plot:
        title = f'站点 {df["station_id"].iloc[0]} 第{week_num}周潮汐分析'
    else:
        title = f'站点 {df["station_id"].iloc[0]} 潮汐水位分析'
    
    plt.title(title, fontsize=14)
    plt.ylabel('水位 (m)' if USE_CHINESE else 'Water Level (m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 智能图例（避免过多条目）
    handles, labels = ax1.get_legend_handles_labels()
    if len(handles) > 6:
        # 只显示前5个图例
        plt.legend(handles[:5], labels[:5], loc='upper right')
    else:
        plt.legend(loc='upper right')
    
    # 设置时间格式 - 确保显示完整日期时间
    _set_time_format(ax1, df, weekly_plot)
    
    plt.xticks(rotation=45)
    
    # 变化率曲线
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(df['time'], df['change_rate'], 'm-', label='水位变化率' if USE_CHINESE else 'Change Rate')
    
    if not tp.empty:
        plt.scatter(tp['time'], tp['change_rate'], c='purple', s=60, marker='*', 
                   label='拐点' if USE_CHINESE else 'Turning Points')
        
        # 添加变化方向标签
        for i, row in tp.iterrows():
            direction = "↑涨" if row['change_rate'] > 0 else "↓落"
            eng_direction = "↑Rise" if row['change_rate'] > 0 else "↓Fall"
            
            label = f"{direction}{abs(row['change_rate']):.2f}m/h" if USE_CHINESE else f"{eng_direction}{abs(row['change_rate']):.2f}m/h"
            
            plt.annotate(label, 
                        (row['time'], row['change_rate']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, 
                        color='green' if row['change_rate'] > 0 else 'brown')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title('水位变化率分析' if USE_CHINESE else 'Water Level Change Rate', fontsize=14)
    plt.ylabel('变化率 (m/h)' if USE_CHINESE else 'Change Rate (m/h)', fontsize=12)
    plt.xlabel('时间' if USE_CHINESE else 'Time', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 设置变化率图的时间格式（与水位图一致）
    _set_time_format(ax2, df, weekly_plot)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 安全保存图像
    try:
        if weekly_plot:
            filename = f'tidal_analysis_station_{df["station_id"].iloc[0]}_week{week_num}.png'
        else:
            filename = f'tidal_analysis_station_{df["station_id"].iloc[0]}.png'
        save_path = os.path.join(save_dir, filename) if save_dir else filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    except Exception as e:
        print(f"保存图像失败: {str(e)}")
    
    plt.close(fig)
    
    # 打印涨落潮时间分析
    if tidal_periods['rise_fall_times']:
        print("\n涨落潮持续时间分析:" if USE_CHINESE else "\nTidal Duration Analysis:")
        for i, (tide_type, duration, start, end) in enumerate(tidal_periods['rise_fall_times']):
            tide_name = "涨" if tide_type == 'rise' else "落"
            eng_tide = "Rise" if tide_type == 'rise' else "Fall"
            
            print(f"{i+1}. {tide_name}潮: {duration:.2f}小时 ({start.strftime('%m-%d %H:%M')} → {end.strftime('%m-%d %H:%M')})"
                  if USE_CHINESE else 
                  f"{i+1}. {eng_tide} Tide: {duration:.2f} hours ({start.strftime('%m-%d %H:%M')} → {end.strftime('%m-%d %H:%M')})")

def _plot_single_day(tidal_features, tidal_periods, save_dir, day):
    """绘制单日潮汐分析图表"""
    df = tidal_features['data']
    ht = tidal_features['high_tides']
    lt = tidal_features['low_tides']
    tp = tidal_features['turning_points']
    
    if len(df) == 0:
        return
    
    fig = plt.figure(figsize=(14, 10))
    
    # 水位曲线
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(df['time'], df['water_level'], 'b-', label='实际水位', alpha=0.7)
    plt.plot(df['time'], df['smoothed'], 'g-', label='平滑曲线', linewidth=1.5)
    
    # 标记高低潮
    if not ht.empty:
        plt.scatter(ht['time'], ht['water_level'], c='red', s=80, marker='^', label='高潮位', zorder=5)
        for i, row in ht.iterrows():
            label = f"H: {row['water_level']:.2f}m"
            plt.annotate(label, 
                        (row['time'], row['water_level']),
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', fontsize=9, color='darkred')
    
    if not lt.empty:
        plt.scatter(lt['time'], lt['water_level'], c='blue', s=80, marker='v', label='低潮位', zorder=5)
        for i, row in lt.iterrows():
            label = f"L: {row['water_level']:.2f}m"
            plt.annotate(label, 
                        (row['time'], row['water_level']),
                        xytext=(0, -25), textcoords='offset points',
                        ha='center', fontsize=9, color='darkblue')
    
    # 标记拐点
    if not tp.empty:
        plt.scatter(tp['time'], tp['water_level'], c='purple', s=100, marker='*', 
                   label='变化拐点', edgecolors='gold', zorder=6)
    
    title = f'站点 {df["station_id"].iloc[0]} 潮汐水位分析 ({day})'
    plt.title(title, fontsize=14)
    plt.ylabel('水位 (m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # 设置时间格式为小时:分钟
    ax1.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
    ax1.xaxis.set_major_locator(HourLocator(interval=3))
    plt.xticks(rotation=45)
    
    # 变化率曲线
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(df['time'], df['change_rate'], 'm-', label='水位变化率')
    
    if not tp.empty:
        plt.scatter(tp['time'], tp['change_rate'], c='purple', s=60, marker='*', 
                   label='拐点')
        
        # 添加变化方向标签
        for i, row in tp.iterrows():
            direction = "↑涨" if row['change_rate'] > 0 else "↓落"
            label = f"{direction}{abs(row['change_rate']):.2f}m/h"
            
            plt.annotate(label, 
                        (row['time'], row['change_rate']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, 
                        color='green' if row['change_rate'] > 0 else 'brown')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title('水位变化率分析', fontsize=14)
    plt.ylabel('变化率 (m/h)', fontsize=12)
    plt.xlabel('时间', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 设置变化率图的时间格式
    ax2.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 安全保存图像
    try:
        filename = f'tidal_analysis_station_{df["station_id"].iloc[0]}_{day}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"单日图表已保存至: {save_path}")
    except Exception as e:
        print(f"保存单日图像失败: {str(e)}")
    
    plt.close(fig)

def _set_time_format(ax, df, is_weekly=False):
    """智能设置时间轴格式，确保显示完整日期时间"""
    time_range = df['time'].max() - df['time'].min()
    total_days = time_range.total_seconds() / (3600 * 24)
    
    if total_days > 30:
        # 长时间范围：按月显示
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    elif total_days > 7 or is_weekly:
        # 周图或中等时间范围：按天显示带日期
        ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    elif total_days > 1:
        # 多天范围：按天显示带日期和时间
        ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    else:
        # 短期范围：按小时显示
        ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

# 主函数 (优化版)
def main():
    try:
        data_path = "exports"
        file_pattern = "*闸下潮位.csv"
        output_dir = "tidal_analysis_results"  # 指定图片保存目录
        
        print("开始加载数据...")
        df = load_and_preprocess(data_path, file_pattern)
        print(f"成功加载 {len(df)} 条记录，去重后保留 {len(df)} 条")
        
        # 分析主要站点 (3018)
        station_id = 3018
        # station_id = 7308
        
        # 自动确定时间范围（最近7天）
        latest_time = df['time'].max()
        time_range = (latest_time - pd.Timedelta(days=7), latest_time)
        
        print(f"\n分析站点 {station_id}，时间范围: {time_range[0].strftime('%Y-%m-%d')} 至 {time_range[1].strftime('%Y-%m-%d')}")
        
        tidal_features = detect_tidal_features(df, station_id, prominence=0.5, time_range=time_range)
        
        print(f"检测到高潮: {len(tidal_features['high_tides'])} 次")
        print(f"检测到低潮: {len(tidal_features['low_tides'])} 次")
        print(f"检测到拐点: {len(tidal_features['turning_points'])} 个")
        
        tidal_periods = analyze_tidal_periods(tidal_features)
        
        # 输出统计信息
        print(f"\n站点 {station_id} 潮汐特征:")
        if tidal_periods['high_tide_periods'] is not None:
            print(f"平均高潮间隔: {tidal_periods['high_tide_periods']:.2f}小时")
        else:
            print("未计算出有效的高潮间隔")
            
        if tidal_periods['low_tide_periods'] is not None:
            print(f"平均低潮间隔: {tidal_periods['low_tide_periods']:.2f}小时")
        else:
            print("未计算出有效的低潮间隔")
        
        # 可视化分析 - 生成整段时间的图表
        plot_tidal_analysis(tidal_features, tidal_periods, save_dir=output_dir)
        
        # 生成多天的潮汐曲线图（每天一张）
        daily_dir = os.path.join(output_dir, "daily")
        plot_tidal_analysis(tidal_features, tidal_periods, save_dir=daily_dir, daily_plot=True)
        
        # 新增：绘制最近三个月的每周图（每隔7天一张）
        weekly_dir = os.path.join(output_dir, "weekly")
        if not os.path.exists(weekly_dir):
            os.makedirs(weekly_dir)
        
        # 计算最近三个月的时间范围
        three_months_ago = latest_time - pd.Timedelta(days=90)
        
        # 生成每周的时间段（每周7天）
        current_start = three_months_ago
        week_num = 1
        
        while current_start < latest_time:
            current_end = current_start + pd.Timedelta(days=7)
            if current_end > latest_time:
                current_end = latest_time
                
            print(f"\n分析第{week_num}周: {current_start.strftime('%Y-%m-%d')} 至 {current_end.strftime('%Y-%m-%d')}")
            
            try:
                weekly_features = detect_tidal_features(
                    df, station_id, prominence=0.5, 
                    time_range=(current_start, current_end)
                )
                weekly_periods = analyze_tidal_periods(weekly_features)
                
                # 绘制周图
                plot_tidal_analysis(
                    weekly_features, weekly_periods, 
                    save_dir=weekly_dir, weekly_plot=True, week_num=week_num
                )
                
            except Exception as e:
                print(f"第{week_num}周分析失败: {str(e)}")
            
            current_start = current_end
            week_num += 1
        
        print(f"\n所有分析已完成，结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"处理出错: {str(e)}")

if __name__ == "__main__":
    main()