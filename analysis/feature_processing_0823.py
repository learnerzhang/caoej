import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime, timedelta
import joblib
import os
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def load_data():
    # 加载水文数据
    # 加载水文数据
    df = pd.read_csv('imports/闸下潮位.csv', parse_dates=['time'])
    print(f"潮位数据记录数: {len(df)}, 时间范围: {df['time'].min()} 到 {df['time'].max()}")
    
    df_water_level = df[df['station_id'] == 3018].copy()
    print(f"站点3018记录数: {len(df_water_level)}, 缺失值: {df_water_level['water_level'].isnull().sum()}")
    # 对潮位数据按时间排序并插值
    df_water_level.sort_values('time', inplace=True)
    df_water_level['water_level'] = df_water_level['water_level'].interpolate(method='linear')
    # 如果还有缺失，用前一个值填充
    df_water_level['water_level'].fillna(method='ffill', inplace=True)
    df_water_level['water_level'].fillna(method='bfill', inplace=True)

    df_flow = pd.read_csv('imports/实测流量.csv', parse_dates=['监测日期'])
    df_flow.sort_values('监测日期', inplace=True)
    df_flow['流量'] = df_flow['流量'].interpolate(method='linear')
    df_flow['流量'].fillna(method='ffill', inplace=True)
    df_flow['流量'].fillna(method='bfill', inplace=True)

    df_rain_actual = pd.read_csv('imports/实测降雨.csv', parse_dates=['监测日期'])
    # 对于实测降雨，用0填充缺失的雨量
    df_rain_actual['雨量'].fillna(0, inplace=True)

    df_rain_forecast = pd.read_csv('imports/降雨预报.csv', parse_dates=['预计开始时间'])
    df_rain_forecast['降雨量'].fillna(0, inplace=True)

    df_water_status = pd.read_csv('imports/水位工况.csv', parse_dates=['监测日期'])
    df_water_status.sort_values('监测日期', inplace=True)
    df_water_status['水位'] = df_water_status['水位'].interpolate(method='linear')
    df_water_status['水位'].fillna(method='ffill', inplace=True)
    df_water_status['水位'].fillna(method='bfill', inplace=True)
    
    return df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status

def safe_linregress(x, y):
    """安全的线性回归计算，处理x值相同的情况"""
    if len(x) < 2 or np.var(x) == 0:
        return 0, 0, 0, 0, 0  # 斜率, 截距, r值, p值, 标准误
    
    try:
        return linregress(x, y)
    except:
        return 0, 0, 0, 0, 0

def extract_hourly_tidal_features(water_data, period_hours=24):
    """提取小时级别潮汐特征"""
    # 定义默认特征值
    default_features = {
        'mean': 0,
        'max': 0,
        'min': 0,
        'range': 0,
        'slope': 0,
        'r_squared': 0,
        'cycle_count': 0,
        'rise_rate': 0,
        'fall_rate': 0,
        'phase': 0
    }
    
    if len(water_data) < 2:
        return default_features
    
    # 确保数据按时间排序
    water_data = water_data.sort_values('time')
    levels = water_data['water_level'].values
    
    features = {}
    
    # 基础统计特征
    features['mean'] = np.mean(levels)
    features['max'] = np.max(levels)
    features['min'] = np.min(levels)
    features['range'] = features['max'] - features['min']
    
    # 趋势特征 - 使用安全的线性回归
    time_numeric = (water_data['time'] - water_data['time'].min()).dt.total_seconds().values
    slope, intercept, r_value, p_value, std_err = safe_linregress(time_numeric, levels)
    features['slope'] = slope
    features['r_squared'] = r_value**2
    
    # 潮汐周期检测
    diffs = np.diff(levels)
    turning_points = []
    
    for i in range(1, len(diffs)):
        if diffs[i] * diffs[i-1] < 0:  # 符号变化表示转折点
            turning_points.append(i)
    
    # 计算潮汐周期数量
    features['cycle_count'] = len(turning_points) // 2
    
    # 计算涨落潮速率
    rise_rates = []
    fall_rates = []
    
    for i in range(len(turning_points) - 1):
        start_idx = turning_points[i]
        end_idx = turning_points[i+1]
        level_diff = levels[end_idx] - levels[start_idx]
        
        # 计算时间差（小时）
        time_diff_hours = (water_data['time'].iloc[end_idx] - water_data['time'].iloc[start_idx]).total_seconds() / 3600
        
        if time_diff_hours > 0:
            rate = level_diff / time_diff_hours
            if level_diff > 0:
                rise_rates.append(rate)
            else:
                fall_rates.append(abs(rate))
    
    features['rise_rate'] = np.mean(rise_rates) if rise_rates else 0
    features['fall_rate'] = np.mean(fall_rates) if fall_rates else 0
    
    # 潮汐相位 (基于时间)
    hour_of_day = water_data['time'].iloc[0].hour
    features['phase'] = hour_of_day % 12 / 12.0  # 半日潮相位
    
    return features

def extract_features(df_order, df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status):
    features = []
    targets = []
    regions = ['绍兴平原', '嵊州', '虞南山区', '新昌', '虞北平原']
    
    # 创建空值统计器
    missing_stats = {
        'water_level': 0,
        'flow': 0,
        'rain_actual': 0,
        'rain_forecast': 0,
        'water_status': 0,
        'future_water_level': 0
    }

    df_order['目标水位'] = pd.to_numeric(df_order['目标水位'], errors='coerce')
    df_order['目标水位'] = df_order['目标水位'].fillna(1)
    df_order['目标水位'] = df_order['目标水位'].astype(float)
    
    # 过滤无效记录：开闸时间早于调令时间
    df_order = df_order[df_order['开闸时间'] > df_order['SIGNTM']]
    
    for idx, row in df_order.iterrows():
        base_time = row['SIGNTM']
        open_time = row['开闸时间']
        start_time_24h = base_time - timedelta(hours=24)
        start_time_week = base_time - timedelta(days=7)  # 新增：一周前的时间
        start_time_48h = base_time - timedelta(hours=48)
        
        # 检查所有目标变量是否有效
        if any(pd.isnull(row[['开闸时间', '开闸时长', '开闸孔数', '目标水位']])):
            continue
            
        # 处理开闸孔数
        raw_gate_count = row['开闸孔数']
        processed_gate_count = raw_gate_count / 2
        processed_gate_count = max(1, min(10, processed_gate_count))
        
        # 计算开闸时间（小时+分钟）
        open_hour = open_time.hour + open_time.minute/60.0
        target_water_level = row['目标水位']

        # 联合目标
        dura_dot_num = float(row['开闸时长']) * processed_gate_count
        if dura_dot_num <= 0:
            dura_dot_num = 1e-5
        log_dura_dot_num = np.log(dura_dot_num)

        targets.append([
            open_hour,
            row['开闸时长'],
            processed_gate_count,
            target_water_level,
            dura_dot_num,
            log_dura_dot_num
        ])
        
        feat_dict = {}
        feat_dict['date'] = base_time.date()

        # 时间特征
        feat_dict['hour_of_day'] = base_time.hour
        feat_dict['day_of_week'] = base_time.weekday()
        feat_dict['month'] = base_time.month
        feat_dict['is_weekend'] = 1 if base_time.weekday() >= 5 else 0
        feat_dict['hour_sin'] = np.sin(2 * np.pi * base_time.hour / 24)
        feat_dict['hour_cos'] = np.cos(2 * np.pi * base_time.hour / 24)
        feat_dict['day_of_year'] = base_time.timetuple().tm_yday
        
        # 历史操作特征 - 修改为最近一周
        prev_orders = df_order[df_order['SIGNTM'] < base_time].sort_values('SIGNTM', ascending=False)
        
        if len(prev_orders) > 0:
            latest = prev_orders.iloc[0]
            feat_dict['prev_gate_count'] = latest['开闸孔数']
            feat_dict['prev_duration'] = latest['开闸时长'] if not pd.isnull(latest['开闸时长']) else 0
            feat_dict['prev_op_hour'] = latest['开闸时间'].hour if not pd.isnull(latest['开闸时间']) else 0
        else:
            feat_dict['prev_gate_count'] = 0
            feat_dict['prev_duration'] = 0
            feat_dict['prev_op_hour'] = 0
        
        # 过去一周操作统计 - 修改为168小时（7天）
        last_week_orders = df_order[
            (df_order['SIGNTM'] >= start_time_week) & 
            (df_order['SIGNTM'] < base_time)
        ]
        feat_dict['ops_week_count'] = len(last_week_orders)
        feat_dict['ops_week_avg_gates'] = last_week_orders['开闸孔数'].mean() if len(last_week_orders) > 0 else 0
        feat_dict['ops_week_total_duration'] = last_week_orders['开闸时长'].sum() if len(last_week_orders) > 0 else 0
        
        # 潮位数据查询 - 优先使用48小时内数据
        water_data_24h = df_water_level[
            (df_water_level['time'] >= start_time_48h) & 
            (df_water_level['time'] <= base_time)
        ]

        # 如果数据不足，尝试使用更宽的时间范围
        if len(water_data_24h) < 12:  # 至少需要12个数据点
            start_time_72h = base_time - timedelta(hours=72)
            water_data_24h = df_water_level[
                (df_water_level['time'] >= start_time_72h) & 
                (df_water_level['time'] <= base_time)
            ]

        water_missing = 0
        if len(water_data_24h) > 0:
            tidal_features = extract_hourly_tidal_features(water_data_24h, 24)
            # 添加前缀以区分不同时间段
            for key, value in tidal_features.items():
                feat_dict[f'tide_24h_{key}'] = value
        else:
            missing_stats['water_level'] += 1
            water_missing = 1
            # 添加默认值
            default_features = extract_hourly_tidal_features(pd.DataFrame(), 24)
            for key, value in default_features.items():
                feat_dict[f'tide_24h_{key}'] = value

        feat_dict['water_missing'] = water_missing
        
        # 过去12小时潮位特征
        start_time_12h = base_time - timedelta(hours=12)
        water_data_12h = df_water_level[
            (df_water_level['time'] >= start_time_12h) & 
            (df_water_level['time'] <= base_time)
        ]
        
        if len(water_data_12h) > 0:
            tidal_features = extract_hourly_tidal_features(water_data_12h, 12)
            for key, value in tidal_features.items():
                feat_dict[f'tide_12h_{key}'] = value
        else:
            # 使用24小时数据的后半部分作为近似
            if len(water_data_24h) > 12:  # 至少有12小时数据
                mid_point = len(water_data_24h) // 2
                water_data_12h_approx = water_data_24h.iloc[mid_point:]
                tidal_features = extract_hourly_tidal_features(water_data_12h_approx, 12)
                for key, value in tidal_features.items():
                    feat_dict[f'tide_12h_{key}'] = value
            else:
                # 使用默认值
                default_features = extract_hourly_tidal_features(pd.DataFrame(), 12)
                for key, value in default_features.items():
                    feat_dict[f'tide_12h_{key}'] = value
        
        # 未来12小时潮位特征
        # 在extract_features函数中添加更稳健的未来潮位预测
        future_water_data = df_water_level[
            (df_water_level['time'] >= base_time) & 
            (df_water_level['time'] <= base_time + timedelta(hours=12))
        ]

        if len(future_water_data) > 0:
            future_tidal_features = extract_hourly_tidal_features(future_water_data, 12)
            for key, value in future_tidal_features.items():
                feat_dict[f'future_tide_{key}'] = value
        else:
            missing_stats['future_water_level'] += 1
            future_water_missing = 1
            
            # 改进的预测方法 - 使用更长的历史数据和周期性模式
            historical_data = df_water_level[
                (df_water_level['time'] >= base_time - timedelta(days=7)) & 
                (df_water_level['time'] <= base_time)
            ]
            
            if len(historical_data) > 12:
                # 使用历史同期数据预测
                same_period_data = historical_data[
                    historical_data['time'].dt.time.isin([
                        (base_time + timedelta(hours=i)).time() for i in range(1, 13)
                    ])
                ]
                
                if len(same_period_data) > 0:
                    # 计算平均变化模式
                    future_levels = []
                    for hour_offset in range(1, 13):
                        target_time = (base_time + timedelta(hours=hour_offset)).time()
                        same_time_data = historical_data[historical_data['time'].dt.time == target_time]
                        if len(same_time_data) > 0:
                            future_level = same_time_data['water_level'].mean()
                            future_levels.append(future_level)
                    
                    if len(future_levels) == 12:
                        # 创建模拟的未来数据
                        future_times = [base_time + timedelta(hours=i) for i in range(1, 13)]
                        simulated_data = pd.DataFrame({
                            'time': future_times,
                            'water_level': future_levels
                        })
                        future_tidal_features = extract_hourly_tidal_features(simulated_data, 12)
                        for key, value in future_tidal_features.items():
                            feat_dict[f'future_tide_{key}'] = value
                    else:
                        # 使用默认值
                        default_features = extract_hourly_tidal_features(pd.DataFrame(), 12)
                        for key, value in default_features.items():
                            feat_dict[f'future_tide_{key}'] = value
                else:
                    # 使用线性外推
                    last_6h_data = historical_data[historical_data['time'] >= base_time - timedelta(hours=6)]
                    if len(last_6h_data) > 1:
                        time_numeric = (last_6h_data['time'] - last_6h_data['time'].min()).dt.total_seconds().values
                        slope, intercept, _, _, _ = safe_linregress(time_numeric, last_6h_data['water_level'])
                        
                        future_times = [base_time + timedelta(hours=i) for i in range(1, 13)]
                        future_levels = [intercept + slope * (i * 3600) for i in range(1, 13)]
                        
                        simulated_data = pd.DataFrame({
                            'time': future_times,
                            'water_level': future_levels
                        })
                        
                        future_tidal_features = extract_hourly_tidal_features(simulated_data, 12)
                        for key, value in future_tidal_features.items():
                            feat_dict[f'future_tide_{key}'] = value
                    else:
                        default_features = extract_hourly_tidal_features(pd.DataFrame(), 12)
                        for key, value in default_features.items():
                            feat_dict[f'future_tide_{key}'] = value
            else:
                default_features = extract_hourly_tidal_features(pd.DataFrame(), 12)
                for key, value in default_features.items():
                    feat_dict[f'future_tide_{key}'] = value
        
            feat_dict['future_water_missing'] = future_water_missing
        
        # 流量特征
        flow_data = df_flow[
            (df_flow['监测日期'] >= start_time_24h) & 
            (df_flow['监测日期'] <= base_time)
        ]
        
        flow_missing = 0
        if len(flow_data) > 0:
            feat_dict['flow_mean'] = flow_data['流量'].mean()
            feat_dict['flow_max'] = flow_data['流量'].max()
            feat_dict['flow_min'] = flow_data['流量'].min()
            feat_dict['flow_range'] = feat_dict['flow_max'] - feat_dict['flow_min']
            feat_dict['flow_var'] = flow_data['流量'].var()
            feat_dict['flow_skew'] = flow_data['流量'].skew() if len(flow_data) > 2 else 0
        else:
            missing_stats['flow'] += 1
            flow_missing = 1
            feat_dict.update({f'flow_{stat}': 0 for stat in ['mean', 'max', 'min', 'range', 'var', 'skew']})
        
        feat_dict['flow_missing'] = flow_missing
        
        # 降雨特征 - 修改缺失统计逻辑
        rain_actual_data = df_rain_actual[
            (df_rain_actual['监测日期'] >= start_time_24h) & 
            (df_rain_actual['监测日期'] <= base_time)
        ]
        
        rain_forecast_data = df_rain_forecast[
            (df_rain_forecast['预计开始时间'] >= base_time) & 
            (df_rain_forecast['预计开始时间'] <= base_time + timedelta(hours=24))
        ]
        
        total_rain_actual = 0
        total_rain_forecast = 0
        rain_actual_missing_in_record = False
        rain_forecast_missing_in_record = False
        
        for region in regions:
            # 实际降雨
            region_rain = rain_actual_data[rain_actual_data['所属区域'] == region]
            if len(region_rain) > 0:
                feat_dict[f'rain_actual_{region}_sum'] = region_rain['雨量'].sum()
                feat_dict[f'rain_actual_{region}_max'] = region_rain['雨量'].max()
                feat_dict[f'rain_actual_{region}_mean'] = region_rain['雨量'].mean()
                total_rain_actual += feat_dict[f'rain_actual_{region}_sum']
            else:
                rain_actual_missing_in_record = True
                feat_dict[f'rain_actual_{region}_sum'] = 0
                feat_dict[f'rain_actual_{region}_max'] = 0
                feat_dict[f'rain_actual_{region}_mean'] = 0
            
            # 预测降雨
            region_forecast = rain_forecast_data[rain_forecast_data['大流域'] == region]
            if len(region_forecast) > 0:
                feat_dict[f'rain_forecast_{region}_mean'] = region_forecast['降雨量'].mean()
                feat_dict[f'rain_forecast_{region}_max'] = region_forecast['降雨量'].max()
                feat_dict[f'rain_forecast_{region}_sum'] = region_forecast['降雨量'].sum()
                total_rain_forecast += feat_dict[f'rain_forecast_{region}_sum']
            else:
                rain_forecast_missing_in_record = True
                feat_dict[f'rain_forecast_{region}_mean'] = 0
                feat_dict[f'rain_forecast_{region}_max'] = 0
                feat_dict[f'rain_forecast_{region}_sum'] = 0
        
        # 区域平均降雨特征
        feat_dict['rain_actual_total'] = total_rain_actual
        feat_dict['rain_forecast_total'] = total_rain_forecast
        feat_dict['rain_actual_avg'] = total_rain_actual / len(regions) if len(regions) > 0 else 0
        feat_dict['rain_forecast_avg'] = total_rain_forecast / len(regions) if len(regions) > 0 else 0
        feat_dict['rain_missing'] = 1 if (rain_actual_missing_in_record or rain_forecast_missing_in_record) else 0
        
        # 更新缺失统计（每条记录只计一次）
        if rain_actual_missing_in_record:
            missing_stats['rain_actual'] += 1
        if rain_forecast_missing_in_record:
            missing_stats['rain_forecast'] += 1
        
        # 添加降雨变化率特征
        if len(rain_actual_data) > 1:
            rain_actual_data = rain_actual_data.sort_values('监测日期')
            rain_diff = rain_actual_data['雨量'].diff().dropna()
            feat_dict['rain_change_rate'] = rain_diff.mean() if len(rain_diff) > 0 else 0
        else:
            feat_dict['rain_change_rate'] = 0
        
        # 添加组合特征
        feat_dict['water_rain_ratio'] = feat_dict.get('tide_24h_mean', 0) / (feat_dict['rain_actual_total'] + 1e-5)
        feat_dict['flow_rain_ratio'] = feat_dict.get('flow_mean', 0) / (feat_dict['rain_actual_total'] + 1e-5)
        
        # 添加时间敏感性特征
        feat_dict['is_rush_hour'] = 1 if 7 <= base_time.hour <= 9 or 17 <= base_time.hour <= 19 else 0

        # 水位工况特征
        water_status_data = df_water_status[
            (df_water_status['监测日期'] >= start_time_24h) & 
            (df_water_status['监测日期'] <= base_time)
        ]
        
        water_status_missing = 0
        if len(water_status_data) > 0:
            feat_dict['water_status_mean'] = water_status_data['水位'].mean()
            feat_dict['water_status_max'] = water_status_data['水位'].max()
            feat_dict['water_status_min'] = water_status_data['水位'].min()
            feat_dict['water_status_range'] = feat_dict['water_status_max'] - feat_dict['water_status_min']
            
            if len(water_status_data) > 1:
                times = (water_status_data['监测日期'] - water_status_data['监测日期'].min()).dt.total_seconds().values
                slope, _, _, _, _ = safe_linregress(times, water_status_data['水位'])
                feat_dict['water_status_slope'] = slope
            else:
                feat_dict['water_status_slope'] = 0
        else:
            missing_stats['water_status'] += 1
            water_status_missing = 1
            feat_dict.update({f'water_status_{stat}': 0 for stat in ['mean', 'max', 'min', 'range', 'slope']})
        
        feat_dict['water_status_missing'] = water_status_missing
        
        features.append(feat_dict)
    
    # 打印缺失值统计
    print("\n特征缺失统计:")
    for key, count in missing_stats.items():
        print(f"{key}: {count} 条记录缺失 (占总记录数 {count/len(df_order)*100:.2f}%)")
    return pd.DataFrame(features), np.array(targets)

def save_features(feat_dir="features_0823", prefix="00"):
    # 加载调令信息数据
    if prefix == "11":
        # 加载两个数据源
        df_order = pd.read_csv(
            f"imports/07调令信息.csv",
            parse_dates=['SIGNTM', '开闸时间']
        )
        train_df_order = pd.read_csv(
            f"imports/00调令信息.csv",
            parse_dates=['SIGNTM', '开闸时间']
        )
        
        # 计算需要采样的数量（确保为整数）
        sample_size = int(len(train_df_order) * 0.05)
        # 从训练数据中随机采样，避免使用不适合DataFrame的random.choice
        sampled_data = train_df_order.sample(n=sample_size, random_state=42)  # 固定随机种子，保证可复现性
        
        # 合并数据（修正原代码中错误的+运算）
        df_order = pd.concat([df_order, sampled_data], ignore_index=True)
    else:
        # 加载对应前缀的数据源
        df_order = pd.read_csv(
            f"imports/{prefix}调令信息.csv",
            parse_dates=['SIGNTM', '开闸时间']
        )

    df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status = load_data()
    X, y = extract_features(df_order, df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status)
    
    # 保存特征和目标
    os.makedirs(feat_dir, exist_ok=True)
    X.to_csv(f'{feat_dir}/{prefix}_features.csv', index=False)
    np.save(f'{feat_dir}/{prefix}_target.npy', y)
    
    # 保存缺失值处理信息
    missing_info = pd.DataFrame({
        'feature': X.columns,
        'missing_count': X.isnull().sum(),
        'missing_percent': X.isnull().mean() * 100
    })

    missing_info.to_csv(f'{feat_dir}/{prefix}_feature_missing_info.csv', index=False)
    print(f"\n特征数据已保存: {prefix}_features.csv 和 {prefix}_target.npy")
    print(f"缺失值统计已保存: {prefix}_feature_missing_info.csv")
    print(f"目标变量维度: {y.shape} (开闸时间, 开闸时长, 开闸孔数, 目标水位, 联合时长-孔数, 联合时长-孔数-log)")

if __name__ == '__main__':
    # 训练集
    save_features(prefix="00")
    # 测试集 7+8月
    save_features(prefix="07")
    # 测试集 7+8+随机月
    save_features(prefix="11")