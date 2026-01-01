import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime, timedelta
import joblib
import os
from scipy.signal import find_peaks

def load_data():
    # 加载水文数据（保持不变）
    df_water_level = pd.read_csv('imports/闸下潮位.csv', parse_dates=['time'])
    df_flow = pd.read_csv('imports/实测流量.csv', parse_dates=['监测日期'])
    df_rain_actual = pd.read_csv('imports/实测降雨.csv', parse_dates=['监测日期'])
    df_rain_forecast = pd.read_csv('imports/降雨预报.csv', parse_dates=['预计开始时间'])
    df_water_status = pd.read_csv('imports/水位工况.csv', parse_dates=['监测日期'])
    
    return df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status

def extract_tidal_features(water_data):
    """提取24小时内的潮汐特征 - 增强版"""
    features = {}
    
    if len(water_data) < 2:
        return {
            'tide_range_max': 0,
            'tide_range_avg': 0,
            'tide_rise_avg_rate': 0,
            'tide_fall_avg_rate': 0,
            'tide_cycle_count': 0,
            'tide_type': 0,
            'tide_phase': 0
        }
    
    # 重采样到10分钟间隔
    resampled = water_data.set_index('time').resample('10T').mean().interpolate()
    levels = resampled['water_level'].values
    
    # 寻找波峰和波谷 - 增强检测逻辑
    min_prominence = np.ptp(levels) * 0.2  # 动态设置显著度阈值
    peaks, _ = find_peaks(levels, prominence=min_prominence)
    valleys, _ = find_peaks(-levels, prominence=min_prominence)
    
    # 合并关键点并排序
    key_points = sorted(np.concatenate([peaks, valleys]))
    
    # 特征初始化
    tide_ranges = []
    rise_rates = []
    fall_rates = []
    tide_phases = []
    
    # 分析每个潮汐周期
    for i in range(1, len(key_points)):
        prev_idx = key_points[i-1]
        curr_idx = key_points[i]
        
        prev_level = levels[prev_idx]
        curr_level = levels[curr_idx]
        time_diff = (resampled.index[curr_idx] - resampled.index[prev_idx]).total_seconds() / 3600
        
        # 计算潮汐相位
        phase = (curr_idx - prev_idx) / len(levels) * 6  # 6小时一个相位周期
        tide_phases.append(phase)
        
        # 涨潮特征
        if curr_level > prev_level:
            tide_range = curr_level - prev_level
            tide_ranges.append(tide_range)
            rise_rates.append(tide_range / time_diff if time_diff > 0 else 0)
        
        # 落潮特征
        else:
            tide_range = prev_level - curr_level
            tide_ranges.append(tide_range)
            fall_rates.append(tide_range / time_diff if time_diff > 0 else 0)
    
    # 计算统计特征
    features['tide_range_max'] = max(tide_ranges) if tide_ranges else 0
    features['tide_range_avg'] = np.mean(tide_ranges) if tide_ranges else 0
    features['tide_rise_avg_rate'] = np.mean(rise_rates) if rise_rates else 0
    features['tide_fall_avg_rate'] = np.mean(fall_rates) if fall_rates else 0
    features['tide_cycle_count'] = len(tide_ranges)
    features['tide_phase'] = np.mean(tide_phases) if tide_phases else 0
    
    # 潮汐类型识别 (1=半日潮，2=全日潮，3=混合潮)
    if features['tide_cycle_count'] >= 3:
        features['tide_type'] = 1
    elif features['tide_cycle_count'] == 1:
        features['tide_type'] = 2
    else:
        features['tide_type'] = 3
        
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
        'water_status': 0
    }

    df_order['目标水位'] = pd.to_numeric(df_order['目标水位'], errors='coerce')  # 非数字→NaN
    df_order['目标水位'] = df_order['目标水位'].fillna(1)  # NaN→1
    df_order['目标水位'] = df_order['目标水位'].astype(float)  # 转为浮点型
    
    # 过滤无效记录：开闸时间早于调令时间
    df_order = df_order[df_order['开闸时间'] > df_order['SIGNTM']]
    
    for idx, row in df_order.iterrows():
        base_time = row['SIGNTM']
        open_time = row['开闸时间']
        start_time_24h = base_time - timedelta(hours=24)
        
        # 检查所有目标变量是否有效
        if any(pd.isnull(row[['开闸时间', '开闸时长', '开闸孔数', '目标水位']])):
            continue
            
        # 多目标输出 [开闸时间(小时), 开闸时长, 开闸孔数]
        # 处理开闸孔数：除以2后规范在1-10之间
        raw_gate_count = row['开闸孔数']
        processed_gate_count = raw_gate_count / 2
        processed_gate_count = max(1, min(10, processed_gate_count))
        
        # 计算开闸时间（小时+分钟）
        open_hour = open_time.hour + open_time.minute/60.0
        # 目标水位
        target_water_level = row['目标水位']

        # 联合目标
        dura_dot_num = float(row['开闸时长']) * processed_gate_count
        # 避免零或负值（确保对数变换有效）
        if dura_dot_num <= 0:
            dura_dot_num = 1e-5  # 设置一个很小的正数
        log_dura_dot_num = np.log(dura_dot_num)

        targets.append([
            open_hour,  # 开闸时间(小时)
            row['开闸时长'],  # 开闸时长
            processed_gate_count,  # 开闸孔数
            target_water_level,  # 目标水位
            dura_dot_num,  # 联合目标
            log_dura_dot_num  # 联合目标的对数变换
        ])
        feat_dict = {}
        feat_dict['date'] = base_time.date()

        # 时间特征增强
        feat_dict['hour_of_day'] = base_time.hour
        feat_dict['day_of_week'] = base_time.weekday()
        feat_dict['month'] = base_time.month
        feat_dict['is_weekend'] = 1 if base_time.weekday() >= 5 else 0
        feat_dict['hour_sin'] = np.sin(2 * np.pi * base_time.hour / 24)
        feat_dict['hour_cos'] = np.cos(2 * np.pi * base_time.hour / 24)
        feat_dict['day_of_year'] = base_time.timetuple().tm_yday
        
        # 历史操作特征增强
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
        
        # 过去24小时操作统计
        last_24h_orders = df_order[
            (df_order['SIGNTM'] >= start_time_24h) & 
            (df_order['SIGNTM'] < base_time)
        ]
        feat_dict['ops_24h_count'] = len(last_24h_orders)
        feat_dict['ops_24h_avg_gates'] = last_24h_orders['开闸孔数'].mean() if len(last_24h_orders) > 0 else 0
        feat_dict['ops_24h_total_duration'] = last_24h_orders['开闸时长'].sum() if len(last_24h_orders) > 0 else 0
        
        # 水位特征增强 (24小时窗口)
        water_data_12h = df_water_level[
            (df_water_level['time'] >= start_time_24h) & 
            (df_water_level['time'] <= base_time)
        ]
        
        # 潮汐特征增强 (24小时窗口)
        water_data_24h = df_water_level[
            (df_water_level['time'] >= start_time_24h) & 
            (df_water_level['time'] <= base_time)
        ]
        
        water_missing = 0
        if len(water_data_12h) > 0:
            # 12小时基础特征
            feat_dict['water_mean'] = water_data_12h['water_level'].mean()
            feat_dict['water_max'] = water_data_12h['water_level'].max()
            feat_dict['water_min'] = water_data_12h['water_level'].min()
            feat_dict['water_range'] = feat_dict['water_max'] - feat_dict['water_min']
            
            if len(water_data_12h) > 1:
                times = (water_data_12h['time'] - water_data_12h['time'].min()).dt.total_seconds().values
                slope, intercept, r_value, p_value, std_err = linregress(times, water_data_12h['water_level'])
                feat_dict['water_slope'] = slope
                feat_dict['water_r_squared'] = r_value**2
            else:
                feat_dict['water_slope'] = 0
                feat_dict['water_r_squared'] = 0
            
            # 24小时潮汐特征
            tidal_features = extract_tidal_features(water_data_24h)
            feat_dict.update(tidal_features)
            
            # 添加潮汐相位特征
            feat_dict['tide_phase_sin'] = np.sin(2 * np.pi * feat_dict['tide_phase'] / 6)
            feat_dict['tide_phase_cos'] = np.cos(2 * np.pi * feat_dict['tide_phase'] / 6)
        else:
            missing_stats['water_level'] += 1
            water_missing = 1
            feat_dict.update({f'water_{stat}': 0 for stat in ['mean', 'max', 'min', 'range', 'slope', 'r_squared']})
            feat_dict.update({
                'tide_range_max': 0,
                'tide_range_avg': 0,
                'tide_rise_avg_rate': 0,
                'tide_fall_avg_rate': 0,
                'tide_cycle_count': 0,
                'tide_type': 0,
                'tide_phase': 0,
                'tide_phase_sin': 0,
                'tide_phase_cos': 0
            })
        
        feat_dict['water_missing'] = water_missing
        # 流量特征增强
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
        
        # 降雨特征增强
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
        rain_missing_counter = 0
        
        for region in regions:
            # 实际降雨
            region_rain = rain_actual_data[rain_actual_data['所属区域'] == region]
            if len(region_rain) > 0:
                feat_dict[f'rain_actual_{region}_sum'] = region_rain['雨量'].sum()
                feat_dict[f'rain_actual_{region}_max'] = region_rain['雨量'].max()
                feat_dict[f'rain_actual_{region}_mean'] = region_rain['雨量'].mean()
                total_rain_actual += feat_dict[f'rain_actual_{region}_sum']
            else:
                missing_stats['rain_actual'] += 1
                rain_missing_counter += 1
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
                missing_stats['rain_forecast'] += 1
                rain_missing_counter += 1
                feat_dict[f'rain_forecast_{region}_mean'] = 0
                feat_dict[f'rain_forecast_{region}_max'] = 0
                feat_dict[f'rain_forecast_{region}_sum'] = 0
        
        # 区域平均降雨特征
        feat_dict['rain_actual_total'] = total_rain_actual
        feat_dict['rain_forecast_total'] = total_rain_forecast
        feat_dict['rain_actual_avg'] = total_rain_actual / len(regions) if len(regions) > 0 else 0
        feat_dict['rain_forecast_avg'] = total_rain_forecast / len(regions) if len(regions) > 0 else 0
        feat_dict['rain_missing'] = 1 if rain_missing_counter > 0 else 0
        
        # 添加降雨变化率特征
        if len(rain_actual_data) > 1:
            rain_actual_data = rain_actual_data.sort_values('监测日期')
            rain_diff = rain_actual_data['雨量'].diff().dropna()
            feat_dict['rain_change_rate'] = rain_diff.mean() if len(rain_diff) > 0 else 0
        else:
            feat_dict['rain_change_rate'] = 0
        
        # 添加组合特征
        feat_dict['water_rain_ratio'] = feat_dict['water_mean'] / (feat_dict['rain_actual_total'] + 1e-5)
        feat_dict['flow_rain_ratio'] = feat_dict['flow_mean'] / (feat_dict['rain_actual_total'] + 1e-5)
        
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
                slope, _, _, _, _ = linregress(times, water_status_data['水位'])
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

def save_features():
    feat_dir = "features_v2"
    prefix = "07"
    # prefix = ""
    # 加载所有数据源
    df_order = pd.read_csv(f'imports/{prefix}调令信息.csv', parse_dates=['SIGNTM', '开闸时间'])
    
    # print(f"调令信息数据量: {len(df_order)}")
    # df_order = df_order[df_order['日期'] < '2025-07-01'] 
    # print(f"调令信息数据量: {len(df_order)}")
    df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status = load_data()
    X, y = extract_features(df_order, df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status)
    
    # 保存特征和目标
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
    save_features()