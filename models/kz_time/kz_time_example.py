import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

class SafeSimpleImputer:
    """安全的SimpleImputer实现，处理众数计算中的边界情况"""
    
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        
    def fit(self, X, y=None):
        if hasattr(X, 'iloc'):  # 如果是DataFrame
            X = X.values
        
        n_features = X.shape[1]
        self.statistics_ = np.zeros(n_features)
        
        for i in range(n_features):
            col_data = X[:, i]
            # 移除NaN值
            valid_data = col_data[~np.isnan(col_data)]
            
            if len(valid_data) == 0:
                # 如果所有值都是NaN，使用fill_value或默认值
                if self.fill_value is not None:
                    self.statistics_[i] = self.fill_value
                else:
                    self.statistics_[i] = 0
                continue
                
            if self.strategy == 'mean':
                self.statistics_[i] = np.mean(valid_data)
            elif self.strategy == 'median':
                self.statistics_[i] = np.median(valid_data)
            elif self.strategy == 'most_frequent':
                # 安全的众数计算
                values, counts = np.unique(valid_data, return_counts=True)
                self.statistics_[i] = values[np.argmax(counts)]
            elif self.strategy == 'constant':
                self.statistics_[i] = self.fill_value if self.fill_value is not None else 0
                
        return self
    
    def transform(self, X):
        if hasattr(X, 'iloc'):  # 如果是DataFrame
            X = X.values.copy()
        else:
            X = X.copy()
            
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if np.any(mask):
                X[mask, i] = self.statistics_[i]
                
        return X

def extract_tidal_features(water_data, hours=24):
    """提取指定时间窗口内的潮汐特征"""
    features = {}
    
    if water_data is None or len(water_data) < 2:
        # 返回默认特征值
        return {
            f'tide_{hours}h_mean': 0,
            f'tide_{hours}h_max': 0,
            f'tide_{hours}h_min': 0,
            f'tide_{hours}h_range': 0,
            f'tide_{hours}h_slope': 0,
            f'tide_{hours}h_r_squared': 0,
            f'tide_{hours}h_cycle_count': 0,
            f'tide_{hours}h_rise_rate': 0,
            f'tide_{hours}h_fall_rate': 0,
            f'tide_{hours}h_phase': 0
        }
    
    # 确保数据按时间排序
    water_data = water_data.sort_values('time')
    
    # 重采样到10分钟间隔
    resampled = water_data.set_index('time').resample('10T').mean().interpolate()
    levels = resampled['water_level'].values
    
    # 寻找波峰和波谷
    min_prominence = np.ptp(levels) * 0.2  # 动态设置显著度阈值
    peaks, _ = find_peaks(levels, prominence=min_prominence)
    valleys, _ = find_peaks(-levels, prominence=min_prominence)
    
    # 合并关键点并排序
    key_points = sorted(np.concatenate([peaks, valleys]))
    
    # 特征初始化
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
        
        # 计算潮汐相位 (小时)
        phase = time_diff
        tide_phases.append(phase)
        
        # 涨潮特征
        if curr_level > prev_level:
            tide_range = curr_level - prev_level
            rise_rates.append(tide_range / time_diff if time_diff > 0 else 0)
        
        # 落潮特征
        else:
            tide_range = prev_level - curr_level
            fall_rates.append(tide_range / time_diff if time_diff > 0 else 0)
    
    # 计算统计特征
    features[f'tide_{hours}h_mean'] = np.mean(levels) if len(levels) > 0 else 0
    features[f'tide_{hours}h_max'] = np.max(levels) if len(levels) > 0 else 0
    features[f'tide_{hours}h_min'] = np.min(levels) if len(levels) > 0 else 0
    features[f'tide_{hours}h_range'] = features[f'tide_{hours}h_max'] - features[f'tide_{hours}h_min']
    
    # 计算趋势特征
    if len(levels) > 1:
        time_idx = np.arange(len(levels))
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, levels)
        features[f'tide_{hours}h_slope'] = slope
        features[f'tide_{hours}h_r_squared'] = r_value**2
    else:
        features[f'tide_{hours}h_slope'] = 0
        features[f'tide_{hours}h_r_squared'] = 0
    
    # 计算周期特征
    features[f'tide_{hours}h_cycle_count'] = len(rise_rates) + len(fall_rates)
    features[f'tide_{hours}h_rise_rate'] = np.mean(rise_rates) if rise_rates else 0
    features[f'tide_{hours}h_fall_rate'] = np.mean(fall_rates) if fall_rates else 0
    features[f'tide_{hours}h_phase'] = np.mean(tide_phases) if tide_phases else 0
    
    return features

def identify_tide_type(tide_24h_features, tide_12h_features):
    """识别潮汐类型 (1=半日潮，2=全日潮，3=混合潮)"""
    cycle_count_24h = tide_24h_features.get('tide_24h_cycle_count', 0)
    cycle_count_12h = tide_12h_features.get('tide_12h_cycle_count', 0)
    
    if cycle_count_24h >= 3 or cycle_count_12h >= 2:
        return 1  # 半日潮
    elif cycle_count_24h == 1 or cycle_count_12h == 1:
        return 2  # 全日潮
    else:
        return 3  # 混合潮

def load_regression_model(model_path='outputs/kz_time/pkls/0822_kz_time_regression_model.pkl'):
    """加载预训练的回归模型"""
    try:
        model = joblib.load(model_path)
        print("模型加载成功")
        return model
    except FileNotFoundError:
        print(f"模型文件不存在: {model_path}")
        # 尝试加载默认模型
        default_path = 'outputs/kz_time/pkls/0822_kz_time_regression_model_default.pkl'
        if os.path.exists(default_path):
            return joblib.load(default_path)
        else:
            raise FileNotFoundError("找不到可用的模型文件")
def load_all_models(model_types=['lr', 'mlp', 'rf', 'svm', 'xgb']):
    """加载所有预训练的模型"""
    models = {}
    
    for model_type in model_types:
        model_path = f'outputs/kz_time/pkls/0822_kz_time_{model_type}_regression_model.pkl'
        
        try:
            model = joblib.load(model_path)
            models[model_type.upper()] = model
            print(f"{model_type.upper()}模型加载成功")
        except FileNotFoundError:
            # 尝试加载默认模型
            default_path = f'outputs/kz_time/pkls/0822_kz_time_{model_type}_regression_model_default.pkl'
            if os.path.exists(default_path):
                model = joblib.load(default_path)
                models[model_type.upper()] = model
                print(f"{model_type.upper()}默认模型加载成功")
            else:
                print(f"警告: {model_type.upper()}模型文件不存在")
    
    if not models:
        raise FileNotFoundError("找不到任何可用的模型文件")
    
    return models
def preprocess_data(X):
    """数据预处理：处理缺失值和异常值"""
    X_processed = X.copy()
    
    # 处理数值特征的缺失值
    numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if X_processed[col].isnull().any():
            # 使用中位数填充数值特征的缺失值
            median_val = X_processed[col].median()
            if pd.isna(median_val):  # 如果中位数也是NaN，使用0
                median_val = 0
            X_processed[col].fillna(median_val, inplace=True)
    
    # 处理分类特征的缺失值
    categorical_columns = ['tide_type']
    for col in categorical_columns:
        if col in X_processed.columns and X_processed[col].isnull().any():
            # 使用众数填充分类特征
            mode_val = X_processed[col].mode()
            if len(mode_val) > 0:
                X_processed[col].fillna(mode_val[0], inplace=True)
            else:
                X_processed[col].fillna(0, inplace=True)
    
    return X_processed

def prepare_input_features(input_data):
    """
    准备预测所需的特征数据，与train.py中的特征工程保持一致
    :param input_data: 包含以下字段的字典:
        - 'SIGNTM': 调令时间 (datetime)
        - 'water_level': 闸下潮位数据 (DataFrame with 'time' and 'water_level')
        - 'flow': 实测流量数据 (DataFrame with '监测日期' and '流量')
        - 'rain_actual': 实测降雨数据 (DataFrame with '监测日期', '所属区域' and '雨量')
        - 'rain_forecast': 降雨预报数据 (DataFrame with '时间', '所属区域' and '雨量')
        - 'water_status': 水位工况数据 (DataFrame with '监测日期' and '水位')
        - 'prev_orders': 历史调令数据 (DataFrame with 'SIGNTM', '开闸时间', '开闸时长', '开闸孔数')
    :return: 特征DataFrame
    """
    # 从输入数据中提取各个部分
    base_time = input_data['SIGNTM']
    water_data = input_data.get('water_level', pd.DataFrame())
    flow_data = input_data.get('flow', pd.DataFrame())
    rain_actual = input_data.get('rain_actual', pd.DataFrame())
    rain_forecast = input_data.get('rain_forecast', pd.DataFrame())
    water_status = input_data.get('water_status', pd.DataFrame())
    prev_orders = input_data.get('prev_orders', pd.DataFrame())
    
    # 创建基础特征字典
    feat_dict = {}
    
    # 提取24小时和12小时潮汐特征
    tide_24h_features = extract_tidal_features(water_data, 24)
    tide_12h_features = extract_tidal_features(water_data, 12)
    
    # 更新特征字典
    feat_dict.update(tide_24h_features)
    feat_dict.update(tide_12h_features)
    
    # 识别潮汐类型
    feat_dict['tide_type'] = identify_tide_type(tide_24h_features, tide_12h_features)
    
    # 添加时间特征
    hour = base_time.hour
    day = base_time.weekday()
    month = base_time.month
    day_of_year = base_time.timetuple().tm_yday
    
    feat_dict['hour_of_day'] = hour
    feat_dict['day_of_week'] = day
    feat_dict['month'] = month
    feat_dict['is_weekend'] = 1 if day >= 5 else 0
    feat_dict['day_of_year'] = day_of_year
    
    # 添加小时的正弦/余弦变换
    hour_rad = 2 * np.pi * hour / 24
    feat_dict['hour_sin'] = np.sin(hour_rad)
    feat_dict['hour_cos'] = np.cos(hour_rad)
    
    # 添加高峰时段特征
    feat_dict['is_rush_hour'] = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    
    # 添加缺失值指示器特征
    feat_dict['water_missing'] = 1 if water_data.empty else 0
    feat_dict['flow_missing'] = 1 if flow_data.empty else 0
    feat_dict['rain_missing'] = 1 if (rain_actual.empty and rain_forecast.empty) else 0
    feat_dict['water_status_missing'] = 1 if water_status.empty else 0
    
    # 水位统计特征（最近24小时）
    if not water_data.empty:
        cutoff_time = base_time - timedelta(hours=24)
        water_data_24h = water_data[water_data['time'] >= cutoff_time]
        
        if len(water_data_24h) > 1:
            levels = water_data_24h['water_level'].values
            time_idx = np.arange(len(levels))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, levels)
            feat_dict['water_slope'] = slope
            feat_dict['water_r_squared'] = r_value**2
        else:
            feat_dict['water_slope'] = 0
            feat_dict['water_r_squared'] = 0
        
        # 基本水位统计
        feat_dict['water_mean'] = water_data_24h['water_level'].mean() if len(water_data_24h) > 0 else 0
        feat_dict['water_max'] = water_data_24h['water_level'].max() if len(water_data_24h) > 0 else 0
        feat_dict['water_min'] = water_data_24h['water_level'].min() if len(water_data_24h) > 0 else 0
        feat_dict['water_range'] = feat_dict['water_max'] - feat_dict['water_min'] if len(water_data_24h) > 0 else 0
    else:
        feat_dict['water_slope'] = 0
        feat_dict['water_r_squared'] = 0
        feat_dict['water_mean'] = 0
        feat_dict['water_max'] = 0
        feat_dict['water_min'] = 0
        feat_dict['water_range'] = 0
    
    # 流量统计特征
    if not flow_data.empty:
        cutoff_time = base_time - timedelta(hours=24)
        flow_data_24h = flow_data[flow_data['监测日期'] >= cutoff_time]
        
        if len(flow_data_24h) > 2:
            flows = flow_data_24h['流量'].values
            feat_dict['flow_var'] = np.var(flows)
            feat_dict['flow_skew'] = stats.skew(flows)
        else:
            feat_dict['flow_var'] = 0
            feat_dict['flow_skew'] = 0
        
        # 基本流量统计
        feat_dict['flow_mean'] = flow_data_24h['流量'].mean() if len(flow_data_24h) > 0 else 0
        feat_dict['flow_max'] = flow_data_24h['流量'].max() if len(flow_data_24h) > 0 else 0
        feat_dict['flow_min'] = flow_data_24h['流量'].min() if len(flow_data_24h) > 0 else 0
        feat_dict['flow_range'] = feat_dict['flow_max'] - feat_dict['flow_min'] if len(flow_data_24h) > 0 else 0
    else:
        feat_dict['flow_var'] = 0
        feat_dict['flow_skew'] = 0
        feat_dict['flow_mean'] = 0
        feat_dict['flow_max'] = 0
        feat_dict['flow_min'] = 0
        feat_dict['flow_range'] = 0
    
    # 降雨统计特征
    rain_actual_total = 0
    if not rain_actual.empty:
        cutoff_time = base_time - timedelta(hours=24)
        rain_actual_24h = rain_actual[rain_actual['监测日期'] >= cutoff_time]
        rain_actual_total = rain_actual_24h['雨量'].sum()
        feat_dict['rain_actual_avg'] = rain_actual_24h['雨量'].mean() if len(rain_actual_24h) > 0 else 0
    else:
        feat_dict['rain_actual_avg'] = 0
    
    rain_forecast_total = 0
    if not rain_forecast.empty:
        cutoff_time = base_time + timedelta(hours=24)
        rain_forecast_24h = rain_forecast[rain_forecast['时间'] <= cutoff_time]
        rain_forecast_total = rain_forecast_24h['雨量'].sum()
        feat_dict['rain_forecast_avg'] = rain_forecast_24h['雨量'].mean() if len(rain_forecast_24h) > 0 else 0
    else:
        feat_dict['rain_forecast_avg'] = 0
    
    feat_dict['rain_actual_total'] = rain_actual_total
    feat_dict['rain_forecast_total'] = rain_forecast_total
    
    # 降雨变化率特征
    if not rain_actual.empty:
        # 获取最近24小时的数据
        cutoff_time = base_time - timedelta(hours=24)
        rain_actual_24h = rain_actual[rain_actual['监测日期'] >= cutoff_time]
        
        if len(rain_actual_24h) > 1:
            rain_actual_sorted = rain_actual_24h.sort_values('监测日期')
            rain_diff = rain_actual_sorted['雨量'].diff().dropna()
            feat_dict['rain_change_rate'] = rain_diff.mean() if len(rain_diff) > 0 else 0
        else:
            feat_dict['rain_change_rate'] = 0
    else:
        feat_dict['rain_change_rate'] = 0
    
    # 水位-降雨比率和流量-降雨比率
    eps = 1e-5  # 避免除零错误
    rain_total = feat_dict.get('rain_actual_total', 0)
    
    feat_dict['water_rain_ratio'] = (
        feat_dict.get('water_mean', 0) / (rain_total + eps) if rain_total > 0 else 0
    )
    
    feat_dict['flow_rain_ratio'] = (
        feat_dict.get('flow_mean', 0) / (rain_total + eps) if rain_total > 0 else 0
    )
    
    # 水位工况统计特征
    if not water_status.empty:
        cutoff_time = base_time - timedelta(hours=24)
        water_status_24h = water_status[water_status['监测日期'] >= cutoff_time]
        
        if len(water_status_24h) > 1:
            status_levels = water_status_24h['水位'].values
            time_idx = np.arange(len(status_levels))
            slope, _, _, _, _ = stats.linregress(time_idx, status_levels)
            feat_dict['water_status_slope'] = slope
        else:
            feat_dict['water_status_slope'] = 0
        
        # 基本水位工况统计
        feat_dict['water_status_mean'] = water_status_24h['水位'].mean() if len(water_status_24h) > 0 else 0
        feat_dict['water_status_max'] = water_status_24h['水位'].max() if len(water_status_24h) > 0 else 0
        feat_dict['water_status_min'] = water_status_24h['水位'].min() if len(water_status_24h) > 0 else 0
        feat_dict['water_status_range'] = feat_dict['water_status_max'] - feat_dict['water_status_min'] if len(water_status_24h) > 0 else 0
    else:
        feat_dict['water_status_slope'] = 0
        feat_dict['water_status_mean'] = 0
        feat_dict['water_status_max'] = 0
        feat_dict['water_status_min'] = 0
        feat_dict['water_status_range'] = 0
    
    # 添加历史操作特征（周统计）
    if not prev_orders.empty:
        cutoff_time = base_time - timedelta(days=7)
        prev_orders_week = prev_orders[prev_orders['SIGNTM'] >= cutoff_time]
        
        # 最近一次操作的特征
        if len(prev_orders_week) > 0:
            latest_order = prev_orders_week.iloc[-1]
            feat_dict['prev_gate_count'] = latest_order['开闸孔数']
            feat_dict['prev_duration'] = latest_order['开闸时长']
            feat_dict['prev_op_hour'] = latest_order['开闸时间'].hour if hasattr(latest_order['开闸时间'], 'hour') else 0
        else:
            feat_dict['prev_gate_count'] = 0
            feat_dict['prev_duration'] = 0
            feat_dict['prev_op_hour'] = 0
        
        # 周统计特征
        feat_dict['ops_week_count'] = len(prev_orders_week)
        feat_dict['ops_week_avg_gates'] = prev_orders_week['开闸孔数'].mean() if len(prev_orders_week) > 0 else 0
        feat_dict['ops_week_total_duration'] = prev_orders_week['开闸时长'].sum() if len(prev_orders_week) > 0 else 0
    else:
        feat_dict['prev_gate_count'] = 0
        feat_dict['prev_duration'] = 0
        feat_dict['prev_op_hour'] = 0
        feat_dict['ops_week_count'] = 0
        feat_dict['ops_week_avg_gates'] = 0
        feat_dict['ops_week_total_duration'] = 0
    
    # 未来潮汐特征（使用预报数据）
    future_water_data = input_data.get('future_water_level', pd.DataFrame())
    if not future_water_data.empty:
        # 使用未来潮汐预报数据提取特征，窗口大小为12小时
        future_tide_features = extract_tidal_features(future_water_data, 12)
        for key, value in future_tide_features.items():
            new_key = key.replace('tide_12h_', 'future_tide_')
            feat_dict[new_key] = value
    else:
        # 如果没有提供未来潮汐数据，则使用备用方案：用最近12小时数据模拟
        if not water_data.empty and len(water_data) > 12:
            future_tide_features = extract_tidal_features(water_data.tail(12), 12)
            for key, value in future_tide_features.items():
                new_key = key.replace('tide_12h_', 'future_tide_')
                feat_dict[new_key] = value
        else:
            # 设置默认未来潮汐特征
            future_features = {
                'future_tide_mean': 0,
                'future_tide_max': 0,
                'future_tide_min': 0,
                'future_tide_range': 0,
                'future_tide_slope': 0,
                'future_tide_r_squared': 0,
                'future_tide_cycle_count': 0,
                'future_tide_rise_rate': 0,
                'future_tide_fall_rate': 0,
                'future_tide_phase': 0
            }
            feat_dict.update(future_features)
    
    # 创建特征DataFrame
    features_df = pd.DataFrame([feat_dict])
    
    # 确保包含所有训练时使用的特征
    required_features = [
        # 时间特征
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos',
        
        # 历史操作特征
        'prev_gate_count', 'prev_duration', 'prev_op_hour',
        'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration',
        
        # 潮汐特征
        'tide_24h_phase', 'tide_12h_phase',
        'tide_24h_mean', 'tide_24h_max', 'tide_24h_min', 'tide_24h_range',
        'tide_24h_slope', 'tide_24h_r_squared', 'tide_24h_cycle_count',
        'tide_24h_rise_rate', 'tide_24h_fall_rate',
        'tide_12h_mean', 'tide_12h_max', 'tide_12h_min', 'tide_12h_range',
        'tide_12h_slope', 'tide_12h_r_squared', 'tide_12h_cycle_count',
        'tide_12h_rise_rate', 'tide_12h_fall_rate',
        
        # 未来潮汐特征
        'future_tide_mean', 'future_tide_max', 'future_tide_min', 'future_tide_range',
        'future_tide_slope', 'future_tide_r_squared', 'future_tide_cycle_count',
        'future_tide_rise_rate', 'future_tide_fall_rate', 'future_tide_phase',
        
        # 流量特征
        'flow_mean', 'flow_max', 'flow_min', 'flow_range', 'flow_var', 'flow_skew',
        
        # 降雨特征
        'rain_actual_total', 'rain_forecast_total', 
        'rain_actual_avg', 'rain_forecast_avg',
        'rain_change_rate', 'water_rain_ratio', 'flow_rain_ratio',
        
        # 水位工况特征
        'water_status_mean', 'water_status_max', 'water_status_min', 
        'water_status_range', 'water_status_slope',
        
        # 其他特征
        'is_rush_hour', 'day_of_year',
        
        # 分类特征
        'tide_type',
        
        # 缺失值指示器
        'water_missing', 'flow_missing', 
        'rain_missing', 'water_status_missing'
    ]
    
    # 添加缺失的特征（用0填充）
    for feature in required_features:
        if feature not in features_df.columns:
            features_df[feature] = 0
            
    # 确保特征顺序一致
    features_df = features_df[required_features]
    
    # 数据预处理（与train.py保持一致）
    features_df = preprocess_data(features_df)
    
    return features_df

# 当前错误的转换方式
def convert_predictions(y_pred):
    """安全的时间预测值转换"""
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        # 如果是sin/cos编码
        y_pred_hours = np.arctan2(y_pred[:, 0], y_pred[:, 1]) * 24 / (2 * np.pi)
        y_pred_hours = np.mod(y_pred_hours, 24)
    else:
        # 如果是直接预测的小时数
        y_pred_hours = y_pred.flatten() % 24
    
    # 处理边界情况
    y_pred_hours = np.clip(y_pred_hours, 0, 24)
    
    if y_pred_hours.size == 1:
        return float(y_pred_hours)
    return y_pred_hours
    
def predict_with_all_models(models, input_features, sign_time):
    """
    使用所有模型进行预测
    :param models: 加载的模型字典
    :param input_features: 准备好的特征DataFrame
    :param sign_time: 调令时间 (datetime)
    :return: 各模型的预测结果字典
    """
    predictions = {}
    
    for model_name, model in models.items():
        try:
            # 使用模型预测
            prediction = model.predict(input_features)
            
            # 将预测值转换为小时
            hour_pred = convert_predictions(prediction)
            
            # 计算开闸时间（同一天）
            open_time = sign_time.replace(
                hour=int(hour_pred),
                minute=int((hour_pred % 1) * 60),
                second=0,
                microsecond=0
            )
            
            # 如果开闸时间早于调令时间，则推迟到第二天相同时间
            if open_time < sign_time:
                open_time += timedelta(days=1)
            
            predictions[model_name] = {
                'predicted_time': open_time,
                'predicted_hour': hour_pred,
                'time_difference': (open_time - sign_time).total_seconds() / 3600
            }
            
        except Exception as e:
            print(f"{model_name}模型预测失败: {str(e)}")
            predictions[model_name] = {
                'predicted_time': None,
                'predicted_hour': None,
                'time_difference': None,
                'error': str(e)
            }
    
    return predictions

def predict_opening_time_all_models(input_data, model_types=['lr', 'mlp', 'rf', 'svm', 'xgb']):
    """
    主预测函数 - 使用所有模型进行预测
    :param input_data: 包含所有必要输入数据的字典
    :param model_types: 要使用的模型类型列表
    :return: 各模型的预测结果字典
    """
    # 1. 加载所有模型
    models = load_all_models(model_types)
    
    # 2. 准备特征
    features = prepare_input_features(input_data)
    
    # 3. 进行预测
    sign_time = input_data['SIGNTM']
    predictions = predict_with_all_models(models, features, sign_time)
    
    return predictions

def print_prediction_results(predictions, sign_time):
    """打印预测结果"""
    print(f"\n{'='*60}")
    print("开闸时间预测结果汇总")
    print(f"{'='*60}")
    print(f"调令时间: {sign_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'模型':<10} {'预测开闸时间':<20} {'预测小时数':<12} {'时间差(小时)':<15}")
    print(f"{'-'*60}")
    
    for model_name, result in predictions.items():
        if result['predicted_time'] is not None:
            print(f"{model_name:<10} {result['predicted_time'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{result['predicted_hour']:.2f}{'小时':<10} {result['time_difference']:.2f}{'小时':<10}")
        else:
            print(f"{model_name:<10} {'预测失败':<20} {'N/A':<12} {'N/A':<15}")

# 示例使用
if __name__ == '__main__':
    # 示例输入数据
    sign_time = datetime.now()
    example_input = {
        'SIGNTM': sign_time,
        'water_level': pd.DataFrame({
            'time': [sign_time - timedelta(hours=i) for i in range(24, 0, -1)],
            'water_level': np.random.uniform(1.5, 3.5, 24)
        }),
        'future_water_level': pd.DataFrame({
            'time': [sign_time + timedelta(hours=i) for i in range(1, 13)],
            'water_level': np.random.uniform(1.5, 3.5, 12)
        }),
        'flow': pd.DataFrame({
            '监测日期': [sign_time - timedelta(hours=i) for i in range(24, 0, -1)],
            '流量': np.random.uniform(100, 500, 24)
        }),
        'rain_actual': pd.DataFrame({
            '监测日期': [sign_time - timedelta(hours=i) for i in range(24, 0, -1)],
            '所属区域': ['绍兴平原']*12 + ['新昌']*12,
            '雨量': np.random.uniform(0, 10, 24)
        }),
        'rain_forecast': pd.DataFrame({
            '时间': [sign_time + timedelta(hours=i) for i in range(1, 25)],
            '所属区域': ['绍兴平原']*12 + ['新昌']*12,
            '雨量': np.random.uniform(0, 8, 24)
        }),
        'water_status': pd.DataFrame({
            '监测日期': [sign_time - timedelta(hours=i) for i in range(24, 0, -1)],
            '水位': np.random.uniform(2.0, 3.5, 24)
        }),
        'prev_orders': pd.DataFrame({
            'SIGNTM': [sign_time - timedelta(hours=3), sign_time - timedelta(days=1), sign_time - timedelta(days=3)],
            '开闸时间': [sign_time - timedelta(hours=2.5), sign_time - timedelta(days=1), sign_time - timedelta(days=3)],
            '开闸时长': [3.5, 4.2, 2.8],
            '开闸孔数': [2, 3, 2]
        })
    }
    
    # 进行多模型预测
    predictions = predict_opening_time_all_models(example_input)
    
    # 打印结果
    print_prediction_results(predictions, sign_time)
    
    # 计算平均预测时间（排除失败的预测）
    valid_predictions = [result['predicted_time'] for result in predictions.values() 
                        if result['predicted_time'] is not None]
    
    if valid_predictions:
        # 计算时间戳的平均值
        timestamps = [dt.timestamp() for dt in valid_predictions]
        avg_timestamp = sum(timestamps) / len(timestamps)
        avg_time = datetime.fromtimestamp(avg_timestamp)
        
        print(f"\n平均预测开闸时间: {avg_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"参与平均的模型数量: {len(valid_predictions)}")