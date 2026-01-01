import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

# 添加与train.py一致的SafeSimpleImputer类
class SafeSimpleImputer(BaseEstimator, TransformerMixin):
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

def preprocess_data(X):
    """数据预处理：处理缺失值和异常值，与train.py保持一致"""
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
    categorical_columns = ['tide_type']  # 明确指定分类特征
    for col in categorical_columns:
        if col in X_processed.columns and X_processed[col].isnull().any():
            # 使用众数填充分类特征
            mode_val = X_processed[col].mode()
            if len(mode_val) > 0:
                X_processed[col].fillna(mode_val[0], inplace=True)
            else:
                X_processed[col].fillna(0, inplace=True)
    
    return X_processed

# 分类标签到真实孔数的映射（根据train.py中的处理逻辑）
LABEL_TO_GATES = {0: 4, 1: 8, 2: 10, 3: 12, 4: 20, 5:28 }
def safe_linregress(x, y):
    """安全的线性回归计算，处理x值相同的情况"""
    if len(x) < 2 or np.var(x) == 0:
        return 0, 0, 0, 0, 0  # 斜率, 截距, r值, p值, 标准误
    
    try:
        return stats.linregress(x, y)
    except:
        return 0, 0, 0, 0, 0

def extract_hourly_tidal_features(water_data, period_hours=24):
    """提取小时级别潮汐特征，与train.py中的实现保持一致"""
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
        'phase': 0,
        'tide_type': 0, # 潮汐类型，0表示无潮汐，1表示半日潮，2表示全日潮
    }
    
    if water_data is None or len(water_data) < 2:
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
    cycle_count = len(turning_points) // 2
    features['cycle_count'] = cycle_count

    # 潮汐类型识别 (1=半日潮，2=全日潮，3=混合潮)
    if cycle_count >= 3:
        features['tide_type'] = 1
    elif cycle_count == 1:
        features['tide_type'] = 2
    else:
        features['tide_type'] = 3

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

def load_all_models():
    """加载所有预训练的模型"""
    models = {}
    model_paths = {
        'lr': 'outputs/kz_num/pkls/0822_kz_num_lr_classification_model.pkl',
        'mlp': 'outputs/kz_num/pkls/0822_kz_num_mlp_classification_model.pkl',
        'rf': 'outputs/kz_num/pkls/0822_kz_num_rf_classification_model.pkl',
        'svm': 'outputs/kz_num/pkls/0822_kz_num_svm_classification_model.pkl',
        'xgb': 'outputs/kz_num/pkls/0822_kz_num_xgb_classification_model.pkl'
    }
    
    for model_name, model_path in model_paths.items():
        try:
            model = joblib.load(model_path)
            models[model_name] = model
            print(f"{model_name.upper()} 模型加载成功: {model_path}")
        except Exception as e:
            print(f"{model_name.upper()} 模型加载失败: {str(e)}")
            models[model_name] = None
    
    return models

def predict_gate_count(model, input_features):
    """预测开闸孔数"""
    if model is None:
        return 4  # 默认返回4孔
    
    prediction = model.predict(input_features)
    # 使用映射关系将分类标签转换为真实孔数
    return LABEL_TO_GATES.get(prediction[0], 4)  # 默认返回4孔

def prepare_input_features(input_data):
    """
    准备预测所需的特征数据，与train.py中的特征工程保持一致
    :param input_data: 包含所有必要输入数据的字典
    :return: 特征DataFrame
    """
    base_time = input_data['SIGNTM']
    water_data = input_data.get('water_level', pd.DataFrame())
    future_water_data = input_data.get('future_water_level', pd.DataFrame())
    flow_data = input_data.get('flow', pd.DataFrame())
    rain_actual = input_data.get('rain_actual', pd.DataFrame())
    rain_forecast = input_data.get('rain_forecast', pd.DataFrame())
    water_status = input_data.get('water_status', pd.DataFrame())
    prev_orders = input_data.get('prev_orders', pd.DataFrame())
    
    # 创建基础特征字典
    feat_dict = {}
    
    # 1. 时间特征
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
    
    # 2. 潮汐特征 - 使用与train.py相同的特征提取方法
    if water_data is not None and len(water_data) > 0:
        # 过去24小时潮汐特征
        start_time_24h = base_time - timedelta(hours=24)
        water_data_24h = water_data[water_data['time'] >= start_time_24h]
        
        if len(water_data_24h) > 0:
            tidal_features_24h = extract_hourly_tidal_features(water_data_24h, 24)
            for key, value in tidal_features_24h.items():
                feat_dict[f'tide_24h_{key}'] = value
        else:
            # 使用默认值
            default_features = extract_hourly_tidal_features(pd.DataFrame(), 24)
            for key, value in default_features.items():
                feat_dict[f'tide_24h_{key}'] = value
        
        # 过去12小时潮汐特征
        start_time_12h = base_time - timedelta(hours=12)
        water_data_12h = water_data[water_data['time'] >= start_time_12h]
        
        if len(water_data_12h) > 0:
            tidal_features_12h = extract_hourly_tidal_features(water_data_12h, 12)
            for key, value in tidal_features_12h.items():
                feat_dict[f'tide_12h_{key}'] = value
        else:
            # 使用默认值
            default_features = extract_hourly_tidal_features(pd.DataFrame(), 12)
            for key, value in default_features.items():
                feat_dict[f'tide_12h_{key}'] = value
    else:
        # 使用默认值
        default_features_24h = extract_hourly_tidal_features(pd.DataFrame(), 24)
        for key, value in default_features_24h.items():
            feat_dict[f'tide_24h_{key}'] = value
        
        default_features_12h = extract_hourly_tidal_features(pd.DataFrame(), 12)
        for key, value in default_features_12h.items():
            feat_dict[f'tide_12h_{key}'] = value
    
    # 3. 未来潮汐特征
    if future_water_data is not None and len(future_water_data) > 0:
        future_tidal_features = extract_hourly_tidal_features(future_water_data, 12)
        for key, value in future_tidal_features.items():
            feat_dict[f'future_tide_{key}'] = value
    else:
        # 使用默认值
        default_features = extract_hourly_tidal_features(pd.DataFrame(), 12)
        for key, value in default_features.items():
            feat_dict[f'future_tide_{key}'] = value
    
    # 4. 流量特征
    if flow_data is not None and len(flow_data) > 0:
        start_time_24h = base_time - timedelta(hours=24)
        flow_data_24h = flow_data[flow_data['监测日期'] >= start_time_24h]
        
        if len(flow_data_24h) > 0:
            feat_dict['flow_mean'] = flow_data_24h['流量'].mean()
            feat_dict['flow_max'] = flow_data_24h['流量'].max()
            feat_dict['flow_min'] = flow_data_24h['流量'].min()
            feat_dict['flow_range'] = feat_dict['flow_max'] - feat_dict['flow_min']
            feat_dict['flow_var'] = flow_data_24h['流量'].var() if len(flow_data_24h) > 1 else 0
            feat_dict['flow_skew'] = flow_data_24h['流量'].skew() if len(flow_data_24h) > 2 else 0
        else:
            for key in ['flow_mean', 'flow_max', 'flow_min', 'flow_range', 'flow_var', 'flow_skew']:
                feat_dict[key] = 0
    else:
        for key in ['flow_mean', 'flow_max', 'flow_min', 'flow_range', 'flow_var', 'flow_skew']:
            feat_dict[key] = 0
    
    # 5. 降雨特征
    regions = ['绍兴平原', '嵊州', '虞南山区', '新昌', '虞北平原']
    rain_actual_total = 0
    rain_forecast_total = 0
    
    # 实际降雨
    if rain_actual is not None and len(rain_actual) > 0:
        start_time_24h = base_time - timedelta(hours=24)
        rain_actual_24h = rain_actual[rain_actual['监测日期'] >= start_time_24h]
        
        for region in regions:
            region_rain = rain_actual_24h[rain_actual_24h['所属区域'] == region]
            if len(region_rain) > 0:
                region_sum = region_rain['雨量'].sum()
                feat_dict[f'rain_actual_{region}_sum'] = region_sum
                rain_actual_total += region_sum
            else:
                feat_dict[f'rain_actual_{region}_sum'] = 0
    else:
        for region in regions:
            feat_dict[f'rain_actual_{region}_sum'] = 0
    
    # 预测降雨
    if rain_forecast is not None and len(rain_forecast) > 0:
        end_time_24h = base_time + timedelta(hours=24)
        rain_forecast_24h = rain_forecast[rain_forecast['预计开始时间'] <= end_time_24h]
        
        for region in regions:
            region_forecast = rain_forecast_24h[rain_forecast_24h['大流域'] == region]
            if len(region_forecast) > 0:
                region_sum = region_forecast['降雨量'].sum()
                feat_dict[f'rain_forecast_{region}_sum'] = region_sum
                rain_forecast_total += region_sum
            else:
                feat_dict[f'rain_forecast_{region}_sum'] = 0
    else:
        for region in regions:
            feat_dict[f'rain_forecast_{region}_sum'] = 0
    
    feat_dict['rain_actual_total'] = rain_actual_total
    feat_dict['rain_forecast_total'] = rain_forecast_total
    feat_dict['rain_actual_avg'] = rain_actual_total / len(regions) if len(regions) > 0 else 0
    feat_dict['rain_forecast_avg'] = rain_forecast_total / len(regions) if len(regions) > 0 else 0
    
    # 降雨变化率
    if rain_actual_total > 0:
        feat_dict['rain_change_rate'] = (rain_forecast_total - rain_actual_total) / rain_actual_total
    else:
        feat_dict['rain_change_rate'] = 0
    
    # 水位-降雨比率
    if rain_actual_total > 0:
        feat_dict['water_rain_ratio'] = feat_dict.get('tide_24h_mean', 0) / rain_actual_total
    else:
        feat_dict['water_rain_ratio'] = 0
    
    # 流量-降雨比率
    if rain_actual_total > 0:
        feat_dict['flow_rain_ratio'] = feat_dict.get('flow_mean', 0) / rain_actual_total
    else:
        feat_dict['flow_rain_ratio'] = 0
    
    # 6. 水位工况特征
    if water_status is not None and len(water_status) > 0:
        start_time_24h = base_time - timedelta(hours=24)
        water_status_24h = water_status[water_status['监测日期'] >= start_time_24h]
        
        if len(water_status_24h) > 0:
            feat_dict['water_status_mean'] = water_status_24h['水位'].mean()
            feat_dict['water_status_max'] = water_status_24h['水位'].max()
            feat_dict['water_status_min'] = water_status_24h['水位'].min()
            feat_dict['water_status_range'] = feat_dict['water_status_max'] - feat_dict['water_status_min']
            
            if len(water_status_24h) > 1:
                times = (water_status_24h['监测日期'] - water_status_24h['监测日期'].min()).dt.total_seconds().values
                slope, _, _, _, _ = safe_linregress(times, water_status_24h['水位'])
                feat_dict['water_status_slope'] = slope
            else:
                feat_dict['water_status_slope'] = 0
        else:
            for key in ['water_status_mean', 'water_status_max', 'water_status_min', 
                       'water_status_range', 'water_status_slope']:
                feat_dict[key] = 0
    else:
        for key in ['water_status_mean', 'water_status_max', 'water_status_min', 
                   'water_status_range', 'water_status_slope']:
            feat_dict[key] = 0
    
    # 7. 历史操作特征
    if prev_orders is not None and len(prev_orders) > 0:
        # 最近一次操作
        prev_orders_sorted = prev_orders.sort_values('SIGNTM', ascending=False)
        latest = prev_orders_sorted.iloc[0] if len(prev_orders_sorted) > 0 else None
        
        if latest is not None:
            feat_dict['prev_gate_count'] = latest['开闸孔数']
            feat_dict['prev_duration'] = latest['开闸时长'] if not pd.isnull(latest['开闸时长']) else 0
            feat_dict['prev_op_hour'] = latest['开闸时间'].hour if not pd.isnull(latest['开闸时间']) else 0
        else:
            feat_dict['prev_gate_count'] = 0
            feat_dict['prev_duration'] = 0
            feat_dict['prev_op_hour'] = 0
        
        # 一周内的操作统计
        start_time_week = base_time - timedelta(days=7)
        last_week_orders = prev_orders[prev_orders['SIGNTM'] >= start_time_week]
        
        feat_dict['ops_week_count'] = len(last_week_orders)
        feat_dict['ops_week_avg_gates'] = last_week_orders['开闸孔数'].mean() if len(last_week_orders) > 0 else 0
        feat_dict['ops_week_total_duration'] = last_week_orders['开闸时长'].sum() if len(last_week_orders) > 0 else 0
    else:
        for key in ['prev_gate_count', 'prev_duration', 'prev_op_hour', 
                   'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration']:
            feat_dict[key] = 0
    
    # 8. 缺失值指示器特征
    feat_dict['water_missing'] = 1 if (water_data is None or len(water_data) == 0) else 0
    feat_dict['flow_missing'] = 1 if (flow_data is None or len(flow_data) == 0) else 0
    feat_dict['rain_missing'] = 1 if ((rain_actual is None or len(rain_actual) == 0) and 
                                     (rain_forecast is None or len(rain_forecast) == 0)) else 0
    feat_dict['water_status_missing'] = 1 if (water_status is None or len(water_status) == 0) else 0
    feat_dict['future_water_missing'] = 1 if (future_water_data is None or len(future_water_data) == 0) else 0
    
    # 9. 潮汐类型特征（使用提取的特征）
    feat_dict['tide_type'] = feat_dict.get('tide_24h_tide_type', 0)
    
    # 创建特征DataFrame
    features_df = pd.DataFrame([feat_dict])
    
    # 确保包含所有训练时使用的特征
    required_features = [
        # 时间特征
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_of_year',
        
        # 历史操作特征
        'prev_gate_count', 'prev_duration', 'prev_op_hour',
        'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration',
        
        # 潮汐特征
        'tide_24h_mean', 'tide_24h_max', 'tide_24h_min', 'tide_24h_range',
        'tide_24h_slope', 'tide_24h_r_squared', 'tide_24h_cycle_count',
        'tide_24h_rise_rate', 'tide_24h_fall_rate', 'tide_24h_phase',
        'tide_12h_mean', 'tide_12h_max', 'tide_12h_min', 'tide_12h_range',
        'tide_12h_slope', 'tide_12h_r_squared', 'tide_12h_cycle_count',
        'tide_12h_rise_rate', 'tide_12h_fall_rate', 'tide_12h_phase',
        
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
        'is_rush_hour', 'tide_type',
        
        # 缺失值指示器
        'water_missing', 'flow_missing', 'rain_missing', 'water_status_missing', 'future_water_missing'
    ]
    
    # 添加缺失的特征
    for feature in required_features:
        if feature not in features_df.columns:
            features_df[feature] = 0
    
    # 数据预处理（与train.py保持一致）
    features_df = preprocess_data(features_df)
    
    return features_df[required_features]

def predict_with_all_models(input_data):
    """
    使用所有模型进行预测
    :param input_data: 包含所有必要输入数据的字典
    :return: 所有模型的预测结果字典
    """
    # 1. 加载所有模型
    models = load_all_models()
    
    # 2. 准备特征
    features = prepare_input_features(input_data)
    
    # 3. 进行预测
    predictions = {}
    for model_name, model in models.items():
        if model is not None:
            predicted_count = predict_gate_count(model, features)
            predictions[model_name.upper()] = predicted_count
        else:
            predictions[model_name.upper()] = 4  # 默认值
    
    return predictions

# 真实数据示例
if __name__ == '__main__':
    real_input = {
        'SIGNTM': datetime(2023, 8, 26, 12, 30),
        'water_level': pd.DataFrame({
            'time': [
                datetime(2023, 8, 25, 8, 30), datetime(2023, 8, 25, 10, 30),
                datetime(2023, 8, 25, 12, 30), datetime(2023, 8, 25, 14, 30),
                datetime(2023, 8, 25, 16, 30), datetime(2023, 8, 25, 18, 30),
                datetime(2023, 8, 25, 20, 30), datetime(2023, 8, 25, 22, 30),
                datetime(2023, 8, 26, 0, 30), datetime(2023, 8, 26, 2, 30),
                datetime(2023, 8, 26, 4, 30), datetime(2023, 8, 26, 6, 30)
            ],
            'water_level': [3.2, 3.5, 3.8, 4.1, 4.3, 4.0, 3.7, 3.4, 3.1, 3.3, 3.6, 3.9]
        }),
        'future_water_level': pd.DataFrame({
            'time': [datetime(2023, 8, 26, 12, 30) + timedelta(hours=i) for i in range(1, 13)],
            'water_level': [4.2, 4.5, 4.8, 5.1, 5.3, 5.0, 4.7, 4.4, 4.1, 4.3, 4.6, 4.9]
        }),
        'flow': pd.DataFrame({
            '监测日期': [
                datetime(2023, 8, 25, 20, 0), datetime(2023, 8, 25, 22, 0),
                datetime(2023, 8, 26, 0, 0), datetime(2023, 8, 26, 2, 0),
                datetime(2023, 8, 26, 4, 0), datetime(2023, 8, 26, 6, 0)
            ],
            '流量': [320, 350, 380, 410, 390, 360]
        }),
        'rain_actual': pd.DataFrame({
            '监测日期': [
                datetime(2023, 8, 25, 20, 0), datetime(2023, 8, 25, 22, 0),
                datetime(2023, 8, 26, 0, 0), datetime(2023, 8, 26, 2, 0),
                datetime(2023, 8, 26, 4, 0), datetime(2023, 8, 26, 6, 0)
            ],
            '所属区域': ['绍兴平原'] * 6,
            '雨量': [5.2, 7.8, 10.1, 8.5, 6.2, 4.0]
        }),
        'rain_forecast': pd.DataFrame({
            '预计开始时间': [
                datetime(2023, 8, 26, 9, 0), datetime(2023, 8, 26, 10, 0),
                datetime(2023, 8, 26, 11, 0), datetime(2023, 8, 26, 12, 0),
                datetime(2023, 8, 26, 13, 0), datetime(2023, 8, 26, 14, 0)
            ],
            '大流域': ['绍兴平原'] * 6,
            '降雨量': [3.5, 4.2, 5.0, 4.5, 3.8, 3.0]
        }),
        'water_status': pd.DataFrame({
            '监测日期': [
                datetime(2023, 8, 25, 20, 0), datetime(2023, 8, 25, 22, 0),
                datetime(2023, 8, 26, 0, 0), datetime(2023, 8, 26, 2, 0),
                datetime(2023, 8, 26, 4, 0), datetime(2023, 8, 26, 6, 0)
            ],
            '水位': [2.8, 2.9, 3.0, 3.1, 3.2, 3.3]
        }),
        'prev_orders': pd.DataFrame({
            'SIGNTM': [datetime(2023, 8, 25, 6, 0), datetime(2023, 8, 24, 9, 0)],
            '开闸时间': [datetime(2023, 8, 25, 7, 30), datetime(2023, 8, 24, 10, 0)],
            '开闸时长': [4.5, 5.2],
            '开闸孔数': [8, 12]  # 真实孔数
        })
    }
    
    predictions = predict_with_all_models(real_input)
    
    print("\n" + "="*50)
    print("多模型预测结果")
    print("="*50)
    for model_name, count in predictions.items():
        print(f"{model_name}: {count} 孔")
    
    # 计算平均预测值
    valid_predictions = [count for count in predictions.values() if count != 4]
    if valid_predictions:
        avg_prediction = sum(valid_predictions) / len(valid_predictions)
        print(f"\n平均预测值: {avg_prediction:.1f} 孔")