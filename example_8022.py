import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 添加缺失的类定义
from sklearn.base import BaseEstimator, TransformerMixin

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

def safe_load_model(model_path):
    """安全加载模型，处理自定义类的问题"""
    try:
        return joblib.load(model_path)
    except AttributeError as e:
        if "SafeSimpleImputer" in str(e):
            # 注册自定义类后重新加载
            import sys
            sys.modules[__name__].SafeSimpleImputer = SafeSimpleImputer
            return joblib.load(model_path)
        else:
            raise e

class KZPredictor:
    def __init__(self):
        # 加载所有模型
        self.time_model = joblib.load('outputs/kz_time/pkls/0822_kz_time_regression_model.pkl')
        self.num_model = joblib.load('outputs/kz_num/pkls/0822_kz_num_classification_model.pkl')
        self.level_model = joblib.load('outputs/kz_level/pkls/0822_kz_level_regression_model.pkl')
        self.dura_model = joblib.load('outputs/kz_dura/pkls/0822_kz_dura_classification_model.pkl')
        self.dura_encoder = joblib.load('outputs/kz_dura/pkls/0822_kz_dura_label_encoder.pkl')
        self.comb_model = joblib.load('outputs/kz_comb_log/pkls/0822_kz_comb_log_regression_model.pkl')
        
        # 开闸孔数映射
        self.LABEL_TO_GATES = {0: 4, 1: 8, 2: 10, 3: 12, 4: 20, 5:28 }

    def extract_tidal_features(self, water_data, hours=24):
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
        # 使用'value'列而不是'water_level'列
        levels = resampled['value'].values
        
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

    def identify_tide_type(self, tide_24h_features, tide_12h_features):
        """识别潮汐类型 (1=半日潮，2=全日潮，3=混合潮)"""
        cycle_count_24h = tide_24h_features.get('tide_24h_cycle_count', 0)
        cycle_count_12h = tide_12h_features.get('tide_12h_cycle_count', 0)
        
        if cycle_count_24h >= 3 or cycle_count_12h >= 2:
            return 1  # 半日潮
        elif cycle_count_24h == 1 or cycle_count_12h == 1:
            return 2  # 全日潮
        else:
            return 3  # 混合潮

    def prepare_common_features(self, input_data, time_window=24):
        """统一特征处理方法"""
        # 从输入数据中提取各个部分
        base_time = input_data['sign_time']
        water_data = input_data.get('water_level', pd.DataFrame())
        future_water_data = input_data.get('future_water_level', pd.DataFrame())
        flow_data = input_data.get('flow', pd.DataFrame())
        rain_actual = input_data.get('rain_actual', pd.DataFrame())
        rain_forecast = input_data.get('rain_forecast', pd.DataFrame())
        water_status = input_data.get('water_status', pd.DataFrame())
        prev_orders = input_data.get('prev_orders', pd.DataFrame())
        
        # 创建基础特征字典
        feat_dict = {}
        
        # 提取潮汐特征
        tide_24h_features = self.extract_tidal_features(water_data, 24)
        tide_12h_features = self.extract_tidal_features(water_data, 12)
        
        # 更新特征字典
        feat_dict.update(tide_24h_features)
        feat_dict.update(tide_12h_features)
        
        # 识别潮汐类型
        feat_dict['tide_type'] = self.identify_tide_type(tide_24h_features, tide_12h_features)
        
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
        feat_dict['future_water_missing'] = 1 if future_water_data.empty else 0
        
        # 水位统计特征
        if not water_data.empty:
            cutoff_time = base_time - timedelta(hours=time_window)
            water_data_window = water_data[water_data['time'] >= cutoff_time]
            
            if len(water_data_window) > 1:
                # 使用'value'列而不是'water_level'列
                levels = water_data_window['value'].values
                time_idx = np.arange(len(levels))
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, levels)
                feat_dict['water_slope'] = slope
                feat_dict['water_r_squared'] = r_value**2
            else:
                feat_dict['water_slope'] = 0
                feat_dict['water_r_squared'] = 0
            
            # 基本水位统计
            feat_dict['water_mean'] = water_data_window['value'].mean() if len(water_data_window) > 0 else 0
            feat_dict['water_max'] = water_data_window['value'].max() if len(water_data_window) > 0 else 0
            feat_dict['water_min'] = water_data_window['value'].min() if len(water_data_window) > 0 else 0
            feat_dict['water_range'] = feat_dict['water_max'] - feat_dict['water_min'] if len(water_data_window) > 0 else 0
        else:
            feat_dict['water_slope'] = 0
            feat_dict['water_r_squared'] = 0
            feat_dict['water_mean'] = 0
            feat_dict['water_max'] = 0
            feat_dict['water_min'] = 0
            feat_dict['water_range'] = 0
        
        # 流量统计特征
        if not flow_data.empty:
            cutoff_time = base_time - timedelta(hours=time_window)
            flow_data_window = flow_data[flow_data['time'] >= cutoff_time]
            
            if len(flow_data_window) > 2:
                flows = flow_data_window['value'].values
                feat_dict['flow_var'] = np.var(flows)
                feat_dict['flow_skew'] = stats.skew(flows)
            else:
                feat_dict['flow_var'] = 0
                feat_dict['flow_skew'] = 0
            
            # 基本流量统计
            feat_dict['flow_mean'] = flow_data_window['value'].mean() if len(flow_data_window) > 0 else 0
            feat_dict['flow_max'] = flow_data_window['value'].max() if len(flow_data_window) > 0 else 0
            feat_dict['flow_min'] = flow_data_window['value'].min() if len(flow_data_window) > 0 else 0
            feat_dict['flow_range'] = feat_dict['flow_max'] - feat_dict['flow_min'] if len(flow_data_window) > 0 else 0
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
            cutoff_time = base_time - timedelta(hours=time_window)
            rain_actual_window = rain_actual[rain_actual['time'] >= cutoff_time]
            rain_actual_total = rain_actual_window['value'].sum()
            feat_dict['rain_actual_avg'] = rain_actual_window['value'].mean() if len(rain_actual_window) > 0 else 0
        else:
            feat_dict['rain_actual_avg'] = 0
        
        rain_forecast_total = 0
        if not rain_forecast.empty:
            cutoff_time = base_time + timedelta(hours=time_window)
            rain_forecast_window = rain_forecast[rain_forecast['time'] <= cutoff_time]
            rain_forecast_total = rain_forecast_window['value'].sum()
            feat_dict['rain_forecast_avg'] = rain_forecast_window['value'].mean() if len(rain_forecast_window) > 0 else 0
        else:
            feat_dict['rain_forecast_avg'] = 0
        
        feat_dict['rain_actual_total'] = rain_actual_total
        feat_dict['rain_forecast_total'] = rain_forecast_total
        
        # 降雨变化率特征
        if not rain_actual.empty:
            # 获取指定时间窗口的数据
            cutoff_time = base_time - timedelta(hours=time_window)
            rain_actual_window = rain_actual[rain_actual['time'] >= cutoff_time]
            
            if len(rain_actual_window) > 1:
                rain_actual_sorted = rain_actual_window.sort_values('time')
                rain_diff = rain_actual_sorted['value'].diff().dropna()
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
            cutoff_time = base_time - timedelta(hours=time_window)
            water_status_window = water_status[water_status['time'] >= cutoff_time]
            
            if len(water_status_window) > 1:
                status_levels = water_status_window['value'].values
                time_idx = np.arange(len(status_levels))
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, status_levels)
                feat_dict['water_status_slope'] = slope
            else:
                feat_dict['water_status_slope'] = 0
            
            # 基本水位工况统计
            feat_dict['water_status_mean'] = water_status_window['value'].mean() if len(water_status_window) > 0 else 0
            feat_dict['water_status_max'] = water_status_window['value'].max() if len(water_status_window) > 0 else 0
            feat_dict['water_status_min'] = water_status_window['value'].min() if len(water_status_window) > 0 else 0
            feat_dict['water_status_range'] = feat_dict['water_status_max'] - feat_dict['water_status_min'] if len(water_status_window) > 0 else 0
        else:
            feat_dict['water_status_slope'] = 0
            feat_dict['water_status_mean'] = 0
            feat_dict['water_status_max'] = 0
            feat_dict['water_status_min'] = 0
            feat_dict['water_status_range'] = 0
        
        # 添加历史操作特征（周统计）
        if not prev_orders.empty:
            cutoff_time = base_time - timedelta(days=7)
            prev_orders_week = prev_orders[prev_orders['sign_time'] >= cutoff_time]
            
            # 最近一次操作的特征
            if len(prev_orders_week) > 0:
                latest_order = prev_orders_week.iloc[-1]
                feat_dict['prev_gate_count'] = latest_order['gate_count']
                feat_dict['prev_duration'] = latest_order['duration']
                feat_dict['prev_op_hour'] = latest_order['opening_time'].hour if hasattr(latest_order['opening_time'], 'hour') else 0
            else:
                feat_dict['prev_gate_count'] = 0
                feat_dict['prev_duration'] = 0
                feat_dict['prev_op_hour'] = 0
            
            # 周统计特征
            feat_dict['ops_week_count'] = len(prev_orders_week)
            feat_dict['ops_week_avg_gates'] = prev_orders_week['gate_count'].mean() if len(prev_orders_week) > 0 else 0
            feat_dict['ops_week_total_duration'] = prev_orders_week['duration'].sum() if len(prev_orders_week) > 0 else 0
        else:
            feat_dict['prev_gate_count'] = 0
            feat_dict['prev_duration'] = 0
            feat_dict['prev_op_hour']  = 0
            feat_dict['ops_week_count'] = 0
            feat_dict['ops_week_avg_gates'] = 0
            feat_dict['ops_week_total_duration'] = 0
        
        # 未来潮汐特征（使用预报数据）
        if not future_water_data.empty:
            # 使用未来潮汐预报数据提取特征，窗口大小为12小时
            future_tide_features = self.extract_tidal_features(future_water_data, 12)
            for key, value in future_tide_features.items():
                new_key = key.replace('tide_12h_', 'future_tide_')
                feat_dict[new_key] = value
        else:
            # 如果没有提供未来潮汐数据，则使用备用方案：用最近12小时数据模拟
            if not water_data.empty and len(water_data) > 12:
                future_tide_features = self.extract_tidal_features(water_data.tail(12), 12)
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
        
        return feat_dict

    def prepare_features(self, input_data, model_type='time'):
        """为不同模型准备特征"""
        # 根据模型类型设置时间窗口
        time_window = 24 if model_type in ['time', 'level', 'comb'] else 12
        
        # 获取通用特征
        feat_dict = self.prepare_common_features(input_data, time_window)
        
        # 创建特征DataFrame
        features_df = pd.DataFrame([feat_dict])
        
        # 定义不同模型所需的特征列表
        feature_sets = {
            'time': [
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
            ],
            'num': [
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
                'is_rush_hour', 'day_of_year', 'tide_type',
                
                # 缺失值指示器
                'water_missing', 'flow_missing', 'rain_missing', 'water_status_missing', 'future_water_missing'
            ],
            'level': [
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
            ],
            'dura': [
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
                'is_rush_hour', 'tide_type', 'day_of_year',
                
                # 缺失值指示器
                'water_missing', 'flow_missing', 'rain_missing', 'water_status_missing'
            ],
            'comb': [
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
        }
        
        # 获取当前模型所需的特征列表
        required_features = feature_sets.get(model_type, feature_sets['time'])
        
        # 添加缺失的特征（用0填充）
        for feature in required_features:
            if feature not in features_df.columns:
                features_df[feature] = 0
                
        # 确保特征顺序一致
        features_df = features_df[required_features]
        
        return features_df

    def prepare_time_features(self, input_data):
        """为开闸时间预测准备特征"""
        return self.prepare_features(input_data, 'time')

    def prepare_num_features(self, input_data):
        """为开闸孔数预测准备特征"""
        return self.prepare_features(input_data, 'num')

    def prepare_level_features(self, input_data):
        """为目标水位预测准备特征"""
        return self.prepare_features(input_data, 'level')

    def prepare_dura_features(self, input_data):
        """为开闸时长预测准备特征"""
        return self.prepare_features(input_data, 'dura')

    def prepare_comb_features(self, input_data):
        """为开闸时长*开孔数量预测准备特征"""
        return self.prepare_features(input_data, 'comb')

    def predict_time(self, input_data):
        """预测开闸时间"""
        features = self.prepare_time_features(input_data)
        prediction = self.time_model.predict(features)
        
        # 预测结果是正弦和余弦值
        sin_value = prediction[0][0]
        cos_value = prediction[0][1]
        
        # 将正弦和余弦值转换为小时
        hour_rad = np.arctan2(sin_value, cos_value)
        hour_pred = hour_rad * 24 / (2 * np.pi)
        
        # 确保在0-24范围内
        if hour_pred < 0:
            hour_pred += 24
        
        # 计算开闸时间（同一天）
        sign_time = input_data['sign_time']
        open_time = sign_time.replace(
            hour=int(hour_pred),
            minute=int((hour_pred % 1) * 60),
            second=0,
            microsecond=0
        )
        
        # 如果开闸时间早于调令时间，则推迟到第二天相同时间
        if open_time < sign_time:
            open_time += timedelta(days=1)
        
        return open_time

    def predict_num(self, input_data):
        """预测开闸孔数"""
        features = self.prepare_num_features(input_data)
        prediction = self.num_model.predict(features)
        # 使用映射关系将分类标签转换为真实孔数
        return self.LABEL_TO_GATES.get(prediction[0], 4)  # 默认返回4孔

    def predict_level(self, input_data):
        """预测目标水位"""
        features = self.prepare_level_features(input_data)
        prediction = self.level_model.predict(features)
        return prediction[0]

    def predict_duration(self, input_data):
        """预测开闸时长类别"""
        features = self.prepare_dura_features(input_data)
        prediction = self.dura_model.predict(features)
        # 将编码后的类别转换回原始类别标签
        predicted_class = self.dura_encoder.inverse_transform(prediction)
        return predicted_class[0]

    def predict_comb(self, input_data):
        """预测开闸时长*开孔数量的log值"""
        features = self.prepare_comb_features(input_data)
        prediction = self.comb_model.predict(features)
        # 返回实际的开闸时长*开孔数量（取指数）
        return np.exp(prediction[0])

    def predict_all(self, input_data):
        """预测所有开闸参数"""
        opening_time = self.predict_time(input_data)
        gate_count = self.predict_num(input_data)
        duration_category = self.predict_duration(input_data)
        target_water_level = self.predict_level(input_data)
        comb_value = self.predict_comb(input_data)
        
        return {
            'opening_time': opening_time,
            'gate_count': gate_count,
            'target_water_level': target_water_level,
            'duration_category': duration_category,
            'duration_gate_combination': comb_value
        }

# 示例使用
if __name__ == '__main__':
    # 创建预测器实例
    predictor = KZPredictor()
    
    # 示例输入数据（使用英文参数名）
    sign_time = datetime.now()
    example_input = {
        'sign_time': sign_time,
        'water_level': pd.DataFrame({
            'time': [sign_time - timedelta(hours=i) for i in range(24, 0, -1)],
            'value': np.random.uniform(1.5, 3.5, 24),  # 改为'value'列
        }),
        'future_water_level': pd.DataFrame({  # 添加未来潮汐预报数据
            'time': [sign_time + timedelta(hours=i) for i in range(1, 13)],
            'value': np.random.uniform(1.5, 3.5, 12),  # 改为'value'列
        }),
        'flow': pd.DataFrame({
            'time': [sign_time - timedelta(hours=i) for i in range(24, 0, -1)],
            'value': np.random.uniform(100, 500, 24)
        }),
        'rain_actual': pd.DataFrame({
            'time': [sign_time - timedelta(hours=i) for i in range(24, 0, -1)],
            'value': np.random.uniform(0, 10, 24)
        }),
        'rain_forecast': pd.DataFrame({
            'time': [sign_time + timedelta(hours=i) for i in range(1, 25)],
            'value': np.random.uniform(0, 8, 24)
        }),
        'water_status': pd.DataFrame({
            'time': [sign_time - timedelta(hours=i) for i in range(24, 0, -1)],
            'value': np.random.uniform(2.0, 3.5, 24)
        }),
        'prev_orders': pd.DataFrame({
            'sign_time': [sign_time - timedelta(hours=3), sign_time - timedelta(days=1), sign_time - timedelta(days=3)],
            'opening_time': [sign_time - timedelta(hours=2.5), sign_time - timedelta(days=1), sign_time - timedelta(days=3)],
            'duration': [3.5, 4.2, 2.8],
            'gate_count': [2, 3, 2]
        })
    }
    # print(example_input)
    # 进行预测
    predictions = predictor.predict_all(example_input)
    
    print(f"调令时间: {sign_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"预测开闸时间: {predictions['opening_time'].strftime('%Y-%m-%d %H:%M')}")
    print(f"预测开闸孔数: {predictions['gate_count']}孔")
    print(f"预测目标水位: {predictions['target_water_level']:.2f}米")
    print(f"预测开闸时长: {predictions['duration_category']}小时")
    print(f"预测开闸时长*开孔数量: {predictions['duration_gate_combination']:.2f}")