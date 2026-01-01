from example import KZPredictor
from example_bin import KZBinaryPredictor
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from scipy import stats
import logging

# 添加 SafeSimpleImputer 类定义
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
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# 全局模型变量
model_time = None
model_num = None
model_duration = None
model_level = None  # 新增
binary_predictor = None  # 新增：二分类预测器

from datetime import datetime, timedelta

class TimeValidator:
    @staticmethod
    def adjust_predicted_time(pred_hour, sign_tm):
        """调整预测时间确保晚于调令时间"""
        sign_hour = sign_tm.hour + sign_tm.minute/60
        
        # 处理跨天情况
        if pred_hour < sign_hour:
            pred_hour += 24
        
        # 强制最小延迟30分钟
        min_delay = 0.5  # 0.5小时 = 30分钟
        if pred_hour - sign_hour < min_delay:
            pred_hour = sign_hour + min_delay
        
        # 转换为具体时间
        hours = int(pred_hour)
        minutes = int((pred_hour - hours) * 60)
        
        # 计算日期（如果需要跨天）
        base_date = sign_tm.date()
        if hours >= 24:
            hours -= 24
            base_date += timedelta(days=1)
        
        return datetime(
            base_date.year, base_date.month, base_date.day,
            hours, minutes
        )

def convert_numpy_types(obj):
    """将numpy数据类型转换为Python原生类型以确保JSON可序列化"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return round(float(obj), 2)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
def standardize_datetime(date_obj):
    """标准化日期时间对象"""
    if isinstance(date_obj, str):
        try:
            # 尝试解析ISO格式日期
            return datetime.fromisoformat(date_obj)
        except ValueError:
            try:
                # 尝试解析常见格式
                return datetime.strptime(date_obj, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    return datetime.strptime(date_obj, "%Y/%m/%d %H:%M:%S")
                except ValueError:
                    app.logger.warning(f"无法解析的日期格式: {date_obj}")
                    return None
    return date_obj

def standardize_dataframes(input_data):
    """标准化输入数据中的DataFrames"""
    for key in ['water_level', 'future_water_level', 'flow', 'rain_actual', 'rain_forecast', 'water_status', 'prev_orders']:
        if key in input_data:
            # 如果数据是列表形式，转换为DataFrame
            if isinstance(input_data[key], list):
                input_data[key] = pd.DataFrame(input_data[key])
                
            # 标准化日期列
            if 'time' in input_data[key].columns:
                input_data[key]['time'] = pd.to_datetime(input_data[key]['time'])
            elif 'sign_time' in input_data[key].columns:
                input_data[key]['sign_time'] = pd.to_datetime(input_data[key]['sign_time'])
                if 'kz_time' in input_data[key].columns:
                    input_data[key]['kz_time'] = pd.to_datetime(input_data[key]['kz_time'])
    
    # 标准化基础时间
    if 'sign_time' in input_data:
        input_data['sign_time'] = pd.to_datetime(input_data['sign_time'])
    
    return input_data

def extract_tidal_features(water_data):
    """提取24小时内的潮汐特征"""
    features = {}
    
    if water_data is None or len(water_data) < 2:
        return {
            'tide_range_max': 0,
            'tide_range_avg': 0,
            'tide_rise_avg_rate': 0,
            'tide_fall_avg_rate': 0,
            'tide_cycle_count': 0,
            'tide_type': 0
        }
    
    # 确保数据按时间排序
    water_data = water_data.sort_values('time')
    
    # 只保留需要的列（与 example.py 一致）
    required_cols = ['time', 'value']
    if all(col in water_data.columns for col in required_cols):
        water_data = water_data[required_cols]
    else:
        return {
            'tide_range_max': 0,
            'tide_range_avg': 0,
            'tide_rise_avg_rate': 0,
            'tide_fall_avg_rate': 0,
            'tide_cycle_count': 0,
            'tide_type': 0
        }
    
    # 重采样到10分钟间隔
    resampled = water_data.set_index('time').resample('10min').mean().interpolate()
    levels = resampled['value'].values
    
    # 寻找波峰和波谷
    peaks, _ = find_peaks(levels, prominence=0.1)
    valleys, _ = find_peaks(-levels, prominence=0.1)
    
    # 合并关键点并排序
    key_points = sorted(np.concatenate([peaks, valleys]))
    
    # 特征初始化
    tide_ranges = []
    rise_rates = []
    fall_rates = []
    
    # 分析每个潮汐周期
    for i in range(1, len(key_points)):
        prev_idx = key_points[i-1]
        curr_idx = key_points[i]
        
        prev_level = levels[prev_idx]
        curr_level = levels[curr_idx]
        time_diff = (resampled.index[curr_idx] - resampled.index[prev_idx]).total_seconds() / 3600
        
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
    
    # 潮汐类型识别
    if features['tide_cycle_count'] >= 3:
        features['tide_type'] = 1
    elif features['tide_cycle_count'] == 1:
        features['tide_type'] = 2
    else:
        features['tide_type'] = 3
        
    return features

def prepare_input_features(input_data):
    """
    准备预测所需的特征数据
    :param input_data: 包含所有必要输入数据的字典
    :return: 特征DataFrame
    """
    base_time = input_data['sign_time']
    water_data = input_data.get('water_level', pd.DataFrame())
    flow_data = input_data.get('flow', pd.DataFrame())
    rain_actual = input_data.get('rain_actual', pd.DataFrame())
    rain_forecast = input_data.get('rain_forecast', pd.DataFrame())
    water_status = input_data.get('water_status', pd.DataFrame())
    prev_orders = input_data.get('prev_orders', pd.DataFrame())
    
    # 创建基础特征字典
    feat_dict = {}
    
    # 提取潮汐特征
    tidal_features = extract_tidal_features(water_data)
    feat_dict.update(tidal_features)
    
    # 添加时间特征
    hour = base_time.hour
    day = base_time.weekday()
    month = base_time.month
    
    feat_dict['hour_of_day'] = hour
    feat_dict['day_of_week'] = day
    feat_dict['month'] = month
    feat_dict['is_weekend'] = 1 if day >= 5 else 0
    
    # 添加小时的正弦/余弦变换
    hour_rad = 2 * np.pi * hour / 24
    feat_dict['hour_sin'] = np.sin(hour_rad)
    feat_dict['hour_cos'] = np.cos(hour_rad)
    
    # 添加高峰时段特征
    feat_dict['is_rush_hour'] = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    
    # 缺失值指示器特征
    feat_dict['water_missing'] = 1 if water_data.empty else 0
    feat_dict['flow_missing'] = 1 if flow_data.empty else 0
    feat_dict['rain_missing'] = 1 if (rain_actual.empty and rain_forecast.empty) else 0
    feat_dict['water_status_missing'] = 1 if water_status.empty else 0
    
    # 水位统计特征（最近12小时）
    if not water_data.empty:
        cutoff_time = base_time - timedelta(hours=12)
        water_data_12h = water_data[water_data['time'] >= cutoff_time]
        
        if len(water_data_12h) > 1:
            levels = water_data_12h['value'].values
            time_idx = np.arange(len(levels))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, levels)
            feat_dict['water_slope'] = slope
            feat_dict['water_r_squared'] = r_value**2
        else:
            feat_dict['water_slope'] = 0
            feat_dict['water_r_squared'] = 0
        
        # 基本水位统计
        feat_dict['water_mean'] = water_data_12h['value'].mean() if len(water_data_12h) > 0 else 0
        feat_dict['water_max'] = water_data_12h['value'].max() if len(water_data_12h) > 0 else 0
        feat_dict['water_min'] = water_data_12h['value'].min() if len(water_data_12h) > 0 else 0
        feat_dict['water_range'] = feat_dict['water_max'] - feat_dict['water_min']
    else:
        feat_dict['water_slope'] = 0
        feat_dict['water_r_squared'] = 0
        feat_dict['water_mean'] = 0
        feat_dict['water_max'] = 0
        feat_dict['water_min'] = 0
        feat_dict['water_range'] = 0
    
    # 流量统计特征
    if not flow_data.empty:
        cutoff_time = base_time - timedelta(hours=12)
        flow_data_12h = flow_data[flow_data['time'] >= cutoff_time]
        
        if len(flow_data_12h) > 2:
            flows = flow_data_12h['value'].values
            feat_dict['flow_var'] = np.var(flows)
            feat_dict['flow_skew'] = stats.skew(flows)
        else:
            feat_dict['flow_var'] = 0
            feat_dict['flow_skew'] = 0
        
        # 基本流量统计
        feat_dict['flow_mean'] = flow_data_12h['value'].mean() if len(flow_data_12h) > 0 else 0
        feat_dict['flow_max'] = flow_data_12h['value'].max() if len(flow_data_12h) > 0 else 0
        feat_dict['flow_min'] = flow_data_12h['value'].min() if len(flow_data_12h) > 0 else 0
        feat_dict['flow_range'] = feat_dict['flow_max'] - feat_dict['flow_min']
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
        cutoff_time = base_time - timedelta(hours=12)
        rain_actual_12h = rain_actual[rain_actual['time'] >= cutoff_time]
        rain_actual_total = rain_actual_12h['value'].sum()
        feat_dict['rain_actual_avg'] = rain_actual_12h['value'].mean() if len(rain_actual_12h) > 0 else 0
    
    rain_forecast_total = 0
    if not rain_forecast.empty:
        cutoff_time = base_time + timedelta(hours=12)
        rain_forecast_12h = rain_forecast[rain_forecast['time'] <= cutoff_time]
        rain_forecast_total = rain_forecast_12h['value'].sum()
        feat_dict['rain_forecast_avg'] = rain_forecast_12h['value'].mean() if len(rain_forecast_12h) > 0 else 0
    
    feat_dict['rain_actual_total'] = rain_actual_total
    feat_dict['rain_forecast_total'] = rain_forecast_total
    
    # 水位工况统计特征
    if not water_status.empty:
        cutoff_time = base_time - timedelta(hours=12)
        water_status_12h = water_status[water_status['time'] >= cutoff_time]
        
        if len(water_status_12h) > 1:
            status_levels = water_status_12h['value'].values
            time_idx = np.arange(len(status_levels))
            slope, _, _, _, _ = stats.linregress(time_idx, status_levels)
            feat_dict['water_status_slope'] = slope
        else:
            feat_dict['water_status_slope'] = 0
        
        # 基本水位工况统计
        feat_dict['water_status_mean'] = water_status_12h['value'].mean() if len(water_status_12h) > 0 else 0
        feat_dict['water_status_max'] = water_status_12h['value'].max() if len(water_status_12h) > 0 else 0
        feat_dict['water_status_min'] = water_status_12h['value'].min() if len(water_status_12h) > 0 else 0
        feat_dict['water_status_range'] = feat_dict['water_status_max'] - feat_dict['water_status_min']
    else:
        feat_dict['water_status_slope'] = 0
        feat_dict['water_status_mean'] = 0
        feat_dict['water_status_max'] = 0
        feat_dict['water_status_min'] = 0
        feat_dict['water_status_range'] = 0
    
    # 添加历史操作特征
    if not prev_orders.empty:
        cutoff_time = base_time - timedelta(hours=24)
        prev_orders_24h = prev_orders[prev_orders['sign_time'] >= cutoff_time]
        
        feat_dict['prev_gate_count'] = prev_orders_24h['kz_num'].mean() if len(prev_orders_24h) > 0 else 0
        feat_dict['prev_duration'] = prev_orders_24h['kz_dura'].mean() if len(prev_orders_24h) > 0 else 0
        feat_dict['prev_op_hour'] = prev_orders_24h['kz_time'].apply(lambda x: x.hour).mean() if len(prev_orders_24h) > 0 else 0
        feat_dict['ops_24h_count'] = len(prev_orders_24h)
        feat_dict['ops_24h_avg_gates'] = prev_orders_24h['kz_num'].mean() if len(prev_orders_24h) > 0 else 0
        feat_dict['ops_24h_total_duration'] = prev_orders_24h['kz_dura'].sum()
    else:
        feat_dict['prev_gate_count'] = 0
        feat_dict['prev_duration'] = 0
        feat_dict['prev_op_hour'] = 0
        feat_dict['ops_24h_count'] = 0
        feat_dict['ops_24h_avg_gates'] = 0
        feat_dict['ops_24h_total_duration'] = 0
    
    # 降雨相关比率特征
    eps = 1e-5
    
    # 水位-降雨比率
    if feat_dict['rain_actual_total'] > eps:
        feat_dict['water_rain_ratio'] = feat_dict['water_mean'] / (feat_dict['rain_actual_total'] + eps)
    else:
        feat_dict['water_rain_ratio'] = 0
    
    # 流量-降雨比率
    if feat_dict['rain_actual_total'] > eps:
        feat_dict['flow_rain_ratio'] = feat_dict['flow_mean'] / (feat_dict['rain_actual_total'] + eps)
    else:
        feat_dict['flow_rain_ratio'] = 0
    
    # 降雨变化率（预测/实际）
    if feat_dict['rain_actual_total'] > eps:
        feat_dict['rain_change_rate'] = (feat_dict['rain_forecast_total'] - feat_dict['rain_actual_total']) / (feat_dict['rain_actual_total'] + eps)
    else:
        feat_dict['rain_change_rate'] = 0
    
    # 添加潮汐相位特征
    tide_phase = (base_time.hour % 6)
    feat_dict['tide_phase_sin'] = np.sin(2 * np.pi * tide_phase / 6)
    feat_dict['tide_phase_cos'] = np.cos(2 * np.pi * tide_phase / 6)
    
    # 创建特征DataFrame
    features_df = pd.DataFrame([feat_dict])
    
    # 确保包含所有训练时使用的特征
    required_features = [
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'tide_phase_sin', 'tide_phase_cos',
        'prev_gate_count', 'prev_duration', 'prev_op_hour',
        'ops_24h_count', 'ops_24h_avg_gates', 'ops_24h_total_duration',
        'tide_range_max', 'tide_range_avg', 'tide_rise_avg_rate', 
        'tide_fall_avg_rate', 'tide_cycle_count',
        'water_mean', 'water_max', 'water_min', 'water_range', 
        'water_slope', 'water_r_squared',
        'flow_mean', 'flow_max', 'flow_min', 'flow_range', 'flow_var', 'flow_skew',
        'rain_actual_total', 'rain_forecast_total', 
        'rain_actual_avg', 'rain_forecast_avg',
        'rain_change_rate', 'water_rain_ratio', 'flow_rain_ratio',
        'water_status_mean', 'water_status_max', 'water_status_min', 
        'water_status_range', 'water_status_slope',
        'is_rush_hour', 'tide_type'
    ]
    
    # 添加缺失的特征
    for feature in required_features:
        if feature not in features_df.columns:
            features_df[feature] = 0
    
    return features_df

@app.route('/predict', methods=['POST'])
def predict():
    """统一预测接口"""
    # 获取并标准化输入数据
    data = request.json
    data = standardize_dataframes(data)
    
    # 检查基础时间
    if 'sign_time' not in data:
        return jsonify({
            'error': "Missing required sign_time field",
            'status': 'error'
        }), 400
    
    if data['sign_time'] is None:
        return jsonify({
            'error': "Invalid sign_time datetime format",
            'status': 'error'
        }), 400
    
    # 创建预测器实例并准备特征
    predictor = KZPredictor()
    
    # 进行预测
    results = {
        "opening_time": None,
        # "opening_time_in_hours": None,
        "gate_count": None,
        "duration_in_hours": None,
        "target_water_level": None,
        "closing_time": None,
        "dura_dot_num": None  # 新增字段
    }

    # 预测开闸时间
    try:
        opening_time = predictor.predict_time(data)
        results['opening_time'] = opening_time.strftime("%Y-%m-%d %H:%M:00")
        # results['opening_time_in_hours'] = round(opening_time.hour + opening_time.minute/60.0, 2)
    except Exception as e:
        app.logger.error(f"Error predicting opening time: {str(e)}")
        return jsonify({
            'error': f"Error predicting opening time: {str(e)}",
            'status': 'error'
        }), 500

    # 预测开闸孔数
    try:
        gate_count = predictor.predict_num(data)
        results['gate_count'] = int(gate_count)
    except Exception as e:
        app.logger.error(f"Error predicting gate count: {str(e)}")
        return jsonify({
            'error': f"Error predicting gate count: {str(e)}",
            'status': 'error'
        }), 500

    # 预测目标水位
    try:
        target_water_level = predictor.predict_level(data)
        if target_water_level <= 1:
            results['target_water_level'] = "最低水位不做限制"
        else:
            # 确保转换为Python float
            results['target_water_level'] = float(round(target_water_level, 2))
    except Exception as e:
        app.logger.error(f"Error predicting target water level: {str(e)}")
        return jsonify({
            'error': f"Error predicting target water level: {str(e)}",
            'status': 'error'
        }), 500

    # 预测开闸时长
    try:
        duration_category = predictor.predict_duration(data)
        results['duration_in_hours'] = int(duration_category)
    except Exception as e:
        app.logger.error(f"Error predicting duration: {str(e)}")
        return jsonify({
            'error': f"Error predicting duration: {str(e)}",
            'status': 'error'
        }), 500

    # 预测开闸时长*开孔数量
    try:
        comb_value = predictor.predict_comb(data)
        # 确保转换为Python float
        results['dura_dot_num'] = float(round(comb_value, 2))
    except Exception as e:
        app.logger.error(f"Error predicting duration*gate combination: {str(e)}")
        return jsonify({
            'error': f"Error predicting duration*gate combination: {str(e)}",
            'status': 'error'
        }), 500

    # 计算关闸时间
    try:
        if results['opening_time'] and results['duration_in_hours']:
            opening_time_obj = datetime.strptime(results['opening_time'], "%Y-%m-%d %H:%M:00")
            closing_time = opening_time_obj + timedelta(hours=results['duration_in_hours'])
            results['closing_time'] = closing_time.strftime("%Y-%m-%d %H:%M:00")
    except Exception as e:
        app.logger.error(f"Error calculating closing time: {str(e)}")
        # 不返回错误，只是不设置closing_time字段

    results = convert_numpy_types(results)
    # 返回所有预测结果
    return jsonify({
        'status': 'success',
        'prediction': results,
        'metadata': {
            'model_version': '1.0',
            'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    })


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """统一预测接口"""
    # 获取并标准化输入数据
    data = request.json
    data = standardize_dataframes(data)
    
    # 检查基础时间
    if 'sign_time' not in data:
        return jsonify({
            'error': "Missing required sign_time field",
            'status': 'error'
        }), 400
    
    if data['sign_time'] is None:
        return jsonify({
            'error': "Invalid sign_time datetime format",
            'status': 'error'
        }), 400
    
    # 创建预测器实例并准备特征
    predictor = KZPredictor()
    results = predictor.batch_predict(data)

    results = convert_numpy_types(results)
    # 返回所有预测结果
    return jsonify({
        'status': 'success',
        'prediction': results,
        'metadata': {
            'model_version': '2.0',
            'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    })


@app.route('/predict_bin', methods=['POST'])
def predict_binary():
    """开闸与否二分类预测接口"""
    # 获取并标准化输入数据
    data = request.json
    data = standardize_dataframes(data)
    
    # 检查基础时间
    if 'sign_time' not in data:
        return jsonify({
            'error': "Missing required sign_time field",
            'status': 'error'
        }), 400
    
    if data['sign_time'] is None:
        return jsonify({
            'error': "Invalid sign_time datetime format",
            'status': 'error'
        }), 400
    
    # 检查二分类预测器是否已加载
    if binary_predictor is None or binary_predictor.model is None:
        return jsonify({
            'error': "Binary prediction model not loaded",
            'status': 'error'
        }), 500
    
    # 获取阈值参数（可选，默认0.5）
    threshold = request.json.get('threshold', 0.5)
    
    try:
        # 进行二分类预测
        result = binary_predictor.predict_with_explanation(data, threshold)
        
        if result is None:
            return jsonify({
                'error': "Binary prediction failed",
                'status': 'error'
            }), 500
        
        # 构建响应结果
        prediction_result = {
            'prediction': result['prediction'],  # 0=不开闸, 1=开闸
            'probability': round(float(result['probability']), 4),
            'confidence': round(float(result['confidence']), 4),
            'features_used': result['features_used'],
            'explanation': result['explanation'],
            'recommendation': 'Issue gate opening order' if result['prediction'] == 1 else 'Continue monitoring without opening'
        }
        
        # 添加详细解释
        if result['prediction'] == 1:
            if result['probability'] > 0.8:
                prediction_result['confidence_level'] = 'high'
            elif result['probability'] > 0.6:
                prediction_result['confidence_level'] = 'medium'
            else:
                prediction_result['confidence_level'] = 'low'
        else:
            if result['probability'] < 0.2:
                prediction_result['confidence_level'] = 'high'
            elif result['probability'] < 0.4:
                prediction_result['confidence_level'] = 'medium'
            else:
                prediction_result['confidence_level'] = 'low'
        
        prediction_result = convert_numpy_types(prediction_result)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction_result,
            'metadata': {
                'model_version': 'binary_1.0',
                'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'threshold_used': threshold
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error in binary prediction: {str(e)}")
        return jsonify({
            'error': f"Binary prediction error: {str(e)}",
            'status': 'error'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    # 检查模型是否加载完成
    models_loaded = all([
        model_time is not None,
        model_num is not None,
        model_duration is not None,
        binary_predictor is not None
    ])
    
    # 设置状态
    status = {
        'model_time_loaded': model_time is not None,
        'model_num_loaded': model_num is not None,
        'model_duration_loaded': model_duration is not None,
        'binary_predictor_loaded': binary_predictor  is not None,
        'status': 'ready' if models_loaded else 'loading'
    }
    
    # 返回状态
    return jsonify(status)

def load_models():
    """加载所有模型（在应用启动时调用）"""
    global binary_predictor
    
    try:
        # 加载二分类模型
        binary_predictor = KZBinaryPredictor()
        if binary_predictor.model is not None:
            app.logger.info("Binary prediction model loaded successfully")
        else:
            app.logger.warning("Binary prediction model failed to load")
    except Exception as e:
        app.logger.error(f"Error loading binary model: {str(e)}")
        
if __name__ == '__main__':
    # 启动时加载所有模型
    load_models()
    app.run(host='0.0.0.0', port=8001, debug=True)
    