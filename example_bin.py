import codecs
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class KZBinaryPredictor:
    def __init__(self, model_path='outputs/binary/pkls/enhanced_binary_xgboost_model.pkl'):
        """
        开闸与否二分类预测器
        """
        # 加载二分类模型
        try:
            self.model = joblib.load(model_path)
            print(f"成功加载二分类模型: {model_path}")
        except FileNotFoundError:
            print(f"模型文件未找到: {model_path}")
            print("请先运行 train_xgboost.py 训练模型")
            self.model = None
        
    def extract_tidal_features(self, water_data, hours=24):
        """提取指定时间窗口内的潮汐特征"""
        features = {}
        
        if water_data is None or len(water_data) < 2:
            # 返回默认特征值 - 包含所有可能的特征
            default_features = {
                f'tide_{hours}h_mean': 0, f'tide_{hours}h_max': 0, f'tide_{hours}h_min': 0, 
                f'tide_{hours}h_range': 0, f'tide_{hours}h_slope': 0, f'tide_{hours}h_r_squared': 0,
                f'tide_{hours}h_cycle_count': 0, f'tide_{hours}h_rise_rate': 0, f'tide_{hours}h_fall_rate': 0,
                f'tide_{hours}h_phase': 0, f'tide_{hours}h_volatility': 0, f'tide_{hours}h_trend_strength': 0,
                # 添加额外的特征以匹配训练时的特征集
                f'tide_{hours}h_diff_mean': 0, f'tide_{hours}h_diff_max': 0, f'tide_{hours}h_diff_min': 0,
                f'tide_{hours}h_diff_std': 0, f'tide_{hours}h_rolling_3_mean': 0, f'tide_{hours}h_rolling_3_std': 0,
                f'tide_{hours}h_rolling_3_min': 0, f'tide_{hours}h_rolling_3_max': 0, f'tide_{hours}h_rolling_6_mean': 0,
                f'tide_{hours}h_rolling_6_std': 0, f'tide_{hours}h_rolling_6_min': 0, f'tide_{hours}h_rolling_6_max': 0,
                f'tide_{hours}h_rolling_12_mean': 0, f'tide_{hours}h_rolling_12_std': 0, f'tide_{hours}h_rolling_12_min': 0,
                f'tide_{hours}h_rolling_12_max': 0, f'tide_{hours}h_hour_sin': 0, f'tide_{hours}h_hour_cos': 0,
                f'tide_{hours}h_seasonal_strength': 0, f'tide_{hours}h_residual_strength': 0
            }
            return default_features
        
        # 确保数据按时间排序
        water_data = water_data.sort_values('time')
        
        # 重采样到10分钟间隔
        resampled = water_data.set_index('time').resample('10T').mean().interpolate()
        levels = resampled['value'].values
        
        # 基础统计特征
        features[f'tide_{hours}h_mean'] = np.mean(levels) if len(levels) > 0 else 0
        features[f'tide_{hours}h_max'] = np.max(levels) if len(levels) > 0 else 0
        features[f'tide_{hours}h_min'] = np.min(levels) if len(levels) > 0 else 0
        features[f'tide_{hours}h_range'] = features[f'tide_{hours}h_max'] - features[f'tide_{hours}h_min']
        
        # 趋势特征
        if len(levels) > 1:
            time_idx = np.arange(len(levels))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, levels)
            features[f'tide_{hours}h_slope'] = slope
            features[f'tide_{hours}h_r_squared'] = r_value**2
            features[f'tide_{hours}h_trend_strength'] = abs(slope) * 100
        else:
            features[f'tide_{hours}h_slope'] = 0
            features[f'tide_{hours}h_r_squared'] = 0
            features[f'tide_{hours}h_trend_strength'] = 0
        
        # 波动性特征
        features[f'tide_{hours}h_volatility'] = np.std(np.diff(levels)) if len(levels) > 1 else 0
        
        # 潮汐周期检测
        diffs = np.diff(levels)
        turning_points = []
        for i in range(1, len(diffs)):
            if diffs[i] * diffs[i-1] < 0:  # 符号变化表示转折点
                turning_points.append(i)
        
        # 计算潮汐周期数量
        cycle_count = len(turning_points) // 2
        features[f'tide_{hours}h_cycle_count'] = cycle_count
        
        # 潮汐类型识别
        if cycle_count >= 2:
            tide_type = 1  # 半日潮
        elif cycle_count == 1:
            tide_type = 2  # 全日潮
        else:
            tide_type = 3  # 混合潮
        features['tide_type'] = tide_type
        
        # 计算涨落潮速率和相位
        rise_rates = []
        fall_rates = []
        tide_phases = []
        
        for i in range(1, len(turning_points)):
            prev_idx = turning_points[i-1]
            curr_idx = turning_points[i]
            
            prev_level = levels[prev_idx]
            curr_level = levels[curr_idx]
            time_diff = (resampled.index[curr_idx] - resampled.index[prev_idx]).total_seconds() / 3600
            
            phase = time_diff
            tide_phases.append(phase)
            
            if curr_level > prev_level:
                tide_range = curr_level - prev_level
                rise_rates.append(tide_range / time_diff if time_diff > 0 else 0)
            else:
                tide_range = prev_level - curr_level
                fall_rates.append(tide_range / time_diff if time_diff > 0 else 0)
        
        features[f'tide_{hours}h_rise_rate'] = np.mean(rise_rates) if rise_rates else 0
        features[f'tide_{hours}h_fall_rate'] = np.mean(fall_rates) if fall_rates else 0
        features[f'tide_{hours}h_phase'] = np.mean(tide_phases) if tide_phases else 0
        
        # 添加差分特征
        if len(levels) > 1:
            diffs = np.diff(levels)
            features[f'tide_{hours}h_diff_mean'] = np.mean(diffs)
            features[f'tide_{hours}h_diff_max'] = np.max(diffs)
            features[f'tide_{hours}h_diff_min'] = np.min(diffs)
            features[f'tide_{hours}h_diff_std'] = np.std(diffs)
        else:
            features[f'tide_{hours}h_diff_mean'] = 0
            features[f'tide_{hours}h_diff_max'] = 0
            features[f'tide_{hours}h_diff_min'] = 0
            features[f'tide_{hours}h_diff_std'] = 0
        
        # 添加滑动窗口统计特征
        if len(levels) >= 12:
            window_sizes = [3, 6, 12]
            for window in window_sizes:
                if len(levels) >= window:
                    features[f'tide_{hours}h_rolling_{window}_mean'] = np.mean(levels[-window:])
                    features[f'tide_{hours}h_rolling_{window}_std'] = np.std(levels[-window:])
                    features[f'tide_{hours}h_rolling_{window}_min'] = np.min(levels[-window:])
                    features[f'tide_{hours}h_rolling_{window}_max'] = np.max(levels[-window:])
                else:
                    features[f'tide_{hours}h_rolling_{window}_mean'] = 0
                    features[f'tide_{hours}h_rolling_{window}_std'] = 0
                    features[f'tide_{hours}h_rolling_{window}_min'] = 0
                    features[f'tide_{hours}h_rolling_{window}_max'] = 0
        else:
            # 为所有窗口大小设置默认值
            for window in [3, 6, 12]:
                features[f'tide_{hours}h_rolling_{window}_mean'] = 0
                features[f'tide_{hours}h_rolling_{window}_std'] = 0
                features[f'tide_{hours}h_rolling_{window}_min'] = 0
                features[f'tide_{hours}h_rolling_{window}_max'] = 0
        
        # 添加周期性特征
        if len(water_data) > 0:
            hour = water_data['time'].iloc[0].hour
            features[f'tide_{hours}h_hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features[f'tide_{hours}h_hour_cos'] = np.cos(2 * np.pi * hour / 24)
        else:
            features[f'tide_{hours}h_hour_sin'] = 0
            features[f'tide_{hours}h_hour_cos'] = 0
        
        # 简化处理时间序列分解特征
        features[f'tide_{hours}h_seasonal_strength'] = 0
        features[f'tide_{hours}h_residual_strength'] = 0
        
        return features

    def prepare_features(self, input_data):
        """准备二分类预测所需的特征"""
        # 从输入数据中提取各个部分
        base_time = input_data['sign_time']
        water_data = input_data.get('water_level', pd.DataFrame())
        flow_data = input_data.get('flow', pd.DataFrame())
        rain_actual = input_data.get('rain_actual', pd.DataFrame())
        rain_forecast = input_data.get('rain_forecast', pd.DataFrame())
        water_status = input_data.get('water_status', pd.DataFrame())
        prev_orders = input_data.get('prev_orders', pd.DataFrame())
        
        water_data = input_data.get('water_level', pd.DataFrame())
        if not isinstance(water_data, pd.Series):
            water_data = pd.DataFrame(water_data)
            water_data['time'] = pd.to_datetime(water_data['time'])
            if 'station_id' in water_data.columns:
                del water_data['station_id']
        # print("water_data:", water_data)

        future_water_data = input_data.get('future_water_level', pd.DataFrame())
        if not isinstance(future_water_data, pd.Series):
            future_water_data = pd.DataFrame(future_water_data)
            if not future_water_data.empty:
                future_water_data['time'] = pd.to_datetime(future_water_data['time'])
                if 'station_id' in future_water_data.columns:
                    del water_data['station_id']
        # print("future_water_data:", future_water_data)

        flow_data = input_data.get('flow', pd.DataFrame())
        if not isinstance(flow_data, pd.Series):
            flow_data = pd.DataFrame(flow_data)
            flow_data['time'] = pd.to_datetime(flow_data['time'])

        rain_actual = input_data.get('rain_actual', pd.DataFrame())
        if not isinstance(rain_actual, pd.Series):
            rain_actual = pd.DataFrame(rain_actual)
            rain_actual['time'] = pd.to_datetime(rain_actual['time'])

        rain_forecast = input_data.get('rain_forecast', pd.DataFrame())
        if not isinstance(rain_forecast, pd.Series):
            rain_forecast = pd.DataFrame(rain_forecast)
            rain_forecast['time'] = pd.to_datetime(rain_forecast['time'])

        water_status = input_data.get('water_status', pd.DataFrame())
        if not isinstance(water_status, pd.Series):
            water_status = pd.DataFrame(water_status)
            water_status['time'] = pd.to_datetime(water_status['time'])

        prev_orders = input_data.get('prev_orders', pd.DataFrame())
        if not isinstance(prev_orders, pd.Series):
            prev_orders = pd.DataFrame(prev_orders)
            if not prev_orders.empty:
                # 使用 sign_time 作为时间列
                prev_orders['sign_time'] = pd.to_datetime(prev_orders['sign_time'])
                if 'opening_time' in prev_orders.columns:
                    prev_orders['opening_time'] = pd.to_datetime(prev_orders['opening_time'])
        # 创建基础特征字典
        feat_dict = {}
        
        # 提取潮汐特征
        tide_24h_features = self.extract_tidal_features(water_data, 24)
        tide_12h_features = self.extract_tidal_features(water_data, 12)
        
        # 更新特征字典
        feat_dict.update(tide_24h_features)
        feat_dict.update(tide_12h_features)
        
        # 添加未来潮汐特征（简化处理）
        future_tide_features = {
            'future_tide_mean': 0, 'future_tide_max': 0, 'future_tide_min': 0, 'future_tide_range': 0,
            'future_tide_slope': 0, 'future_tide_r_squared': 0, 'future_tide_cycle_count': 0,
            'future_tide_rise_rate': 0, 'future_tide_fall_rate': 0, 'future_tide_phase': 0,
            'future_tide_volatility': 0, 'future_tide_trend_strength': 0,
            'future_tide_diff_mean': 0, 'future_tide_diff_max': 0, 'future_tide_diff_min': 0,
            'future_tide_rolling_12_min': 0, 'future_tide_hour_cos': 0
        }
        feat_dict.update(future_tide_features)
        
        # 添加时间特征
        hour = base_time.hour
        day = base_time.weekday()
        month = base_time.month
        
        feat_dict['hour_of_day'] = hour
        feat_dict['day_of_week'] = day
        feat_dict['month'] = month
        feat_dict['is_weekend'] = 1 if day >= 5 else 0
        feat_dict['day_of_year'] = base_time.timetuple().tm_yday
        
        # 添加小时的正弦/余弦变换
        hour_rad = 2 * np.pi * hour / 24
        feat_dict['hour_sin'] = np.sin(hour_rad)
        feat_dict['hour_cos'] = np.cos(hour_rad)
        
        # 添加季节特征
        def get_season(month):
            if month in [12, 1, 2]: return 0  # 冬季
            elif month in [3, 4, 5]: return 1  # 春季
            elif month in [6, 7, 8]: return 2  # 夏季
            else: return 3  # 秋季
        feat_dict['season'] = get_season(month)
        
        # 添加时间敏感性特征
        feat_dict['is_rush_hour'] = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        feat_dict['is_night'] = 1 if hour < 6 or hour > 22 else 0
        
        # 添加缺失值指示器特征
        feat_dict['water_missing'] = 1 if water_data.empty else 0
        feat_dict['flow_missing'] = 1 if flow_data.empty else 0
        feat_dict['rain_missing'] = 1 if (rain_actual.empty and rain_forecast.empty) else 0
        feat_dict['water_status_missing'] = 1 if water_status.empty else 0
        feat_dict['future_water_missing'] = 1  # 简化处理，假设未来水位数据缺失
        
        # [其余的特征提取代码保持不变...]
        
        # 创建特征DataFrame
        features_df = pd.DataFrame([feat_dict])
        
        # 定义二分类模型所需的完整特征列表
        required_features = [
            # 时间特征
            'hour_of_day', 'day_of_week', 'month', 'season', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_of_year', 'is_rush_hour', 'is_night',
            
            # 历史操作特征
            'prev_gate_count', 'prev_duration', 'prev_op_hour',
            'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration',
            'hours_since_last_op',
            
            # 24小时潮汐基础特征
            'tide_24h_phase', 'tide_24h_mean', 'tide_24h_max', 'tide_24h_min', 'tide_24h_range',
            'tide_24h_slope', 'tide_24h_r_squared', 'tide_24h_cycle_count',
            'tide_24h_rise_rate', 'tide_24h_fall_rate', 'tide_24h_volatility', 'tide_24h_trend_strength',
            'tide_24h_tide_type',  # 新增：24小时潮汐类型特征
            
            # 24小时潮汐差分特征
            'tide_24h_diff_mean', 'tide_24h_diff_max', 'tide_24h_diff_min', 'tide_24h_diff_std',
            
            # 24小时潮汐滑动窗口特征
            'tide_24h_rolling_3_mean', 'tide_24h_rolling_3_std', 'tide_24h_rolling_3_min', 'tide_24h_rolling_3_max',
            'tide_24h_rolling_6_mean', 'tide_24h_rolling_6_std', 'tide_24h_rolling_6_min', 'tide_24h_rolling_6_max',
            'tide_24h_rolling_12_mean', 'tide_24h_rolling_12_std', 'tide_24h_rolling_12_min', 'tide_24h_rolling_12_max',
            
            # 24小时潮汐周期性特征
            'tide_24h_hour_sin', 'tide_24h_hour_cos', 'tide_24h_seasonal_strength', 'tide_24h_residual_strength',
            
            # 12小时潮汐基础特征
            'tide_12h_phase', 'tide_12h_mean', 'tide_12h_max', 'tide_12h_min', 'tide_12h_range',
            'tide_12h_slope', 'tide_12h_r_squared', 'tide_12h_cycle_count',
            'tide_12h_rise_rate', 'tide_12h_fall_rate', 'tide_12h_volatility', 'tide_12h_trend_strength',
            'tide_12h_tide_type',  # 新增：12小时潮汐类型特征
            
            # 12小时潮汐差分特征
            'tide_12h_diff_mean', 'tide_12h_diff_max', 'tide_12h_diff_min', 'tide_12h_diff_std',
            
            # 12小时潮汐滑动窗口特征
            'tide_12h_rolling_3_mean', 'tide_12h_rolling_3_std', 'tide_12h_rolling_3_min', 'tide_12h_rolling_3_max',
            'tide_12h_rolling_6_mean', 'tide_12h_rolling_6_std', 'tide_12h_rolling_6_min', 'tide_12h_rolling_6_max',
            'tide_12h_rolling_12_mean', 'tide_12h_rolling_12_std', 'tide_12h_rolling_12_min', 'tide_12h_rolling_12_max',
            
            # 12小时潮汐周期性特征
            'tide_12h_hour_sin', 'tide_12h_hour_cos', 'tide_12h_seasonal_strength', 'tide_12h_residual_strength',
            
            # 未来潮汐特征
            'future_tide_mean', 'future_tide_max', 'future_tide_min', 'future_tide_range',
            'future_tide_slope', 'future_tide_r_squared', 'future_tide_cycle_count',
            'future_tide_rise_rate', 'future_tide_fall_rate', 'future_tide_phase',
            'future_tide_volatility', 'future_tide_trend_strength',
            'future_tide_diff_mean', 'future_tide_diff_max', 'future_tide_diff_min',
            'future_tide_diff_std',  # 新增：未来潮汐差分标准差
            'future_tide_rolling_12_min', 'future_tide_hour_cos',
            'future_tide_rolling_3_std',  # 新增：未来潮汐3窗口标准差
            'future_tide_rolling_6_std',  # 新增：未来潮汐6窗口标准差
            
            # 流量特征
            'flow_mean', 'flow_max', 'flow_min', 'flow_range', 'flow_var', 'flow_skew', 'flow_trend',
            
            # 降雨特征
            'rain_actual_total', 'rain_forecast_total', 
            'rain_actual_avg', 'rain_forecast_avg',
            'rain_change_rate', 'water_rain_ratio', 'flow_rain_ratio', 'water_flow_ratio',
            
            # 区域降雨特征
            'rain_actual_绍兴平原_sum', 'rain_actual_绍兴平原_max', 'rain_actual_绍兴平原_mean',
            'rain_forecast_绍兴平原_mean', 'rain_forecast_绍兴平原_max', 'rain_forecast_绍兴平原_sum',
            'rain_actual_嵊州_sum', 'rain_actual_嵊州_max', 'rain_actual_嵊州_mean',
            'rain_forecast_嵊州_mean', 'rain_forecast_嵊州_max', 'rain_forecast_嵊州_sum',
            'rain_actual_虞南山区_sum', 'rain_actual_虞南山区_max', 'rain_actual_虞南山区_mean',
            'rain_forecast_虞南山区_mean', 'rain_forecast_虞南山区_max', 'rain_forecast_虞南山区_sum',
            'rain_actual_新昌_sum', 'rain_actual_新昌_max', 'rain_actual_新昌_mean',
            'rain_forecast_新昌_mean', 'rain_forecast_新昌_max', 'rain_forecast_新昌_sum',
            'rain_actual_虞北平原_sum', 'rain_actual_虞北平原_max', 'rain_actual_虞北平原_mean',
            'rain_forecast_虞北平原_mean', 'rain_forecast_虞北平原_max', 'rain_forecast_虞北平原_sum',
            
            # 水位工况特征
            'water_status_mean', 'water_status_max', 'water_status_min', 
            'water_status_range', 'water_status_slope',
            
            # 分类特征和指示器特征
            'tide_type',
            'water_missing', 'flow_missing', 
            'rain_missing', 'water_status_missing', 'future_water_missing'
        ]
        # 添加区域降雨特征
        regions = ['绍兴平原', '嵊州', '虞南山区', '新昌', '虞北平原']
        for region in regions:
            features_df[f'rain_actual_{region}_sum'] = 0
            features_df[f'rain_actual_{region}_max'] = 0
            features_df[f'rain_actual_{region}_mean'] = 0
            features_df[f'rain_forecast_{region}_mean'] = 0
            features_df[f'rain_forecast_{region}_max'] = 0
            features_df[f'rain_forecast_{region}_sum'] = 0
        
        # 添加所有缺失的特征（用0填充）
        for feature in required_features:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # 确保特征顺序一致
        features_df = features_df[required_features]
        return features_df


    def predict(self, input_data, threshold=0.5):
        """预测开闸与否"""
        if self.model is None:
            print("模型未加载，无法进行预测")
            return None
        
        # 准备特征
        features = self.prepare_features(input_data)
        
        # 进行预测
        prediction_proba = self.model.predict_proba(features)[0]
        prediction = 1 if prediction_proba[1] >= threshold else 0
        
        return {
            'prediction': prediction,  # 0=不开闸, 1=开闸
            'probability': prediction_proba[1],  # 开闸概率
            'confidence': max(prediction_proba),  # 置信度
            'features_used': len(features.columns)
        }

    def predict_with_explanation(self, input_data, threshold=0.5):
        """带解释的预测"""
        prediction_result = self.predict(input_data, threshold)
        
        if prediction_result is None:
            return None
        
        # 生成解释文本
        if prediction_result['prediction'] == 1:
            explanation = f"预测开闸 (概率: {prediction_result['probability']:.3f})"
            if prediction_result['probability'] > 0.8:
                explanation += " - 高置信度开闸"
            elif prediction_result['probability'] > 0.6:
                explanation += " - 中等置信度开闸"
            else:
                explanation += " - 低置信度开闸"
        else:
            explanation = f"预测不开闸 (开闸概率: {prediction_result['probability']:.3f})"
            if prediction_result['probability'] < 0.2:
                explanation += " - 高置信度不开闸"
            elif prediction_result['probability'] < 0.4:
                explanation += " - 中等置信度不开闸"
            else:
                explanation += " - 低置信度不开闸"
        
        prediction_result['explanation'] = explanation
        return prediction_result

# 示例使用
if __name__ == '__main__':
    # 创建预测器实例
    predictor = KZBinaryPredictor()
    # 示例输入数据
    with codecs.open('data/input_v2.json', 'r', encoding='utf-8') as f:
        example_input = json.load(f)
    # 进行预测
    if isinstance(example_input['sign_time'], str):
        sign_time = datetime.strptime(example_input['sign_time'], '%Y-%m-%d %H:%M:%S')
        example_input['sign_time'] = sign_time
    # 进行预测
    if predictor.model is not None:
        result = predictor.predict_with_explanation(example_input)
        
        print(f"调令时间: {sign_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"预测结果: {result['explanation']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"使用的特征数量: {result['features_used']}")
        print(f"开闸概率: {result['probability']:.3f}")
        
        # 根据预测结果给出建议
        if result['prediction'] == 1:
            print("建议: 发布开闸调令")
        else:
            print("建议: 暂不开闸，继续监测")
    else:
        print("模型未加载，请检查模型文件路径")