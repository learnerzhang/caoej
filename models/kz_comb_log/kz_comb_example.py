import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from scipy import stats
import logging
import os
from typing import Dict, Any, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义自定义转换器类（与训练脚本保持一致）
class SafeSimpleImputer(BaseEstimator, TransformerMixin):
    """安全的简单填充器，处理全NaN列的情况"""
    
    def __init__(self, strategy='median', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        self.fill_values_ = {}  # 确保在初始化时创建这个属性
        
    def fit(self, X, y=None):
        # 确保fill_values_存在
        if not hasattr(self, 'fill_values_'):
            self.fill_values_ = {}
            
        # 对每列单独处理，避免全NaN列导致的错误
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col in X_df.columns:
            col_data = X_df[col]
            if col_data.isnull().all():
                # 如果整列都是NaN，使用0填充
                self.fill_values_[col] = 0
            else:
                # 使用指定策略填充
                if self.strategy == 'constant' and self.fill_value is not None:
                    self.fill_values_[col] = self.fill_value
                elif self.strategy == 'mean':
                    self.fill_values_[col] = col_data.mean()
                elif self.strategy == 'median':
                    self.fill_values_[col] = col_data.median()
                elif self.strategy == 'most_frequent':
                    self.fill_values_[col] = col_data.mode()[0] if not col_data.mode().empty else 0
                else:
                    self.fill_values_[col] = 0
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # 确保fill_values_存在
        if not hasattr(self, 'fill_values_'):
            self.fill_values_ = {}
            
        for col in X_df.columns:
            if col in self.fill_values_ and X_df[col].isnull().any():
                X_df[col] = X_df[col].fillna(self.fill_values_[col])
            elif X_df[col].isnull().any():
                # 如果列不在fill_values_中但有缺失值，使用中位数填充
                median_val = X_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_df[col] = X_df[col].fillna(median_val)
                
        return X_df.values if not isinstance(X, pd.DataFrame) else X_df

    # 添加get和setstate方法以确保正确序列化
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 确保fill_values_属性存在
        if not hasattr(self, 'fill_values_'):
            self.fill_values_ = {}

class KZCombPredictor:
    """开闸时长×孔数预测器"""
    
    def __init__(self, model_path: str = None):
        """初始化预测器"""
        self.model = None
        self.model_path = model_path or 'outputs/kz_comb_log/pkls/0822_kz_comb_log_regression_model.pkl'
        self.load_model()
    
    def load_model(self) -> bool:
        """加载预训练的回归模型"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"模型文件 {self.model_path} 不存在，尝试加载默认模型...")
                default_path = 'outputs/kz_comb_log/pkls/0822_kz_comb_log_regression_model_default.pkl'
                if not os.path.exists(default_path):
                    raise FileNotFoundError(f"未找到任何模型文件: {self.model_path} 或 {default_path}")
                self.model_path = default_path
            
            # 安全加载模型，传递自定义类
            self.model = joblib.load(self.model_path)
            logger.info(f"成功加载模型: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    @staticmethod
    def extract_tidal_features(water_data: pd.DataFrame) -> Dict[str, float]:
        """提取潮汐特征"""
        features = {
            'tide_24h_mean': 0, 'tide_24h_max': 0, 'tide_24h_min': 0, 'tide_24h_range': 0,
            'tide_24h_slope': 0, 'tide_24h_r_squared': 0, 'tide_24h_cycle_count': 0,
            'tide_24h_rise_rate': 0, 'tide_24h_fall_rate': 0, 'tide_24h_phase': 0,
            'tide_12h_mean': 0, 'tide_12h_max': 0, 'tide_12h_min': 0, 'tide_12h_range': 0,
            'tide_12h_slope': 0, 'tide_12h_r_squared': 0, 'tide_12h_cycle_count': 0,
            'tide_12h_rise_rate': 0, 'tide_12h_fall_rate': 0, 'tide_12h_phase': 0,
            'tide_type': 0
        }
        
        if water_data is None or len(water_data) < 2:
            return features
        
        try:
            # 确保数据按时间排序
            water_data = water_data.sort_values('time')
            
            # 重采样到10分钟间隔
            resampled = water_data.set_index('time').resample('10T').mean().interpolate()
            levels = resampled['water_level'].values
            
            if len(levels) < 2:
                return features
            
            # 基本统计特征
            features['tide_24h_mean'] = np.mean(levels)
            features['tide_24h_max'] = np.max(levels)
            features['tide_24h_min'] = np.min(levels)
            features['tide_24h_range'] = features['tide_24h_max'] - features['tide_24h_min']
            
            # 趋势特征
            time_idx = np.arange(len(levels))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_idx, levels)
            features['tide_24h_slope'] = slope
            features['tide_24h_r_squared'] = r_value**2
            
            # 12小时特征（使用后半段数据）
            if len(levels) >= 72:  # 12小时有72个10分钟间隔
                levels_12h = levels[-72:]
                features['tide_12h_mean'] = np.mean(levels_12h)
                features['tide_12h_max'] = np.max(levels_12h)
                features['tide_12h_min'] = np.min(levels_12h)
                features['tide_12h_range'] = features['tide_12h_max'] - features['tide_12h_min']
                
                time_idx_12h = np.arange(len(levels_12h))
                slope_12h, _, r_value_12h, _, _ = stats.linregress(time_idx_12h, levels_12h)
                features['tide_12h_slope'] = slope_12h
                features['tide_12h_r_squared'] = r_value_12h**2
            
            # 潮汐周期特征
            min_prominence = np.ptp(levels) * 0.2
            peaks, _ = find_peaks(levels, prominence=min_prominence)
            valleys, _ = find_peaks(-levels, prominence=min_prominence)
            
            key_points = sorted(np.concatenate([peaks, valleys]))
            features['tide_24h_cycle_count'] = len(key_points) // 2
            
            # 潮汐类型识别
            if features['tide_24h_cycle_count'] >= 3:
                features['tide_type'] = 1  # 半日潮
            elif features['tide_24h_cycle_count'] == 1:
                features['tide_type'] = 2  # 全日潮
            else:
                features['tide_type'] = 3  # 混合潮
                
        except Exception as e:
            logger.warning(f"潮汐特征提取失败: {e}")
        
        return features
    
    def prepare_input_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """准备预测所需的特征数据"""
        try:
            base_time = input_data['SIGNTM']
            
            # 提取各个数据部分
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
            tide_features = self.extract_tidal_features(water_data)
            feat_dict.update(tide_features)
            
            # 提取未来潮汐特征
            future_tide_features = self.extract_tidal_features(future_water_data)
            future_tide_features_prefixed = {f'future_{k}': v for k, v in future_tide_features.items()}
            feat_dict.update(future_tide_features_prefixed)
            
            # 时间特征
            hour = base_time.hour
            day = base_time.weekday()
            month = base_time.month
            
            feat_dict.update({
                'hour_of_day': hour,
                'day_of_week': day,
                'month': month,
                'is_weekend': 1 if day >= 5 else 0,
                'day_of_year': base_time.timetuple().tm_yday,
                'is_rush_hour': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
            })
            
            # 小时的正弦/余弦变换
            hour_rad = 2 * np.pi * hour / 24
            feat_dict['hour_sin'] = np.sin(hour_rad)
            feat_dict['hour_cos'] = np.cos(hour_rad)
            
            # 缺失值指示器
            feat_dict.update({
                'water_missing': 1 if water_data.empty else 0,
                'flow_missing': 1 if flow_data.empty else 0,
                'rain_missing': 1 if (rain_actual.empty and rain_forecast.empty) else 0,
                'water_status_missing': 1 if water_status.empty else 0
            })
            
            # 流量特征
            if not flow_data.empty:
                cutoff_time = base_time - timedelta(hours=12)
                flow_data_12h = flow_data[flow_data['监测日期'] >= cutoff_time]
                
                if len(flow_data_12h) > 0:
                    flows = flow_data_12h['流量'].values
                    feat_dict.update({
                        'flow_mean': np.mean(flows),
                        'flow_max': np.max(flows),
                        'flow_min': np.min(flows),
                        'flow_range': np.max(flows) - np.min(flows),
                        'flow_var': np.var(flows) if len(flows) > 1 else 0,
                        'flow_skew': stats.skew(flows) if len(flows) > 2 else 0
                    })
            
            # 降雨特征
            rain_actual_total = 0
            if not rain_actual.empty:
                cutoff_time = base_time - timedelta(hours=12)
                rain_actual_12h = rain_actual[rain_actual['监测日期'] >= cutoff_time]
                rain_actual_total = rain_actual_12h['雨量'].sum() if len(rain_actual_12h) > 0 else 0
                feat_dict['rain_actual_avg'] = rain_actual_12h['雨量'].mean() if len(rain_actual_12h) > 0 else 0
            
            rain_forecast_total = 0
            if not rain_forecast.empty:
                cutoff_time = base_time + timedelta(hours=12)
                rain_forecast_12h = rain_forecast[rain_forecast['时间'] <= cutoff_time]
                rain_forecast_total = rain_forecast_12h['雨量'].sum() if len(rain_forecast_12h) > 0 else 0
                feat_dict['rain_forecast_avg'] = rain_forecast_12h['雨量'].mean() if len(rain_forecast_12h) > 0 else 0
            
            feat_dict.update({
                'rain_actual_total': rain_actual_total,
                'rain_forecast_total': rain_forecast_total,
                'rain_change_rate': 0,  # 简化处理
                'water_rain_ratio': feat_dict.get('flow_mean', 0) / (rain_actual_total + 1e-5),
                'flow_rain_ratio': feat_dict.get('flow_mean', 0) / (rain_actual_total + 1e-5)
            })
            
            # 水位工况特征
            if not water_status.empty:
                cutoff_time = base_time - timedelta(hours=12)
                water_status_12h = water_status[water_status['监测日期'] >= cutoff_time]
                
                if len(water_status_12h) > 0:
                    levels = water_status_12h['水位'].values
                    feat_dict.update({
                        'water_status_mean': np.mean(levels),
                        'water_status_max': np.max(levels),
                        'water_status_min': np.min(levels),
                        'water_status_range': np.max(levels) - np.min(levels)
                    })
                    
                    if len(water_status_12h) > 1:
                        time_idx = np.arange(len(levels))
                        slope, _, _, _, _ = stats.linregress(time_idx, levels)
                        feat_dict['water_status_slope'] = slope
            
            # 历史操作特征
            if not prev_orders.empty:
                # 24小时内操作
                cutoff_time_24h = base_time - timedelta(hours=24)
                prev_orders_24h = prev_orders[prev_orders['SIGNTM'] >= cutoff_time_24h]
                
                # 一周内操作
                cutoff_time_week = base_time - timedelta(days=7)
                prev_orders_week = prev_orders[prev_orders['SIGNTM'] >= cutoff_time_week]
                
                feat_dict.update({
                    'prev_gate_count': prev_orders_24h['开闸孔数'].mean() if len(prev_orders_24h) > 0 else 0,
                    'prev_duration': prev_orders_24h['开闸时长'].mean() if len(prev_orders_24h) > 0 else 0,
                    'prev_op_hour': prev_orders_24h['开闸时间'].apply(lambda x: x.hour).mean() if len(prev_orders_24h) > 0 else 0,
                    'ops_week_count': len(prev_orders_week),
                    'ops_week_avg_gates': prev_orders_week['开闸孔数'].mean() if len(prev_orders_week) > 0 else 0,
                    'ops_week_total_duration': prev_orders_week['开闸时长'].sum() if len(prev_orders_week) > 0 else 0
                })
            
            # 创建特征DataFrame
            features_df = pd.DataFrame([feat_dict])
            
            # 确保包含所有训练时使用的特征
            required_features = [
                'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos', 'day_of_year',
                'prev_gate_count', 'prev_duration', 'prev_op_hour', 'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration',
                'tide_24h_mean', 'tide_24h_max', 'tide_24h_min', 'tide_24h_range', 'tide_24h_slope', 'tide_24h_r_squared', 
                'tide_24h_cycle_count', 'tide_24h_rise_rate', 'tide_24h_fall_rate', 'tide_24h_phase',
                'tide_12h_mean', 'tide_12h_max', 'tide_12h_min', 'tide_12h_range', 'tide_12h_slope', 'tide_12h_r_squared',
                'tide_12h_cycle_count', 'tide_12h_rise_rate', 'tide_12h_fall_rate', 'tide_12h_phase',
                'future_tide_mean', 'future_tide_max', 'future_tide_min', 'future_tide_range', 'future_tide_slope',
                'future_tide_r_squared', 'future_tide_cycle_count', 'future_tide_rise_rate', 'future_tide_fall_rate', 'future_tide_phase',
                'flow_mean', 'flow_max', 'flow_min', 'flow_range', 'flow_var', 'flow_skew',
                'rain_actual_total', 'rain_forecast_total', 'rain_actual_avg', 'rain_forecast_avg',
                'rain_change_rate', 'water_rain_ratio', 'flow_rain_ratio',
                'water_status_mean', 'water_status_max', 'water_status_min', 'water_status_range', 'water_status_slope',
                'is_rush_hour', 'tide_type',
                'water_missing', 'flow_missing', 'rain_missing', 'water_status_missing'
            ]
            
            # 添加缺失的特征
            for feature in required_features:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            
            return features_df[required_features]
            
        except Exception as e:
            logger.error(f"特征准备失败: {e}")
            return pd.DataFrame()
    
    def predict(self, input_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """进行预测"""
        if self.model is None:
            logger.error("模型未加载，无法进行预测")
            return None, None
        
        try:
            # 准备特征
            features = self.prepare_input_features(input_data)
            if features.empty:
                logger.error("特征准备失败")
                return None, None
            
            # 进行预测
            predicted_log = self.model.predict(features)[0]
            
            # 还原为实际值
            predicted_comb = np.expm1(predicted_log)
            
            logger.info(f"预测成功: log值={predicted_log:.4f}, 实际值={predicted_comb:.2f}")
            return predicted_log, predicted_comb
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None, None

def create_example_input() -> Dict[str, Any]:
    """创建示例输入数据"""
    sign_time = datetime.now()
    
    return {
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
            '监测日期': [sign_time - timedelta(hours=i) for i in range(12, 0, -1)],
            '流量': np.random.uniform(100, 500, 12)
        }),
        'rain_actual': pd.DataFrame({
            '监测日期': [sign_time - timedelta(hours=i) for i in range(12, 0, -1)],
            '所属区域': ['绍兴平原'] * 12,
            '雨量': np.random.uniform(0, 10, 12)
        }),
        'rain_forecast': pd.DataFrame({
            '时间': [sign_time + timedelta(hours=i) for i in range(1, 13)],
            '所属区域': ['绍兴平原'] * 12,
            '雨量': np.random.uniform(0, 8, 12)
        }),
        'water_status': pd.DataFrame({
            '监测日期': [sign_time - timedelta(hours=i) for i in range(12, 0, -1)],
            '水位': np.random.uniform(2.0, 3.5, 12)
        }),
        'prev_orders': pd.DataFrame({
            'SIGNTM': [sign_time - timedelta(hours=3), sign_time - timedelta(days=1)],
            '开闸时间': [sign_time - timedelta(hours=2.5), sign_time - timedelta(days=1)],
            '开闸时长': [3.5, 4.2],
            '开闸孔数': [12, 20]
        })
    }

def main():
    """主函数示例"""
    logger.info("开闸时长×孔数预测示例")
    
    # 创建预测器
    predictor = KZCombPredictor()
    
    if predictor.model is None:
        logger.error("预测器初始化失败")
        return
    
    # 创建示例输入
    example_input = create_example_input()
    
    # 进行预测
    predicted_log, predicted_comb = predictor.predict(example_input)
    
    if predicted_log is not None and predicted_comb is not None:
        print(f"\n=== 预测结果 ===")
        print(f"调令时间: {example_input['SIGNTM'].strftime('%Y-%m-%d %H:%M')}")
        print(f"预测 log(开闸时长×开孔数量): {predicted_log:.4f}")
        print(f"预测 开闸时长×开孔数量: {predicted_comb:.2f}")
        print(f"示例: 如果开闸时长为{predicted_comb/20:.1f}小时，开孔数为20孔")
        print(f"      或开闸时长为{predicted_comb/15:.1f}小时，开孔数为15孔")
    else:
        print("预测失败")

if __name__ == '__main__':
    main()