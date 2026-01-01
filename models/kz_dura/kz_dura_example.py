import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from scipy import stats
import os

# 模型配置
MODEL_CONFIGS = {
    'lr': {
        'model_path': 'outputs/kz_dura_lr_optimized/pkls/optimized_lr_model.pkl',
        'encoder_path': 'outputs/kz_dura_lr_optimized/pkls/lr_label_encoder.pkl',
        'name': '逻辑回归'
    },
    'mlp': {
        'model_path': 'outputs/kz_dura_class7_mlp/pkls/mlp_classification_model.pkl',
        'encoder_path': 'outputs/kz_dura_class7_mlp/pkls/mlp_label_encoder.pkl',
        'name': 'MLP'
    },
    'rf': {
        'model_path': 'outputs/kz_dura/pkls/rf_optimized_model.pkl',
        'encoder_path': 'outputs/kz_dura/pkls/rf_label_encoder.pkl',
        'name': '随机森林'
    },
    'svm': {
        'model_path': 'outputs/kz_dura_svm_class7/pkls/svm_classification_model.pkl',
        'encoder_path': 'outputs/kz_dura_svm_class7/pkls/svm_label_encoder.pkl',
        'name': 'SVM'
    },
    'xgb': {
        'model_path': 'outputs/kz_dura_xgb_class7/pkls/xgb_classification_model.pkl',
        'encoder_path': 'outputs/kz_dura_xgb_class7/pkls/xgb_label_encoder.pkl',
        'name': 'XGBoost'
    }
}

class MultiModelGateDurationPredictor:
    """多模型开闸时长预测器类"""
    
    def __init__(self, model_types=None):
        self.models = {}
        self.label_encoders = {}
        self.model_types = model_types or list(MODEL_CONFIGS.keys())
        self.load_models()
    
    def load_models(self):
        """加载所有预训练的分类模型和标签编码器"""
        for model_type in self.model_types:
            if model_type in MODEL_CONFIGS:
                config = MODEL_CONFIGS[model_type]
                try:
                    model = joblib.load(config['model_path'])
                    label_encoder = joblib.load(config['encoder_path'])
                    self.models[model_type] = {
                        'model': model,
                        'encoder': label_encoder,
                        'name': config['name']
                    }
                    print(f"✅ {config['name']}模型加载成功")
                except Exception as e:
                    print(f"❌ {config['name']}模型加载失败: {str(e)}")
            else:
                print(f"❌ 未知模型类型: {model_type}")
    
    def prepare_input_features(self, input_data):
        """准备预测所需的特征数据（与训练保持一致）"""
        # 这里需要实现与训练时相同的特征工程逻辑
        # 由于篇幅限制，这里简化实现，实际使用时需要完整实现
        
        base_time = input_data['SIGNTM']
        
        # 创建基础特征字典
        feat_dict = {}
        
        # 时间特征
        hour = base_time.hour
        day = base_time.weekday()
        month = base_time.month
        day_of_year = base_time.timetuple().tm_yday
        
        feat_dict.update({
            'hour_of_day': hour,
            'day_of_week': day,
            'month': month,
            'is_weekend': 1 if day >= 5 else 0,
            'day_of_year': day_of_year,
            'is_rush_hour': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        })
        
        # 小时周期特征
        hour_rad = 2 * np.pi * hour / 24
        feat_dict['hour_sin'] = np.sin(hour_rad)
        feat_dict['hour_cos'] = np.cos(hour_rad)
        
        # 其他特征（简化实现）
        # 实际使用时需要根据具体数据源计算潮汐、流量、降雨等特征
        numeric_features = [
            'prev_gate_count', 'prev_duration', 'prev_op_hour',
            'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration',
            'tide_24h_mean', 'tide_24h_max', 'tide_24h_min', 'tide_24h_range',
            'flow_mean', 'flow_max', 'flow_min', 'flow_range',
            'rain_actual_total', 'rain_forecast_total',
            'water_status_mean', 'water_status_max', 'water_status_min'
        ]
        
        for feature in numeric_features:
            if feature not in feat_dict:
                feat_dict[feature] = 0  # 默认值
        
        # 分类特征
        feat_dict['tide_type'] = 1
        
        # 缺失值指示器
        feat_dict.update({
            'water_missing': 0,
            'flow_missing': 0,
            'rain_missing': 0,
            'water_status_missing': 0
        })
        
        # 创建特征DataFrame
        features_df = pd.DataFrame([feat_dict])
        
        return features_df
    
    def predict_all_models(self, input_data):
        """
        使用所有模型预测开闸时长类别
        :param input_data: 包含所有必要输入数据的字典
        :return: 所有模型的预测结果字典
        """
        if not self.models:
            raise ValueError("没有成功加载任何模型")
        
        # 准备特征
        features = self.prepare_input_features(input_data)
        
        predictions = {}
        
        for model_type, model_info in self.models.items():
            try:
                model = model_info['model']
                label_encoder = model_info['encoder']
                
                # 进行预测
                prediction_encoded = model.predict(features)
                predicted_category = label_encoder.inverse_transform(prediction_encoded)
                
                # 获取概率预测（如果支持）
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)
                    class_labels = label_encoder.classes_
                    probabilities = dict(zip(class_labels, proba[0]))
                
                predictions[model_type] = {
                    'predicted_duration': predicted_category[0],
                    'probabilities': probabilities,
                    'model_name': model_info['name']
                }
                
            except Exception as e:
                print(f"❌ {model_info['name']}模型预测失败: {str(e)}")
                predictions[model_type] = {
                    'predicted_duration': None,
                    'probabilities': None,
                    'model_name': model_info['name'],
                    'error': str(e)
                }
        
        return predictions
    
    def get_model_consensus(self, predictions):
        """
        获取模型共识预测
        :param predictions: 所有模型的预测结果
        :return: 共识预测结果
        """
        valid_predictions = []
        prediction_weights = []
        
        for model_type, pred_info in predictions.items():
            if pred_info['predicted_duration'] is not None:
                valid_predictions.append(pred_info['predicted_duration'])
                # 可以根据模型性能给予不同权重
                prediction_weights.append(1.0)
        
        if not valid_predictions:
            return None
        
        # 使用众数作为共识预测
        from collections import Counter
        counter = Counter(valid_predictions)
        consensus = counter.most_common(1)[0][0]
        
        return consensus
    
    def print_predictions(self, predictions):
        """打印所有模型的预测结果"""
        print("\n📊 多模型预测结果:")
        print("=" * 50)
        
        for model_type, pred_info in predictions.items():
            print(f"\n{pred_info['model_name']}:")
            print(f"  预测开闸时长: {pred_info['predicted_duration']} 小时")
            
            if pred_info['probabilities']:
                print("  各类别概率:")
                for category, prob in sorted(pred_info['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    时长 {category} 小时: {prob:.3f}")
        
        # 显示共识预测
        consensus = self.get_model_consensus(predictions)
        if consensus is not None:
            print(f"\n🎯 模型共识预测: {consensus} 小时")

def predict_with_all_models(input_data, model_types=None):
    """
    主预测函数 - 使用所有模型预测开闸时长类别
    :param input_data: 包含所有必要输入数据的字典
    :param model_types: 要使用的模型类型列表，默认为所有模型
    :return: 所有模型的预测结果
    """
    predictor = MultiModelGateDurationPredictor(model_types)
    return predictor.predict_all_models(input_data)

# 使用示例
if __name__ == '__main__':
    # 创建示例数据
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
            'water_level': np.random.uniform(1.5, 3.5, 12)
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
            '时间': [
                datetime(2023, 8, 26, 9, 0), datetime(2023, 8, 26, 10, 0),
                datetime(2023, 8, 26, 11, 0), datetime(2023, 8, 26, 12, 0),
                datetime(2023, 8, 26, 13, 0), datetime(2023, 8, 26, 14, 0)
            ],
            '所属区域': ['绍兴平原'] * 6,
            '雨量': [3.5, 4.2, 5.0, 4.5, 3.8, 3.0]
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
            '开闸孔数': [3, 4]
        })
    }
    
    try:
        # 使用多模型预测器
        predictor = MultiModelGateDurationPredictor()
        predictions = predictor.predict_all_models(real_input)
        
        # 打印预测结果
        predictor.print_predictions(predictions)
        
    except Exception as e:
        print(f"❌ 预测失败: {str(e)}")