import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# 设置字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

def load_all_models():
    """加载所有模型"""
    model_paths = {
        'Linear Regression': 'outputs/kz_level/pkls/0822_kz_level_lr_model.pkl',
        'MLP': 'outputs/kz_level/pkls/0822_kz_level_mlp_model.pkl', 
        'Random Forest': 'outputs/kz_level/pkls/0822_kz_level_rf_model.pkl',
        'SVM': 'outputs/kz_level/pkls/0822_kz_level_svm_model.pkl',
        'XGBoost': 'outputs/kz_level/pkls/0822_kz_level_xgb_model.pkl'
    }
    
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = joblib.load(path)
            print(f"✓ 已加载: {name}")
        except Exception as e:
            print(f"✗ 加载失败 {name}: {e}")
    
    return models

def preprocess_input_data(X):
    """预处理输入数据（与训练时一致）"""
    X_processed = X.copy()
    
    # 处理数值特征的缺失值
    numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if X_processed[col].isnull().any():
            median_val = X_processed[col].median()
            if pd.isna(median_val):
                median_val = 0
            X_processed[col].fillna(median_val, inplace=True)
    
    # 处理分类特征的缺失值
    categorical_columns = ['tide_type']
    for col in categorical_columns:
        if col in X_processed.columns and X_processed[col].isnull().any():
            mode_val = X_processed[col].mode()
            if len(mode_val) > 0:
                X_processed[col].fillna(mode_val[0], inplace=True)
            else:
                X_processed[col].fillna(0, inplace=True)
    
    return X_processed

def create_sample_data():
    """创建示例数据"""
    # 这里创建一个示例数据点，实际使用时应该替换为真实数据
    sample_data = {
        'hour_of_day': 14,
        'day_of_week': 3,
        'month': 8,
        'is_weekend': 0,
        'hour_sin': 0.5,
        'hour_cos': -0.866,
        'day_of_year': 234,
        'prev_gate_count': 2,
        'prev_duration': 30,
        'prev_op_hour': 1,
        'ops_week_count': 8,
        'ops_week_avg_gates': 2.5,
        'ops_week_total_duration': 240,
        'tide_24h_mean': 2.1,
        'tide_24h_max': 3.2,
        'tide_24h_min': 1.0,
        'tide_24h_range': 2.2,
        'tide_24h_slope': 0.02,
        'tide_24h_r_squared': 0.85,
        'tide_24h_cycle_count': 2,
        'tide_24h_rise_rate': 0.1,
        'tide_24h_fall_rate': -0.08,
        'tide_24h_phase': 0.75,
        'tide_12h_mean': 2.2,
        'tide_12h_max': 3.0,
        'tide_12h_min': 1.4,
        'tide_12h_range': 1.6,
        'tide_12h_slope': 0.03,
        'tide_12h_r_squared': 0.78,
        'tide_12h_cycle_count': 1,
        'tide_12h_rise_rate': 0.12,
        'tide_12h_fall_rate': -0.09,
        'tide_12h_phase': 0.6,
        'future_tide_mean': 2.3,
        'future_tide_max': 3.1,
        'future_tide_min': 1.5,
        'future_tide_range': 1.6,
        'future_tide_slope': 0.01,
        'future_tide_r_squared': 0.82,
        'future_tide_cycle_count': 1,
        'future_tide_rise_rate': 0.08,
        'future_tide_fall_rate': -0.07,
        'future_tide_phase': 0.7,
        'flow_mean': 1500,
        'flow_max': 1800,
        'flow_min': 1200,
        'flow_range': 600,
        'flow_var': 50000,
        'flow_skew': 0.2,
        'rain_actual_total': 5.2,
        'rain_forecast_total': 4.8,
        'rain_actual_avg': 0.2,
        'rain_forecast_avg': 0.18,
        'rain_change_rate': 0.1,
        'water_rain_ratio': 10.5,
        'flow_rain_ratio': 288.5,
        'water_status_mean': 2.8,
        'water_status_max': 3.5,
        'water_status_min': 2.1,
        'water_status_range': 1.4,
        'water_status_slope': 0.005,
        'is_rush_hour': 0,
        'tide_type': 1,
        'water_missing': 0,
        'flow_missing': 0,
        'rain_missing': 0,
        'water_status_missing': 0
    }
    
    return pd.DataFrame([sample_data])

def predict_with_all_models(models, X):
    """使用所有模型进行预测"""
    predictions = {}
    
    for name, model in models.items():
        try:
            pred = model.predict(X)[0]
            predictions[name] = pred
            print(f"{name}: {pred:.4f}米")
        except Exception as e:
            print(f"{name} 预测失败: {e}")
            predictions[name] = None
    
    return predictions

def plot_predictions_comparison(predictions):
    """绘制多模型预测比较图"""
    # 创建输出目录
    os.makedirs('outputs/kz_level/example', exist_ok=True)
    
    # 过滤掉预测失败的结果
    valid_predictions = {k: v for k, v in predictions.items() if v is not None}
    
    if not valid_predictions:
        print("没有有效的预测结果可展示")
        return
    
    # 创建比较图
    plt.figure(figsize=(12, 8))
    
    # 1. 柱状图比较
    plt.subplot(2, 2, 1)
    names = list(valid_predictions.keys())
    values = list(valid_predictions.values())
    
    bars = plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.title('各模型目标水位预测结果')
    plt.ylabel('预测水位 (米)')
    plt.xticks(rotation=45)
    
    # 在柱子上添加数值
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    
    # 2. 雷达图比较（相对性能）
    plt.subplot(2, 2, 2)
    if len(valid_predictions) >= 3:
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        
        if range_val > 0:
            normalized_values = [(v - min_val) / range_val for v in values]
        else:
            normalized_values = [0.5] * len(values)
        
        # 雷达图
        angles = np.linspace(0, 2*np.pi, len(names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        normalized_values += normalized_values[:1]
        
        ax = plt.subplot(2, 2, 2, polar=True)
        ax.plot(angles, normalized_values, 'o-', linewidth=2)
        ax.fill(angles, normalized_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names)
        ax.set_title('模型预测相对比较')
    
    # 3. 箱线图样式展示
    plt.subplot(2, 2, 3)
    plt.boxplot(values, labels=['所有模型'])
    plt.title('预测值分布')
    plt.ylabel('水位 (米)')
    
    # 添加具体数值点
    for i, (name, value) in enumerate(valid_predictions.items(), 1):
        plt.scatter(1, value, label=name, s=100, alpha=0.7)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 预测一致性分析
    plt.subplot(2, 2, 4)
    avg_pred = np.mean(values)
    std_pred = np.std(values)
    
    plt.axhline(y=avg_pred, color='r', linestyle='-', label=f'平均值: {avg_pred:.3f}米')
    plt.axhline(y=avg_pred + std_pred, color='r', linestyle='--', alpha=0.5, label=f'±标准差')
    plt.axhline(y=avg_pred - std_pred, color='r', linestyle='--', alpha=0.5)
    
    for i, (name, value) in enumerate(valid_predictions.items()):
        plt.scatter(i, value, s=100, label=name)
        plt.text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom')
    
    plt.xticks(range(len(valid_predictions)), list(valid_predictions.keys()), rotation=45)
    plt.ylabel('预测水位 (米)')
    plt.title('模型预测一致性分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/kz_level/example/multi_model_prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存预测结果
    result_df = pd.DataFrame(list(valid_predictions.items()), columns=['Model', 'Prediction'])
    result_df['Average'] = avg_pred
    result_df['Std'] = std_pred
    result_df.to_csv('outputs/kz_level/example/multi_model_predictions.csv', index=False)
    
    return valid_predictions, avg_pred, std_pred

def main():
    """主函数"""
    print("=== 多模型预测示例 ===\n")
    
    # 1. 加载所有模型
    print("1. 加载模型...")
    models = load_all_models()
    
    if not models:
        print("没有可用的模型！")
        return
    
    print(f"成功加载 {len(models)} 个模型\n")
    
    # 2. 准备输入数据
    print("2. 准备输入数据...")
    sample_data = create_sample_data()
    print(f"输入特征数量: {sample_data.shape[1]}")
    print("数据预处理...")
    processed_data = preprocess_input_data(sample_data)
    
    # 3. 进行预测
    print("\n3. 各模型预测结果:")
    print("-" * 40)
    predictions = predict_with_all_models(models, processed_data)
    
    # 4. 可视化比较
    print("\n4. 生成预测比较图...")
    valid_predictions, avg_pred, std_pred = plot_predictions_comparison(predictions)
    
    # 5. 总结
    print("\n5. 预测总结:")
    print("-" * 40)
    print(f"有效模型数量: {len(valid_predictions)}")
    print(f"预测平均值: {avg_pred:.4f}米")
    print(f"预测标准差: {std_pred:.4f}米")
    print(f"预测范围: {min(valid_predictions.values()):.4f} - {max(valid_predictions.values()):.4f}米")
    
    # 推荐结果
    if std_pred < 0.1:  # 如果标准差很小，说明模型一致性高
        print(f"\n🎯 推荐结果: {avg_pred:.4f}米 (模型一致性高)")
    else:
        best_model = min(valid_predictions.items(), key=lambda x: abs(x[1] - avg_pred))
        print(f"\n🎯 推荐结果: {best_model[1]:.4f}米 (来自 {best_model[0]}, 最接近平均值)")
    
    print(f"\n📊 详细结果已保存到: outputs/kz_level/example/")

if __name__ == '__main__':
    main()