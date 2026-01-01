import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import os
from utils import SafeSimpleImputer
import warnings
warnings.filterwarnings("ignore")

# 设置字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

def preprocess_data(X):
    """数据预处理：处理缺失值和异常值（与train.py保持一致）"""
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

def convert_predictions(y_pred):
    """将预测值转换回小时"""
    y_pred_hours = np.arctan2(y_pred[:, 0], y_pred[:, 1]) * 24 / (2 * np.pi)
    y_pred_hours = np.mod(y_pred_hours, 24)
    return y_pred_hours

def evaluate_single_model(model_path, features_path, target_path, output_suffix="eval"):
    """
    评估单个模型性能
    """
    # 确保输出目录存在
    os.makedirs('outputs/kz_time/plots', exist_ok=True)
    os.makedirs('outputs/kz_time/outputs', exist_ok=True)
    
    print(f"正在评估模型: {model_path}")
    print(f"特征数据: {features_path}")
    print(f"目标数据: {target_path}")
    
    # 加载特征数据
    data = pd.read_csv(features_path)
    
    # 提取日期列（如果存在）
    if 'date' in data.columns:
        dates = data['date']
        X = data.drop(columns=['date'])
    else:
        dates = pd.Series([f"eval_{i}" for i in range(len(data))])
        X = data.copy()
    
    # 加载目标数据
    y_targets = np.load(target_path)
    y_true = y_targets[:, 0]  # 只取开闸时间列
    
    # 数据预处理
    X = preprocess_data(X)
    
    print(f"数据形状: X={X.shape}, y={y_true.shape}")
    
    # 加载模型
    try:
        model = joblib.load(model_path)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None
    
    # 进行预测
    y_pred_transformed = model.predict(X)
    y_pred_hours = convert_predictions(y_pred_transformed)
    
    # 计算评估指标
    mae = mean_absolute_error(y_true, y_pred_hours)
    mse = mean_squared_error(y_true, y_pred_hours)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred_hours)
    
    print(f"\n=== 评估结果 ===")
    print(f"平均绝对误差(MAE): {mae:.4f}小时")
    print(f"均方误差(MSE): {mse:.4f}小时²")
    print(f"均方根误差(RMSE): {rmse:.4f}小时")
    print(f"决定系数(R²): {r2:.4f}")
    
    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred_hours, alpha=0.6, s=50)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('实际开闸时间(小时)')
    plt.ylabel('预测开闸时间(小时)')
    plt.title(f'开闸时间预测: 实际值 vs 预测值 ({output_suffix})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/kz_time/plots/kz_time_{output_suffix}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 误差分布图
    errors = y_pred_hours - y_true
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', label='零误差线')
    plt.xlabel('预测误差(小时)')
    plt.ylabel('频率')
    plt.title(f'预测误差分布 ({output_suffix})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/kz_time/plots/kz_time_{output_suffix}_error_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存详细结果
    results_df = pd.DataFrame({
        '实际开闸时间': y_true,
        '预测开闸时间': y_pred_hours,
        '绝对误差': np.abs(y_pred_hours - y_true),
        '相对误差': np.abs(y_pred_hours - y_true) / (y_true + 1e-8)
    })
    
    # 添加日期信息（如果可用）
    if 'date' in data.columns:
        results_df['日期'] = dates.values
    
    # 保存结果到CSV
    results_df.to_csv(f'outputs/kz_time/outputs/kz_time_{output_suffix}_detailed_results.csv', index=False, encoding='utf-8-sig')
    
    # 保存评估指标摘要
    metrics_summary = pd.DataFrame({
        '指标': ['MAE', 'MSE', 'RMSE', 'R²'],
        '值': [mae, mse, rmse, r2],
        '说明': ['平均绝对误差(小时)', '均方误差(小时²)', '均方根误差(小时)', '决定系数']
    })
    
    metrics_summary.to_csv(f'outputs/kz_time/outputs/kz_time_{output_suffix}_metrics.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n评估完成！结果已保存到 outputs/kz_time/outputs/")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred_hours,
        'true_values': y_true
    }

def evaluate_all_models(datasets, model_types=['lr', 'mlp', 'rf', 'svm', 'xgb']):
    """评估所有模型在多个数据集上的性能"""
    
    all_results = {}
    
    for dataset in datasets:
        dataset_name = dataset['name']
        features_path = dataset['features_path']
        target_path = dataset['target_path']
        
        print(f"\n{'='*50}")
        print(f"评估数据集: {dataset_name}")
        print(f"{'='*50}")
        
        # 检查数据文件是否存在
        if not os.path.exists(features_path):
            print(f"警告: 特征文件不存在 - {features_path}")
            continue
            
        if not os.path.exists(target_path):
            print(f"警告: 目标文件不存在 - {target_path}")
            continue
        
        dataset_results = {}
        
        for model_type in model_types:
            model_path = f'outputs/kz_time/pkls/0822_kz_time_{model_type}_regression_model.pkl'
            
            # 如果主模型不存在，尝试加载默认模型
            if not os.path.exists(model_path):
                model_path = f'outputs/kz_time/pkls/0822_kz_time_{model_type}_regression_model_default.pkl'
                if not os.path.exists(model_path):
                    print(f"警告: {model_type.upper()}模型文件不存在")
                    continue
            
            output_suffix = f"{dataset_name}_{model_type}"
            
            # 评估模型
            result = evaluate_single_model(
                model_path=model_path,
                features_path=features_path,
                target_path=target_path,
                output_suffix=output_suffix
            )
            
            if result is not None:
                dataset_results[model_type.upper()] = result
        
        all_results[dataset_name] = dataset_results
    
    # 生成综合比较报告
    if len(all_results) > 0:
        generate_comprehensive_report(all_results)
    
    return all_results

def generate_comprehensive_report(all_results):
    """生成综合比较报告"""
    print(f"\n{'='*60}")
    print("模型在不同数据集上的性能比较")
    print(f"{'='*60}")
    
    comparison_data = []
    
    for dataset_name, model_results in all_results.items():
        for model_name, metrics in model_results.items():
            comparison_data.append({
                '数据集': dataset_name,
                '模型': model_name,
                'MAE(小时)': f"{metrics['mae']:.4f}",
                'RMSE(小时)': f"{metrics['rmse']:.4f}", 
                'R²': f"{metrics['r2']:.4f}"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n性能比较表:")
    print(comparison_df.to_string(index=False))
    
    # 保存比较结果
    comparison_df.to_csv('outputs/kz_time/outputs/all_models_performance_comparison.csv', 
                       index=False, encoding='utf-8-sig')
    
    # 可视化比较结果
    plt.figure(figsize=(15, 10))
    
    # 为每个数据集创建子图
    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    
    for i, dataset_name in enumerate(datasets):
        model_results = all_results[dataset_name]
        
        if not model_results:
            continue
            
        model_names = list(model_results.keys())
        mae_values = [model_results[name]['mae'] for name in model_names]
        rmse_values = [model_results[name]['rmse'] for name in model_names]
        r2_values = [model_results[name]['r2'] for name in model_names]
        
        # MAE比较
        plt.subplot(n_datasets, 3, i*3 + 1)
        plt.bar(model_names, mae_values, alpha=0.7)
        plt.title(f'{dataset_name} - MAE比较')
        plt.ylabel('MAE (小时)')
        plt.xticks(rotation=45)
        
        # RMSE比较
        plt.subplot(n_datasets, 3, i*3 + 2)
        plt.bar(model_names, rmse_values, alpha=0.7)
        plt.title(f'{dataset_name} - RMSE比较')
        plt.ylabel('RMSE (小时)')
        plt.xticks(rotation=45)
        
        # R²比较
        plt.subplot(n_datasets, 3, i*3 + 3)
        plt.bar(model_names, r2_values, alpha=0.7)
        plt.title(f'{dataset_name} - R²比较')
        plt.ylabel('R²')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/kz_time/plots/all_models_performance_comparison.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n性能比较图表已保存到: outputs/kz_time/plots/all_models_performance_comparison.png")

def evaluate_multiple_datasets():
    """评估多个数据集"""
    
    # 定义要评估的数据集
    datasets = [
        {
            'name': '07测试集',
            'features_path': 'features/features_0822/07_features.csv',
            'target_path': 'features/features_0822/07_target.npy'
        },
        {
            'name': '11测试集', 
            'features_path': 'features/features_0822/11_features.csv',
            'target_path': 'features/features_0822/11_target.npy'
        }
    ]
    
    # 评估所有模型
    all_results = evaluate_all_models(datasets)
    
    return all_results

if __name__ == '__main__':
    print("开始评估所有开闸时间预测模型...")
    results = evaluate_multiple_datasets()
    print("\n所有模型评估完成！")