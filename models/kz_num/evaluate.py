import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import os
# 设置字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

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

def evaluate_model(model_path, test_data_prefix='07', model_name='model'):
    """评估单个模型"""
    print(f"\n{'='*50}")
    print(f"评估模型: {model_name}")
    print(f"{'='*50}")
    
    # 加载测试数据
    try:
        X_test = pd.read_csv(f'features/features_0822/{test_data_prefix}_features.csv')
        dates_test = X_test['date']
        X_test = X_test.drop(columns=['date'])
        y_test = np.load(f'features/features_0822/{test_data_prefix}_target.npy')[:, 2].astype(int)
    except Exception as e:
        print(f"加载测试数据失败: {str(e)}")
        return None
    
    # 数据预处理
    X_test = preprocess_data(X_test)
    
    # 加载模型
    try:
        model = joblib.load(model_path)
        print(f"模型加载成功: {model_path}")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    print(f"准确率: {accuracy:.4f}")
    print(f"加权F1分数: {f1:.4f}")
    
    # 可视化混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), 
                yticklabels=np.unique(y_test))
    plt.xlabel('预测孔数')
    plt.ylabel('实际孔数')
    plt.title(f'{model_name} - 开闸孔数预测混淆矩阵')
    os.makedirs('outputs/kz_num/plots', exist_ok=True)
    plt.savefig(f'outputs/kz_num/plots/kz_num_{model_name}_confusion_matrix.png')
    plt.close()
    
    # 保存每个样本的结果
    results_df = pd.DataFrame({
        'date': dates_test.values,
        'true_gate_count': y_test * 2,  # 恢复原始孔数
        'pred_gate_count': y_pred * 2
    })
    
    os.makedirs('outputs/kz_num/outputs', exist_ok=True)
    results_df.to_csv(f'outputs/kz_num/outputs/kz_num_{model_name}_sample_predictions.csv', index=False)
    
    # 按日期分组计算每日平均
    daily_results = results_df.groupby('date').agg({
        'true_gate_count': 'mean',
        'pred_gate_count': 'mean'
    }).reset_index()
    
    # 可视化每日预测对比
    plt.figure(figsize=(14, 7))
    plt.plot(pd.to_datetime(daily_results['date']), 
             daily_results['true_gate_count'], 'o-', label='真实开闸孔数')
    plt.plot(pd.to_datetime(daily_results['date']), 
             daily_results['pred_gate_count'], 's--', label='预测开闸孔数')
    plt.xlabel('日期')
    plt.ylabel('开闸孔数')
    plt.title(f'{model_name} - 每日真实开闸孔数 vs 预测开闸孔数')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'outputs/kz_num/plots/kz_num_{model_name}_daily_comparison.png')
    plt.close()
    
    print(f"评估完成，结果已保存\n样本数量: {len(results_df)}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'results_df': results_df,
        'daily_results': daily_results
    }

def evaluate_all_models(test_data_prefix='07'):
    """评估所有模型"""
    # 定义模型路径和名称
    models = {
        'lr': 'outputs/kz_num/pkls/0822_kz_num_lr_classification_model.pkl',
        'mlp': 'outputs/kz_num/pkls/0822_kz_num_mlp_classification_model.pkl',
        'rf': 'outputs/kz_num/pkls/0822_kz_num_rf_classification_model.pkl',
        'svm': 'outputs/kz_num/pkls/0822_kz_num_svm_classification_model.pkl',
        'xgb': 'outputs/kz_num/pkls/0822_kz_num_xgb_classification_model.pkl'
    }
    
    results = {}
    
    for model_key, model_path in models.items():
        model_name = model_key.upper()
        result = evaluate_model(model_path, test_data_prefix, model_name)
        if result is not None:
            results[model_key] = result
    
    # 生成模型比较报告
    if results:
        print("\n" + "="*60)
        print("模型性能比较")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': [results[model]['model_name'] for model in results],
            'Accuracy': [results[model]['accuracy'] for model in results],
            'F1-Score': [results[model]['f1_score'] for model in results]
        }).sort_values('Accuracy', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # 保存比较结果
        comparison_df.to_csv('outputs/kz_num/outputs/model_comparison.csv', index=False)
        
        # 可视化模型比较
        plt.figure(figsize=(12, 6))
        
        # 准确率比较
        plt.subplot(1, 2, 1)
        models_names = [results[model]['model_name'] for model in results]
        accuracies = [results[model]['accuracy'] for model in results]
        bars = plt.bar(models_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.xlabel('模型')
        plt.ylabel('准确率')
        plt.title('模型准确率比较')
        plt.xticks(rotation=45)
        
        # 在柱状图上显示数值
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{accuracy:.3f}', ha='center', va='bottom')
        
        # F1分数比较
        plt.subplot(1, 2, 2)
        f1_scores = [results[model]['f1_score'] for model in results]
        bars = plt.bar(models_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.xlabel('模型')
        plt.ylabel('F1分数')
        plt.title('模型F1分数比较')
        plt.xticks(rotation=45)
        
        # 在柱状图上显示数值
        for bar, f1 in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('outputs/kz_num/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n模型比较结果已保存到 outputs/kz_num/outputs/model_comparison.csv")
        print(f"模型比较图已保存到 outputs/kz_num/plots/model_comparison.png")
    
    return results

if __name__ == '__main__':
    # 评估07测试集
    print("评估07测试集...")
    results_07 = evaluate_all_models('07')
    
    # 评估11测试集
    print("\n\n评估11测试集...")
    results_11 = evaluate_all_models('11')