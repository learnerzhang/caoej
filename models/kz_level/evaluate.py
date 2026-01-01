import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# 设置字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

def load_models():
    """加载所有训练好的模型"""
    model_files = {
        'Linear Regression': 'outputs/kz_level/pkls/0822_kz_level_lr_model.pkl',
        'MLP': 'outputs/kz_level/pkls/0822_kz_level_mlp_model.pkl',
        'Random Forest': 'outputs/kz_level/pkls/0822_kz_level_rf_model.pkl',
        'SVM': 'outputs/kz_level/pkls/0822_kz_level_svm_model.pkl',
        'XGBoost': 'outputs/kz_level/pkls/0822_kz_level_xgb_model.pkl'
    }
    
    models = {}
    for name, filepath in model_files.items():
        try:
            models[name] = joblib.load(filepath)
            print(f"成功加载模型: {name}")
        except Exception as e:
            print(f"加载模型 {name} 失败: {e}")
    
    return models

def preprocess_data(X):
    """数据预处理（与训练时一致）"""
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

def evaluate_models(models, X_test, y_test):
    """评估多个模型"""
    results = {}
    predictions = {}
    
    for name, model in models.items():
        try:
            # 预测
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            
            # 计算指标
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            print(f"评估模型 {name} 时出错: {e}")
            results[name] = None
    
    return results, predictions

def plot_comparison(results, predictions, y_test):
    """绘制模型比较图"""
    # 创建输出目录
    os.makedirs('outputs/kz_level/evaluation', exist_ok=True)
    
    # 1. 性能指标比较
    metrics_df = pd.DataFrame(results).T
    metrics_df = metrics_df[['MAE', 'RMSE', 'R2']]
    
    plt.figure(figsize=(12, 8))
    
    # MAE和RMSE比较
    plt.subplot(2, 2, 1)
    metrics_df[['MAE', 'RMSE']].plot(kind='bar', ax=plt.gca())
    plt.title('模型性能比较 (MAE & RMSE)')
    plt.ylabel('误差值')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # R²比较
    plt.subplot(2, 2, 2)
    metrics_df['R2'].plot(kind='bar', color='green', alpha=0.7)
    plt.title('模型性能比较 (R²)')
    plt.ylabel('R²值')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 预测值散点图
    plt.subplot(2, 2, 3)
    for name, y_pred in predictions.items():
        plt.scatter(y_test, y_pred, alpha=0.6, label=name, s=20)
    
    min_val = min(y_test.min(), min([y_pred.min() for y_pred in predictions.values()]))
    max_val = max(y_test.max(), max([y_pred.max() for y_pred in predictions.values()]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='完美预测')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('各模型预测效果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 误差分布
    plt.subplot(2, 2, 4)
    errors = {}
    for name, y_pred in predictions.items():
        errors[name] = y_pred - y_test
    
    plt.boxplot(errors.values(), labels=errors.keys())
    plt.xticks(rotation=45)
    plt.ylabel('预测误差')
    plt.title('各模型误差分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/kz_level/evaluation/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 排名图
    plt.figure(figsize=(10, 6))
    
    # MAE排名
    mae_rank = metrics_df['MAE'].sort_values()
    plt.subplot(1, 2, 1)
    mae_rank.plot(kind='barh', color='skyblue')
    plt.title('模型MAE排名 (越小越好)')
    plt.xlabel('MAE')
    
    # R²排名
    r2_rank = metrics_df['R2'].sort_values(ascending=False)
    plt.subplot(1, 2, 2)
    r2_rank.plot(kind='barh', color='lightgreen')
    plt.title('模型R²排名 (越大越好)')
    plt.xlabel('R²')
    
    plt.tight_layout()
    plt.savefig('outputs/kz_level/evaluation/model_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics_df

def main():
    """主函数"""
    print("开始加载模型...")
    models = load_models()
    
    if not models:
        print("没有找到可用的模型！")
        return
    
    print(f"成功加载 {len(models)} 个模型")
    
    # 加载测试数据
    print("加载测试数据...")
    try:
        # 加载特征数据
        X_test = pd.read_csv('features/features_0822/00_features.csv')
        if 'date' in X_test.columns:
            X_test = X_test.drop(columns=['date'])
        
        # 加载目标数据
        y_targets = np.load('features/features_0822/00_target.npy')
        y_test = y_targets[:, 3].astype(float)
        
        # 预处理数据
        X_test = preprocess_data(X_test)
        
        # 处理异常值（与训练时一致）
        q1 = np.percentile(y_test, 5)
        q3 = np.percentile(y_test, 95)
        y_test = np.clip(y_test, q1, q3)
        
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        return
    
    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
    
    # 评估模型
    print("\n开始评估模型...")
    results, predictions = evaluate_models(models, X_test, y_test)
    
    # 绘制比较图
    print("\n生成比较图表...")
    metrics_df = plot_comparison(results, predictions, y_test)
    
    # 保存结果
    metrics_df.to_csv('outputs/kz_level/evaluation/model_metrics_comparison.csv')
    print("\n评估结果已保存到 outputs/kz_level/evaluation/")
    
    # 打印最佳模型
    best_mae_model = metrics_df['MAE'].idxmin()
    best_r2_model = metrics_df['R2'].idxmax()
    
    print(f"\n最佳模型 (MAE): {best_mae_model} (MAE: {metrics_df.loc[best_mae_model, 'MAE']:.4f})")
    print(f"最佳模型 (R²): {best_r2_model} (R²: {metrics_df.loc[best_r2_model, 'R2']:.4f})")

if __name__ == '__main__':
    main()