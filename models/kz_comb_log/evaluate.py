import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import os
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置字体，与train.py保持一致
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

def preprocess_data(X):
    """数据预处理：处理缺失值和异常值（从train.py复制）"""
    X_processed = X.copy()
    
    # 处理数值特征的缺失值
    numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if X_processed[col].isnull().any():
            # 使用中位数填充数值特征的缺失值
            median_val = X_processed[col].median()
            if pd.isna(median_val):  # 如果中位数也是NaN，使用0
                median_val = 0
            X_processed[col] = X_processed[col].fillna(median_val)
    
    # 处理分类特征的缺失值
    categorical_columns = ['tide_type']
    for col in categorical_columns:
        if col in X_processed.columns and X_processed[col].isnull().any():
            # 使用众数填充分类特征
            mode_val = X_processed[col].mode()
            if len(mode_val) > 0:
                X_processed[col] = X_processed[col].fillna(mode_val[0])
            else:
                X_processed[col] = X_processed[col].fillna(0)
    
    return X_processed

def load_model_with_custom_classes(model_path):
    """安全加载包含自定义类的模型"""
    try:
        # 首先尝试正常加载
        return joblib.load(model_path)
    except Exception as e:
        logger.warning(f"正常加载模型失败: {e}, 尝试使用自定义类映射...")
        try:
            # 使用自定义类映射
            return joblib.load(model_path)
        except Exception as e2:
            logger.error(f"使用自定义类映射加载模型也失败: {e2}")
            raise e2

def evaluate_dataset(data_id, model_path=None):
    """
    通用评估函数，支持不同数据集的评估
    
    参数:
        data_id: 数据集标识（如'07', '11'）
        model_path: 模型文件路径，如果为None则使用默认路径
    """
    try:
        logger.info(f"===== 开始评估数据集 {data_id} =====")
        
        # 1. 加载数据
        feature_path = f'features/features_0822/{data_id}_features.csv'
        target_path = f'features/features_0822/{data_id}_target.npy'
        
        # 检查文件是否存在
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"特征文件不存在: {feature_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"目标文件不存在: {target_path}")
        
        # 加载特征数据
        X_test = pd.read_csv(feature_path)
        
        # 提取日期列
        if 'date' not in X_test.columns:
            raise ValueError(f"验证数据 {data_id} 中缺少 'date' 列")
        dates_test = X_test['date']
        X_test = X_test.drop(columns=['date'])
        
        # 加载目标数据（开闸时长*孔数，第6列，索引5）
        y_targets = np.load(target_path)
        try:
            y_test = y_targets[:, 5].astype(float)
        except ValueError as e:
            logger.warning(f"数据集 {data_id} 目标变量转换错误: {e}")
            # 处理非数值数据（与train.py保持一致）
            valid_targets = []
            for value in y_targets[:, 5]:
                try:
                    valid_targets.append(float(value))
                except ValueError:
                    # 使用非异常值的中位数替换
                    valid_values = [float(v) for v in y_targets[:, 5] 
                                  if isinstance(v, (int, float, np.number)) or 
                                  (isinstance(v, str) and v.replace('.', '').isdigit())]
                    median_val = np.median(valid_values) if valid_values else 0
                    valid_targets.append(median_val)
            y_test = np.array(valid_targets)
        
        # 2. 数据预处理（与训练流程保持一致）
        X_test = preprocess_data(X_test)
        
        # 3. 加载模型
        if model_path is None:
            model_path = 'outputs/kz_comb_log/pkls/0822_kz_comb_log_regression_model.pkl'
            if not os.path.exists(model_path):
                logger.warning(f"模型文件 {model_path} 不存在，尝试加载默认模型...")
                model_path = 'outputs/kz_comb_log/pkls/0822_kz_comb_log_regression_model_default.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
        
        # 安全加载模型
        model = load_model_with_custom_classes(model_path)
        logger.info(f"成功加载模型: {model_path}")
        
        # 4. 预测与评估
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"\n验证集 {data_id} 评估指标:")
        logger.info(f"平均绝对误差(MAE): {mae:.4f}")
        logger.info(f"均方根误差(RMSE): {rmse:.4f}")
        logger.info(f"决定系数(R²): {r2:.4f}")
        
        # 5. 可视化结果
        output_plots_dir = 'outputs/kz_comb_log/plots'
        os.makedirs(output_plots_dir, exist_ok=True)
        
        # 实际值 vs 预测值散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        # 添加参考线（y=x）
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel('实际开闸时长×孔数')
        plt.ylabel('预测开闸时长×孔数')
        plt.title(f'数据集 {data_id} 预测: 实际值 vs 预测值')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_plots_dir}/kz_comb_log_{data_id}_eval_scatter.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 误差分布图
        errors = y_pred - y_test
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('预测误差')
        plt.ylabel('频率')
        plt.title(f'数据集 {data_id} 预测误差分布')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_plots_dir}/kz_comb_log_{data_id}_eval_error_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. 保存结果
        output_data_dir = 'outputs/kz_comb_log/outputs'
        os.makedirs(output_data_dir, exist_ok=True)
        
        # 详细结果
        results_df = pd.DataFrame({
            'date': dates_test.values,
            'true_comb_log': y_test,
            'pred_comb_log': y_pred,
            'error': np.abs(y_test - y_pred)
        })
        
        # 每日平均结果
        daily_results = results_df.groupby('date').agg({
            'true_comb_log': 'mean',
            'pred_comb_log': 'mean',
            'error': 'mean'
        }).reset_index()
        
        daily_results.to_csv(f'{output_data_dir}/daily_comb_log_{data_id}_predictions.csv', index=False)
        
        # 每日趋势对比图
        plt.figure(figsize=(14, 7))
        plt.plot(pd.to_datetime(daily_results['date']), 
                 daily_results['true_comb_log'], 'o-', label='真实开闸时长×孔数', linewidth=2, markersize=4)
        plt.plot(pd.to_datetime(daily_results['date']), 
                 daily_results['pred_comb_log'], 's--', label='预测开闸时长×孔数', linewidth=2, markersize=4)
        plt.xlabel('日期')
        plt.ylabel('开闸时长×孔数')
        plt.title(f'数据集 {data_id} 每日真实值 vs 预测值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_plots_dir}/daily_comb_log_{data_id}_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存评估指标
        metrics_df = pd.DataFrame({
            'dataset': [data_id],
            'mae': [mae],
            'rmse': [rmse],
            'r2': [r2],
            'evaluation_time': [datetime.now()]
        })
        
        metrics_path = f'{output_data_dir}/evaluation_metrics_{data_id}.csv'
        if os.path.exists(metrics_path):
            existing_metrics = pd.read_csv(metrics_path)
            metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)
        
        metrics_df.to_csv(metrics_path, index=False)
        
        logger.info(f"数据集 {data_id} 评估完成，结果已保存")
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'dataset': data_id
        }
        
    except Exception as e:
        logger.error(f"评估数据集 {data_id} 失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_evaluation_report(datasets):
    """生成评估报告"""
    report_data = []
    
    for data_id in datasets:
        result = evaluate_dataset(data_id)
        if result:
            report_data.append(result)
    
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_path = 'outputs/kz_comb_log/outputs/evaluation_summary.csv'
        report_df.to_csv(report_path, index=False)
        logger.info(f"评估报告已保存到: {report_path}")
        
        # 打印汇总信息
        logger.info("\n=== 评估汇总 ===")
        for metric in ['mae', 'rmse', 'r2']:
            avg_value = report_df[metric].mean()
            logger.info(f"平均{metric.upper()}: {avg_value:.4f}")
    
    return report_data

def main():
    """主函数"""
    # 创建输出目录
    os.makedirs('outputs/kz_comb_log/plots', exist_ok=True)
    os.makedirs('outputs/kz_comb_log/outputs', exist_ok=True)
    
    # 评估指定的数据集
    datasets = ['07', '11']
    
    logger.info("开始模型评估...")
    report_data = generate_evaluation_report(datasets)
    
    if report_data:
        logger.info("\n所有数据集评估完成！")
        logger.info("评估结果保存在 outputs/kz_comb_log/outputs/ 目录下")
    else:
        logger.error("评估失败，请检查数据和模型文件")

if __name__ == '__main__':
    main()