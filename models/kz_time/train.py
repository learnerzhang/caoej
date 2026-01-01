import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")
import os
from utils import SafeSimpleImputer
# 设置使用的CPU核心数
n_jobs = max(1, os.cpu_count() // 2)

# 设置字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

class NormalizerSaver(BaseEstimator, TransformerMixin):
    def __init__(self, model_type='default'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self
        
    def transform(self, X):
        return self.scaler.transform(X)
    
    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

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

def data_quality_check(X, y):
    """数据质量检查"""
    print("=== 数据质量检查 ===")
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    
    # 检查缺失值
    missing_info = pd.DataFrame({
        'feature': X.columns,
        'missing_count': X.isnull().sum(),
        'missing_percent': X.isnull().mean() * 100
    })
    
    print("\n缺失值统计:")
    print(missing_info[missing_info['missing_count'] > 0])
    
    # 检查分类特征的值
    categorical_features = ['tide_type']
    for feature in categorical_features:
        if feature in X.columns:
            print(f"\n{feature} 的值分布:")
            print(X[feature].value_counts())
    
    # 检查目标变量的分布
    print(f"\n目标变量统计:")
    print(f"最小值: {y.min():.2f}, 最大值: {y.max():.2f}, 均值: {y.mean():.2f}, 标准差: {y.std():.2f}")
    
    return X, y

def create_preprocessor(use_normalizer_saver=False):
    """创建预处理管道"""
    numeric_features = [
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
        'is_rush_hour',
        'day_of_year'
    ]
    
    categorical_features = ['tide_type']
    
    # 根据参数选择使用哪种标准化器
    if use_normalizer_saver:
        scaler = NormalizerSaver()
    else:
        scaler = StandardScaler()
    
    # 使用安全的数值型转换器
    numeric_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='median')),
        ('scaler', scaler)
    ])
    
    # 使用安全的分类特征处理器
    categorical_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='constant', fill_value=0)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    indicator_features = [
        'water_missing', 'flow_missing', 
        'rain_missing', 'water_status_missing'
    ]
    
    indicator_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='constant', fill_value=0))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ind', indicator_transformer, indicator_features)
        ])
    
    return preprocessor

def train_and_save_model():
    """训练开闸时间回归模型"""
    # 加载特征数据
    data = pd.read_csv('features/features_0822/00_features.csv')
    # 提取日期列
    dates = data['date']
    X = data.drop(columns=['date'])
    
    # 加载目标数据（只使用开闸时间）
    y_targets = np.load('features/features_0822/00_target.npy')
    y = y_targets[:, 0]  # 只取开闸时间列
    
    # 数据预处理
    X = preprocess_data(X)
    
    # 数据质量检查
    X, y = data_quality_check(X, y)
    
    # 划分数据集 - 使用分层抽样确保时间分布
    # 创建分层变量：基于月份和工作日/周末
    stratify_param = X['month'].astype(str) + '_' + X['is_weekend'].astype(str)
    
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, 
        test_size=0.1, 
        random_state=1122,
        stratify=stratify_param
    )
    
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 优化目标: 开闸时间（小时）的周期性编码
    y_train_sin = np.sin(2 * np.pi * y_train / 24)
    y_train_cos = np.cos(2 * np.pi * y_train / 24)
    y_train_transformed = np.column_stack([y_train_sin, y_train_cos])
    
    # 测试集周期性编码
    y_test_sin = np.sin(2 * np.pi * y_test / 24)
    y_test_cos = np.cos(2 * np.pi * y_test / 24)
    y_test_transformed = np.column_stack([y_test_sin, y_test_cos])

    # 简化参数网格搜索，专注于效果最好的模型
    param_grid = [
        {
            'preprocessor': [create_preprocessor(False)],
            'regressor': [RandomForestRegressor(random_state=42)],
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, None],
        },
        {
            'preprocessor': [create_preprocessor(False)],
            'regressor': [XGBRegressor(random_state=42)],
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [5, 10],
        }
    ]

    # 创建基础模型管道
    base_model = Pipeline(steps=[
        ('preprocessor', create_preprocessor(False)),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,  # 减少交叉验证折数
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    print("开始网格搜索寻找最佳回归模型...")
    
    try:
        grid_search.fit(X_train, y_train_transformed)
        
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        
        # 将预测值转换回小时
        y_pred_hours = np.arctan2(y_pred[:, 0], y_pred[:, 1]) * 24 / (2 * np.pi)
        y_pred_hours = np.mod(y_pred_hours, 24)  # 确保在0-24范围内
        
        # 回归评估指标
        mae = mean_absolute_error(y_test, y_pred_hours)
        mse = mean_squared_error(y_test, y_pred_hours)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_hours)
        
        print(f"\n测试集评估指标:")
        print(f"平均绝对误差(MAE): {mae:.4f}小时")
        print(f"均方根误差(RMSE): {rmse:.4f}小时")
        print(f"决定系数(R²): {r2:.4f}")
        
        # 可视化预测结果
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_hours, alpha=0.5)
        plt.plot([0, 24], [0, 24], 'r--')
        plt.xlabel('实际开闸时间(小时)')
        plt.ylabel('预测开闸时间(小时)')
        plt.title('实际值 vs 预测值')
        plt.grid(True)
        plt.savefig('outputs/kz_time/plots/0822_kz_time_regression_scatter.png')
        plt.close()
        
        # 误差分布图
        errors = y_pred_hours - y_test
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30)
        plt.xlabel('预测误差(小时)')
        plt.ylabel('频率')
        plt.title('预测误差分布')
        plt.grid(True)
        plt.savefig('outputs/kz_time/plots/0822_kz_time_error_distribution.png')
        plt.close()
        
        # 特征重要性分析
        try:
            # 获取特征名称
            preprocessor = best_model.named_steps['preprocessor']
            feature_names = []
            
            # 数值特征
            num_features = preprocessor.transformers_[0][2]
            feature_names.extend(num_features)
            
            # 分类特征
            if len(preprocessor.transformers_) > 1 and preprocessor.transformers_[1][0] == 'cat':
                cat_transformer = preprocessor.transformers_[1][1].named_steps['onehot']
                cat_features = preprocessor.transformers_[1][2]
                cat_names = cat_transformer.get_feature_names_out(cat_features)
                feature_names.extend(cat_names)
            
            # 指示器特征
            if len(preprocessor.transformers_) > 2 and preprocessor.transformers_[2][0] == 'ind':
                ind_features = preprocessor.transformers_[2][2]
                feature_names.extend(ind_features)
            
            # 获取回归器
            regressor = best_model.named_steps['regressor']
            
            # 处理特征重要性
            if hasattr(regressor, 'feature_importances_'):
                importances = regressor.feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_df.to_csv('outputs/kz_time/outputs/0822_kz_time_regression_feature_importance.csv', index=False)
                
                # 可视化最重要的特征
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', 
                            data=importance_df.head(20))
                plt.title('回归模型特征重要性排名')
                plt.tight_layout()
                plt.savefig('outputs/kz_time/plots/0822_kz_time_regression_feature_importance.png')
                plt.close()
                
        except Exception as e:
            print(f"无法获取特征重要性: {str(e)}")
        
        # 保存模型
        os.makedirs('outputs/kz_time/pkls', exist_ok=True)
        joblib.dump(best_model, 'outputs/kz_time/pkls/0822_kz_time_regression_model.pkl')
        print("回归模型已保存到 outputs/kz_time/pkls/0822_kz_time_regression_model.pkl")
        
        # 保存网格搜索结果
        os.makedirs('outputs/kz_time/outputs', exist_ok=True)
        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv('outputs/kz_time/outputs/0822_kz_time_regression_grid_search_results.csv', index=False)
        
        # 保存每天的真实值和预测值
        results_df = pd.DataFrame({
            'date': dates_test.values,
            'true_open_time': y_test,
            'pred_open_time': y_pred_hours,
            'error': np.abs(y_test - y_pred_hours)
        })

        # 按日期分组计算每日平均
        daily_results = results_df.groupby('date').agg({
            'true_open_time': 'mean',
            'pred_open_time': 'mean',
            'error': 'mean'
        }).reset_index()

        # 转换小时数为时间格式
        daily_results['true_time'] = daily_results['true_open_time'].apply(
            lambda x: f"{int(x)}:{int((x % 1) * 60):02d}")
        daily_results['pred_time'] = daily_results['pred_open_time'].apply(
            lambda x: f"{int(x)}:{int((x % 1) * 60):02d}")

        # 保存每日结果
        daily_results[['date', 'true_time', 'pred_time', 'error']].to_csv(
            'outputs/kz_time/outputs/daily_open_time_predictions.csv', index=False)

        print("\n每日开闸时间预测结果已保存到 outputs/kz_time/outputs/daily_open_time_predictions.csv")

        # 可视化每日预测对比
        plt.figure(figsize=(14, 7))
        plt.plot(pd.to_datetime(daily_results['date']), daily_results['true_open_time'], 'o-', label='真实开闸时间')
        plt.plot(pd.to_datetime(daily_results['date']), daily_results['pred_open_time'], 's--', label='预测开闸时间')
        plt.xlabel('日期')
        plt.ylabel('开闸时间(小时)')
        plt.title('每日真实开闸时间 vs 预测开闸时间')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/kz_time/plots/daily_time_comparison.png')
        plt.close()
        
        return best_model
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("尝试使用默认参数训练...")
        
        # 使用默认参数训练
        default_model = Pipeline(steps=[
            ('preprocessor', create_preprocessor(False)),
            ('regressor', RandomForestRegressor(
                random_state=42,
                n_estimators=100,
                max_depth=10
            ))
        ])
        
        default_model.fit(X_train, y_train_transformed)
        y_pred = default_model.predict(X_test)
        
        # 将预测值转换回小时
        y_pred_hours = np.arctan2(y_pred[:, 0], y_pred[:, 1]) * 24 / (2 * np.pi)
        y_pred_hours = np.mod(y_pred_hours, 24)
        
        mae = mean_absolute_error(y_test, y_pred_hours)
        print(f"默认模型测试集MAE: {mae:.4f}小时")
        
        # 保存默认模型
        joblib.dump(default_model, 'outputs/kz_time/pkls/0822_kz_time_regression_model_default.pkl')
        print("默认回归模型已保存")
        
        return default_model

if __name__ == '__main__':
    # 创建输出目录
    os.makedirs('outputs/kz_time/plots', exist_ok=True)
    os.makedirs('outputs/kz_time/pkls', exist_ok=True)
    os.makedirs('outputs/kz_time/outputs', exist_ok=True)
    
    model = train_and_save_model()
    print("训练完成！")