import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from utils import SafeSimpleImputer
from scipy.stats import randint, uniform
import os
import warnings

# 过滤xgboost中关于pkg_resources的弃用警告
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.",
    category=UserWarning
)

# 设置使用的CPU核心数
n_jobs = max(1, os.cpu_count() // 2)

# 设置字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False


def create_preprocessor():
    """创建预处理管道 - 与train.py保持一致的特征集"""
    numeric_features = [
        # 时间特征
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_of_year',
        
        # 历史操作特征
        'prev_gate_count', 'prev_duration', 'prev_op_hour',
        'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration',
        
        # 潮汐特征
        'tide_24h_mean', 'tide_24h_max', 'tide_24h_min', 'tide_24h_range',
        'tide_24h_slope', 'tide_24h_r_squared', 'tide_24h_cycle_count',
        'tide_24h_rise_rate', 'tide_24h_fall_rate', 'tide_24h_phase',
        'tide_12h_mean', 'tide_12h_max', 'tide_12h_min', 'tide_12h_range',
        'tide_12h_slope', 'tide_12h_r_squared', 'tide_12h_cycle_count',
        'tide_12h_rise_rate', 'tide_12h_fall_rate', 'tide_12h_phase',
        
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
        'is_rush_hour'
    ]
    
    categorical_features = ['tide_type']
    
    # 使用安全的数值型转换器
    numeric_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 使用安全的分类特征处理器
    categorical_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='most_frequent')),
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

def data_quality_check(X, y):
    """数据质量检查（与train.py保持一致）"""
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
    print(f"最小值: {np.min(y):.4f}")
    print(f"最大值: {np.max(y):.4f}")
    print(f"平均值: {np.mean(y):.4f}")
    print(f"标准差: {np.std(y):.4f}")
    print(f"中位数: {np.median(y):.4f}")
    
    # 检查目标变量的异常值
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = y[(y < lower_bound) | (y > upper_bound)]
    print(f"异常值数量: {len(outliers)} ({len(outliers)/len(y)*100:.2f}%)")
    
    return X, y

def train_xgb_model():
    """训练基于XGBoost的开闸时长*孔数回归模型（参考train.py实现）"""
    print("开始训练XGBoost模型...")
    
    # 加载特征数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    
    # 提取日期列（用于后续分析）
    if 'date' in X.columns:
        dates = X['date']
        X = X.drop(columns=['date'])
    else:
        dates = None
    
    # 加载目标数据 - 开闸时长*孔数（第6列，索引5）
    y_targets = np.load('features/features_0822/00_target.npy')
    
    # 确保目标变量是数值类型
    try:
        y = y_targets[:, 5].astype(float)
    except ValueError as e:
        print(f"开闸时长*孔数 数据转换错误: {e}")
        # 处理非数值数据
        valid_targets = []
        for i, value in enumerate(y_targets[:, 5]):
            try:
                valid_targets.append(float(value))
            except ValueError:
                print(f"索引 {i} 的无效开闸时长*孔数值: '{value}'，替换为中位数")
                # 使用非异常值的中位数替换
                valid_values = [float(v) for v in y_targets[:, 5] 
                              if isinstance(v, (int, float, np.number)) or 
                              (isinstance(v, str) and v.replace('.', '').isdigit())]
                median_val = np.median(valid_values) if valid_values else 0
                valid_targets.append(median_val)
        
        y = np.array(valid_targets)
    
    # 数据预处理
    X = preprocess_data(X)
    
    # 数据质量检查
    X, y = data_quality_check(X, y)
    
    # 处理目标变量的异常值 - 使用Winsorization方法
    q1 = np.percentile(y, 5)  # 使用5%和95%分位数
    q3 = np.percentile(y, 95)
    y_processed = np.clip(y, q1, q3)
    
    if np.any(y != y_processed):
        print(f"处理了 {np.sum(y != y_processed)} 个目标变量异常值")
        y = y_processed
    
    # 划分数据集
    # 创建分层变量确保时间分布
    if 'month' in X.columns and 'is_weekend' in X.columns:
        stratify_param = X['month'].astype(str) + '_' + X['is_weekend'].astype(str)
    else:
        stratify_param = None
        print("警告: 无法创建分层变量，使用随机划分")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1122, stratify=stratify_param
    )
    
    # 如果dates存在，也相应划分
    if dates is not None:
        dates_train, dates_test = train_test_split(
            dates, test_size=0.1, random_state=1122, stratify=stratify_param
        )
    else:
        dates_train, dates_test = None, None
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"目标变量范围: {np.min(y):.2f}-{np.max(y):.2f}")
    print(f"目标变量均值: {np.mean(y):.2f}, 标准差: {np.std(y):.2f}")
    
    # 创建预处理管道
    preprocessor = create_preprocessor()
    
    # 创建XGBoost模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            random_state=42,
            n_jobs=n_jobs,
            tree_method='hist',
            predictor='cpu_predictor'
        ))
    ])
    
    # 参数分布设置
    param_dist = {
        'regressor__n_estimators': randint(50, 200),
        'regressor__max_depth': randint(3, 8),
        'regressor__learning_rate': uniform(0.01, 0.3),
        'regressor__subsample': uniform(0.7, 0.3),
        'regressor__colsample_bytree': uniform(0.7, 0.3),
        'regressor__reg_lambda': uniform(0, 2),
        'regressor__reg_alpha': uniform(0, 1)
    }
    
    # 随机搜索
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1,
        random_state=42
    )
    
    print("开始随机搜索寻找最佳XGBoost模型...")
    
    try:
        random_search.fit(X_train, y_train)
        
        print(f"最佳XGB模型: {random_search.best_estimator_}")
        print(f"最佳MAE: {-random_search.best_score_:.4f}")
        
        best_model = random_search.best_estimator_
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        
        # 回归评估指标
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n测试集评估指标:")
        print(f"平均绝对误差(MAE): {mae:.4f}")
        print(f"均方根误差(RMSE): {rmse:.4f}")
        print(f"决定系数(R²): {r2:.4f}")
        
        # 可视化预测结果
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # 添加参考线（y=x）
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('实际开闸时长×孔数')
        plt.ylabel('预测开闸时长×孔数')
        plt.title('XGB模型: 实际值 vs 预测值')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('outputs/kz_comb_log/plots/0822_kz_comb_log_xgb_regression_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 误差分布图
        errors = y_pred - y_test
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('预测误差')
        plt.ylabel('频率')
        plt.title('XGB模型: 开闸时长×孔数预测误差分布')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/kz_comb_log/plots/0822_kz_comb_log_xgb_error_distribution.png', dpi=300, bbox_inches='tight')
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
            
            # 获取XGB模型
            xgb_regressor = best_model.named_steps['regressor']
            
            if hasattr(xgb_regressor, 'feature_importances_'):
                importances = xgb_regressor.feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_df.to_csv('outputs/kz_comb_log/outputs/0822_kz_comb_log_xgb_feature_importance.csv', index=False)
                
                # 可视化最重要的特征
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', 
                            data=importance_df.head(20))
                plt.title('XGB模型: 开闸时长×孔数预测特征重要性排名')
                plt.tight_layout()
                plt.savefig('outputs/kz_comb_log/plots/0822_kz_comb_log_xgb_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"无法获取特征重要性: {str(e)}")
        
        # 保存模型
        os.makedirs('outputs/kz_comb_log/pkls', exist_ok=True)
        joblib.dump(best_model, 'outputs/kz_comb_log/pkls/0822_kz_comb_log_xgb_regression_model.pkl')
        print("XGB模型已保存到 outputs/kz_comb_log/pkls/0822_kz_comb_log_xgb_regression_model.pkl")
        
        # 保存随机搜索结果
        results_df = pd.DataFrame(random_search.cv_results_)
        os.makedirs('outputs/kz_comb_log/outputs', exist_ok=True)
        results_df.to_csv('outputs/kz_comb_log/outputs/0822_kz_comb_log_xgb_regression_random_search_results.csv', index=False)
        
        # 保存预测结果
        if dates_test is not None:
            results_df = pd.DataFrame({
                'date': dates_test.values,
                'true_comb_log': y_test,
                'pred_comb_log': y_pred,
                'error': np.abs(y_test - y_pred)
            })
            
            # 按日期分组计算每日平均
            daily_results = results_df.groupby('date').agg({
                'true_comb_log': 'mean',
                'pred_comb_log': 'mean',
                'error': 'mean'
            }).reset_index()

            daily_results.to_csv('outputs/kz_comb_log/outputs/0822_daily_comb_log_xgb_predictions.csv', index=False)
            print("每日开闸时长×孔数XGB预测结果已保存")
            
            # 可视化每日预测对比
            plt.figure(figsize=(14, 7))
            plt.plot(pd.to_datetime(daily_results['date']), daily_results['true_comb_log'], 'o-', label='真实开闸时长×孔数')
            plt.plot(pd.to_datetime(daily_results['date']), daily_results['pred_comb_log'], 's--', label='预测开闸时长×孔数')
            plt.xlabel('日期')
            plt.ylabel('开闸时长×孔数')
            plt.title('XGB模型: 每日真实开闸时长×孔数 vs 预测开闸时长×孔数')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('outputs/kz_comb_log/plots/0822_daily_comb_log_xgb_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return best_model
        
    except Exception as e:
        print(f"随机搜索失败: {str(e)}")
        print("尝试使用默认参数训练XGB模型...")
        
        # 使用默认参数训练
        default_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                random_state=42,
                n_jobs=n_jobs,
                tree_method='hist',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1
            ))
        ])
        
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"默认XGB模型测试集MAE: {mae:.4f}")
        print(f"默认XGB模型测试集R²: {r2:.4f}")
        
        # 保存默认模型
        joblib.dump(default_model, 'outputs/kz_comb_log/pkls/0822_kz_comb_log_xgb_regression_model_default.pkl')
        print("默认XGB模型已保存")
        
        return default_model

if __name__ == '__main__':
    # 创建输出目录
    os.makedirs('outputs/kz_comb_log/plots', exist_ok=True)
    os.makedirs('outputs/kz_comb_log/pkls', exist_ok=True)
    os.makedirs('outputs/kz_comb_log/outputs', exist_ok=True)
    
    model = train_xgb_model()
    print("XGB模型训练完成！")