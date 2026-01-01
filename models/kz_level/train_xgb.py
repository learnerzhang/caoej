import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import xgboost as xgb
import os
from sklearn.base import BaseEstimator, TransformerMixin
from utils import SafeSimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 设置使用的CPU核心数
n_jobs = max(1, os.cpu_count() - 1)  # 保留一个核心给系统

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

def create_preprocessor():
    """创建预处理管道 - 优化性能的版本"""
    # 精简特征集，移除相关性较低的特征
    numeric_features = [
        # 时间特征
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_of_year',
        
        # 历史操作特征
        'prev_gate_count', 'prev_duration', 'prev_op_hour',
        'ops_week_count', 'ops_week_avg_gates',
        
        # 潮汐特征 - 只保留关键特征
        'tide_24h_mean', 'tide_24h_max', 'tide_24h_min', 
        'tide_24h_phase', 'tide_12h_mean', 'tide_12h_phase',
        
        # 未来潮汐特征
        'future_tide_mean', 'future_tide_max', 'future_tide_min',
        
        # 流量特征
        'flow_mean', 'flow_max', 'flow_min',
        
        # 降雨特征
        'rain_actual_total', 'rain_forecast_total',
        
        # 水位工况特征
        'water_status_mean', 'water_status_max', 'water_status_min',
        
        # 其他特征
        'is_rush_hour'
    ]
    
    categorical_features = ['tide_type']
    
    indicator_features = [
        'water_missing', 'flow_missing', 
        'rain_missing', 'water_status_missing'
    ]
    
    # 对于XGBoost，不需要对数值特征进行标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='median'))
        # 移除StandardScaler以加速XGBoost训练
    ])
    
    # 使用安全的分类特征处理器
    categorical_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='constant', fill_value=0)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first'))  # 减少维度
    ])
    
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

def train_regression_model():
    """训练XGBoost回归模型 - 优化版本"""
    # 创建输出目录
    os.makedirs('outputs/kz_level/plots', exist_ok=True)
    os.makedirs('outputs/kz_level/pkls', exist_ok=True)
    os.makedirs('outputs/kz_level/outputs', exist_ok=True)
    
    print("开始加载数据...")
    
    # 加载特征数据
    data = pd.read_csv('features/features_0822/00_features.csv')
    
    # 提取日期列
    if 'date' in data.columns:
        dates = data['date']
        X = data.drop(columns=['date'])
    else:
        dates = None
        X = data
    
    # 加载目标数据（只使用目标水位）
    y_targets = np.load('features/features_0822/00_target.npy')
    
    # 确保目标水位是数值类型
    try:
        y = y_targets[:, 3].astype(np.float32)  # 使用float32加速
    except ValueError as e:
        print(f"目标水位数据转换错误: {e}")
        # 处理非数值数据
        valid_targets = []
        for i, value in enumerate(y_targets[:, 3]):
            try:
                valid_targets.append(float(value))
            except ValueError:
                print(f"索引 {i} 的无效目标水位值: '{value}'，替换为中位数")
                # 使用非异常值的中位数替换
                valid_values = [float(v) for v in y_targets[:, 3] 
                              if isinstance(v, (int, float, np.number)) or 
                              (isinstance(v, str) and v.replace('.', '').isdigit())]
                median_val = np.median(valid_values) if valid_values else 0
                valid_targets.append(median_val)
        
        y = np.array(valid_targets, dtype=np.float32)
    
    # 转换特征数据类型为float32以加速训练
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].astype(np.float32)
    
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
    
    # 划分数据集 - 使用分层抽样确保时间分布
    if 'month' in X.columns and 'is_weekend' in X.columns:
        stratify_param = X['month'].astype(str) + '_' + X['is_weekend'].astype(str)
    else:
        stratify_param = None
        print("警告: 无法创建分层变量，使用随机划分")
    
    if dates is not None:
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, 
            test_size=0.1, 
            random_state=1122,
            stratify=stratify_param
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.1, 
            random_state=1122,
            stratify=stratify_param
        )
        dates_train, dates_test = None, None
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"目标水位范围: {np.min(y)}-{np.max(y)}米")
    print(f"目标水位均值: {np.mean(y):.2f}米, 标准差: {np.std(y):.2f}米")

    # 创建预处理管道
    preprocessor = create_preprocessor()
    
    # 创建XGBoost回归模型管道 - 优化参数
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            random_state=42,
            n_jobs=n_jobs,
            tree_method='hist',  # 使用更快的直方图算法
            predictor='cpu_predictor',  # 使用CPU预测器
            enable_categorical=False,  # 禁用分类特征支持以加速
            verbosity=0  # 减少日志输出
        ))
    ])
    
    # 简化的参数网格搜索 - 减少组合数量
    param_grid = {
        'regressor__n_estimators': [100, 150],  # 减少选项
        'regressor__max_depth': [4, 6],         # 适中的深度
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__subsample': [0.8, 0.9],     # 减少选项
    }
    
    # 使用更快的网格搜索设置
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=2,  # 减少交叉验证折数
        scoring='neg_mean_absolute_error',
        n_jobs=max(1, n_jobs//2),  # 减少并行作业数以避免内存问题
        verbose=1,
        pre_dispatch='2*n_jobs'  # 控制预分配内存
    )
    
    print("开始优化后的网格搜索寻找最佳XGBoost模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        # 使用最佳模型
        best_model = grid_search.best_estimator_
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        
        # 回归评估指标
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n测试集评估指标:")
        print(f"平均绝对误差(MAE): {mae:.4f}米")
        print(f"均方根误差(RMSE): {rmse:.4f}米")
        print(f"决定系数(R²): {r2:.4f}")
        
        # 可视化预测结果
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # 添加参考线（y=x）
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('实际目标水位(米)')
        plt.ylabel('预测目标水位(米)')
        plt.title('XGBoost模型: 实际值 vs 预测值')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('outputs/kz_level/plots/0822_kz_level_xgb_regression_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 误差分布图
        errors = y_pred - y_test
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('预测误差(米)')
        plt.ylabel('频率')
        plt.title('XGBoost模型: 目标水位预测误差分布')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/kz_level/plots/0822_kz_level_xgb_error_distribution.png', dpi=300, bbox_inches='tight')
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
            
            if hasattr(regressor, 'feature_importances_'):
                importances = regressor.feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_df.to_csv('outputs/kz_level/outputs/0822_kz_level_xgb_regression_feature_importance.csv', index=False)
                
                # 可视化最重要的特征
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', 
                            data=importance_df.head(20))
                plt.title('XGBoost模型特征重要性排名')
                plt.tight_layout()
                plt.savefig('outputs/kz_level/plots/0822_kz_level_xgb_regression_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"无法获取特征重要性: {str(e)}")
        
        # 保存模型
        joblib.dump(best_model, 'outputs/kz_level/pkls/0822_kz_level_xgb_regression_model.pkl')
        print("XGBoost模型已保存到 outputs/kz_level/pkls/0822_kz_level_xgb_regression_model.pkl")
        
        # 保存网格搜索结果
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv('outputs/kz_level/outputs/0822_kz_level_xgb_regression_grid_search_results.csv', index=False)
        
        # 保存预测结果
        if dates_test is not None:
            results_df = pd.DataFrame({
                'date': dates_test.values,
                'true_water_level': y_test,
                'pred_water_level': y_pred,
                'error': np.abs(y_test - y_pred)
            })
            
            # 按日期分组计算每日平均
            daily_results = results_df.groupby('date').agg({
                'true_water_level': 'mean',
                'pred_water_level': 'mean',
                'error': 'mean'
            }).reset_index()

            daily_results.to_csv('outputs/kz_level/outputs/0822_daily_water_level_xgb_predictions.csv', index=False)
            print("每日目标水位预测结果已保存")
            
            # 可视化每日预测对比
            plt.figure(figsize=(14, 7))
            plt.plot(pd.to_datetime(daily_results['date']), daily_results['true_water_level'], 'o-', label='真实目标水位')
            plt.plot(pd.to_datetime(daily_results['date']), daily_results['pred_water_level'], 's--', label='预测目标水位')
            plt.xlabel('日期')
            plt.ylabel('目标水位(米)')
            plt.title('XGBoost模型: 每日真实目标水位 vs 预测目标水位')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('outputs/kz_level/plots/0822_daily_water_level_xgb_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return best_model
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("尝试使用优化后的默认参数训练...")
        
        # 使用优化后的默认参数训练
        default_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                random_state=42,
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                n_jobs=n_jobs,
                tree_method='hist',
                predictor='cpu_predictor',
                verbosity=0
            ))
        ])
        
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"默认模型测试集MAE: {mae:.4f}")
        print(f"默认模型测试集R²: {r2:.4f}")
        
        # 保存默认模型
        joblib.dump(default_model, 'outputs/kz_level/pkls/0822_kz_level_xgb_regression_model_default.pkl')
        print("默认XGBoost模型已保存")
        
        return default_model

def train_fast_model():
    """快速训练版本 - 使用更激进的优化"""
    print("开始快速训练XGBoost模型...")
    
    # 加载特征数据
    data = pd.read_csv('features/features_0822/00_features.csv')
    
    # 提取日期列
    if 'date' in data.columns:
        dates = data['date']
        X = data.drop(columns=['date'])
    else:
        dates = None
        X = data
    
    # 加载目标数据
    y_targets = np.load('features/features_0822/00_target.npy')
    y = y_targets[:, 3].astype(np.float32)
    
    # 快速预处理
    X = preprocess_data(X)
    
    # 快速异常值处理
    q1, q3 = np.percentile(y, [5, 95])
    y = np.clip(y, q1, q3)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1122
    )
    
    # 使用精简的特征集
    important_features = [
        'hour_of_day', 'month', 'is_weekend',
        'prev_gate_count', 'prev_duration',
        'tide_24h_mean', 'tide_24h_phase',
        'future_tide_mean', 'flow_mean',
        'rain_actual_total', 'water_status_mean',
        'tide_type'
    ]
    
    X_train = X_train[important_features]
    X_test = X_test[important_features]
    
    # 快速预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), X_train.select_dtypes(include=[np.number]).columns),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
            ]), ['tide_type'])
        ])
    
    # 快速模型
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            tree_method='hist',
            n_jobs=n_jobs,
            random_state=42
        ))
    ])
    
    # 快速训练
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    print(f"快速模型MAE: {mae:.4f}")
    
    # 保存快速模型
    joblib.dump(model, 'outputs/kz_level/pkls/0822_kz_level_xgb_fast_model.pkl')
    print("快速XGBoost模型已保存")
    
    return model

if __name__ == '__main__':
    # 提供两种训练模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'fast':
        model = train_fast_model()
        print("快速XGBoost模型训练完成！")
    else:
        model = train_regression_model()
        print("XGBoost模型训练完成！")