import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from utils import SafeSimpleImputer
import os
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)

# 设置使用的CPU核心数
n_jobs = max(1, os.cpu_count() // 2)

# 设置字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False


def create_preprocessor():
    """创建预处理管道 - 与目标水位模型保持一致的特征集"""
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

def asymmetric_mae(y_true, y_pred):
    """不对称MAE，对高估和低估给予不同权重"""
    error = y_pred - y_true
    overestimation_penalty = 1.5  # 高估惩罚系数
    underestimation_penalty = 1.0  # 低估惩罚系数
    
    weighted_errors = np.where(
        error > 0, 
        overestimation_penalty * np.abs(error), 
        underestimation_penalty * np.abs(error)
    )
    return np.mean(weighted_errors)

def train_mlp_comb_model():
    """训练基于MLP的开闸时长*孔数回归模型"""
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
    
    # 创建基础MLP模型
    base_mlp = MLPRegressor(
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        learning_rate='adaptive',
        n_iter_no_change=50
    )
    
    # 创建MLP模型管道
    try:
        # 检测scikit-learn版本兼容性
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=42),
                threshold='median'
            )),
            ('regressor', BaggingRegressor(
                estimator=base_mlp,
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=42,
                n_jobs=n_jobs
            ))
        ])
        
        # 新版本参数网格
        param_grid = {
            'regressor__estimator__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'regressor__estimator__activation': ['relu', 'tanh'],
            'regressor__estimator__alpha': [0.0001, 0.001],
            'regressor__estimator__learning_rate_init': [0.001, 0.01],
            'regressor__estimator__solver': ['adam']
        }
        
        # 测试参数兼容性
        test_model = BaggingRegressor(estimator=base_mlp, n_estimators=2)
        test_model.fit(X_train[:10], y_train[:10])
        print("使用新版本scikit-learn参数 (estimator)")
        
    except TypeError:
        # 旧版本参数兼容
        print("检测到旧版本scikit-learn，使用base_estimator参数")
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=42),
                threshold='median'
            )),
            ('regressor', BaggingRegressor(
                base_estimator=base_mlp,
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=42,
                n_jobs=n_jobs
            ))
        ])
        
        # 旧版本参数网格
        param_grid = {
            'regressor__base_estimator__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'regressor__base_estimator__activation': ['relu', 'tanh'],
            'regressor__base_estimator__alpha': [0.0001, 0.001],
            'regressor__base_estimator__learning_rate_init': [0.001, 0.01],
            'regressor__base_estimator__solver': ['adam']
        }
    
    # 创建自定义评分器
    asymmetric_scorer = make_scorer(asymmetric_mae, greater_is_better=False)
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring=asymmetric_scorer,
        n_jobs=n_jobs,
        verbose=1
    )
    
    print("开始网格搜索寻找最佳MLP模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        
        print(f"最佳MLP模型: {grid_search.best_estimator_}")
        print(f"最佳不对称MAE: {-grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        
        # 回归评估指标
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        asym_mae = asymmetric_mae(y_test, y_pred)
        
        print(f"\n测试集评估指标:")
        print(f"平均绝对误差(MAE): {mae:.4f}")
        print(f"不对称MAE: {asym_mae:.4f}")
        print(f"均方根误差(RMSE): {rmse:.4f}")
        print(f"决定系数(R²): {r2:.4f}")
        
        # 可视化预测结果
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('实际开闸时长×孔数')
        plt.ylabel('预测开闸时长×孔数')
        plt.title('MLP模型: 开闸时长×孔数预测 - 实际值 vs 预测值')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('outputs/kz_comb_log/plots/0822_kz_comb_log_mlp_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 误差分布图
        errors = y_pred - y_test
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('预测误差')
        plt.ylabel('频率')
        plt.title('MLP模型: 开闸时长×孔数预测误差分布')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/kz_comb_log/plots/0822_kz_comb_log_mlp_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存模型
        os.makedirs('outputs/kz_comb_log/pkls', exist_ok=True)
        joblib.dump(best_model, 'outputs/kz_comb_log/pkls/0822_kz_comb_log_mlp_model.pkl')
        print("MLP模型已保存到 outputs/kz_comb_log/pkls/0822_kz_comb_log_mlp_model.pkl")
        
        # 保存网格搜索结果
        results_df = pd.DataFrame(grid_search.cv_results_)
        os.makedirs('outputs/kz_comb_log/outputs', exist_ok=True)
        results_df.to_csv('outputs/kz_comb_log/outputs/0822_kz_comb_log_mlp_grid_results.csv', index=False)
        
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

            daily_results.to_csv('outputs/kz_comb_log/outputs/0822_daily_comb_log_mlp_predictions.csv', index=False)
            print("每日开闸时长×孔数MLP预测结果已保存")
            
            # 可视化每日预测对比
            plt.figure(figsize=(14, 7))
            plt.plot(pd.to_datetime(daily_results['date']), daily_results['true_comb_log'], 'o-', label='真实开闸时长×孔数')
            plt.plot(pd.to_datetime(daily_results['date']), daily_results['pred_comb_log'], 's--', label='MLP预测开闸时长×孔数')
            plt.xlabel('日期')
            plt.ylabel('开闸时长×孔数')
            plt.title('每日真实开闸时长×孔数 vs MLP预测值')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('outputs/kz_comb_log/plots/0822_daily_comb_log_mlp_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return best_model
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("尝试使用默认参数训练MLP模型...")
        
        # 使用默认参数训练
        default_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', MLPRegressor(
                random_state=42,
                hidden_layer_sizes=(100,),
                activation='relu',
                alpha=0.001,
                learning_rate_init=0.01,
                max_iter=1000,
                early_stopping=True,
                learning_rate='adaptive'
            ))
        ])
        
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"默认MLP模型测试集MAE: {mae:.4f}")
        print(f"默认MLP模型测试集R²: {r2:.4f}")
        
        # 保存默认模型
        joblib.dump(default_model, 'outputs/kz_comb_log/pkls/0822_kz_comb_log_mlp_model_default.pkl')
        print("默认MLP模型已保存")
        
        return default_model

if __name__ == '__main__':
    # 创建输出目录
    os.makedirs('outputs/kz_comb_log/plots', exist_ok=True)
    os.makedirs('outputs/kz_comb_log/pkls', exist_ok=True)
    os.makedirs('outputs/kz_comb_log/outputs', exist_ok=True)
    
    model = train_mlp_comb_model()
    print("MLP模型训练完成！")