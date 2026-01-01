import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings("ignore")
import os
from utils import SafeSimpleImputer

# 设置使用的CPU核心数
n_jobs = max(1, os.cpu_count() // 2)

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
    print(f"最小值: {y.min():.2f}, 最大值: {y.max():.2f}, 均值: {y.mean():.2f}, 标准差: {y.std():.2f}")
    
    return X, y

def create_preprocessor():
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
    
    indicator_features = [
        'water_missing', 'flow_missing', 
        'rain_missing', 'water_status_missing'
    ]
    
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

def prepare_target_data(y):
    """准备目标数据（周期性编码）"""
    y_sin = np.sin(2 * np.pi * y / 24)
    y_cos = np.cos(2 * np.pi * y / 24)
    y_transformed = np.column_stack([y_sin, y_cos])
    return y_transformed

def convert_predictions(y_pred):
    """将预测值转换回小时"""
    y_pred_hours = np.arctan2(y_pred[:, 0], y_pred[:, 1]) * 24 / (2 * np.pi)
    y_pred_hours = np.mod(y_pred_hours, 24)
    return y_pred_hours

def train_linear_model(X_train, y_train, X_test, y_test, dates_test):
    """训练线性回归模型"""
    print("\n=== 训练线性回归模型 ===")
    
    # 创建回归模型管道
    model = Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('regressor', LinearRegression())
    ])
    
    # 参数网格搜索
    param_grid = [
        {
            'regressor': [LinearRegression()],
            'regressor__fit_intercept': [True, False],
            'regressor__positive': [True, False]
        },
        {
            'regressor': [Ridge()],
            'regressor__alpha': [0.1, 1.0, 10.0, 100.0],
            'regressor__fit_intercept': [True, False],
            'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr']
        },
        {
            'regressor': [Lasso()],
            'regressor__alpha': [0.1, 1.0, 10.0, 100.0],
            'regressor__fit_intercept': [True, False],
            'regressor__selection': ['cyclic', 'random']
        }
    ]

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳线性模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        y_pred_hours = convert_predictions(y_pred)
        
        mae = mean_absolute_error(y_test, y_pred_hours)
        print(f"测试集MAE: {mae:.4f}小时")
        
        return best_model, y_pred_hours
        
    except Exception as e:
        print(f"线性模型训练失败: {str(e)}")
        # 使用默认参数
        default_model = Pipeline(steps=[
            ('preprocessor', create_preprocessor()),
            ('regressor', LinearRegression())
        ])
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        y_pred_hours = convert_predictions(y_pred)
        return default_model, y_pred_hours

def train_mlp_model(X_train, y_train, X_test, y_test, dates_test):
    """训练MLP模型"""
    print("\n=== 训练MLP模型 ===")
    
    # 创建基础MLP模型
    base_mlp = MLPRegressor(
        random_state=42, 
        max_iter=1000, 
        early_stopping=True,
        learning_rate='adaptive',
        n_iter_no_change=50
    )
    
    # 创建MLP回归模型管道
    model = Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('feature_selection', SelectFromModel(
            RandomForestRegressor(n_estimators=100, random_state=42),
            threshold='median'
        )),
        ('regressor', BaggingRegressor(
            base_estimator=base_mlp,
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            random_state=42
        ))
    ])
    
    # MLP参数网格搜索
    param_grid = {
        'regressor__base_estimator__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'regressor__base_estimator__activation': ['relu', 'tanh'],
        'regressor__base_estimator__alpha': [0.0001, 0.001],
        'regressor__base_estimator__learning_rate_init': [0.001, 0.01],
        'regressor__base_estimator__solver': ['adam']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳MLP模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        y_pred_hours = convert_predictions(y_pred)
        
        mae = mean_absolute_error(y_test, y_pred_hours)
        print(f"测试集MAE: {mae:.4f}小时")
        
        return best_model, y_pred_hours
        
    except Exception as e:
        print(f"MLP模型训练失败: {str(e)}")
        # 使用默认参数
        default_model = Pipeline(steps=[
            ('preprocessor', create_preprocessor()),
            ('regressor', MLPRegressor(
                random_state=42,
                hidden_layer_sizes=(100,),
                max_iter=1000,
                early_stopping=True
            ))
        ])
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        y_pred_hours = convert_predictions(y_pred)
        return default_model, y_pred_hours

def train_rf_model(X_train, y_train, X_test, y_test, dates_test):
    """训练随机森林模型"""
    print("\n=== 训练随机森林模型 ===")
    
    # 创建预处理管道
    preprocessor = create_preprocessor()
    
    # 创建随机森林回归模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=1122, n_jobs=n_jobs))
    ])
    
    # 参数网格搜索
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
        'regressor__max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=1,
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳随机森林模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        y_pred_hours = convert_predictions(y_pred)
        
        mae = mean_absolute_error(y_test, y_pred_hours)
        print(f"测试集MAE: {mae:.4f}小时")
        
        return best_model, y_pred_hours
        
    except Exception as e:
        print(f"随机森林模型训练失败: {str(e)}")
        # 使用默认参数
        default_model = Pipeline(steps=[
            ('preprocessor', create_preprocessor()),
            ('regressor', RandomForestRegressor(
                random_state=1122,
                n_estimators=100,
                max_depth=20,
                n_jobs=n_jobs
            ))
        ])
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        y_pred_hours = convert_predictions(y_pred)
        return default_model, y_pred_hours

def train_svm_model(X_train, y_train, X_test, y_test, dates_test):
    """训练SVM模型"""
    print("\n=== 训练SVM模型 ===")
    
    # 创建预处理管道
    preprocessor = create_preprocessor()
    
    # 创建SVM回归模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MultiOutputRegressor(SVR()))
    ])
    
    # SVM参数网格搜索
    param_grid = {
        'regressor__estimator__kernel': ['linear', 'rbf'],
        'regressor__estimator__C': [0.1, 1, 10],
        'regressor__estimator__gamma': ['scale', 'auto'],
        'regressor__estimator__epsilon': [0.1, 0.5]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳SVM模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        y_pred_hours = convert_predictions(y_pred)
        
        mae = mean_absolute_error(y_test, y_pred_hours)
        print(f"测试集MAE: {mae:.4f}小时")
        
        return best_model, y_pred_hours
        
    except Exception as e:
        print(f"SVM模型训练失败: {str(e)}")
        # 使用默认参数
        default_model = Pipeline(steps=[
            ('preprocessor', create_preprocessor()),
            ('regressor', MultiOutputRegressor(SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            )))
        ])
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        y_pred_hours = convert_predictions(y_pred)
        return default_model, y_pred_hours

def train_xgb_model(X_train, y_train, X_test, y_test, dates_test):
    """训练XGBoost模型"""
    print("\n=== 训练XGBoost模型 ===")
    
    # 创建预处理管道
    preprocessor = create_preprocessor()
    
    # 创建XGBoost回归模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            random_state=1122, 
            n_jobs=n_jobs,
            tree_method='hist',
            predictor='cpu_predictor'
        ))
    ])
    
    # 使用随机搜索
    param_dist = {
        'regressor__n_estimators': randint(50, 200),
        'regressor__max_depth': randint(3, 8),
        'regressor__learning_rate': uniform(0.01, 0.3),
        'regressor__subsample': uniform(0.7, 0.3),
        'regressor__colsample_bytree': uniform(0.7, 0.3),
        'regressor__reg_lambda': uniform(0, 2),
        'regressor__reg_alpha': uniform(0, 1)
    }

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=1,
        verbose=1,
        random_state=1122
    )
    
    try:
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        print(f"最佳XGBoost模型: {random_search.best_estimator_}")
        print(f"最佳MAE: {-random_search.best_score_:.4f}")
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        y_pred_hours = convert_predictions(y_pred)
        
        mae = mean_absolute_error(y_test, y_pred_hours)
        print(f"测试集MAE: {mae:.4f}小时")
        
        return best_model, y_pred_hours
        
    except Exception as e:
        print(f"XGBoost模型训练失败: {str(e)}")
        # 使用默认参数
        default_model = Pipeline(steps=[
            ('preprocessor', create_preprocessor()),
            ('regressor', XGBRegressor(
                random_state=1122,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=n_jobs
            ))
        ])
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        y_pred_hours = convert_predictions(y_pred)
        return default_model, y_pred_hours

def train_and_save_all_models():
    """训练所有模型"""
    # 创建输出目录
    os.makedirs('outputs/kz_time/plots', exist_ok=True)
    os.makedirs('outputs/kz_time/pkls', exist_ok=True)
    os.makedirs('outputs/kz_time/outputs', exist_ok=True)
    
    # 加载特征数据
    data = pd.read_csv('features/features_0822/00_features.csv')
    dates = data['date']
    X = data.drop(columns=['date'])
    
    # 加载目标数据（只使用开闸时间）
    y_targets = np.load('features/features_0822/00_target.npy')
    y = y_targets[:, 0]  # 只取开闸时间列
    
    # 数据预处理
    X = preprocess_data(X)
    
    # 数据质量检查
    X, y = data_quality_check(X, y)
    
    # 划分数据集
    stratify_param = X['month'].astype(str) + '_' + X['is_weekend'].astype(str)
    
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, 
        test_size=0.1, 
        random_state=1122,
        stratify=stratify_param
    )
    
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 准备目标数据
    y_train_transformed = prepare_target_data(y_train)
    y_test_transformed = prepare_target_data(y_test)
    
    # 训练所有模型
    models = {}
    predictions = {}
    
    # 线性回归
    lr_model, lr_pred = train_linear_model(X_train, y_train_transformed, X_test, y_test, dates_test)
    models['lr'] = lr_model
    predictions['lr'] = lr_pred
    
    # MLP
    mlp_model, mlp_pred = train_mlp_model(X_train, y_train_transformed, X_test, y_test, dates_test)
    models['mlp'] = mlp_model
    predictions['mlp'] = mlp_pred
    
    # 随机森林
    rf_model, rf_pred = train_rf_model(X_train, y_train_transformed, X_test, y_test, dates_test)
    models['rf'] = rf_model
    predictions['rf'] = rf_pred
    
    # SVM
    svm_model, svm_pred = train_svm_model(X_train, y_train_transformed, X_test, y_test, dates_test)
    models['svm'] = svm_model
    predictions['svm'] = svm_pred
    
    # XGBoost
    xgb_model, xgb_pred = train_xgb_model(X_train, y_train_transformed, X_test, y_test, dates_test)
    models['xgb'] = xgb_model
    predictions['xgb'] = xgb_pred
    
    # 保存所有模型
    for model_name, model in models.items():
        model_path = f'outputs/kz_time/pkls/0822_kz_time_{model_name}_regression_model.pkl'
        joblib.dump(model, model_path)
        print(f"{model_name.upper()}模型已保存到: {model_path}")
    
    # 生成综合比较报告
    generate_comparison_report(y_test, predictions, dates_test)
    
    return models

def generate_comparison_report(y_true, predictions, dates_test):
    """生成模型比较报告"""
    print("\n=== 模型性能比较 ===")
    
    results = []
    for model_name, y_pred in predictions.items():
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        results.append({
            'Model': model_name.upper(),
            'MAE': f"{mae:.4f}",
            'RMSE': f"{rmse:.4f}",
            'R2': f"{r2:.4f}"
        })
        
        print(f"{model_name.upper()}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    # 保存比较结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/kz_time/outputs/model_comparison_results.csv', index=False)
    
    # 可视化比较
    plt.figure(figsize=(12, 8))
    
    # MAE比较
    plt.subplot(2, 2, 1)
    mae_values = [mean_absolute_error(y_true, pred) for pred in predictions.values()]
    plt.bar(predictions.keys(), mae_values)
    plt.title('模型MAE比较')
    plt.ylabel('MAE (小时)')
    
    # RMSE比较
    plt.subplot(2, 2, 2)
    rmse_values = [np.sqrt(mean_squared_error(y_true, pred)) for pred in predictions.values()]
    plt.bar(predictions.keys(), rmse_values)
    plt.title('模型RMSE比较')
    plt.ylabel('RMSE (小时)')
    
    # R2比较
    plt.subplot(2, 2, 3)
    r2_values = [r2_score(y_true, pred) for pred in predictions.values()]
    plt.bar(predictions.keys(), r2_values)
    plt.title('模型R²比较')
    plt.ylabel('R²')
    
    # 预测值对比
    plt.subplot(2, 2, 4)
    for model_name, y_pred in predictions.items():
        plt.scatter(y_true, y_pred, alpha=0.5, label=model_name.upper())
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('预测值对比')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/kz_time/plots/model_comparison.png')
    plt.close()
    
    # 保存每日预测结果
    daily_results = pd.DataFrame({'date': dates_test.values, 'true': y_true})
    for model_name, y_pred in predictions.items():
        daily_results[model_name] = y_pred
    
    daily_results.to_csv('outputs/kz_time/outputs/daily_predictions_all_models.csv', index=False)

if __name__ == '__main__':
    models = train_and_save_all_models()
    print("\n所有模型训练完成！")