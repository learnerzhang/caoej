import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
import xgboost as xgb
from utils import SafeSimpleImputer
import warnings
warnings.filterwarnings('ignore')
import os

# 设置使用的CPU核心数
n_jobs = max(1, os.cpu_count() // 2)

# 设置字体，确保中文正常显示
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

class StableRidge(Ridge):
    """稳定的Ridge回归实现"""
    def fit(self, X, y, sample_weight=None):
        solvers = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        for solver in solvers:
            try:
                self.solver = solver
                return super().fit(X, y, sample_weight)
            except Exception as e:
                print(f"求解器 {solver} 失败: {e}")
                continue
        print("所有Ridge求解器失败，使用LinearRegression")
        from sklearn.linear_model import LinearRegression
        self.fallback_model = LinearRegression()
        self.fallback_model.fit(X, y)
        self.coef_ = self.fallback_model.coef_
        self.intercept_ = self.fallback_model.intercept_
        return self
    
    def predict(self, X):
        if hasattr(self, 'fallback_model'):
            return self.fallback_model.predict(X)
        return super().predict(X)

class StableLasso(Lasso):
    """稳定的Lasso回归实现"""
    def __init__(self, alpha=1.0, max_iter=10000, tol=1e-4, random_state=None):
        super().__init__(alpha=alpha, max_iter=max_iter, tol=tol, random_state=random_state)
    
    def fit(self, X, y, sample_weight=None):
        original_max_iter = self.max_iter
        self.max_iter = 20000
        try:
            result = super().fit(X, y, sample_weight)
            self.max_iter = original_max_iter
            return result
        except Exception as e:
            print(f"Lasso拟合失败: {e}")
            from sklearn.linear_model import ElasticNet
            self.fallback_model = ElasticNet(alpha=self.alpha, l1_ratio=1.0, max_iter=20000)
            self.fallback_model.fit(X, y)
            self.coef_ = self.fallback_model.coef_
            self.intercept_ = self.fallback_model.intercept_
            return self
    
    def predict(self, X):
        if hasattr(self, 'fallback_model'):
            return self.fallback_model.predict(X)
        return super().predict(X)

def preprocess_data(X):
    """数据预处理：处理缺失值和异常值"""
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
    
    # 检查目标变量的分布
    print(f"\n目标变量统计:")
    print(f"最小值: {np.min(y):.4f}")
    print(f"最大值: {np.max(y):.4f}")
    print(f"平均值: {np.mean(y):.4f}")
    print(f"标准差: {np.std(y):.4f}")
    print(f"中位数: {np.median(y):.4f}")
    
    return X, y

def create_preprocessor(model_type='lr'):
    """创建预处理管道"""
    numeric_features = [
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_of_year',
        'prev_gate_count', 'prev_duration', 'prev_op_hour',
        'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration',
        'tide_24h_mean', 'tide_24h_max', 'tide_24h_min', 'tide_24h_range',
        'tide_24h_slope', 'tide_24h_r_squared', 'tide_24h_cycle_count',
        'tide_24h_rise_rate', 'tide_24h_fall_rate', 'tide_24h_phase',
        'tide_12h_mean', 'tide_12h_max', 'tide_12h_min', 'tide_12h_range',
        'tide_12h_slope', 'tide_12h_r_squared', 'tide_12h_cycle_count',
        'tide_12h_rise_rate', 'tide_12h_fall_rate', 'tide_12h_phase',
        'future_tide_mean', 'future_tide_max', 'future_tide_min', 'future_tide_range',
        'future_tide_slope', 'future_tide_r_squared', 'future_tide_cycle_count',
        'future_tide_rise_rate', 'future_tide_fall_rate', 'future_tide_phase',
        'flow_mean', 'flow_max', 'flow_min', 'flow_range', 'flow_var', 'flow_skew',
        'rain_actual_total', 'rain_forecast_total', 
        'rain_actual_avg', 'rain_forecast_avg',
        'rain_change_rate', 'water_rain_ratio', 'flow_rain_ratio',
        'water_status_mean', 'water_status_max', 'water_status_min', 
        'water_status_range', 'water_status_slope',
        'is_rush_hour'
    ]
    
    categorical_features = ['tide_type']
    indicator_features = ['water_missing', 'flow_missing', 'rain_missing', 'water_status_missing']
    
    # 对于树模型不需要标准化
    if model_type in ['rf', 'xgb']:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SafeSimpleImputer(strategy='median'))
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SafeSimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    
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

def load_and_prepare_data():
    """加载和准备数据"""
    # 创建输出目录
    os.makedirs('outputs/kz_level/plots', exist_ok=True)
    os.makedirs('outputs/kz_level/pkls', exist_ok=True)
    os.makedirs('outputs/kz_level/outputs', exist_ok=True)
    
    # 加载特征数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    
    # 提取日期列
    if 'date' in X.columns:
        dates = X['date']
        X = X.drop(columns=['date'])
    else:
        dates = None
    
    # 加载目标数据
    y_targets = np.load('features/features_0822/00_target.npy')
    
    # 确保目标水位是数值类型
    try:
        y = y_targets[:, 3].astype(float)
    except ValueError as e:
        print(f"目标水位数据转换错误: {e}")
        valid_targets = []
        for i, value in enumerate(y_targets[:, 3]):
            try:
                valid_targets.append(float(value))
            except ValueError:
                valid_values = [float(v) for v in y_targets[:, 3] 
                              if isinstance(v, (int, float, np.number)) or 
                              (isinstance(v, str) and v.replace('.', '').isdigit())]
                median_val = np.median(valid_values) if valid_values else 0
                valid_targets.append(median_val)
        y = np.array(valid_targets)
    
    # 数据预处理
    X = preprocess_data(X)
    
    # 数据质量检查
    X, y = data_quality_check(X, y)
    
    # 处理目标变量的异常值
    q1 = np.percentile(y, 5)
    q3 = np.percentile(y, 95)
    y_processed = np.clip(y, q1, q3)
    
    if np.any(y != y_processed):
        print(f"处理了 {np.sum(y != y_processed)} 个目标变量异常值")
        y = y_processed
    
    return X, y, dates

def train_lr_model():
    """训练线性回归模型"""
    print("=== 训练线性回归模型 ===")
    X, y, dates = load_and_prepare_data()
    
    # 划分数据集
    if 'month' in X.columns and 'is_weekend' in X.columns:
        stratify_param = X['month'].astype(str) + '_' + X['is_weekend'].astype(str)
    else:
        stratify_param = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1122, stratify=stratify_param
    )
    
    # 创建预处理管道
    preprocessor = create_preprocessor('lr')
    
    # 创建模型管道
    base_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', StableRidge(random_state=42))
    ])
    
    # 参数网格
    param_grid = [
        {
            'regressor': [StableRidge(random_state=42)],
            'regressor__alpha': [0.1, 1, 10],
        },
        {
            'regressor': [StableLasso(random_state=42)],
            'regressor__alpha': [0.1, 1, 10],
        },
        {
            'regressor': [LinearRegression()],
        }
    ]
    
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    print("开始网格搜索寻找最佳线性回归模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"测试集MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # 保存模型
        joblib.dump(best_model, 'outputs/kz_level/pkls/0822_kz_level_lr_model.pkl')
        print("线性回归模型已保存")
        
        return best_model
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("使用简单线性回归...")
        simple_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        simple_model.fit(X_train, y_train)
        joblib.dump(simple_model, 'outputs/kz_level/pkls/0822_kz_level_lr_model.pkl')
        return simple_model

def train_mlp_model():
    """训练MLP模型"""
    print("=== 训练MLP模型 ===")
    X, y, dates = load_and_prepare_data()
    
    # 划分数据集
    if 'month' in X.columns and 'is_weekend' in X.columns:
        stratify_param = X['month'].astype(str) + '_' + X['is_weekend'].astype(str)
    else:
        stratify_param = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1122, stratify=stratify_param
    )
    
    # 创建预处理管道
    preprocessor = create_preprocessor('mlp')
    
    # 创建MLP模型
    base_mlp = MLPRegressor(
        random_state=42, 
        max_iter=1000, 
        early_stopping=True,
        learning_rate='adaptive',
        n_iter_no_change=50
    )
    
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
    
    # 参数网格
    param_grid = {
        'regressor__base_estimator__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'regressor__base_estimator__activation': ['relu', 'tanh'],
        'regressor__base_estimator__alpha': [0.0001, 0.001],
        'regressor__base_estimator__learning_rate_init': [0.001, 0.01]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    print("开始网格搜索寻找最佳MLP模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"测试集MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # 保存模型
        joblib.dump(best_model, 'outputs/kz_level/pkls/0822_kz_level_mlp_model.pkl')
        print("MLP模型已保存")
        
        return best_model
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("使用默认MLP模型...")
        default_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=42),
                threshold='median'
            )),
            ('regressor', BaggingRegressor(
                base_estimator=MLPRegressor(random_state=42, hidden_layer_sizes=(100,), max_iter=1000),
                n_estimators=10,
                random_state=42,
                n_jobs=n_jobs
            ))
        ])
        default_model.fit(X_train, y_train)
        joblib.dump(default_model, 'outputs/kz_level/pkls/0822_kz_level_mlp_model.pkl')
        return default_model

def train_rf_model():
    """训练随机森林模型"""
    print("=== 训练随机森林模型 ===")
    X, y, dates = load_and_prepare_data()
    
    # 划分数据集
    if 'month' in X.columns and 'is_weekend' in X.columns:
        stratify_param = X['month'].astype(str) + '_' + X['is_weekend'].astype(str)
    else:
        stratify_param = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1122, stratify=stratify_param
    )
    
    # 创建预处理管道
    preprocessor = create_preprocessor('rf')
    
    # 创建随机森林模型
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=1122, n_jobs=n_jobs))
    ])
    
    # 参数网格
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
        'regressor__max_features': ['auto', 'sqrt']
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    print("开始网格搜索寻找最佳随机森林模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"测试集MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # 保存模型
        joblib.dump(best_model, 'outputs/kz_level/pkls/0822_kz_level_rf_model.pkl')
        print("随机森林模型已保存")
        
        return best_model
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("使用默认随机森林模型...")
        default_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=1122, n_estimators=100, n_jobs=n_jobs))
        ])
        default_model.fit(X_train, y_train)
        joblib.dump(default_model, 'outputs/kz_level/pkls/0822_kz_level_rf_model.pkl')
        return default_model

def train_svm_model():
    """训练SVM模型"""
    print("=== 训练SVM模型 ===")
    X, y, dates = load_and_prepare_data()
    
    # 划分数据集
    if 'month' in X.columns and 'is_weekend' in X.columns:
        stratify_param = X['month'].astype(str) + '_' + X['is_weekend'].astype(str)
    else:
        stratify_param = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1122, stratify=stratify_param
    )
    
    # 创建预处理管道
    preprocessor = create_preprocessor('svm')
    
    # 创建SVM模型
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR())
    ])
    
    # 参数网格
    param_grid = {
        'regressor__kernel': ['linear', 'rbf'],
        'regressor__C': [0.1, 1, 10],
        'regressor__gamma': ['scale', 'auto'],
        'regressor__epsilon': [0.01, 0.1]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    print("开始网格搜索寻找最佳SVM模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"测试集MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # 保存模型
        joblib.dump(best_model, 'outputs/kz_level/pkls/0822_kz_level_svm_model.pkl')
        print("SVM模型已保存")
        
        return best_model
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("使用默认SVM模型...")
        default_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1))
        ])
        default_model.fit(X_train, y_train)
        joblib.dump(default_model, 'outputs/kz_level/pkls/0822_kz_level_svm_model.pkl')
        return default_model

def train_xgb_model():
    """训练XGBoost模型"""
    print("=== 训练XGBoost模型 ===")
    X, y, dates = load_and_prepare_data()
    
    # 转换特征数据类型以加速训练
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].astype(np.float32)
    y = y.astype(np.float32)
    
    # 划分数据集
    if 'month' in X.columns and 'is_weekend' in X.columns:
        stratify_param = X['month'].astype(str) + '_' + X['is_weekend'].astype(str)
    else:
        stratify_param = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1122, stratify=stratify_param
    )
    
    # 创建预处理管道
    preprocessor = create_preprocessor('xgb')
    
    # 创建XGBoost模型
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            random_state=42,
            n_jobs=n_jobs,
            tree_method='hist',
            predictor='cpu_predictor',
            enable_categorical=False,
            verbosity=0
        ))
    ])
    
    # 参数网格
    param_grid = {
        'regressor__n_estimators': [100, 150],
        'regressor__max_depth': [4, 6],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__subsample': [0.8, 0.9],
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=2,
        scoring='neg_mean_absolute_error',
        n_jobs=max(1, n_jobs//2),
        verbose=1
    )
    
    print("开始网格搜索寻找最佳XGBoost模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"测试集MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # 保存模型
        joblib.dump(best_model, 'outputs/kz_level/pkls/0822_kz_level_xgb_model.pkl')
        print("XGBoost模型已保存")
        
        return best_model
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("使用默认XGBoost模型...")
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
                verbosity=0
            ))
        ])
        default_model.fit(X_train, y_train)
        joblib.dump(default_model, 'outputs/kz_level/pkls/0822_kz_level_xgb_model.pkl')
        return default_model

def train_all_models():
    """训练所有模型"""
    models = {}
    
    # 训练各个模型
    models['lr'] = train_lr_model()
    models['mlp'] = train_mlp_model()
    models['rf'] = train_rf_model()
    models['svm'] = train_svm_model()
    models['xgb'] = train_xgb_model()
    
    print("所有模型训练完成！")
    return models

if __name__ == '__main__':
    models = train_all_models()