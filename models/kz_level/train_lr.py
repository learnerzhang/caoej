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
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from utils import SafeSimpleImputer
import warnings
warnings.filterwarnings('ignore')
import os

# 设置使用的CPU核心数
n_jobs = max(1, os.cpu_count() // 2)

# 设置字体，确保中文正常显示
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

def create_enhanced_preprocessor():
    """创建增强的预处理管道"""
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
    
    # 使用安全的数值特征处理器
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

class StableRidge(Ridge):
    """稳定的Ridge回归实现，避免scipy版本兼容性问题"""
    
    def fit(self, X, y, sample_weight=None):
        # 尝试不同的求解器来避免兼容性问题
        solvers = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        
        for solver in solvers:
            try:
                self.solver = solver
                return super().fit(X, y, sample_weight)
            except Exception as e:
                print(f"求解器 {solver} 失败: {e}")
                continue
        
        # 如果所有求解器都失败，使用LinearRegression作为后备
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
    """稳定的Lasso回归实现，增加迭代次数避免收敛问题"""
    
    def __init__(self, alpha=1.0, max_iter=10000, tol=1e-4, random_state=None):
        super().__init__(alpha=alpha, max_iter=max_iter, tol=tol, random_state=random_state)
    
    def fit(self, X, y, sample_weight=None):
        # 增加最大迭代次数避免收敛警告
        original_max_iter = self.max_iter
        self.max_iter = 20000  # 进一步增加迭代次数
        
        try:
            result = super().fit(X, y, sample_weight)
            # 如果收敛，恢复原始max_iter
            self.max_iter = original_max_iter
            return result
        except Exception as e:
            print(f"Lasso拟合失败: {e}")
            # 使用弹性网作为后备
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

class StableElasticNet(ElasticNet):
    """稳定的ElasticNet回归实现"""
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=10000, tol=1e-4, random_state=None):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol, random_state=random_state)
    
    def fit(self, X, y, sample_weight=None):
        # 增加最大迭代次数避免收敛警告
        original_max_iter = self.max_iter
        self.max_iter = 20000
        
        try:
            result = super().fit(X, y, sample_weight)
            self.max_iter = original_max_iter
            return result
        except Exception as e:
            print(f"ElasticNet拟合失败: {e}")
            # 使用Ridge作为后备
            from sklearn.linear_model import Ridge
            self.fallback_model = Ridge(alpha=self.alpha, max_iter=1000)
            self.fallback_model.fit(X, y)
            self.coef_ = self.fallback_model.coef_
            self.intercept_ = self.fallback_model.intercept_
            return self
    
    def predict(self, X):
        if hasattr(self, 'fallback_model'):
            return self.fallback_model.predict(X)
        return super().predict(X)

def train_stable_regression_model():
    """训练稳定的目标水位回归模型"""
    # 创建输出目录
    os.makedirs('outputs/kz_level/plots', exist_ok=True)
    os.makedirs('outputs/kz_level/pkls', exist_ok=True)
    os.makedirs('outputs/kz_level/outputs', exist_ok=True)
    
    # 加载特征数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    
    # 提取日期列（用于后续分析）
    if 'date' in X.columns:
        dates = X['date']
        X = X.drop(columns=['date'])
    else:
        dates = None
    
    # 加载目标数据 - 目标水位（第4列，索引3）
    y_targets = np.load('features/features_0822/00_target.npy')
    
    # 确保目标水位是数值类型
    try:
        y = y_targets[:, 3].astype(float)
    except ValueError as e:
        print(f"目标水位数据转换错误: {e}")
        # 处理非数值数据
        valid_targets = []
        for i, value in enumerate(y_targets[:, 3]):
            try:
                valid_targets.append(float(value))
            except ValueError:
                print(f"索引 {i} 的无效目标水位值: '{value}'，替换为中位数")
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
    
    # 划分数据集
    if 'month' in X.columns and 'is_weekend' in X.columns:
        stratify_param = X['month'].astype(str) + '_' + X['is_weekend'].astype(str)
    else:
        stratify_param = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1122, stratify=stratify_param
    )
    
    if dates is not None:
        dates_train, dates_test = train_test_split(
            dates, test_size=0.1, random_state=1122, stratify=stratify_param
        )
    else:
        dates_train, dates_test = None, None
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建预处理管道
    preprocessor = create_enhanced_preprocessor()
    
    # 创建稳定的回归模型管道
    base_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', StableRidge(random_state=42))
    ])
    
    # 简化的参数网格，避免兼容性问题
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
    
    # 使用简单的KFold而不是StratifiedKFold（回归问题）
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    print("开始网格搜索寻找最佳回归模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳MAE: {-grid_search.best_score_:.4f}")
        
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
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线')
        plt.xlabel('实际目标水位(米)')
        plt.ylabel('预测目标水位(米)')
        plt.title('稳定回归模型目标水位预测')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('outputs/kz_level/plots/0822_kz_level_stable_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存模型
        joblib.dump(best_model, 'outputs/kz_level/pkls/0822_kz_level_stable_regression_model.pkl')
        print("稳定回归模型已保存")
        
        # 保存网格搜索结果
        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv('outputs/kz_level/outputs/0822_kz_level_stable_grid_search_results.csv', index=False)
        
        # 保存预测结果
        test_results = pd.DataFrame({
            'true_water_level': y_test,
            'pred_water_level': y_pred,
            'residual': y_test - y_pred
        })
        
        if dates_test is not None:
            test_results['date'] = dates_test.values
        
        test_results.to_csv('outputs/kz_level/outputs/0822_kz_level_stable_predictions.csv', index=False)
        print("测试集预测结果已保存")
        
        return best_model
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("尝试使用简单线性回归...")
        
        # 使用简单的线性回归作为后备
        simple_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        simple_model.fit(X_train, y_train)
        y_pred = simple_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"简单线性回归测试集MAE: {mae:.4f}")
        print(f"简单线性回归测试集R²: {r2:.4f}")
        
        # 保存简单模型
        joblib.dump(simple_model, 'outputs/kz_level/pkls/0822_kz_level_simple_regression_model.pkl')
        print("简单线性回归模型已保存")
        
        return simple_model

if __name__ == '__main__':
    model = train_stable_regression_model()
    print("稳定回归模型训练完成！")