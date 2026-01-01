import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import uniform, randint
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
    print(f"\n目标变量分布:")
    target_counts = pd.Series(y).value_counts().sort_index()
    for class_id, count in target_counts.items():
        print(f"类别 {class_id}: {count} 样本 ({count/len(y)*100:.2f}%)")
    
    return X, y

def create_preprocessor(model_type='default'):
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
    
    # 根据不同模型类型调整预处理
    if model_type == 'rf':
        # 随机森林不需要标准化
        numeric_transformer = Pipeline(steps=[
            ('imputer', SafeSimpleImputer(strategy='median'))
        ])
    else:
        # 其他模型需要标准化
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

def train_lr_model():
    """训练逻辑回归模型"""
    print("=" * 50)
    print("开始训练逻辑回归模型...")
    
    # 创建输出目录
    os.makedirs('outputs/kz_num/pkls', exist_ok=True)
    
    # 加载特征数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    y = np.load('features/features_0822/00_target.npy')[:, 2].astype(int)
    
    # 数据预处理
    X = preprocess_data(X)
    X, y = data_quality_check(X, y)
    
    # 处理罕见类别
    class_counts = pd.Series(y).value_counts()
    rare_classes = class_counts[class_counts <= 2].index.tolist()
    if rare_classes:
        mask = ~pd.Series(y).isin(rare_classes)
        X = X[mask].reset_index(drop=True)
        y = y[mask]
        print(f"移除了罕见类别: {rare_classes}")
    
    # 划分数据集
    remaining_counts = pd.Series(y).value_counts()
    stratify_param = y if all(remaining_counts > 2) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=3333, stratify=stratify_param
    )
    
    # 计算类别权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # 创建预处理管道
    preprocessor = create_preprocessor('lr')
    
    # 创建逻辑回归模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict),
            threshold='median'
        )),
        ('classifier', LogisticRegression(random_state=42, max_iter=5000, class_weight=class_weight_dict))
    ])
    
    # 参数网格
    param_grid = [
        {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['saga']
        }
    ]
    
    # 网格搜索
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳逻辑回归模型: {grid_search.best_estimator_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
    except Exception as e:
        print(f"网格搜索失败: {str(e)}，使用默认参数")
        best_model = model
        best_model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"测试集F1分数: {f1:.4f}")
    
    # 保存模型
    joblib.dump(best_model, 'outputs/kz_num/pkls/0822_kz_num_lr_classification_model.pkl')
    print("逻辑回归模型已保存")
    
    return best_model

def train_mlp_model():
    """训练MLP模型"""
    print("=" * 50)
    print("开始训练MLP模型...")
    
    # 创建输出目录
    os.makedirs('outputs/kz_num/pkls', exist_ok=True)
    
    # 加载特征数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    y = np.load('features/features_0822/00_target.npy')[:, 2].astype(int)
    
    # 数据预处理
    X = preprocess_data(X)
    X, y = data_quality_check(X, y)
    
    # 处理罕见类别
    class_counts = pd.Series(y).value_counts()
    rare_classes = class_counts[class_counts <= 1].index.tolist()
    if rare_classes:
        mask = ~pd.Series(y).isin(rare_classes)
        X = X[mask].reset_index(drop=True)
        y = y[mask]
        print(f"移除了罕见类别: {rare_classes}")
    
    # 划分数据集
    remaining_counts = pd.Series(y).value_counts()
    stratify_param = y if all(remaining_counts > 1) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=3333, stratify=stratify_param
    )
    
    # 计算类别权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # 创建预处理管道
    preprocessor = create_preprocessor('mlp')
    
    # 创建基础MLP分类器
    base_mlp = MLPClassifier(
        random_state=42, 
        max_iter=1000, 
        early_stopping=True,
        learning_rate='adaptive',
        n_iter_no_change=50
    )
    
    # 创建MLP分类模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', BaggingClassifier(
            base_estimator=base_mlp,
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            random_state=42
        ))
    ])
    
    # MLP参数网格搜索
    param_grid = {
        'classifier__base_estimator__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'classifier__base_estimator__activation': ['relu', 'tanh'],
        'classifier__base_estimator__alpha': [0.0001, 0.001],
        'classifier__base_estimator__learning_rate_init': [0.001, 0.01]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳MLP模型: {grid_search.best_estimator_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
    except Exception as e:
        print(f"网格搜索失败: {str(e)}，使用默认参数")
        best_model = model
        best_model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"测试集F1分数: {f1:.4f}")
    
    # 保存模型
    joblib.dump(best_model, 'outputs/kz_num/pkls/0822_kz_num_mlp_classification_model.pkl')
    print("MLP模型已保存")
    
    return best_model

def train_rf_model():
    """训练随机森林模型"""
    print("=" * 50)
    print("开始训练随机森林模型...")
    
    # 创建输出目录
    os.makedirs('outputs/kz_num/pkls', exist_ok=True)
    
    # 加载特征数据
    data = pd.read_csv('features/features_0822/11_features.csv')
    dates = data['date']
    X = data.drop(columns=['date'])
    y_targets = np.load('features/features_0822/11_target.npy')
    y = y_targets[:, 2].astype(int)
    
    # 数据预处理
    X = preprocess_data(X)
    X, y = data_quality_check(X, y)
    
    # 处理罕见类别
    class_counts = pd.Series(y).value_counts()
    rare_classes = class_counts[class_counts <= 2].index.tolist()
    if rare_classes:
        mask = ~pd.Series(y).isin(rare_classes)
        X = X[mask].reset_index(drop=True)
        y = y[mask]
        print(f"移除了罕见类别: {rare_classes}")
    
    # 划分数据集
    remaining_counts = pd.Series(y).value_counts()
    stratify_param = y if all(remaining_counts > 2) else None
    
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, test_size=0.1, random_state=3333, stratify=stratify_param
    )
    
    # 计算类别权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # 创建预处理管道
    preprocessor = create_preprocessor('rf')
    
    # 创建随机森林分类模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight=class_weight_dict, n_jobs=n_jobs))
    ])
    
    # 参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['sqrt', 'log2']
    }
    
    # 使用分层K折交叉验证
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=1,
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳随机森林模型: {grid_search.best_estimator_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
    except Exception as e:
        print(f"网格搜索失败: {str(e)}，使用默认参数")
        best_model = model
        best_model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"测试集F1分数: {f1:.4f}")
    
    # 保存模型
    joblib.dump(best_model, 'outputs/kz_num/pkls/0822_kz_num_rf_classification_model.pkl')
    print("随机森林模型已保存")
    
    return best_model

def train_svm_model():
    """训练SVM模型"""
    print("=" * 50)
    print("开始训练SVM模型...")
    
    # 创建输出目录
    os.makedirs('outputs/kz_num/pkls', exist_ok=True)
    
    # 加载特征数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    y = np.load('features/features_0822/00_target.npy')[:, 2].astype(int)
    
    # 数据预处理
    X = preprocess_data(X)
    X, y = data_quality_check(X, y)
    
    # 处理罕见类别
    class_counts = pd.Series(y).value_counts()
    rare_classes = class_counts[class_counts <= 1].index.tolist()
    if rare_classes:
        mask = ~pd.Series(y).isin(rare_classes)
        X = X[mask].reset_index(drop=True)
        y = y[mask]
        print(f"移除了罕见类别: {rare_classes}")
    
    # 划分数据集
    remaining_counts = pd.Series(y).value_counts()
    stratify_param = y if all(remaining_counts > 1) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1122, stratify=stratify_param
    )
    
    # 计算类别权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # 创建预处理管道
    preprocessor = create_preprocessor('svm')
    
    # 创建SVM分类模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=42, probability=True, class_weight=class_weight_dict))
    ])
    
    # SVM参数网格搜索
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'rbf', 'poly'],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'classifier__degree': [2, 3, 4],
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳SVM模型: {grid_search.best_estimator_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
    except Exception as e:
        print(f"网格搜索失败: {str(e)}，使用默认参数")
        best_model = model
        best_model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"测试集F1分数: {f1:.4f}")
    
    # 保存模型
    joblib.dump(best_model, 'outputs/kz_num/pkls/0822_kz_num_svm_classification_model.pkl')
    print("SVM模型已保存")
    
    return best_model

def train_xgb_model():
    """训练XGBoost模型"""
    print("=" * 50)
    print("开始训练XGBoost模型...")
    
    # 创建输出目录
    os.makedirs('outputs/kz_num/pkls', exist_ok=True)
    
    # 加载特征数据
    data = pd.read_csv('features/features_0822/00_features.csv')
    dates = data['date']
    X = data.drop(columns=['date'])
    y_targets = np.load('features/features_0822/00_target.npy')
    y = y_targets[:, 2].astype(int)
    
    # 数据预处理
    X = preprocess_data(X)
    X, y = data_quality_check(X, y)
    
    # 处理罕见类别
    class_counts = pd.Series(y).value_counts()
    rare_classes = class_counts[class_counts <= 1].index.tolist()
    if rare_classes:
        mask = ~pd.Series(y).isin(rare_classes)
        X = X[mask].reset_index(drop=True)
        y = y[mask]
        dates = dates[mask]
        print(f"移除了罕见类别: {rare_classes}")
    
    # 划分数据集
    remaining_counts = pd.Series(y).value_counts()
    stratify_param = y if all(remaining_counts > 1) else None
    
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, 
        test_size=0.1, 
        random_state=3333,
        stratify=stratify_param
    )
    
    # 计算类别权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # 创建预处理管道
    preprocessor = create_preprocessor('xgb')
    
    # 创建XGBoost分类模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, n_jobs=n_jobs, eval_metric='mlogloss'))
    ])
    
    # XGBoost参数分布 - 使用随机搜索
    param_dist = {
        'classifier__n_estimators': randint(50, 300),
        'classifier__max_depth': randint(3, 10),
        'classifier__learning_rate': uniform(0.01, 0.3),
        'classifier__subsample': uniform(0.6, 0.4),
        'classifier__colsample_bytree': uniform(0.6, 0.4),
        'classifier__reg_alpha': uniform(0, 1),
        'classifier__reg_lambda': uniform(0, 1),
        'classifier__min_child_weight': randint(1, 10)
    }

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=1,
        random_state=42
    )
    
    try:
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        print(f"最佳XGBoost模型: {random_search.best_estimator_}")
        print(f"最佳F1分数: {random_search.best_score_:.4f}")
    except Exception as e:
        print(f"随机搜索失败: {str(e)}，使用默认参数")
        best_model = model
        best_model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"测试集F1分数: {f1:.4f}")
    
    # 保存模型
    joblib.dump(best_model, 'outputs/kz_num/pkls/0822_kz_num_xgb_classification_model.pkl')
    print("XGBoost模型已保存")
    
    return best_model

def train_all_models():
    """训练所有模型"""
    print("开始训练所有模型...")
    
    # 训练各个模型
    lr_model = train_lr_model()
    mlp_model = train_mlp_model()
    rf_model = train_rf_model()
    svm_model = train_svm_model()
    xgb_model = train_xgb_model()
    
    print("\n" + "=" * 50)
    print("所有模型训练完成！")
    print("=" * 50)
    
    return {
        'lr': lr_model,
        'mlp': mlp_model,
        'rf': rf_model,
        'svm': svm_model,
        'xgb': xgb_model
    }

if __name__ == '__main__':
    train_all_models()