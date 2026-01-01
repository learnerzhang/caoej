import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

from train_base import (
    n_jobs, load_and_preprocess_data, create_data_splits, 
    create_advanced_preprocessor, evaluate_model, feature_selection
)

# 修改 train_lightgbm.py 文件中的特征选择部分

def train_lightgbm():
    """训练LightGBM模型"""
    print("=== 训练LightGBM模型 ===")
    
    # 加载和预处理数据
    X, y_binary = load_and_preprocess_data()
    
    # 特征选择 - 修改为只选择数值型特征
    print("进行特征选择...")
    
    # 只选择数值型特征进行特征选择
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    X_numeric = X[numeric_features]
    
    selected_features = feature_selection(X_numeric, y_binary, n_features=100)  # 减少特征数量
    
    # 确保选中的特征在原始数据中存在
    existing_features = [feat for feat in selected_features if feat in X.columns]
    print(f"原始选中特征: {len(selected_features)}, 实际存在特征: {len(existing_features)}")
    
    X = X[existing_features]
    
    # 划分数据
    X_train, X_test, y_train, y_test = create_data_splits(X, y_binary, strategy='time_series')
    
    # 创建预处理和模型管道
    preprocessor = create_advanced_preprocessor(X_train.columns)
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(
            random_state=42,
            n_estimators=200,
            class_weight='balanced',
            n_jobs=n_jobs
        ))
    ])
    
    # 调整参数网格 - 简化参数网格以加快训练
    param_grid = {
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__num_leaves': [20, 31],
        'classifier__max_depth': [5, 7],
        'classifier__min_child_samples': [20, 30],
    }
    
    # 使用分层K折交叉验证
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=n_jobs,
        verbose=1
    )
    
    # 训练模型
    print("开始网格搜索...")
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证AUC分数: {grid_search.best_score_:.4f}")
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 评估模型
    metrics = evaluate_model(best_model, X_test, y_test, "lightgbm")
    
    # 保存模型
    joblib.dump(best_model, 'outputs/binary/pkls/enhanced_binary_lightgbm_model.pkl')
    print("LightGBM模型已保存")
    
    return best_model, metrics

if __name__ == '__main__':
    train_lightgbm()