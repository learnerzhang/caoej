import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from train_base import (
    n_jobs, load_and_preprocess_data, create_data_splits, 
    create_advanced_preprocessor, evaluate_model
)

def train_catboost():
    """训练CatBoost模型"""
    print("=== 训练CatBoost模型 ===")
    
    # 加载和预处理数据
    X, y_binary = load_and_preprocess_data()
    
    # 划分数据
    X_train, X_test, y_train, y_test = create_data_splits(X, y_binary, strategy='time_series')
    
    # 创建预处理和模型管道
    preprocessor = create_advanced_preprocessor(X_train.columns)
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(
            random_state=42,
            n_estimators=200,
            verbose=0,
            auto_class_weights='Balanced'
        ))
    ])
    
    # 参数网格
    param_grid = {
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__depth': [3, 5, 7],
        'classifier__subsample': [0.8, 0.9, 1.0],
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
    metrics = evaluate_model(best_model, X_test, y_test, "catboost")
    
    # 保存模型
    joblib.dump(best_model, 'outputs/binary/pkls/enhanced_binary_catboost_model.pkl')
    print("CatBoost模型已保存")
    
    return best_model, metrics

if __name__ == '__main__':
    train_catboost()