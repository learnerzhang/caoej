import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from train_base import (
    n_jobs, load_and_preprocess_data, create_data_splits, 
    create_advanced_preprocessor, evaluate_model
)

def train_xgboost():
    """训练XGBoost模型"""
    print("=== 训练XGBoost模型 ===")
    
    # 加载和预处理数据
    X, y_binary = load_and_preprocess_data()
    
    # 划分数据
    X_train, X_test, y_train, y_test = create_data_splits(X, y_binary, strategy='time_series')
    
    # 计算正负样本比例
    scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1])
    
    # 创建预处理和模型管道
    preprocessor = create_advanced_preprocessor(X_train.columns)
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            random_state=42,
            n_estimators=200,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        ))
    ])
    
    # 参数网格
    param_grid = {
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
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
    metrics = evaluate_model(best_model, X_test, y_test, "xgboost")
    
    # 保存模型
    joblib.dump(best_model, 'outputs/binary/pkls/enhanced_binary_xgboost_model.pkl')
    print("XGBoost模型已保存")
    
    return best_model, metrics

if __name__ == '__main__':
    train_xgboost()