import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_class_weight

from train_base import (
    n_jobs, load_and_preprocess_data, create_data_splits, 
    create_advanced_preprocessor, evaluate_model
)

def train_logistic_regression():
    """训练逻辑回归模型"""
    print("=== 训练逻辑回归模型 ===")
    
    # 加载和预处理数据
    X, y_binary = load_and_preprocess_data()
    
    # 划分数据
    X_train, X_test, y_train, y_test = create_data_splits(X, y_binary, strategy='time_series')
    
    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"类别权重: {class_weight_dict}")
    
    # 创建预处理和模型管道
    preprocessor = create_advanced_preprocessor(X_train.columns)
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            random_state=42, 
            max_iter=1000, 
            class_weight=class_weight_dict,
            solver='liblinear'
        ))
    ])
    
    # 参数网格
    param_grid = {
        'classifier__penalty': ['l1', 'l2'],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
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
    metrics = evaluate_model(best_model, X_test, y_test, "logistic_regression")
    
    # 保存模型
    joblib.dump(best_model, 'outputs/binary/pkls/enhanced_binary_logistic_regression_model.pkl')
    print("逻辑回归模型已保存")
    
    return best_model, metrics

if __name__ == '__main__':
    train_logistic_regression()