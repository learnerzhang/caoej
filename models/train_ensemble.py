import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold

from train_base import (
    n_jobs, load_and_preprocess_data, create_data_splits, 
    create_advanced_preprocessor, evaluate_model
)

def train_ensemble(models):
    """训练集成模型"""
    print("=== 训练集成模型 ===")
    
    # 加载和预处理数据
    X, y_binary = load_and_preprocess_data()
    
    # 划分数据
    X_train, X_test, y_train, y_test = create_data_splits(X, y_binary, strategy='time_series')
    
    # 创建预处理管道
    preprocessor = create_advanced_preprocessor(X_train.columns)
    
    # 创建集成模型
    estimators = []
    for name, model_path in models.items():
        # 加载预训练的模型
        model = joblib.load(model_path)
        # 获取模型中的分类器
        classifier = model.named_steps['classifier']
        estimators.append((name, classifier))
    
    # 创建投票分类器
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    
    # 创建完整的管道
    ensemble_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', voting_clf)
    ])
    
    # 训练集成模型
    print("开始训练集成模型...")
    ensemble_model.fit(X_train, y_train)
    
    # 评估模型
    metrics = evaluate_model(ensemble_model, X_test, y_test, "ensemble")
    
    # 保存模型
    joblib.dump(ensemble_model, 'outputs/binary/pkls/enhanced_binary_ensemble_model.pkl')
    print("集成模型已保存")
    
    return ensemble_model, metrics

if __name__ == '__main__':
    # 定义要集成的模型
    models = {
        'random_forest': 'outputs/binary/pkls/enhanced_binary_random_forest_model.pkl',
        'xgboost': 'outputs/binary/pkls/enhanced_binary_xgboost_model.pkl',
        'lightgbm': 'outputs/binary/pkls/enhanced_binary_lightgbm_model.pkl'
    }
    
    train_ensemble(models)