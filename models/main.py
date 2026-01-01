import pandas as pd
import numpy as np
import joblib
import os

from train_logistic import train_logistic_regression
from train_random_forest import train_random_forest
from train_gradient_boosting import train_gradient_boosting
from train_xgboost import train_xgboost
from train_lightgbm import train_lightgbm
from train_catboost import train_catboost
from train_ensemble import train_ensemble

def main():
    """主程序，训练所有模型并比较结果"""
    # 创建输出目录
    os.makedirs('outputs/binary/pkls', exist_ok=True)
    os.makedirs('outputs/binary/plots', exist_ok=True)
    os.makedirs('outputs/binary/outputs', exist_ok=True)
    os.makedirs('features/features_0822', exist_ok=True)
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 存储所有模型的结果
    results = {}
    
    # 训练所有模型
    print("开始训练所有模型...")
    
    try:
        # 逻辑回归
        lr_model, lr_metrics = train_logistic_regression()
        results['logistic_regression'] = lr_metrics
        
        # 随机森林
        rf_model, rf_metrics = train_random_forest()
        results['random_forest'] = rf_metrics
        
        # 梯度提升
        gb_model, gb_metrics = train_gradient_boosting()
        results['gradient_boosting'] = gb_metrics
        
        # XGBoost
        xgb_model, xgb_metrics = train_xgboost()
        results['xgboost'] = xgb_metrics
        
        # LightGBM
        lgb_model, lgb_metrics = train_lightgbm()
        results['lightgbm'] = lgb_metrics
        
        # CatBoost
        cb_model, cb_metrics = train_catboost()
        results['catboost'] = cb_metrics
        
        # 集成模型
        models_for_ensemble = {
            'random_forest': 'outputs/binary/pkls/enhanced_binary_random_forest_model.pkl',
            'xgboost': 'outputs/binary/pkls/enhanced_binary_xgboost_model.pkl',
            'lightgbm': 'outputs/binary/pkls/enhanced_binary_lightgbm_model.pkl'
        }
        ensemble_model, ensemble_metrics = train_ensemble(models_for_ensemble)
        results['ensemble'] = ensemble_metrics
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        return
    
    # 比较所有模型的结果
    print("\n=== 所有模型性能比较 ===")
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('roc_auc', ascending=False)
    print(results_df)
    
    # 保存比较结果
    results_df.to_csv('outputs/binary/outputs/model_comparison.csv')
    print("模型比较结果已保存")
    
    # 找出最佳模型
    best_model_name = results_df.index[0]
    print(f"\n最佳模型: {best_model_name}, AUC: {results_df.iloc[0]['roc_auc']:.4f}")

if __name__ == '__main__':
    main()