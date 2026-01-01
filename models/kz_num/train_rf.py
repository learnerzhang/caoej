import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.base import BaseEstimator, TransformerMixin
from utils import SafeSimpleImputer
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
    print(f"\n目标变量分布:")
    target_counts = pd.Series(y).value_counts().sort_index()
    for class_id, count in target_counts.items():
        print(f"类别 {class_id}: {count} 样本 ({count/len(y)*100:.2f}%)")
    
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
    
    # 使用安全的数值型转换器 - 随机森林不需要标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='median'))
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

def analyze_errors(test_results, y_test, y_pred):
    """分析预测错误的情况"""
    errors = test_results[test_results['true_gate_count'] != test_results['pred_gate_count']]
    print(f"\n错误分析: 总共 {len(test_results)} 个样本，{len(errors)} 个预测错误")
    
    if len(errors) > 0:
        # 分析最常见的错误类型
        error_pairs = errors.groupby(['true_gate_count', 'pred_gate_count']).size().reset_index(name='count')
        error_pairs = error_pairs.sort_values('count', ascending=False)
        print("\n最常见的错误类型:")
        for _, row in error_pairs.head(10).iterrows():
            print(f"真实孔数 {row['true_gate_count']} -> 预测孔数 {row['pred_gate_count']}: {row['count']} 次")
        
        # 保存错误分析结果
        error_pairs.to_csv('outputs/kz_num/outputs/0822_kz_num_error_analysis.csv', index=False)
        
        # 可视化错误分布
        plt.figure(figsize=(12, 8))
        error_counts = errors['true_gate_count'].value_counts().sort_index()
        total_counts = test_results['true_gate_count'].value_counts().sort_index()
        error_rates = (error_counts / total_counts * 100).fillna(0)
        
        x_pos = np.arange(len(error_rates))
        plt.bar(x_pos, error_rates.values)
        plt.xlabel('真实孔数')
        plt.ylabel('错误率 (%)')
        plt.title('各类别的错误率')
        plt.xticks(x_pos, error_rates.index)
        plt.tight_layout()
        plt.savefig('outputs/kz_num/plots/0822_kz_num_error_rates.png', dpi=300, bbox_inches='tight')
        plt.close()

def train_and_save_model():
    """训练随机森林分类模型"""
    # 创建输出目录
    os.makedirs('outputs/kz_num/plots', exist_ok=True)
    os.makedirs('outputs/kz_num/pkls', exist_ok=True)
    os.makedirs('outputs/kz_num/outputs', exist_ok=True)
    
    # 加载特征数据
    data = pd.read_csv('features/features_0822/11_features.csv')
    # 提取日期列
    dates = data['date']
    X = data.drop(columns=['date'])
    
    # 加载目标数据（开闸孔数）
    y_targets = np.load('features/features_0822/11_target.npy')
    y = y_targets[:, 2].astype(int)  # 开闸孔数列
    
    # 数据预处理
    X = preprocess_data(X)
    
    # 数据质量检查
    X, y = data_quality_check(X, y)
    
    # 处理罕见类别
    class_counts = pd.Series(y).value_counts()
    print("\n类别分布详情:")
    for class_id, count in class_counts.items():
        print(f"类别 {class_id}: {count} 样本 ({count/len(y)*100:.2f}%)")
    
    rare_classes = class_counts[class_counts <= 2].index.tolist()
    
    if rare_classes:
        mask = ~pd.Series(y).isin(rare_classes)
        X = X[mask].reset_index(drop=True)
        y = y[mask]
        dates = dates[mask].reset_index(drop=True)
        print(f"移除了罕见类别: {rare_classes}")
    
    # 检查剩余类别的样本量
    remaining_counts = pd.Series(y).value_counts()
    if any(remaining_counts <= 2):
        stratify_param = None
        print("警告: 存在样本量不足的类别，禁用分层抽样")
    else:
        stratify_param = y
    
    # 划分数据集
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, test_size=0.1, random_state=3333, stratify=stratify_param
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 计算类别权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=classes, 
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    print("类别权重:", class_weight_dict)
    
    # 对第5类（大孔数）给予额外权重
    if 5 in class_weight_dict:
        class_weight_dict[5] = class_weight_dict[5] * 1.5
        print("调整后第5类权重:", class_weight_dict[5])
    
    # 创建预处理管道
    preprocessor = create_preprocessor()
    
    # 创建随机森林分类模型管道
    base_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight=class_weight_dict, n_jobs=n_jobs))
    ])
    
    # 参数网格 - 针对随机森林优化
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['sqrt', 'log2']
    }
    
    # 使用分层K折交叉验证
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 减少折数提高速度
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=1,  # 随机森林已经使用了n_jobs，这里设为1避免冲突
        verbose=1
    )
    
    print("开始网格搜索寻找最佳随机森林模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        
        print(f"最佳随机森林模型: {grid_search.best_estimator_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        
        # 确保预测结果是整数类型
        y_pred = y_pred.astype(int)
        
        # 分类评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n测试集准确率: {accuracy:.4f}")
        print(f"测试集F1分数: {f1:.4f}")
        
        # 可视化混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y), 
                    yticklabels=np.unique(y))
        plt.xlabel('预测孔数')
        plt.ylabel('实际孔数')
        plt.title('随机森林开闸孔数预测混淆矩阵')
        plt.tight_layout()
        plt.savefig('outputs/kz_num/plots/0822_kz_num_rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 特征重要性分析
        try:
            # 获取预处理后的特征名称
            preprocessor = best_model.named_steps['preprocessor']
            feature_names = []
            
            # 数值特征
            num_features = preprocessor.transformers_[0][2]
            feature_names.extend(num_features)
            
            # 分类特征
            if len(preprocessor.transformers_) > 1 and preprocessor.transformers_[1][0] == 'cat':
                cat_transformer = preprocessor.transformers_[1][1].named_steps['onehot']
                cat_features = preprocessor.transformers_[1][2]
                cat_names = cat_transformer.get_feature_names_out(cat_features)
                feature_names.extend(cat_names)
            
            # 指示器特征
            if len(preprocessor.transformers_) > 2 and preprocessor.transformers_[2][0] == 'ind':
                ind_features = preprocessor.transformers_[2][2]
                feature_names.extend(ind_features)
            
            # 获取随机森林特征重要性
            classifier = best_model.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_df.to_csv('outputs/kz_num/outputs/0822_kz_num_rf_feature_importance.csv', index=False)
                
                # 可视化最重要的特征
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', 
                            data=importance_df.head(20))
                plt.title('随机森林开闸孔数预测模型特征重要性排名')
                plt.tight_layout()
                plt.savefig('outputs/kz_num/plots/0822_kz_num_rf_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            print(f"特征重要性分析失败: {str(e)}")
        
        # 保存模型
        joblib.dump(best_model, 'outputs/kz_num/pkls/0822_kz_num_rf_classification_model.pkl')
        print("随机森林分类模型已保存")
        
        # 保存网格搜索结果
        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv('outputs/kz_num/outputs/0822_kz_num_rf_grid_search_results.csv', index=False)
        
        # 返回测试集预测结果
        test_results = pd.DataFrame({
            'date': dates_test.values,
            'true_gate_count': y_test,
            'pred_gate_count': y_pred
        })
        test_results.to_csv('outputs/kz_num/outputs/test_predictions_rf_num.csv', index=False)
        print("测试集预测结果已保存")
        
        # 分析错误预测
        analyze_errors(test_results, y_test, y_pred)
        
        # 按日期分组的预测结果可视化
        try:
            # 按日期分组计算每日平均开闸孔数
            daily_results = test_results.groupby('date').agg({
                'true_gate_count': 'mean',
                'pred_gate_count': 'mean'
            }).reset_index()
            
            # 保存每日结果
            daily_results.to_csv('outputs/kz_num/outputs/daily_gate_count_rf_predictions.csv', index=False)
            print("每日开闸孔数预测结果已保存")
            
            # 可视化每日预测对比
            plt.figure(figsize=(14, 7))
            plt.plot(pd.to_datetime(daily_results['date']), daily_results['true_gate_count'], 'o-', label='真实开闸孔数')
            plt.plot(pd.to_datetime(daily_results['date']), daily_results['pred_gate_count'], 's--', label='预测开闸孔数')
            plt.xlabel('日期')
            plt.ylabel('开闸孔数')
            plt.title('每日真实开闸孔数 vs 预测开闸孔数 (随机森林)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('outputs/kz_num/plots/daily_gate_count_rf_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("每日开闸孔数对比图已保存")
            
        except Exception as e:
            print(f"生成每日对比图时出错: {str(e)}")
            
        return best_model
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("尝试使用默认参数训练随机森林模型...")
        
        # 使用默认参数训练
        default_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                random_state=42, 
                class_weight=class_weight_dict,
                n_estimators=100,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                n_jobs=n_jobs
            ))
        ])
        
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"默认随机森林模型测试集准确率: {accuracy:.4f}")
        print(f"默认随机森林模型测试集F1分数: {f1:.4f}")
        
        # 保存默认模型
        joblib.dump(default_model, 'outputs/kz_num/pkls/0822_kz_num_rf_classification_model_default.pkl')
        print("默认随机森林分类模型已保存")
        
        return default_model

if __name__ == '__main__':
    model = train_and_save_model()
    print("随机森林分类模型训练完成！")