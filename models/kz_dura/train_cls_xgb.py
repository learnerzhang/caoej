import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import os
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from utils import SafeSimpleImputer
# 设置使用的CPU核心数
n_jobs = max(1, os.cpu_count() // 2)

# 设置字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False


def create_enhanced_preprocessor():
    """创建增强的预处理管道，特别关注类别7相关特征"""
    numeric_features = [
        # 时间特征
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_of_year',
        
        # 历史操作特征 - 特别关注与长时间操作相关的特征
        'prev_gate_count', 'prev_duration', 'prev_op_hour',
        'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration',
        
        # 潮汐特征 - 长时间操作可能与特定潮汐模式相关
        'tide_24h_mean', 'tide_24h_max', 'tide_24h_min', 'tide_24h_range',
        'tide_24h_slope', 'tide_24h_r_squared', 'tide_24h_cycle_count',
        'tide_24h_rise_rate', 'tide_24h_fall_rate', 'tide_24h_phase',
        'tide_12h_mean', 'tide_12h_max', 'tide_12h_min', 'tide_12h_range',
        'tide_12h_slope', 'tide_12h_r_squared', 'tide_12h_cycle_count',
        'tide_12h_rise_rate', 'tide_12h_fall_rate', 'tide_12h_phase',
        
        # 未来潮汐特征
        'future_tide_mean', 'future_tide_max', 'future_tide_min', 'future_tide_range',
        'future_tide_slope', 'future_tide_r_squared', 'future_tide_cycle_count',
        'future_tide_rise_rate', 'future_tide_fall_rate', 'future_tide_phase',
        
        # 流量特征 - 高流量可能与长时间操作相关
        'flow_mean', 'flow_max', 'flow_min', 'flow_range', 'flow_var', 'flow_skew',
        
        # 降雨特征
        'rain_actual_total', 'rain_forecast_total', 
        'rain_actual_avg', 'rain_forecast_avg',
        'rain_change_rate', 'water_rain_ratio', 'flow_rain_ratio',
        
        # 水位工况特征
        'water_status_mean', 'water_status_max', 'water_status_min', 
        'water_status_range', 'water_status_slope',
        
        # 其他特征
        'is_rush_hour'
    ]
    
    categorical_features = ['tide_type']
    
    # 使用安全的数值型转换器
    numeric_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 使用安全的分类特征处理器
    categorical_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='constant', fill_value=0)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    indicator_features = [
        'water_missing', 'flow_missing', 
        'rain_missing', 'water_status_missing'
    ]
    
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

def analyze_class7_patterns(X, y_duration):
    """分析类别7（长时间开闸）的模式"""
    print("\n=== 类别7（长时间开闸）模式分析 ===")
    
    # 标记类别7的样本
    is_class7 = (y_duration == 7)
    class7_samples = X[is_class7]
    other_samples = X[~is_class7]
    
    print(f"类别7样本数量: {len(class7_samples)}")
    print(f"其他类别样本数量: {len(other_samples)}")
    
    # 分析类别7的特征分布
    numeric_features = X.select_dtypes(include=[np.number]).columns
    
    # 计算类别7与其他类别的特征差异
    feature_differences = {}
    for feature in numeric_features[:20]:  # 只分析前20个特征避免过多输出
        if feature in X.columns:
            class7_mean = class7_samples[feature].mean()
            other_mean = other_samples[feature].mean()
            
            if other_mean != 0:
                difference = (class7_mean - other_mean) / other_mean * 100
                feature_differences[feature] = difference
    
    # 按差异大小排序
    sorted_differences = sorted(feature_differences.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\n类别7特征差异最大的前10个特征:")
    for feature, diff in sorted_differences[:10]:
        print(f"{feature}: {diff:.2f}%")
    
    return sorted_differences

def create_class7_enhanced_features(X, y_duration):
    """创建增强特征，特别关注类别7的模式"""
    X_enhanced = X.copy()
    
    # 添加类别7相关的交互特征
    is_class7_related = (y_duration >= 5)  # 假设5及以上都是较长时间操作
    
    # 计算与长时间操作相关的特征组合
    if 'flow_mean' in X.columns and 'tide_24h_mean' in X.columns:
        X_enhanced['flow_tide_interaction'] = X['flow_mean'] * X['tide_24h_mean']
    
    if 'rain_actual_total' in X.columns and 'water_status_mean' in X.columns:
        X_enhanced['rain_water_interaction'] = X['rain_actual_total'] * X['water_status_mean']
    
    # 添加长时间操作的指示特征
    X_enhanced['is_potential_long_op'] = 0
    if 'prev_duration' in X.columns:
        X_enhanced.loc[X['prev_duration'] > 120, 'is_potential_long_op'] = 1
    
    return X_enhanced

def create_sample_weights(y, class_weight_dict):
    """为每个样本创建权重，替代scale_pos_weight"""
    sample_weights = np.ones(len(y))
    for class_label, weight in class_weight_dict.items():
        sample_weights[y == class_label] = weight
    return sample_weights

def train_xgb_classification_model_with_class7_focus():
    """训练XGBoost开闸时长分类模型，特别关注类别7"""
    # 创建输出目录
    os.makedirs('outputs/kz_dura_xgb_class7/plots', exist_ok=True)
    os.makedirs('outputs/kz_dura_xgb_class7/pkls', exist_ok=True)
    os.makedirs('outputs/kz_dura_xgb_class7/outputs', exist_ok=True)
    
    # 加载特征数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    y = np.load('features/features_0822/00_target.npy')
    
    # 提取开闸时长作为目标变量
    y_duration = y[:, 1].astype(int)  # 开闸时长
    
    # 数据预处理
    X = preprocess_data(X)
    
    # 分析类别7的模式
    feature_differences = analyze_class7_patterns(X, y_duration)
    
    # 创建增强特征
    X_enhanced = create_class7_enhanced_features(X, y_duration)
    
    # 数据质量检查
    X_enhanced, y_duration = data_quality_check(X_enhanced, y_duration)
    
    # 直接将开闸时长作为类别标签
    # 编码类别标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_duration)
    
    # 处理罕见类别 - 对类别7进行特殊处理
    class_counts = pd.Series(y_encoded).value_counts()
    print("\n类别分布详情:")
    for class_id, count in class_counts.items():
        original_duration = label_encoder.inverse_transform([class_id])[0]
        print(f"时长 {original_duration} (编码 {class_id}): {count} 样本 ({count/len(y_encoded)*100:.2f}%)")
    
    # 对类别7进行过采样
    class7_encoded = label_encoder.transform([7])[0] if 7 in label_encoder.classes_ else -1
    
    if class7_encoded != -1 and class_counts[class7_encoded] < 20:  # 如果类别7样本少于20个
        print(f"\n类别7样本较少 ({class_counts[class7_encoded]}个)，进行SMOTE过采样...")
        
        # 使用SMOTE进行过采样
        smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts[class7_encoded]-1))
        X_resampled, y_resampled = smote.fit_resample(X_enhanced, y_encoded)
        
        print(f"过采样后样本数量: {len(X_resampled)}")
        X_enhanced = X_resampled
        y_encoded = y_resampled
        y_duration = label_encoder.inverse_transform(y_encoded)
    
    # 检查剩余类别的样本量
    remaining_counts = pd.Series(y_encoded).value_counts()
    if any(remaining_counts <= 1):
        stratify_param = None
        print("警告: 存在样本量不足的类别，禁用分层抽样")
    else:
        stratify_param = y_encoded
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y_encoded, test_size=0.1, random_state=1122, stratify=stratify_param
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 计算类别权重 - 特别加强类别7的权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=classes, 
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    
    # 特别加强类别7的权重
    if class7_encoded in class_weight_dict:
        class_weight_dict[class7_encoded] = class_weight_dict[class7_encoded] * 3.0  # 3倍权重
        print(f"类别7权重加强: {class_weight_dict[class7_encoded]:.2f}")
    
    # 对长时间类别给予额外权重
    long_duration_classes = [cls for cls in classes if label_encoder.inverse_transform([cls])[0] >= 5]  # 5及以上为长时间
    for cls in long_duration_classes:
        if cls in class_weight_dict and cls != class7_encoded:
            class_weight_dict[cls] = class_weight_dict[cls] * 1.5
            original_duration = label_encoder.inverse_transform([cls])[0]
            print(f"调整长时间类别 {original_duration} 分钟权重: {class_weight_dict[cls]:.2f}")
    
    print("最终类别权重:", class_weight_dict)
    
    # 创建样本权重，替代scale_pos_weight
    sample_weights = create_sample_weights(y_train, class_weight_dict)
    
    # 使用增强的预处理管道
    preprocessor = create_enhanced_preprocessor()
    
    # XGBoost专用的参数网格，特别优化类别7的检测
    # 移除了scale_pos_weight参数，使用样本权重替代
    xgb_param_grid = {
        'classifier__n_estimators': [200, 300, 400],
        'classifier__max_depth': [5, 7, 9],
        'classifier__learning_rate': [0.05, 0.1, 0.15],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__reg_alpha': [0, 0.1, 0.5],
        'classifier__reg_lambda': [1, 1.5, 2]
    }
    
    # 创建XGBoost分类模型管道
    xgb_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=n_jobs,
            use_label_encoder=False
        ))
    ])
    
    # 使用分层K折交叉验证
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=xgb_param_grid,
        cv=cv_strategy,
        scoring='f1_weighted',
        n_jobs=1,  # XGBoost内部已使用多线程，此处设为1避免冲突
        verbose=1
    )
    
    print("开始网格搜索寻找最佳XGBoost分类模型（类别7优化）...")
    
    try:
        # 使用样本权重进行训练
        grid_search.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        
        # 分类评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 特别关注类别7的指标
        class7_f1 = f1_score(y_test, y_pred, average=None)[class7_encoded] if class7_encoded in np.unique(y_test) else 0
        
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, 
                                  target_names=[str(cls) for cls in label_encoder.classes_]))
        
        print(f"\n测试集准确率: {accuracy:.4f}")
        print(f"测试集加权F1分数: {f1:.4f}")
        print(f"类别7的F1分数: {class7_f1:.4f}")
        
        # 可视化混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_, 
                    yticklabels=label_encoder.classes_)
        plt.xlabel('预测时长')
        plt.ylabel('实际时长')
        plt.title('XGBoost开闸时长预测混淆矩阵（类别7优化）')
        plt.tight_layout()
        plt.savefig('outputs/kz_dura_xgb_class7/plots/0822_kz_dura_xgb_class7_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 特征重要性分析
        try:
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
            
            classifier = best_model.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_df.to_csv('outputs/kz_dura_xgb_class7/outputs/0822_kz_dura_xgb_class7_feature_importance.csv', index=False)
                
                # 可视化最重要的特征
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=importance_df.head(20))
                plt.title('XGBoost开闸时长分类模型特征重要性排名（类别7优化）')
                plt.tight_layout()
                plt.savefig('outputs/kz_dura_xgb_class7/plots/0822_kz_dura_xgb_class7_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"无法获取特征重要性: {str(e)}")
        
        # 保存模型和标签编码器
        joblib.dump(best_model, 'outputs/kz_dura_xgb_class7/pkls/0822_kz_dura_xgb_class7_classification_model.pkl')
        joblib.dump(label_encoder, 'outputs/kz_dura_xgb_class7/pkls/0822_kz_dura_xgb_class7_label_encoder.pkl')
        print("XGBoost分类模型已保存到 outputs/kz_dura_xgb_class7/pkls/0822_kz_dura_xgb_class7_classification_model.pkl")
        print("标签编码器已保存到 outputs/kz_dura_xgb_class7/pkls/0822_kz_dura_xgb_class7_label_encoder.pkl")
        
        # 保存网格搜索结果
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv('outputs/kz_dura_xgb_class7/outputs/0822_kz_dura_xgb_class7_grid_search_results.csv', index=False)
        
        # 特别分析类别7的预测结果
        test_results = pd.DataFrame({
            'date': X_test['date'] if 'date' in X_test.columns else X_test.index,
            'true_class': label_encoder.inverse_transform(y_test),
            'pred_class': label_encoder.inverse_transform(y_pred),
            'true_duration': y_duration[X_test.index] if hasattr(X_test, 'index') else y_duration[len(y_train):],
            'correct': (y_test == y_pred)
        })
        
        # 分析类别7的预测情况
        class7_results = test_results[test_results['true_class'] == 7]
        if len(class7_results) > 0:
            class7_accuracy = class7_results['correct'].mean()
            print(f"\n类别7预测准确率: {class7_accuracy:.4f} ({len(class7_results)}个样本)")
            
            # 保存类别7详细预测结果
            class7_results.to_csv('outputs/kz_dura_xgb_class7/outputs/0822_xgb_class7_detailed_predictions.csv', index=False)
        
        test_results.to_csv('outputs/kz_dura_xgb_class7/outputs/0822_test_xgb_class7_predictions.csv', index=False)
        print("测试集预测结果已保存")
        
        return best_model, label_encoder
        
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("尝试使用默认参数训练XGBoost...")
        
        # 使用默认参数训练
        default_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                random_state=42,
                n_estimators=300,
                max_depth=7,
                learning_rate=0.1,
                eval_metric='mlogloss',
                n_jobs=n_jobs,
                use_label_encoder=False
            ))
        ])
        
        # 使用样本权重训练默认模型
        default_model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        y_pred = default_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        class7_f1 = f1_score(y_test, y_pred, average=None)[class7_encoded] if class7_encoded in np.unique(y_test) else 0
        
        print(f"默认XGBoost模型测试集准确率: {accuracy:.4f}")
        print(f"默认XGBoost模型测试集F1分数: {f1:.4f}")
        print(f"默认XGBoost模型类别7的F1分数: {class7_f1:.4f}")
        
        # 保存默认模型
        joblib.dump(default_model, 'outputs/kz_dura_xgb_class7/pkls/0822_kz_dura_xgb_class7_classification_model_default.pkl')
        joblib.dump(label_encoder, 'outputs/kz_dura_xgb_class7/pkls/0822_kz_dura_xgb_class7_label_encoder_default.pkl')
        print("默认XGBoost分类模型已保存")
        
        return default_model, label_encoder

if __name__ == '__main__':
    model, encoder = train_xgb_classification_model_with_class7_focus()
    print("XGBoost开闸时长分类模型（类别7优化）训练完成！")