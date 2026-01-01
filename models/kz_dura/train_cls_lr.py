import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import os
from sklearn.base import BaseEstimator, TransformerMixin
from utils import SafeSimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# 设置使用的CPU核心数
n_jobs = max(1, os.cpu_count() // 2)

# 设置字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """特征工程：创建交互特征和多项式特征"""
    
    def __init__(self, create_interaction=True, create_polynomial=False):
        self.create_interaction = create_interaction
        self.create_polynomial = create_polynomial
        self.interaction_pairs_ = []
        
    def fit(self, X, y=None):
        if self.create_interaction:
            # 自动选择数值型特征创建交互项
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            # 选择与目标变量相关性较高的特征
            if y is not None and len(numeric_cols) > 1:
                correlations = []
                for col in numeric_cols:
                    if len(np.unique(X[col])) > 1:  # 避免常数特征
                        try:
                            corr = np.corrcoef(X[col], y)[0, 1]
                            correlations.append((col, abs(corr)))
                        except:
                            continue
                
                # 选择相关性最高的前5个特征创建交互项
                correlations.sort(key=lambda x: x[1], reverse=True)
                top_features = [corr[0] for corr in correlations[:5]]
                
                # 生成特征对
                for i in range(len(top_features)):
                    for j in range(i+1, len(top_features)):
                        self.interaction_pairs_.append((top_features[i], top_features[j]))
        
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # 创建交互特征
        if self.create_interaction and self.interaction_pairs_:
            for feat1, feat2 in self.interaction_pairs_:
                if feat1 in X.columns and feat2 in X.columns:
                    interaction_name = f"{feat1}_x_{feat2}"
                    X_transformed[interaction_name] = X[feat1] * X[feat2]
        
        return X_transformed

def create_enhanced_preprocessor(use_pca=False, pca_components=0.95):
    """创建增强的预处理管道"""
    numeric_features = [
        # 时间特征
        'hour_of_day', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_of_year',
        
        # 历史操作特征
        'prev_gate_count', 'prev_duration', 'prev_op_hour',
        'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration',
        
        # 潮汐特征
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
        'is_rush_hour'
    ]
    
    categorical_features = ['tide_type']
    
    # 数值型特征处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SafeSimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 分类特征处理管道
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
    
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('ind', indicator_transformer, indicator_features)
    ]
    
    if use_pca:
        # 添加PCA降维
        transformers.append(('pca', PCA(n_components=pca_components), numeric_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor

def create_advanced_lr_pipeline(preprocessor, feature_selection_method='rfe', class_weight=None):
    """创建高级逻辑回归管道"""
    
    # 特征选择器
    if feature_selection_method == 'rfe':
        # 修正：n_features_to_select 不能为 'auto'，需要设置为具体的数值
        selector = RFE(
            estimator=LogisticRegression(random_state=42, max_iter=1000),
            n_features_to_select=0.5,  # 选择50%的特征，或者可以设置为固定数值如50
            step=0.1
        )
    else:  # selectfrommodel
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold='median'
        )
    
    # 逻辑回归分类器
    lr_classifier = LogisticRegression(
        random_state=42,
        max_iter=2000,
        class_weight=class_weight,
        multi_class='multinomial',
        solver='saga',  # saga支持所有正则化类型
        penalty='elasticnet',
        l1_ratio=0.5
    )
    
    # 构建管道
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', selector),
        ('classifier', lr_classifier)
    ])
    
    return pipeline

def optimized_lr_param_grid():
    """优化的逻辑回归参数网格"""
    return [
        {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['saga'],
            'classifier__l1_ratio': [0.1, 0.5, 0.9],
            'classifier__max_iter': [2000, 3000]
        },
        {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs', 'newton-cg'],
            'classifier__max_iter': [2000, 3000]
        }
    ]

def evaluate_model_performance(model, X_test, y_test, label_encoder, class7_encoded, output_dir):
    """全面评估模型性能"""
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # 基础指标
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    
    # 类别7特定指标
    class7_metrics = {}
    if class7_encoded != -1 and class7_encoded in np.unique(y_test):
        class7_f1 = f1_score(y_test, y_pred, average=None)[class7_encoded]
        class7_precision = precision_score(y_test, y_pred, average=None)[class7_encoded]
        class7_recall = recall_score(y_test, y_pred, average=None)[class7_encoded]
        class7_metrics = {
            'f1': class7_f1,
            'precision': class7_precision,
            'recall': class7_recall
        }
    
    print("\n=== 模型性能评估 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"加权F1分数: {f1_weighted:.4f}")
    print(f"宏平均F1分数: {f1_macro:.4f}")
    print(f"加权精确率: {precision_weighted:.4f}")
    print(f"加权召回率: {recall_weighted:.4f}")
    
    if class7_metrics:
        print(f"\n类别7性能:")
        print(f"F1分数: {class7_metrics['f1']:.4f}")
        print(f"精确率: {class7_metrics['precision']:.4f}")
        print(f"召回率: {class7_metrics['recall']:.4f}")
    
    # 分类报告
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, 
                              target_names=[str(cls) for cls in label_encoder.classes_]))
    
    # 保存评估结果
    metrics_df = pd.DataFrame({
        'metric': ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'recall_weighted'],
        'value': [accuracy, f1_weighted, f1_macro, precision_weighted, recall_weighted]
    })
    
    if class7_metrics:
        class7_metrics_df = pd.DataFrame({
            'metric': ['class7_f1', 'class7_precision', 'class7_recall'],
            'value': [class7_metrics['f1'], class7_metrics['precision'], class7_metrics['recall']]
        })
        metrics_df = pd.concat([metrics_df, class7_metrics_df], ignore_index=True)
    
    metrics_df.to_csv(f'{output_dir}/0822_model_metrics.csv', index=False)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'class7_metrics': class7_metrics
    }

def plot_advanced_visualizations(y_test, y_pred, y_pred_proba, label_encoder, output_dir):
    """创建高级可视化"""
    
    # 1. 混淆矩阵
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('逻辑回归分类混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/0822_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 类别概率分布
    plt.figure(figsize=(15, 8))
    for i, class_label in enumerate(label_encoder.classes_):
        plt.subplot(2, 4, i+1)
        plt.hist(y_pred_proba[:, i], bins=20, alpha=0.7, label=f'Class {class_label}')
        plt.title(f'类别 {class_label} 预测概率分布')
        plt.xlabel('预测概率')
        plt.ylabel('频数')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/0822_class_probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def handle_class_imbalance(X, y, label_encoder, target_class=7, min_samples=10):
    """处理类别不平衡问题"""
    
    class7_encoded = label_encoder.transform([target_class])[0] if target_class in label_encoder.classes_ else -1
    
    if class7_encoded != -1:
        class_counts = pd.Series(y).value_counts()
        class7_count = class_counts.get(class7_encoded, 0)
        
        if class7_count < min_samples:
            print(f"\n类别7样本较少 ({class7_count}个)，应用高级过采样...")
            
            # 尝试不同的过采样方法
            try:
                # 方法1: SMOTE
                smote = SMOTE(
                    random_state=42, 
                    k_neighbors=min(5, class7_count-1),
                    sampling_strategy={class7_encoded: max(min_samples, class7_count * 3)}
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)
                print("SMOTE过采样完成")
            except:
                # 方法2: ADASYN
                try:
                    adasyn = ADASYN(random_state=42, n_neighbors=min(3, class7_count-1))
                    X_resampled, y_resampled = adasyn.fit_resample(X, y)
                    print("ADASYN过采样完成")
                except:
                    # 方法3: 简单复制
                    print("使用简单复制方法")
                    X_class7 = X[y == class7_encoded]
                    y_class7 = y[y == class7_encoded]
                    
                    n_needed = min_samples - class7_count
                    if n_needed > 0:
                        # 复制样本
                        X_duplicated = pd.concat([X_class7] * (n_needed // class7_count + 1), ignore_index=True)
                        y_duplicated = np.tile(y_class7, (n_needed // class7_count + 1))
                        
                        # 合并
                        X_resampled = pd.concat([X, X_duplicated.iloc[:n_needed]], ignore_index=True)
                        y_resampled = np.concatenate([y, y_duplicated[:n_needed]])
                    else:
                        X_resampled, y_resampled = X, y
            
            return X_resampled, y_resampled
    
    return X, y

def train_optimized_lr_classification_model():
    """训练优化的逻辑回归分类模型"""
    
    # 创建输出目录
    output_base = 'outputs/kz_dura_lr_optimized'
    os.makedirs(f'{output_base}/plots', exist_ok=True)
    os.makedirs(f'{output_base}/pkls', exist_ok=True)
    os.makedirs(f'{output_base}/outputs', exist_ok=True)
    
    # 加载数据
    print("=== 加载数据 ===")
    X = pd.read_csv('features/features_0822/00_features.csv')
    y = np.load('features/features_0822/00_target.npy')
    y_duration = y[:, 1].astype(int)
    
    # 基础预处理
    X_processed = X.copy()
    numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if X_processed[col].isnull().any():
            median_val = X_processed[col].median()
            if pd.isna(median_val):
                median_val = 0
            X_processed[col].fillna(median_val, inplace=True)
    
    # 特征工程
    print("=== 特征工程 ===")
    feature_engineer = FeatureEngineer(create_interaction=True)
    X_enhanced = feature_engineer.fit_transform(X_processed, y_duration)
    
    # 编码目标变量
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_duration)
    
    # 处理类别不平衡
    X_balanced, y_balanced = handle_class_imbalance(X_enhanced, y_encoded, label_encoder)
    
    # 数据质量检查
    print("=== 数据质量检查 ===")
    print(f"特征形状: {X_balanced.shape}")
    print(f"目标变量分布:")
    target_counts = pd.Series(y_balanced).value_counts().sort_index()
    for class_id, count in target_counts.items():
        original_duration = label_encoder.inverse_transform([class_id])[0]
        print(f"时长 {original_duration}: {count} 样本 ({count/len(y_balanced)*100:.2f}%)")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.15, random_state=1122, stratify=y_balanced
    )
    
    print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 计算类别权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=classes, 
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    
    # 加强类别7权重
    class7_encoded = label_encoder.transform([7])[0] if 7 in label_encoder.classes_ else -1
    if class7_encoded in class_weight_dict:
        class_weight_dict[class7_encoded] *= 3.0
        print(f"类别7权重加强: {class_weight_dict[class7_encoded]:.2f}")
    
    # 创建预处理管道和模型
    preprocessor = create_enhanced_preprocessor(use_pca=False)
    model = create_advanced_lr_pipeline(preprocessor, 'rfe', class_weight_dict)
    
    # 网格搜索优化
    print("\n=== 开始网格搜索 ===")
    param_grid = optimized_lr_param_grid()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True
    )
    
    try:
        grid_search.fit(X_train, y_train)
        
        print(f"\n最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # 全面评估
        metrics = evaluate_model_performance(
            best_model, X_test, y_test, label_encoder, class7_encoded, 
            f'{output_base}/outputs'
        )
        
        # 可视化
        y_pred_proba = best_model.predict_proba(X_test)
        plot_advanced_visualizations(y_test, grid_search.predict(X_test), 
                                   y_pred_proba, label_encoder, 
                                   f'{output_base}/plots')
        
        # 保存结果
        joblib.dump(best_model, f'{output_base}/pkls/0822_optimized_lr_model.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/0822_label_encoder.pkl')
        
        # 保存网格搜索详情
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(f'{output_base}/outputs/0822_grid_search_results.csv', index=False)
        
        print(f"\n优化后的逻辑回归模型训练完成!")
        print(f"模型保存位置: {output_base}/pkls/")
        
        return best_model, label_encoder, metrics
        
    except Exception as e:
        print(f"网格搜索失败: {e}")
        print("使用默认参数训练...")
        
        # 使用默认参数
        default_model = create_advanced_lr_pipeline(preprocessor, 'rfe', class_weight_dict)
        default_model.fit(X_train, y_train)
        
        metrics = evaluate_model_performance(
            default_model, X_test, y_test, label_encoder, class7_encoded,
            f'{output_base}/outputs'
        )
        
        joblib.dump(default_model, f'{output_base}/pkls/0822_default_lr_model.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/0822_label_encoder.pkl')
        
        return default_model, label_encoder, metrics

if __name__ == '__main__':
    print("开始训练优化的逻辑回归分类模型...")
    model, encoder, metrics = train_optimized_lr_classification_model()
    print("训练完成!")