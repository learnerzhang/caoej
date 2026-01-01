import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from utils import SafeSimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

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
    
    missing_features = missing_info[missing_info['missing_count'] > 0]
    if len(missing_features) > 0:
        print("\n缺失值统计:")
        print(missing_features)
    
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

def evaluate_model_performance(model, X_test, y_test, label_encoder, class7_encoded, output_dir, model_name):
    """全面评估模型性能"""
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
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
    
    print(f"\n=== {model_name} 模型性能评估 ===")
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
    print(f"\n{model_name} 详细分类报告:")
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
    
    metrics_df.to_csv(f'{output_dir}/{model_name}_model_metrics.csv', index=False)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'class7_metrics': class7_metrics
    }

def plot_advanced_visualizations(y_test, y_pred, y_pred_proba, label_encoder, output_dir, model_name):
    """创建高级可视化"""
    
    # 1. 混淆矩阵
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name}分类混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 类别概率分布
    if y_pred_proba is not None:
        plt.figure(figsize=(15, 8))
        n_classes = len(label_encoder.classes_)
        n_cols = min(4, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        for i, class_label in enumerate(label_encoder.classes_):
            plt.subplot(n_rows, n_cols, i+1)
            plt.hist(y_pred_proba[:, i], bins=20, alpha=0.7, label=f'Class {class_label}')
            plt.title(f'类别 {class_label} 预测概率分布')
            plt.xlabel('预测概率')
            plt.ylabel('频数')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_class_probability_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

# ==================== 逻辑回归模型 ====================
def create_advanced_lr_pipeline(preprocessor, feature_selection_method='rfe', class_weight=None):
    """创建高级逻辑回归管道"""
    
    # 特征选择器
    if feature_selection_method == 'rfe':
        selector = RFE(
            estimator=LogisticRegression(random_state=42, max_iter=1000),
            n_features_to_select=0.5,
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
        solver='saga',
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
    """优化的逻辑回归参数网格 - 修复参数冲突"""
    return [
        {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'elasticnet'],
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

def train_lr_model():
    """训练逻辑回归分类模型"""
    print("=== 开始训练逻辑回归分类模型 ===")
    
    # 创建输出目录
    output_base = 'outputs/kz_dura_lr_optimized'
    os.makedirs(f'{output_base}/plots', exist_ok=True)
    os.makedirs(f'{output_base}/pkls', exist_ok=True)
    os.makedirs(f'{output_base}/outputs', exist_ok=True)
    
    # 加载数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    y = np.load('features/features_0822/00_target.npy')
    y_duration = y[:, 1].astype(int)
    
    # 基础预处理
    X_processed = preprocess_data(X)
    
    # 特征工程
    feature_engineer = FeatureEngineer(create_interaction=True)
    X_enhanced = feature_engineer.fit_transform(X_processed, y_duration)
    
    # 编码目标变量
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_duration)
    
    # 处理类别不平衡
    X_balanced, y_balanced = handle_class_imbalance(X_enhanced, y_encoded, label_encoder)
    
    # 数据质量检查
    data_quality_check(X_balanced, y_balanced)
    
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
    print("\n=== 开始逻辑回归网格搜索 ===")
    param_grid = optimized_lr_param_grid()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=1,  # 减少输出详细程度
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
            f'{output_base}/outputs', 'lr'
        )
        
        # 可视化
        y_pred_proba = best_model.predict_proba(X_test)
        plot_advanced_visualizations(y_test, grid_search.predict(X_test), 
                                   y_pred_proba, label_encoder, 
                                   f'{output_base}/plots', 'lr')
        
        # 保存结果
        joblib.dump(best_model, f'{output_base}/pkls/optimized_lr_model.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/lr_label_encoder.pkl')
        
        # 保存网格搜索详情
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(f'{output_base}/outputs/lr_grid_search_results.csv', index=False)
        
        print(f"\n✅ 逻辑回归模型训练完成!")
        print(f"模型保存位置: {output_base}/pkls/")
        
        return best_model, label_encoder, metrics
        
    except Exception as e:
        print(f"❌ 网格搜索失败: {e}")
        print("使用默认参数训练...")
        
        # 使用默认参数
        default_model = create_advanced_lr_pipeline(preprocessor, 'rfe', class_weight_dict)
        default_model.fit(X_train, y_train)
        
        metrics = evaluate_model_performance(
            default_model, X_test, y_test, label_encoder, class7_encoded,
            f'{output_base}/outputs', 'lr_default'
        )
        
        joblib.dump(default_model, f'{output_base}/pkls/default_lr_model.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/lr_label_encoder.pkl')
        
        return default_model, label_encoder, metrics

# ==================== MLP模型 ====================
def train_mlp_model():
    """训练MLP分类模型"""
    print("=== 开始训练MLP分类模型 ===")
    
    # 创建输出目录
    output_base = 'outputs/kz_dura_class7_mlp'
    os.makedirs(f'{output_base}/plots', exist_ok=True)
    os.makedirs(f'{output_base}/pkls', exist_ok=True)
    os.makedirs(f'{output_base}/outputs', exist_ok=True)
    
    # 加载数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    y = np.load('features/features_0822/00_target.npy')
    y_duration = y[:, 1].astype(int)
    
    # 数据预处理
    X_processed = preprocess_data(X)
    
    # 分析类别7的模式
    analyze_class7_patterns(X_processed, y_duration)
    
    # 创建增强特征
    X_enhanced = create_class7_enhanced_features(X_processed, y_duration)
    
    # 数据质量检查
    data_quality_check(X_enhanced, y_duration)
    
    # 编码目标变量
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_duration)
    
    # 处理类别不平衡
    class_counts = pd.Series(y_encoded).value_counts()
    class7_encoded = label_encoder.transform([7])[0] if 7 in label_encoder.classes_ else -1
    
    if class7_encoded != -1 and class_counts[class7_encoded] < 20:
        print(f"\n类别7样本较少 ({class_counts[class7_encoded]}个)，进行SMOTE过采样...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts[class7_encoded]-1))
        X_enhanced, y_encoded = smote.fit_resample(X_enhanced, y_encoded)
    
    # 划分数据集
    remaining_counts = pd.Series(y_encoded).value_counts()
    stratify_param = y_encoded if all(remaining_counts > 1) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y_encoded, test_size=0.1, random_state=1122, stratify=stratify_param
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
    
    # 加强类别7权重
    if class7_encoded in class_weight_dict:
        class_weight_dict[class7_encoded] = class_weight_dict[class7_encoded] * 3.0
        print(f"类别7权重加强: {class_weight_dict[class7_encoded]:.2f}")
    
    # 创建基础MLP分类器
    base_mlp = MLPClassifier(
        random_state=42, 
        max_iter=2000,
        early_stopping=True,
        learning_rate='adaptive',
        n_iter_no_change=50,
        validation_fraction=0.1
    )
    
    # 创建MLP分类模型管道
    model = Pipeline(steps=[
        ('preprocessor', create_enhanced_preprocessor()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict),
            threshold='median'
        )),
        ('classifier', BaggingClassifier(
            base_estimator=base_mlp,
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            n_jobs=n_jobs
        ))
    ])
    
    # MLP参数网格搜索
    param_grid = {
        'classifier__base_estimator__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'classifier__base_estimator__activation': ['relu', 'tanh'],
        'classifier__base_estimator__alpha': [0.0001, 0.001, 0.01],
        'classifier__base_estimator__learning_rate_init': [0.001, 0.01],
        'classifier__base_estimator__batch_size': [32, 64]
    }
    
    # 网格搜索
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=1,  # 减少输出详细程度
        error_score='raise'
    )
    
    print("开始网格搜索寻找最佳MLP模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        # 分类评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        class7_f1 = f1_score(y_test, y_pred, average=None)[class7_encoded] if class7_encoded in np.unique(y_test) else 0
        
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, 
                                  target_names=[str(cls) for cls in label_encoder.classes_]))
        
        print(f"\n测试集准确率: {accuracy:.4f}")
        print(f"测试集加权F1分数: {f1:.4f}")
        print(f"类别7的F1分数: {class7_f1:.4f}")
        
        # 可视化
        plot_advanced_visualizations(y_test, y_pred, y_pred_proba, label_encoder, 
                                   f'{output_base}/plots', 'mlp')
        
        # 保存模型
        joblib.dump(best_model, f'{output_base}/pkls/mlp_classification_model.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/mlp_label_encoder.pkl')
        
        # 保存网格搜索结果
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(f'{output_base}/outputs/mlp_grid_search_results.csv', index=False)
        
        print("✅ MLP分类模型训练完成!")
        return best_model, label_encoder
        
    except Exception as e:
        print(f"❌ 网格搜索失败: {str(e)}")
        print("尝试使用默认参数训练...")
        
        # 使用默认参数训练
        default_mlp = MLPClassifier(
            random_state=42,
            hidden_layer_sizes=(100,),
            max_iter=2000,
            early_stopping=True,
            learning_rate='adaptive'
        )
        
        default_model = Pipeline(steps=[
            ('preprocessor', create_enhanced_preprocessor()),
            ('feature_selection', SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict),
                threshold='median'
            )),
            ('classifier', BaggingClassifier(
                base_estimator=default_mlp,
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=42
            ))
        ])
        
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"默认模型测试集准确率: {accuracy:.4f}")
        print(f"默认模型测试集F1分数: {f1:.4f}")
        
        # 保存默认模型
        joblib.dump(default_model, f'{output_base}/pkls/mlp_classification_model_default.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/mlp_label_encoder_default.pkl')
        
        return default_model, label_encoder

# ==================== 随机森林模型 ====================
def train_rf_model():
    """训练随机森林分类模型"""
    print("=== 开始训练随机森林分类模型 ===")
    
    # 创建输出目录
    output_base = 'outputs/kz_dura'
    os.makedirs(f'{output_base}/plots', exist_ok=True)
    os.makedirs(f'{output_base}/pkls', exist_ok=True)
    os.makedirs(f'{output_base}/outputs', exist_ok=True)
    
    # 加载数据
    X = pd.read_csv('features/features_0822/11_features.csv')
    y = np.load('features/features_0822/11_target.npy')
    y_duration = y[:, 1].astype(int)
    
    # 数据预处理
    X_processed = preprocess_data(X)
    
    # 分析类别7的模式
    analyze_class7_patterns(X_processed, y_duration)
    
    # 创建增强特征
    X_enhanced = create_class7_enhanced_features(X_processed, y_duration)
    
    # 数据质量检查
    data_quality_check(X_enhanced, y_duration)
    
    # 编码目标变量
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_duration)
    
    # 处理类别不平衡
    class_counts = pd.Series(y_encoded).value_counts()
    class7_encoded = label_encoder.transform([7])[0] if 7 in label_encoder.classes_ else -1
    
    if class7_encoded != -1 and class_counts[class7_encoded] < 20:
        print(f"\n类别7样本较少 ({class_counts[class7_encoded]}个)，进行SMOTE过采样...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts[class7_encoded]-1))
        X_enhanced, y_encoded = smote.fit_resample(X_enhanced, y_encoded)
    
    # 划分数据集
    remaining_counts = pd.Series(y_encoded).value_counts()
    stratify_param = y_encoded if all(remaining_counts > 1) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y_encoded, test_size=0.1, random_state=1122, stratify=stratify_param
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
    
    # 加强类别7权重
    if class7_encoded in class_weight_dict:
        class_weight_dict[class7_encoded] = class_weight_dict[class7_encoded] * 3.0
        print(f"类别7权重加强: {class_weight_dict[class7_encoded]:.2f}")
    
    # 创建随机森林模型管道
    model = Pipeline(steps=[
        ('preprocessor', create_enhanced_preprocessor()),
        ('classifier', RandomForestClassifier(
            random_state=42, 
            class_weight=class_weight_dict,
            n_jobs=n_jobs
        ))
    ])
    
    # 优化的参数网格
    param_grid = {
        'classifier__n_estimators': [200, 300, 400],
        'classifier__max_depth': [15, 20, 25, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', 0.8],
        'classifier__bootstrap': [True, False]
    }
    
    # 网格搜索
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=1  # 减少输出详细程度
    )
    
    print("开始网格搜索寻找最佳随机森林分类模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        # 分类评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        class7_f1 = f1_score(y_test, y_pred, average=None)[class7_encoded] if class7_encoded in np.unique(y_test) else 0
        
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, 
                                  target_names=[str(cls) for cls in label_encoder.classes_]))
        
        print(f"\n测试集准确率: {accuracy:.4f}")
        print(f"测试集加权F1分数: {f1:.4f}")
        print(f"类别7的F1分数: {class7_f1:.4f}")
        
        # 可视化
        plot_advanced_visualizations(y_test, y_pred, y_pred_proba, label_encoder, 
                                   f'{output_base}/plots', 'rf')
        
        # 保存模型
        joblib.dump(best_model, f'{output_base}/pkls/rf_optimized_model.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/rf_label_encoder.pkl')
        
        # 保存网格搜索结果
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(f'{output_base}/outputs/rf_grid_search_results.csv', index=False)
        
        print("✅ 随机森林分类模型训练完成!")
        return best_model, label_encoder
        
    except Exception as e:
        print(f"❌ 网格搜索失败: {str(e)}")
        print("尝试使用默认参数训练...")
        
        # 使用默认参数训练
        default_model = Pipeline(steps=[
            ('preprocessor', create_enhanced_preprocessor()),
            ('classifier', RandomForestClassifier(
                random_state=42, 
                class_weight=class_weight_dict,
                n_estimators=300,
                max_depth=20,
                n_jobs=n_jobs
            ))
        ])
        
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"默认模型测试集准确率: {accuracy:.4f}")
        print(f"默认模型测试集F1分数: {f1:.4f}")
        
        # 保存默认模型
        joblib.dump(default_model, f'{output_base}/pkls/rf_optimized_model_default.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/rf_label_encoder_default.pkl')
        
        return default_model, label_encoder

# ==================== SVM模型 ====================
def train_svm_model():
    """训练SVM分类模型"""
    print("=== 开始训练SVM分类模型 ===")
    
    # 创建输出目录
    output_base = 'outputs/kz_dura_svm_class7'
    os.makedirs(f'{output_base}/plots', exist_ok=True)
    os.makedirs(f'{output_base}/pkls', exist_ok=True)
    os.makedirs(f'{output_base}/outputs', exist_ok=True)
    
    # 加载数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    y = np.load('features/features_0822/00_target.npy')
    y_duration = y[:, 1].astype(int)
    
    # 数据预处理
    X_processed = preprocess_data(X)
    
    # 分析类别7的模式
    analyze_class7_patterns(X_processed, y_duration)
    
    # 创建增强特征
    X_enhanced = create_class7_enhanced_features(X_processed, y_duration)
    
    # 数据质量检查
    data_quality_check(X_enhanced, y_duration)
    
    # 编码目标变量
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_duration)
    
    # 处理类别不平衡
    class_counts = pd.Series(y_encoded).value_counts()
    class7_encoded = label_encoder.transform([7])[0] if 7 in label_encoder.classes_ else -1
    
    if class7_encoded != -1 and class_counts[class7_encoded] < 20:
        print(f"\n类别7样本较少 ({class_counts[class7_encoded]}个)，进行SMOTE过采样...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts[class7_encoded]-1))
        X_enhanced, y_encoded = smote.fit_resample(X_enhanced, y_encoded)
    
    # 划分数据集
    remaining_counts = pd.Series(y_encoded).value_counts()
    stratify_param = y_encoded if all(remaining_counts > 1) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y_encoded, test_size=0.1, random_state=1122, stratify=stratify_param
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
    
    # 加强类别7权重
    if class7_encoded in class_weight_dict:
        class_weight_dict[class7_encoded] = class_weight_dict[class7_encoded] * 3.0
        print(f"类别7权重加强: {class_weight_dict[class7_encoded]:.2f}")
    
    # 创建SVM分类模型管道
    model = Pipeline(steps=[
        ('preprocessor', create_enhanced_preprocessor()),
        ('classifier', SVC(
            random_state=42, 
            probability=True,
            class_weight=class_weight_dict,
            kernel='rbf',
            C=10,
            gamma='scale'
        ))
    ])
    
    # SVM参数网格
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['rbf', 'linear'],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
        'classifier__class_weight': [class_weight_dict, 'balanced']
    }
    
    # 网格搜索
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1_weighted',
        n_jobs=n_jobs,
        verbose=1,  # 减少输出详细程度
        error_score='raise'
    )
    
    print("开始网格搜索寻找最佳SVM模型...")
    
    try:
        grid_search.fit(X_train, y_train)
        
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        # 分类评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        class7_f1 = f1_score(y_test, y_pred, average=None)[class7_encoded] if class7_encoded in np.unique(y_test) else 0
        
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, 
                                  target_names=[str(cls) for cls in label_encoder.classes_]))
        
        print(f"\n测试集准确率: {accuracy:.4f}")
        print(f"测试集加权F1分数: {f1:.4f}")
        print(f"类别7的F1分数: {class7_f1:.4f}")
        
        # 可视化
        plot_advanced_visualizations(y_test, y_pred, y_pred_proba, label_encoder, 
                                   f'{output_base}/plots', 'svm')
        
        # 保存模型
        joblib.dump(best_model, f'{output_base}/pkls/svm_classification_model.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/svm_label_encoder.pkl')
        
        # 保存网格搜索结果
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(f'{output_base}/outputs/svm_grid_search_results.csv', index=False)
        
        print("✅ SVM分类模型训练完成!")
        return best_model, label_encoder
        
    except Exception as e:
        print(f"❌ 网格搜索失败: {str(e)}")
        print("尝试使用默认参数训练...")
        
        # 使用默认参数训练
        default_model = Pipeline(steps=[
            ('preprocessor', create_enhanced_preprocessor()),
            ('classifier', SVC(
                random_state=42, 
                probability=True,
                class_weight=class_weight_dict,
                kernel='rbf',
                C=10,
                gamma='scale'
            ))
        ])
        
        default_model.fit(X_train, y_train)
        y_pred = default_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"默认模型测试集准确率: {accuracy:.4f}")
        print(f"默认模型测试集F1分数: {f1:.4f}")
        
        # 保存默认模型
        joblib.dump(default_model, f'{output_base}/pkls/svm_classification_model_default.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/svm_label_encoder_default.pkl')
        
        return default_model, label_encoder

# ==================== XGBoost模型 ====================
def create_sample_weights(y, class_weight_dict):
    """为每个样本创建权重，替代scale_pos_weight"""
    sample_weights = np.ones(len(y))
    for class_label, weight in class_weight_dict.items():
        sample_weights[y == class_label] = weight
    return sample_weights

def train_xgb_model():
    """训练XGBoost分类模型"""
    print("=== 开始训练XGBoost分类模型 ===")
    
    # 创建输出目录
    output_base = 'outputs/kz_dura_xgb_class7'
    os.makedirs(f'{output_base}/plots', exist_ok=True)
    os.makedirs(f'{output_base}/pkls', exist_ok=True)
    os.makedirs(f'{output_base}/outputs', exist_ok=True)
    
    # 加载数据
    X = pd.read_csv('features/features_0822/00_features.csv')
    y = np.load('features/features_0822/00_target.npy')
    y_duration = y[:, 1].astype(int)
    
    # 数据预处理
    X_processed = preprocess_data(X)
    
    # 分析类别7的模式
    analyze_class7_patterns(X_processed, y_duration)
    
    # 创建增强特征
    X_enhanced = create_class7_enhanced_features(X_processed, y_duration)
    
    # 数据质量检查
    data_quality_check(X_enhanced, y_duration)
    
    # 编码目标变量
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_duration)
    
    # 处理类别不平衡
    class_counts = pd.Series(y_encoded).value_counts()
    class7_encoded = label_encoder.transform([7])[0] if 7 in label_encoder.classes_ else -1
    
    if class7_encoded != -1 and class_counts[class7_encoded] < 20:
        print(f"\n类别7样本较少 ({class_counts[class7_encoded]}个)，进行SMOTE过采样...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts[class7_encoded]-1))
        X_enhanced, y_encoded = smote.fit_resample(X_enhanced, y_encoded)
    
    # 划分数据集
    remaining_counts = pd.Series(y_encoded).value_counts()
    stratify_param = y_encoded if all(remaining_counts > 1) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y_encoded, test_size=0.1, random_state=1122, stratify=stratify_param
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
    
    # 加强类别7权重
    if class7_encoded in class_weight_dict:
        class_weight_dict[class7_encoded] = class_weight_dict[class7_encoded] * 3.0
        print(f"类别7权重加强: {class_weight_dict[class7_encoded]:.2f}")
    
    # 创建样本权重
    sample_weights = create_sample_weights(y_train, class_weight_dict)
    
    # XGBoost参数网格
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
        ('preprocessor', create_enhanced_preprocessor()),
        ('classifier', XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=n_jobs,
            use_label_encoder=False
        ))
    ])
    
    # 网格搜索
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=xgb_param_grid,
        cv=cv_strategy,
        scoring='f1_weighted',
        n_jobs=1,  # XGBoost内部已使用多线程，此处设为1避免冲突
        verbose=1  # 减少输出详细程度
    )
    
    print("开始网格搜索寻找最佳XGBoost分类模型...")
    
    try:
        # 使用样本权重进行训练
        grid_search.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        
        print(f"最佳模型: {grid_search.best_estimator_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # 评估模型
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        # 分类评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        class7_f1 = f1_score(y_test, y_pred, average=None)[class7_encoded] if class7_encoded in np.unique(y_test) else 0
        
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, 
                                  target_names=[str(cls) for cls in label_encoder.classes_]))
        
        print(f"\n测试集准确率: {accuracy:.4f}")
        print(f"测试集加权F1分数: {f1:.4f}")
        print(f"类别7的F1分数: {class7_f1:.4f}")
        
        # 可视化
        plot_advanced_visualizations(y_test, y_pred, y_pred_proba, label_encoder, 
                                   f'{output_base}/plots', 'xgb')
        
        # 保存模型
        joblib.dump(best_model, f'{output_base}/pkls/xgb_classification_model.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/xgb_label_encoder.pkl')
        
        # 保存网格搜索结果
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(f'{output_base}/outputs/xgb_grid_search_results.csv', index=False)
        
        print("✅ XGBoost分类模型训练完成!")
        return best_model, label_encoder
        
    except Exception as e:
        print(f"❌ 网格搜索失败: {str(e)}")
        print("尝试使用默认参数训练...")
        
        # 使用默认参数训练
        default_model = Pipeline(steps=[
            ('preprocessor', create_enhanced_preprocessor()),
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
        
        print(f"默认模型测试集准确率: {accuracy:.4f}")
        print(f"默认模型测试集F1分数: {f1:.4f}")
        
        # 保存默认模型
        joblib.dump(default_model, f'{output_base}/pkls/xgb_classification_model_default.pkl')
        joblib.dump(label_encoder, f'{output_base}/pkls/xgb_label_encoder_default.pkl')
        
        return default_model, label_encoder

# ==================== 主函数 ====================
def train_all_models():
    """训练所有模型"""
    print("开始训练所有分类模型...")
    
    models = {}
    
    # 训练逻辑回归模型
    try:
        print("\n" + "="*50)
        print("训练逻辑回归模型")
        print("="*50)
        lr_model, lr_encoder, lr_metrics = train_lr_model()
        models['lr'] = {'model': lr_model, 'encoder': lr_encoder, 'metrics': lr_metrics}
        print("✅ 逻辑回归模型训练完成")
    except Exception as e:
        print(f"❌ 逻辑回归模型训练失败: {e}")
    
    # 训练MLP模型
    try:
        print("\n" + "="*50)
        print("训练MLP模型")
        print("="*50)
        mlp_model, mlp_encoder = train_mlp_model()
        models['mlp'] = {'model': mlp_model, 'encoder': mlp_encoder}
        print("✅ MLP模型训练完成")
    except Exception as e:
        print(f"❌ MLP模型训练失败: {e}")
    
    # 训练随机森林模型
    try:
        print("\n" + "="*50)
        print("训练随机森林模型")
        print("="*50)
        rf_model, rf_encoder = train_rf_model()
        models['rf'] = {'model': rf_model, 'encoder': rf_encoder}
        print("✅ 随机森林模型训练完成")
    except Exception as e:
        print(f"❌ 随机森林模型训练失败: {e}")
    
    # 训练SVM模型
    try:
        print("\n" + "="*50)
        print("训练SVM模型")
        print("="*50)
        svm_model, svm_encoder = train_svm_model()
        models['svm'] = {'model': svm_model, 'encoder': svm_encoder}
        print("✅ SVM模型训练完成")
    except Exception as e:
        print(f"❌ SVM模型训练失败: {e}")
    
    # 训练XGBoost模型
    try:
        print("\n" + "="*50)
        print("训练XGBoost模型")
        print("="*50)
        xgb_model, xgb_encoder = train_xgb_model()
        models['xgb'] = {'model': xgb_model, 'encoder': xgb_encoder}
        print("✅ XGBoost模型训练完成")
    except Exception as e:
        print(f"❌ XGBoost模型训练失败: {e}")
    
    print(f"\n🎉 模型训练完成总结:")
    print(f"成功训练的模型: {list(models.keys())}")
    
    return models

if __name__ == '__main__':
    models = train_all_models()
    print("所有模型训练完成!")