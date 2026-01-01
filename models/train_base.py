import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 设置使用的CPU核心数
n_jobs = max(1, os.cpu_count() // 2)

# 设置字体，确保中文正常显示
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

# 自定义特征选择器，用于调试
class DebugSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selector):
        self.selector = selector
        
    def fit(self, X, y):
        self.selector.fit(X, y)
        print(f"特征选择器保留了 {sum(self.selector.get_support())} / {X.shape[1]} 个特征")
        return self
        
    def transform(self, X):
        return self.selector.transform(X)
    
    def get_support(self):
        return self.selector.get_support()


# 在train_base.py中添加特征选择函数
def feature_selection(X, y, n_features=200):
    """使用LightGBM内置的特征重要性进行特征选择"""
    from lightgbm import LGBMClassifier
    
    # 简单预处理
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # 兼容不同版本的 scikit-learn
    try:
        # 新版本 scikit-learn
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        # 旧版本 scikit-learn
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    # 构建转换器列表（只包含有特征的类型）
    transformers = []
    if len(numeric_features) > 0:
        transformers.append(('num', SimpleImputer(strategy='median'), numeric_features))
    if len(categorical_features) > 0:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('onehot', onehot_encoder)
        ]), categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    X_processed = preprocessor.fit_transform(X)
    
    # 训练一个简单的LightGBM模型获取特征重要性
    model = LGBMClassifier(n_estimators=100, random_state=42)
    model.fit(X_processed, y)
    
    # 获取特征重要性
    feature_importance = model.feature_importances_
    
    # 获取特征名称（关键修改：只处理存在的特征类型）
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat' and len(categorical_features) > 0:
            # 只有存在分类特征时才尝试获取编码后的特征名称
            try:
                # 新版本 scikit-learn
                cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
            except AttributeError:
                # 旧版本 scikit-learn
                cat_features = transformer.named_steps['onehot'].get_feature_names(features)
            feature_names.extend(cat_features)
    
    # 选择最重要的特征
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    selected_features = importance_df.head(n_features)['feature'].tolist()
    
    return selected_features

def create_enhanced_preprocessor(X_columns):
    """创建增强的预处理管道，基于实际存在的特征"""
    # 根据实际存在的特征动态构建特征列表
    numeric_features = [
        # 时间特征
        'hour_of_day', 'day_of_week', 'month', 'season', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_of_year',
        
        # 历史操作特征
        'prev_gate_count', 'prev_duration', 'prev_op_hour',
        'ops_week_count', 'ops_week_avg_gates', 'ops_week_total_duration',
        'hours_since_last_op',
        
        # 潮汐特征
        'tide_24h_phase', 'tide_12h_phase',
        'tide_24h_mean', 'tide_24h_max', 'tide_24h_min', 'tide_24h_range',
        'tide_24h_slope', 'tide_24h_r_squared', 'tide_24h_cycle_count',
        'tide_24h_rise_rate', 'tide_24h_fall_rate', 'tide_24h_volatility', 'tide_24h_trend_strength',
        'tide_12h_mean', 'tide_12h_max', 'tide_12h_min', 'tide_12h_range',
        'tide_12h_slope', 'tide_12h_r_squared', 'tide_12h_cycle_count',
        'tide_12h_rise_rate', 'tide_12h_fall_rate', 'tide_12h_volatility', 'tide_12h_trend_strength',
        
        # 未来潮汐特征
        'future_tide_mean', 'future_tide_max', 'future_tide_min', 'future_tide_range',
        'future_tide_slope', 'future_tide_r_squared', 'future_tide_cycle_count',
        'future_tide_rise_rate', 'future_tide_fall_rate', 'future_tide_phase',
        'future_tide_volatility', 'future_tide_trend_strength',
        
        # 流量特征
        'flow_mean', 'flow_max', 'flow_min', 'flow_range', 'flow_var', 'flow_skew', 'flow_trend',
        
        # 降雨特征
        'rain_actual_total', 'rain_forecast_total', 
        'rain_actual_avg', 'rain_forecast_avg',
        'rain_change_rate', 'water_rain_ratio', 'flow_rain_ratio', 'water_flow_ratio',
        
        # 水位工况特征
        'water_status_mean', 'water_status_max', 'water_status_min', 
        'water_status_range', 'water_status_slope',
        
        # 其他特征
        'is_rush_hour', 'is_night'
    ]
    
    categorical_features = ['tide_type']
    
    indicator_features = [
        'water_missing', 'flow_missing', 
        'rain_missing', 'water_status_missing', 'future_water_missing'
    ]
    
    # 只保留实际存在的特征
    numeric_features = [feat for feat in numeric_features if feat in X_columns]
    categorical_features = [feat for feat in categorical_features if feat in X_columns]
    indicator_features = [feat for feat in indicator_features if feat in X_columns]
    
    print(f"数值特征: {len(numeric_features)}个")
    print(f"分类特征: {len(categorical_features)}个")
    print(f"指示器特征: {len(indicator_features)}个")
    
    # 数值特征处理
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 分类特征处理
    try:
        # 尝试使用新版本的参数
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        # 如果失败，使用旧版本的参数
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # 使用更安全的分类特征处理
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # 先用常量填充
        ('onehot', onehot_encoder)
    ])
        
    indicator_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ind', indicator_transformer, indicator_features)
        ])
    
    return preprocessor

def create_advanced_preprocessor(feature_names):
    """创建高级预处理管道"""
    # 分离数值和分类特征
    numeric_features = feature_names[feature_names.str.contains('|'.join([
        'hour', 'day', 'month', 'season', 'mean', 'max', 'min', 'range',
        'slope', 'rate', 'count', 'ratio', 'trend', 'volatility', 'var', 'skew'
    ]))].tolist()
    
    categorical_features = feature_names[~feature_names.isin(numeric_features)].tolist()
    
    # 兼容不同版本的 scikit-learn
    try:
        # 新版本 scikit-learn
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        # 旧版本 scikit-learn
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('onehot', onehot_encoder)
            ]), categorical_features)
        ])
    
    return preprocessor

def analyze_features(X, y):
    """深入分析特征与目标的关系"""
    print("=== 特征分析 ===")
    
    # 检查特征分布
    print("\n1. 特征分布统计:")
    print(X.describe())
    
    # 检查特征与目标的相关性（更详细）
    print("\n2. 特征与目标的相关性 (前20):")
    numeric_X = X.select_dtypes(include=[np.number])
    y_series = pd.Series(y, index=X.index)
    corr_with_target = numeric_X.corrwith(y_series).abs().sort_values(ascending=False)
    print(corr_with_target.head(20))
    
    # 检查特征之间的相关性
    print("\n3. 特征间相关性 (前10对最高相关):")
    corr_matrix = numeric_X.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)  # 将对角线设为0
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_val = corr_matrix.loc[col1, col2]
            high_corr_pairs.append((col1, col2, corr_val))
    
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    for col1, col2, corr_val in high_corr_pairs[:10]:
        print(f"{col1} - {col2}: {corr_val:.4f}")
    
    # 检查特征的方差
    print("\n4. 特征方差 (最低的10个):")
    variances = numeric_X.var().sort_values()
    print(variances.head(10))
    
    return corr_with_target, variances

def check_high_correlation(X, threshold=0.95):
    """检查高度相关的特征对"""
    numeric_X = X.select_dtypes(include=[np.number])
    corr_matrix = numeric_X.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)  # 将对角线设为0
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_val = corr_matrix.loc[col1, col2]
            if corr_val > threshold:
                high_corr_pairs.append((col1, col2, corr_val))
    
    return high_corr_pairs

def create_data_splits(X, y, strategy='time_series'):
    """创建不同的数据分割策略"""
    if strategy == 'time_series' and 'date' in X.columns:
        # 时间序列分割
        X_sorted = X.sort_values('date')
        y_sorted = y[X_sorted.index]
        
        tscv = TimeSeriesSplit(n_splits=5)
        train_index, test_index = list(tscv.split(X_sorted))[-1]
        
        X_train, X_test = X_sorted.iloc[train_index], X_sorted.iloc[test_index]
        y_train, y_test = y_sorted[train_index], y_sorted[test_index]
        
        # 移除日期列
        if 'date' in X_train.columns:
            X_train = X_train.drop('date', axis=1)
            X_test = X_test.drop('date', axis=1)
            
    elif strategy == 'stratified':
        # 分层随机分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        # 简单随机分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    return X_train, X_test, y_train, y_test

def analyze_errors_binary(test_results, X_test, y_test, y_pred, y_proba):
    """分析二分类预测错误的情况"""
    errors = test_results[test_results['true_label'] != test_results['pred_label']]
    print(f"\n错误分析: 总共 {len(test_results)} 个样本，{len(errors)} 个预测错误")
    
    if len(errors) > 0:
        # 分析错误类型
        fp = errors[(errors['true_label'] == 0) & (errors['pred_label'] == 1)]
        fn = errors[(errors['true_label'] == 1) & (errors['pred_label'] == 0)]
        
        print(f"假阳性 (False Positives): {len(fp)} 个")
        print(f"假阴性 (False Negatives): {len(fn)} 个")
        
        # 分析假阳性和假阴性的概率分布
        plt.figure(figsize=(10, 6))
        plt.hist(fp['pred_prob'], alpha=0.5, label='假阳性', bins=20)
        plt.hist(fn['pred_prob'], alpha=0.5, label='假阴性', bins=20)
        plt.xlabel('预测概率')
        plt.ylabel('频数')
        plt.title('错误预测的概率分布')
        plt.legend()
        plt.tight_layout()
        plt.savefig('outputs/binary/plots/enhanced_binary_error_prob_dist.png', dpi=300)
        plt.close()
        
        # 保存错误分析结果
        error_analysis = pd.DataFrame({
            'error_type': ['FP', 'FN'],
            'count': [len(fp), len(fn)]
        })
        error_analysis.to_csv('outputs/binary/outputs/enhanced_binary_error_analysis.csv', index=False)

def plot_roc_curve(y_test, y_proba, model_name):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'{model_name} - ROC曲线')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'outputs/binary/plots/enhanced_binary_{model_name}_roc_curve.png', dpi=300)
    plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_test, y_proba, model_name):
    """绘制精确率-召回率曲线"""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AP = {avg_precision:.2f})')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title(f'{model_name} - 精确率-召回率曲线')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f'outputs/binary/plots/enhanced_binary_{model_name}_pr_curve.png', dpi=300)
    plt.close()
    
    return avg_precision

def analyze_feature_importance(model, X, y, feature_names):
    """分析特征重要性"""
    try:
        # 使用排列重要性
        result = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=n_jobs
        )
        
        # 创建特征重要性DataFrame
        importances = result.importances_mean
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 保存特征重要性
        importance_df.to_csv('outputs/binary/outputs/enhanced_binary_feature_importance.csv', index=False)
        
        # 可视化最重要的特征
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', 
                    data=importance_df.head(20))
        plt.title('特征重要性排名 (排列重要性)')
        plt.tight_layout()
        plt.savefig('outputs/binary/plots/enhanced_binary_feature_importance.png', dpi=300)
        plt.close()
        
        return importance_df
    except Exception as e:
        print(f"特征重要性分析失败: {str(e)}")
        return None

def get_feature_names(preprocessor):
    """获取预处理后的特征名称"""
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
    
    return feature_names

def explain_model_predictions(model, X_test, y_test):
    """使用SHAP值解释模型预测"""
    try:
        import shap
        
        # 获取预处理后的特征
        preprocessor = model.named_steps['preprocessor']
        X_processed = preprocessor.transform(X_test)
        
        # 获取特征名称
        feature_names = get_feature_names(preprocessor)
        
        # 创建解释器
        explainer = shap.Explainer(model.named_steps['classifier'], X_processed, feature_names=feature_names)
        shap_values = explainer(X_processed)
        
        # 绘制摘要图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('outputs/binary/plots/enhanced_binary_shap_summary.png', dpi=300)
        plt.close()
        
        # 绘制特征重要性
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('outputs/binary/plots/enhanced_binary_shap_importance.png', dpi=300)
        plt.close()
        
        print("SHAP分析完成")
        
    except ImportError:
        print("SHAP库未安装，跳过模型解释")
    except Exception as e:
        print(f"SHAP分析失败: {str(e)}")

def load_and_preprocess_data():
    """加载和预处理数据"""
    # 加载特征数据
    X = pd.read_csv('features/features_enhanced/all_features.csv')
    
    # 检查数据基本信息
    print(f"数据集形状: {X.shape}")
    print(f"特征列: {X.columns.tolist()}")
    
    # 移除缺失值过多的特征
    missing_ratio = X.isnull().sum() / len(X)
    high_missing_features = missing_ratio[missing_ratio > 0.3].index.tolist()  # 降低阈值到30%
    print(f"移除缺失值比例超过30%的特征: {high_missing_features}")
    X = X.drop(columns=high_missing_features)
    
    # 移除常数特征
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    print(f"移除常数特征: {constant_features}")
    X = X.drop(columns=constant_features)
    
    # 新增：移除全部为NaN的列
    all_nan_columns = X.columns[X.isnull().all()].tolist()
    if all_nan_columns:
        print(f"移除全部为NaN的列: {all_nan_columns}")
        X = X.drop(columns=all_nan_columns)
    
    # 目标变量 - 转换为二分类问题
    y = np.load('features/features_enhanced/all_target.npy')[:, 0].astype(int)
    
    # 转换为二分类问题：是否开闸（0=不开闸，1=开闸）
    y_binary = (y > 0).astype(int)
    print(f"\n二分类目标分布:")
    print(pd.Series(y_binary).value_counts())
    print(f"正样本比例: {y_binary.mean():.4f}")
    
    # 深入分析特征
    corr_with_target, variances = analyze_features(X, y_binary)
    
    # 检查是否存在高度相关的特征
    high_corr_features = check_high_correlation(X, threshold=0.9)  # 降低阈值到0.9
    print(f"高度相关的特征对: {high_corr_features}")
    
    # 移除高度相关的特征
    features_to_remove = set()
    for col1, col2, corr_val in high_corr_features:
        # 保留与目标相关性更高的特征
        if corr_with_target.get(col1, 0) > corr_with_target.get(col2, 0):
            features_to_remove.add(col2)
        else:
            features_to_remove.add(col1)
    
    print(f"移除高度相关的特征: {features_to_remove}")
    X = X.drop(columns=list(features_to_remove))
    
    return X, y_binary

def evaluate_model(model, X_test, y_test, model_name):
    """评估模型并保存结果"""
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 确保预测结果是整数类型
    y_pred = y_pred.astype(int)
    
    # 分类评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    print(f"\n{model_name} 测试集表现:")
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC分数: {roc_auc:.4f}")
    print(f"平均精确率: {avg_precision:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 可视化ROC曲线
    plot_roc_curve(y_test, y_proba, f"{model_name}_enhanced")
    
    # 可视化精确率-召回率曲线
    plot_precision_recall_curve(y_test, y_proba, f"{model_name}_enhanced")
    
    # 可视化混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['不开闸', '开闸'], 
                yticklabels=['不开闸', '开闸'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} - 二分类预测混淆矩阵')
    plt.savefig(f'outputs/binary/plots/enhanced_binary_{model_name}_confusion_matrix.png', dpi=300)
    plt.close()
    
    # 保存测试集预测结果
    test_results = pd.DataFrame({
        'true_label': y_test,
        'pred_label': y_pred,
        'pred_prob': y_proba
    })
    test_results.to_csv(f'outputs/binary/outputs/test_predictions_{model_name}_enhanced_binary.csv', index=False)
    print("测试集预测结果已保存")
    
    # 分析错误预测
    analyze_errors_binary(test_results, X_test, y_test, y_pred, y_proba)
    
    # 绘制预测概率分布
    plt.figure(figsize=(10, 6))
    plt.hist(y_proba[y_test == 0], alpha=0.5, label='真实负例', bins=20)
    plt.hist(y_proba[y_test == 1], alpha=0.5, label='真正例', bins=20)
    plt.xlabel('预测概率')
    plt.ylabel('频数')
    plt.title('预测概率分布')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'outputs/binary/plots/enhanced_binary_{model_name}_probability_distribution.png', dpi=300)
    plt.close()
    
    # 分析不同阈值下的性能
    thresholds = np.linspace(0, 1, 21)
    results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        precision = precision_score(y_test, y_pred_thresh, zero_division=0)
        recall = recall_score(y_test, y_pred_thresh, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
    
    threshold_df = pd.DataFrame(results)
    threshold_df.to_csv(f'outputs/binary/outputs/enhanced_binary_{model_name}_threshold_analysis.csv', index=False)
    
    # 绘制阈值分析图
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['threshold'], threshold_df['accuracy'], label='准确率')
    plt.plot(threshold_df['threshold'], threshold_df['f1'], label='F1分数')
    plt.plot(threshold_df['threshold'], threshold_df['precision'], label='精确率')
    plt.plot(threshold_df['threshold'], threshold_df['recall'], label='召回率')
    plt.xlabel('阈值')
    plt.ylabel('分数')
    plt.title('不同阈值下的性能指标')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'outputs/binary/plots/enhanced_binary_{model_name}_threshold_analysis.png', dpi=300)
    plt.close()
    
    # 模型解释
    explain_model_predictions(model, X_test, y_test)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }