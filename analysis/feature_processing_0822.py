import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime, timedelta
import joblib
import os
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# 新增可视化库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# 新增模型导入
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# 分类模型导入
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import f_regression, mutual_info_regression

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_data():
    # 加载水文数据
    df = pd.read_csv('imports/闸下潮位.csv', parse_dates=['time'])
    print(f"潮位数据记录数: {len(df)}, 时间范围: {df['time'].min()} 到 {df['time'].max()}")
    
    df_water_level = df[df['station_id'] == 3018].copy()
    print(f"站点3018记录数: {len(df_water_level)}, 缺失值: {df_water_level['water_level'].isnull().sum()}")
    # 对潮位数据按时间排序并插值
    df_water_level.sort_values('time', inplace=True)
    df_water_level['water_level'] = df_water_level['water_level'].interpolate(method='linear')
    # 如果还有缺失，用前一个值填充
    df_water_level['water_level'].fillna(method='ffill', inplace=True)
    df_water_level['water_level'].fillna(method='bfill', inplace=True)

    df_flow = pd.read_csv('imports/实测流量.csv', parse_dates=['监测日期'])
    df_flow.sort_values('监测日期', inplace=True)
    df_flow['流量'] = df_flow['流量'].interpolate(method='linear')
    df_flow['流量'].fillna(method='ffill', inplace=True)
    df_flow['流量'].fillna(method='bfill', inplace=True)

    df_rain_actual = pd.read_csv('imports/实测降雨.csv', parse_dates=['监测日期'])
    # 对于实测降雨，用0填充缺失的雨量
    df_rain_actual['雨量'].fillna(0, inplace=True)

    df_rain_forecast = pd.read_csv('imports/降雨预报.csv', parse_dates=['预计开始时间'])
    df_rain_forecast['降雨量'].fillna(0, inplace=True)

    df_water_status = pd.read_csv('imports/水位工况.csv', parse_dates=['监测日期'])
    df_water_status.sort_values('监测日期', inplace=True)
    df_water_status['水位'] = df_water_status['水位'].interpolate(method='linear')
    df_water_status['水位'].fillna(method='ffill', inplace=True)
    df_water_status['水位'].fillna(method='bfill', inplace=True)
    
    return df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status

def safe_linregress(x, y):
    """安全的线性回归计算，处理x值相同的情况"""
    if len(x) < 2 or np.var(x) == 0:
        return 0, 0, 0, 0, 0  # 斜率, 截距, r值, p值, 标准误
    
    try:
        return linregress(x, y)
    except:
        return 0, 0, 0, 0, 0

def extract_hourly_tidal_features(water_data, period_hours=24):
    """提取小时级别潮汐特征"""
    # 定义默认特征值
    default_features = {
        'mean': 0,
        'max': 0,
        'min': 0,
        'range': 0,
        'slope': 0,
        'r_squared': 0,
        'cycle_count': 0,
        'rise_rate': 0,
        'fall_rate': 0,
        'phase': 0,
        'tide_type': 0, # 潮汐类型，0表示无潮汐，1表示半日潮，2表示全日潮
    }
    
    if len(water_data) < 2:
        return default_features
    
    # 确保数据按时间排序
    water_data = water_data.sort_values('time')
    levels = water_data['water_level'].values
    
    features = {}
    
    # 基础统计特征
    features['mean'] = np.mean(levels)
    features['max'] = np.max(levels)
    features['min'] = np.min(levels)
    features['range'] = features['max'] - features['min']
    
    # 趋势特征 - 使用安全的线性回归
    time_numeric = (water_data['time'] - water_data['time'].min()).dt.total_seconds().values
    slope, intercept, r_value, p_value, std_err = safe_linregress(time_numeric, levels)
    features['slope'] = slope
    features['r_squared'] = r_value**2
    
    # 潮汐周期检测
    diffs = np.diff(levels)
    turning_points = []
    
    for i in range(1, len(diffs)):
        if diffs[i] * diffs[i-1] < 0:  # 符号变化表示转折点
            turning_points.append(i)
    
    # 计算潮汐周期数量
    cycle_count = len(turning_points) // 2
    features['cycle_count'] = cycle_count

    # 潮汐类型识别 (1=半日潮，2=全日潮，3=混合潮)
    if cycle_count >= 3:
        features['tide_type'] = 1
    elif cycle_count == 1:
        features['tide_type'] = 2
    else:
        features['tide_type'] = 3

    # 计算涨落潮速率
    rise_rates = []
    fall_rates = []
    
    for i in range(len(turning_points) - 1):
        start_idx = turning_points[i]
        end_idx = turning_points[i+1]
        level_diff = levels[end_idx] - levels[start_idx]
        
        # 计算时间差（小时）
        time_diff_hours = (water_data['time'].iloc[end_idx] - water_data['time'].iloc[start_idx]).total_seconds() / 3600
        
        if time_diff_hours > 0:
            rate = level_diff / time_diff_hours
            if level_diff > 0:
                rise_rates.append(rate)
            else:
                fall_rates.append(abs(rate))
    
    features['rise_rate'] = np.mean(rise_rates) if rise_rates else 0
    features['fall_rate'] = np.mean(fall_rates) if fall_rates else 0
    
    # 潮汐相位 (基于时间)
    hour_of_day = water_data['time'].iloc[0].hour
    features['phase'] = hour_of_day % 12 / 12.0  # 半日潮相位
    
    return features


def analyze_feature_correlation(X, y, target_name, save_path):
    """分析特征与目标变量的相关性"""
    print(f"\n=== 特征相关性分析: {target_name} ===")
    
    # 确保X是数值型数据
    X_numeric = X.copy()
    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
            X_numeric[col] = pd.factorize(X_numeric[col])[0]
    
    # 处理缺失值
    X_numeric = X_numeric.fillna(X_numeric.mean())
    
    # 移除NaN值
    valid_indices = ~np.isnan(y)
    X_valid = X_numeric[valid_indices]
    y_valid = y[valid_indices]
    
    if len(y_valid) == 0:
        print(f"  跳过 {target_name} - 无有效数据")
        return
    
    # 归一化处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    
    # 计算皮尔逊相关系数
    correlations = {}
    for i, col in enumerate(X_valid.columns):
        corr = np.corrcoef(X_scaled[:, i], y_valid)[0, 1]
        correlations[col] = corr
    
    # 转换为DataFrame并排序
    corr_df = pd.DataFrame({
        'feature': list(correlations.keys()),
        'correlation': list(correlations.values())
    }).sort_values('correlation', key=abs, ascending=False)
    
    # 保存相关性结果
    corr_df.to_csv(f'{save_path}/feature_correlation_{target_name}.csv', index=False)
    
    # 绘制相关性图
    plt.figure(figsize=(12, 8))
    top_corr = corr_df.head(20)
    colors = ['red' if x < 0 else 'blue' for x in top_corr['correlation']]
    plt.barh(top_corr['feature'], top_corr['correlation'], color=colors)
    plt.xlabel('相关系数')
    plt.title(f'{target_name} - 特征相关性(top 20)')
    plt.tight_layout()
    plt.savefig(f'{save_path}/feature_correlation_{target_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 使用线性回归计算R²
    lr = LinearRegression()
    r2_scores = {}
    
    for col in X_valid.columns:
        X_feature = X_scaled[:, X_valid.columns.get_loc(col)].reshape(-1, 1)
        lr.fit(X_feature, y_valid)
        y_pred = lr.predict(X_feature)
        r2 = r2_score(y_valid, y_pred)
        r2_scores[col] = r2
    
    # 转换为DataFrame并排序
    r2_df = pd.DataFrame({
        'feature': list(r2_scores.keys()),
        'r2_score': list(r2_scores.values())
    }).sort_values('r2_score', ascending=False)
    
    # 保存R²结果
    r2_df.to_csv(f'{save_path}/feature_r2_{target_name}.csv', index=False)
    
    # 绘制R²图
    plt.figure(figsize=(12, 8))
    top_r2 = r2_df.head(20)
    plt.barh(top_r2['feature'], top_r2['r2_score'], color='green')
    plt.xlabel('R²得分')
    plt.title(f'{target_name} - 特征R²得分(top 20)')
    plt.tight_layout()
    plt.savefig(f'{save_path}/feature_r2_{target_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  特征相关性分析完成，已保存至 {save_path}")


def fit_models_for_targets(X, y, save_path):
    """为目标变量拟合模型并评估，增加更多回归模型和分类分析"""
    print("\n=== 目标变量模型拟合 ===")
    
    # 目标变量名称
    target_names = ['开闸时间(小时)', '开闸时长', '开闸孔数', '目标水位', '联合时长-孔数', '联合时长-孔数-log']
    
    # 确保X是数值型数据
    X_numeric = X.copy()
    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
            X_numeric[col] = pd.factorize(X_numeric[col])[0]
    
    # 处理缺失值
    X_numeric = X_numeric.fillna(X_numeric.mean())
    
    results = []
    
    # 模型列表
    regression_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42),
        'SVM': SVR(),
        'Lasso': Lasso(random_state=42),
        'Ridge': Ridge(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'MLP': MLPRegressor(random_state=42, max_iter=1000)
    }
    
    classification_models = {
        'Random Forest Classifier': RandomForestClassifier(random_state=42),
        'XGBoost Classifier': XGBClassifier(random_state=42, verbosity=0),
        'LightGBM Classifier': LGBMClassifier(random_state=42, verbose=-1),
        'SVM Classifier': SVC(),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42)
    }
    
    for i, target_name in enumerate(target_names[:6]):  # 只对前6个目标变量建模
        print(f"\n建模目标变量: {target_name}")
        
        # 特征相关性分析
        analyze_feature_correlation(X, y[:, i], target_name, save_path)
        
        # 准备数据
        y_target = y[:, i]
        
        # 移除NaN值
        valid_indices = ~np.isnan(y_target)
        X_valid = X_numeric[valid_indices]
        y_valid = y_target[valid_indices]
        
        if len(y_valid) == 0:
            print(f"  跳过 {target_name} - 无有效数据")
            continue
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=0.2, random_state=42
        )
        
        # 回归模型评估
        model_results = {}
        
        for model_name, model in regression_models.items():
            try:
                # 对需要归一化的模型进行归一化处理
                if model_name in ['Linear Regression', 'MLP']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                model_results[model_name] = {
                    'rmse': rmse,
                    'r2': r2
                }
                
                print(f"  {model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            except Exception as e:
                print(f"  {model_name} 训练失败: {str(e)}")
                model_results[model_name] = {
                    'rmse': np.nan,
                    'r2': np.nan
                }
        
        # 保存最佳回归模型
        best_reg_model = None
        best_reg_score = float('inf')
        for model_name, scores in model_results.items():
            if not np.isnan(scores['rmse']) and scores['rmse'] < best_reg_score:
                best_reg_score = scores['rmse']
                best_reg_model = model_name
        
        # 对开闸孔数进行额外分类分析
        classification_results = {}
        if target_name == '开闸孔数':
            print(f"  对 {target_name} 进行分类分析")
            
            # 对开闸孔数进行重新编号，使其从0开始
            y_train_clf = y_train.astype(int)
            y_test_clf = y_test.astype(int)
            
            # 获取唯一类别并重新映射
            unique_classes = np.unique(y_train_clf)
            class_mapping = {orig: new for new, orig in enumerate(unique_classes)}
            
            # 应用映射
            y_train_clf = np.array([class_mapping[x] for x in y_train_clf])
            y_test_clf = np.array([class_mapping[x] for x in y_test_clf])
            
            # 训练分类模型
            for model_name, model in classification_models.items():
                try:
                    model.fit(X_train, y_train_clf)
                    y_pred_clf = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test_clf, y_pred_clf)
                    f1 = f1_score(y_test_clf, y_pred_clf, average='weighted')
                    precision = precision_score(y_test_clf, y_pred_clf, average='weighted')
                    recall = recall_score(y_test_clf, y_pred_clf, average='weighted')
                    
                    classification_results[model_name] = {
                        'accuracy': accuracy,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall
                    }
                    
                    print(f"    {model_name} - 准确率: {accuracy:.4f}, F1: {f1:.4f}")
                    
                    # 保存分类报告
                    if model_name == 'Random Forest Classifier':  # 只保存一个模型的详细报告
                        clf_report = classification_report(y_test_clf, y_pred_clf, output_dict=True)
                        clf_report_df = pd.DataFrame(clf_report).transpose()
                        clf_report_df.to_csv(f'{save_path}/classification_report_{target_name}.csv')
                        
                except Exception as e:
                    print(f"    {model_name} 分类训练失败: {str(e)}")
                    classification_results[model_name] = {
                        'accuracy': np.nan,
                        'f1': np.nan,
                        'precision': np.nan,
                        'recall': np.nan
                    }
        
        # 保存结果
        results.append({
            'target': target_name,
            'best_regression_model': best_reg_model,
            'best_regression_rmse': best_reg_score,
            'regression_results': model_results,
            'classification_results': classification_results if classification_results else None,
            'sample_size': len(y_valid)
        })
        
        # 绘制预测 vs 实际值图（只绘制前6个模型）
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # 选择前6个模型进行可视化
        top_models = list(regression_models.keys())[:6]
        
        for j, model_name in enumerate(top_models):
            if j >= len(axes):
                break
                
            if model_name in model_results and not np.isnan(model_results[model_name]['rmse']):
                # 重新训练模型用于绘图
                model = regression_models[model_name]
                
                # 对需要归一化的模型使用归一化数据
                if model_name in ['Linear Regression', 'MLP']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                axes[j].scatter(y_test, y_pred, alpha=0.6)
                axes[j].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[j].set_xlabel('实际值')
                axes[j].set_ylabel('预测值')
                axes[j].set_title(f'{model_name}\nRMSE: {model_results[model_name]["rmse"]:.4f}, R²: {model_results[model_name]["r2"]:.4f}')
                axes[j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/model_performance_{target_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制特征重要性（对于树模型和线性模型）
        tree_models = ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']
        linear_models = ['Linear Regression', 'Lasso', 'Ridge']
        
        # 处理树模型的特征重要性
        if best_reg_model in tree_models:
            try:
                model = regression_models[best_reg_model]
                model.fit(X_train, y_train)
                
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    feature_names = X_numeric.columns
                    
                    # 创建特征重要性 DataFrame
                    feat_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False).head(20)
                    
                    # 绘制特征重要性图
                    plt.figure(figsize=(10, 8))
                    sns.barplot(x='importance', y='feature', data=feat_importance)
                    plt.title(f'{best_reg_model} - {target_name} 特征重要性')
                    plt.tight_layout()
                    plt.savefig(f'{save_path}/feature_importance_{target_name}_{best_reg_model}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # 保存特征重要性数据
                    feat_importance.to_csv(f'{save_path}/feature_importance_{target_name}_{best_reg_model}.csv', index=False)
            except Exception as e:
                print(f"  绘制特征重要性失败: {str(e)}")
        
        # 处理线性模型的系数
        elif best_reg_model in linear_models:
            try:
                model = regression_models[best_reg_model]
                
                # 对线性回归使用归一化数据
                if best_reg_model == 'Linear Regression':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    model.fit(X_train_scaled, y_train)
                    coefficients = model.coef_
                else:
                    model.fit(X_train, y_train)
                    coefficients = model.coef_
                
                feature_names = X_numeric.columns
                
                # 创建系数 DataFrame
                coef_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefficients
                }).sort_values('coefficient', key=abs, ascending=False).head(20)
                
                # 绘制系数图
                plt.figure(figsize=(10, 8))
                colors = ['red' if x < 0 else 'blue' for x in coef_df['coefficient']]
                plt.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
                plt.title(f'{best_reg_model} - {target_name} 特征系数')
                plt.tight_layout()
                plt.savefig(f'{save_path}/feature_coefficients_{target_name}_{best_reg_model}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 保存系数数据
                coef_df.to_csv(f'{save_path}/feature_coefficients_{target_name}_{best_reg_model}.csv', index=False)
            except Exception as e:
                print(f"  绘制特征系数失败: {str(e)}")
    
    # 保存模型性能结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{save_path}/model_performance_summary.csv', index=False)
    print(f"\n模型性能摘要已保存至: {save_path}/model_performance_summary.csv")
    
    # 创建模型比较图
    model_names = list(regression_models.keys())
    metric_data = {model: {'rmse': [], 'r2': []} for model in model_names}
    
    for result in results:
        for model_name in model_names:
            if model_name in result['regression_results']:
                metric_data[model_name]['rmse'].append(result['regression_results'][model_name]['rmse'])
                metric_data[model_name]['r2'].append(result['regression_results'][model_name]['r2'])
            else:
                metric_data[model_name]['rmse'].append(np.nan)
                metric_data[model_name]['r2'].append(np.nan)
    
    # 绘制模型比较图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE比较
    rmse_data = []
    for model_name in model_names:
        rmse_data.append(metric_data[model_name]['rmse'])
    
    ax1.boxplot(rmse_data, labels=model_names)
    ax1.set_title('各模型RMSE分布比较')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # R²比较
    r2_data = []
    for model_name in model_names:
        r2_data.append(metric_data[model_name]['r2'])
    
    ax2.boxplot(r2_data, labels=model_names)
    ax2.set_title('各模型R²分布比较')
    ax2.set_ylabel('R²')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results_df

def extract_features(df_order, df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status):
    features = []
    targets = []

    # 统计24/28孔的历史模式
    large_gate_history = df_order[df_order['开闸孔数'].isin([24, 28])]
    regions = ['绍兴平原', '嵊州', '虞南山区', '新昌', '虞北平原']
    
    # 创建空值统计器
    missing_stats = {
        'water_level': 0,
        'flow': 0,
        'rain_actual': 0,
        'rain_forecast': 0,
        'water_status': 0,
        'future_water_level': 0
    }

    # 过滤无效记录：开闸时间早于调令时间
    df_order = df_order[df_order['开闸时间'] > df_order['SIGNTM']]
    
    for idx, row in df_order.iterrows():
        base_time = row['SIGNTM']
        open_time = row['开闸时间']
        start_time_24h = base_time - timedelta(hours=24)
        start_time_week = base_time - timedelta(days=7)  # 新增：一周前的时间
        start_time_48h = base_time - timedelta(hours=48)
        
        # 检查所有目标变量是否有效
        if any(pd.isnull(row[['开闸小时', '处理后的开闸时长', '处理后的开闸孔数', '处理后的目标水位']])):
            continue
            
        # 处理开闸孔数 - 按照给定的分桶边界 [2, 8, 10, 12, 24, 28]
        gate_count = row['处理后的开闸孔数']
        
        # 开闸时长优化处理
        duration = row['处理后的开闸时长']
        
        # 计算开闸时间（小时+分钟）
        open_hour = open_time.hour + open_time.minute/60.0
        target_water_level = row['处理后的目标水位']

        # 联合目标
        dura_dot_num = row['处理后的开闸时长_孔洞']
        
        log_dura_dot_num = row['处理后的开闸时长_孔洞_log']

        targets.append([
            open_hour,
            duration,  # 使用处理后的开闸时长
            gate_count-1,  # 使用处理后的开闸孔数
            target_water_level,
            dura_dot_num,
            log_dura_dot_num
        ])
        feat_dict = {}
        
        feat_dict['date'] = base_time.date()
        # 时间特征
        feat_dict['hour_of_day'] = base_time.hour
        feat_dict['day_of_week'] = base_time.weekday()
        feat_dict['month'] = base_time.month
        feat_dict['is_weekend'] = 1 if base_time.weekday() >= 5 else 0
        feat_dict['hour_sin'] = np.sin(2 * np.pi * base_time.hour / 24)
        feat_dict['hour_cos'] = np.cos(2 * np.pi * base_time.hour / 24)
        feat_dict['day_of_year'] = base_time.timetuple().tm_yday
        
        # 关键修改位置：移除 gate_bin 特征
        # 原代码：feat_dict['gate_bin'] = row['gate_bin']
        # 已移除 gate_bin 特征
        
        # 历史操作特征 - 修改为最近一周
        prev_orders = df_order[df_order['SIGNTM'] < base_time].sort_values('SIGNTM', ascending=False)
        
        if len(prev_orders) > 0:
            latest = prev_orders.iloc[0]
            feat_dict['prev_gate_count'] = latest['开闸孔数']
            feat_dict['prev_duration'] = latest['开闸时长'] if not pd.isnull(latest['开闸时长']) else 0
            feat_dict['prev_op_hour'] = latest['开闸时间'].hour if not pd.isnull(latest['开闸时间']) else 0
        else:
            feat_dict['prev_gate_count'] = 0
            feat_dict['prev_duration'] = 0
            feat_dict['prev_op_hour'] = 0
        
        # 过去一周操作统计 - 修改为168小时（7天）
        last_week_orders = df_order[
            (df_order['SIGNTM'] >= start_time_week) & 
            (df_order['SIGNTM'] < base_time)
        ]
        feat_dict['ops_week_count'] = len(last_week_orders)
        feat_dict['ops_week_avg_gates'] = last_week_orders['开闸孔数'].mean() if len(last_week_orders) > 0 else 0
        feat_dict['ops_week_total_duration'] = last_week_orders['开闸时长'].sum() if len(last_week_orders) > 0 else 0
        
        # 潮位数据查询 - 优先使用48小时内数据
        water_data_24h = df_water_level[
            (df_water_level['time'] >= start_time_48h) & 
            (df_water_level['time'] <= base_time)
        ]

        # 如果数据不足，尝试使用更宽的时间范围
        if len(water_data_24h) < 12:  # 至少需要12个数据点
            start_time_72h = base_time - timedelta(hours=72)
            water_data_24h = df_water_level[
                (df_water_level['time'] >= start_time_72h) & 
                (df_water_level['time'] <= base_time)
            ]

        water_missing = 0
        if len(water_data_24h) > 0:
            tidal_features = extract_hourly_tidal_features(water_data_24h, 24)
            # 添加前缀以区分不同时间段
            for key, value in tidal_features.items():
                feat_dict[f'tide_24h_{key}'] = value
        else:
            missing_stats['water_level'] += 1
            water_missing = 1
            # 添加默认值
            default_features = extract_hourly_tidal_features(pd.DataFrame(), 24)
            for key, value in default_features.items():
                feat_dict[f'tide_24h_{key}'] = value

        feat_dict['water_missing'] = water_missing
        feat_dict['tide_type'] = tidal_features.get('tide_type', 0)
        
        # 过去12小时潮位特征
        start_time_12h = base_time - timedelta(hours=12)
        water_data_12h = df_water_level[
            (df_water_level['time'] >= start_time_12h) & 
            (df_water_level['time'] <= base_time)
        ]
        
        if len(water_data_12h) > 0:
            tidal_features = extract_hourly_tidal_features(water_data_12h, 12)
            for key, value in tidal_features.items():
                feat_dict[f'tide_12h_{key}'] = value
        else:
            # 使用24小时数据的后半部分作为近似
            if len(water_data_24h) > 12:  # 至少有12小时数据
                mid_point = len(water_data_24h) // 2
                water_data_12h_approx = water_data_24h.iloc[mid_point:]
                tidal_features = extract_hourly_tidal_features(water_data_12h_approx, 12)
                for key, value in tidal_features.items():
                    feat_dict[f'tide_12h_{key}'] = value
            else:
                # 使用默认值
                default_features = extract_hourly_tidal_features(pd.DataFrame(), 12)
                for key, value in default_features.items():
                    feat_dict[f'tide_12h_{key}'] = value
        
        # 未来12小时潮位特征
        # 在extract_features函数中添加更稳健的未来潮位预测
        future_water_data = df_water_level[
            (df_water_level['time'] >= base_time) & 
            (df_water_level['time'] <= base_time + timedelta(hours=12))
        ]

        if len(future_water_data) > 0:
            future_tidal_features = extract_hourly_tidal_features(future_water_data, 12)
            for key, value in future_tidal_features.items():
                feat_dict[f'future_tide_{key}'] = value
        else:
            missing_stats['future_water_level'] += 1
            future_water_missing = 1
            
            # 改进的预测方法 - 使用更长的历史数据和周期性模式
            historical_data = df_water_level[
                (df_water_level['time'] >= base_time - timedelta(days=7)) & 
                (df_water_level['time'] <= base_time)
            ]
            
            if len(historical_data) > 12:
                # 使用历史同期数据预测
                same_period_data = historical_data[
                    historical_data['time'].dt.time.isin([
                        (base_time + timedelta(hours=i)).time() for i in range(1, 13)
                    ])
                ]
                
                if len(same_period_data) > 0:
                    # 计算平均变化模式
                    future_levels = []
                    for hour_offset in range(1, 13):
                        target_time = (base_time + timedelta(hours=hour_offset)).time()
                        same_time_data = historical_data[historical_data['time'].dt.time == target_time]
                        if len(same_time_data) > 0:
                            future_level = same_time_data['water_level'].mean()
                            future_levels.append(future_level)
                    
                    if len(future_levels) == 12:
                        # 创建模拟的未来数据
                        future_times = [base_time + timedelta(hours=i) for i in range(1, 13)]
                        simulated_data = pd.DataFrame({
                            'time': future_times,
                            'water_level': future_levels
                        })
                        future_tidal_features = extract_hourly_tidal_features(simulated_data, 12)
                        for key, value in future_tidal_features.items():
                            feat_dict[f'future_tide_{key}'] = value
                    else:
                        # 使用默认值
                        default_features = extract_hourly_tidal_features(pd.DataFrame(), 12)
                        for key, value in default_features.items():
                            feat_dict[f'future_tide_{key}'] = value
                else:
                    # 使用线性外推
                    last_6h_data = historical_data[historical_data['time'] >= base_time - timedelta(hours=6)]
                    if len(last_6h_data) > 1:
                        time_numeric = (last_6h_data['time'] - last_6h_data['time'].min()).dt.total_seconds().values
                        slope, intercept, _, _, _ = safe_linregress(time_numeric, last_6h_data['water_level'])
                        
                        future_times = [base_time + timedelta(hours=i) for i in range(1, 13)]
                        future_levels = [intercept + slope * (i * 3600) for i in range(1, 13)]
                        
                        simulated_data = pd.DataFrame({
                            'time': future_times,
                            'water_level': future_levels
                        })
                        
                        future_tidal_features = extract_hourly_tidal_features(simulated_data, 12)
                        for key, value in future_tidal_features.items():
                            feat_dict[f'future_tide_{key}'] = value
                    else:
                        default_features = extract_hourly_tidal_features(pd.DataFrame(), 12)
                        for key, value in default_features.items():
                            feat_dict[f'future_tide_{key}'] = value
            else:
                default_features = extract_hourly_tidal_features(pd.DataFrame(), 12)
                for key, value in default_features.items():
                    feat_dict[f'future_tide_{key}'] = value
        
            feat_dict['future_water_missing'] = future_water_missing
        
        # 流量特征
        flow_data = df_flow[
            (df_flow['监测日期'] >= start_time_24h) & 
            (df_flow['监测日期'] <= base_time)
        ]
        
        flow_missing = 0
        if len(flow_data) > 0:
            feat_dict['flow_mean'] = flow_data['流量'].mean()
            feat_dict['flow_max'] = flow_data['流量'].max()
            feat_dict['flow_min'] = flow_data['流量'].min()
            feat_dict['flow_range'] = feat_dict['flow_max'] - feat_dict['flow_min']
            feat_dict['flow_var'] = flow_data['流量'].var()
            feat_dict['flow_skew'] = flow_data['流量'].skew() if len(flow_data) > 2 else 0
        else:
            missing_stats['flow'] += 1
            flow_missing = 1
            feat_dict.update({f'flow_{stat}': 0 for stat in ['mean', 'max', 'min', 'range', 'var', 'skew']})
        
        feat_dict['flow_missing'] = flow_missing
        
        # 降雨特征 - 修改缺失统计逻辑
        rain_actual_data = df_rain_actual[
            (df_rain_actual['监测日期'] >= start_time_24h) & 
            (df_rain_actual['监测日期'] <= base_time)
        ]
        
        rain_forecast_data = df_rain_forecast[
            (df_rain_forecast['预计开始时间'] >= base_time) & 
            (df_rain_forecast['预计开始时间'] <= base_time + timedelta(hours=24))
        ]
        
        total_rain_actual = 0
        total_rain_forecast = 0
        rain_actual_missing_in_record = False
        rain_forecast_missing_in_record = False
        
        for region in regions:
            # 实际降雨
            region_rain = rain_actual_data[rain_actual_data['所属区域'] == region]
            if len(region_rain) > 0:
                feat_dict[f'rain_actual_{region}_sum'] = region_rain['雨量'].sum()
                feat_dict[f'rain_actual_{region}_max'] = region_rain['雨量'].max()
                feat_dict[f'rain_actual_{region}_mean'] = region_rain['雨量'].mean()
                total_rain_actual += feat_dict[f'rain_actual_{region}_sum']
            else:
                rain_actual_missing_in_record = True
                feat_dict[f'rain_actual_{region}_sum'] = 0
                feat_dict[f'rain_actual_{region}_max'] = 0
                feat_dict[f'rain_actual_{region}_mean'] = 0
            
            # 预测降雨
            region_forecast = rain_forecast_data[rain_forecast_data['大流域'] == region]
            if len(region_forecast) > 0:
                feat_dict[f'rain_forecast_{region}_mean'] = region_forecast['降雨量'].mean()
                feat_dict[f'rain_forecast_{region}_max'] = region_forecast['降雨量'].max()
                feat_dict[f'rain_forecast_{region}_sum'] = region_forecast['降雨量'].sum()
                total_rain_forecast += feat_dict[f'rain_forecast_{region}_sum']
            else:
                rain_forecast_missing_in_record = True
                feat_dict[f'rain_forecast_{region}_mean'] = 0
                feat_dict[f'rain_forecast_{region}_max'] = 0
                feat_dict[f'rain_forecast_{region}_sum'] = 0
        
        # 区域平均降雨特征
        feat_dict['rain_actual_total'] = total_rain_actual
        feat_dict['rain_forecast_total'] = total_rain_forecast
        feat_dict['rain_actual_avg'] = total_rain_actual / len(regions) if len(regions) > 0 else 0
        feat_dict['rain_forecast_avg'] = total_rain_forecast / len(regions) if len(regions) > 0 else 0
        feat_dict['rain_missing'] = 1 if (rain_actual_missing_in_record or rain_forecast_missing_in_record) else 0
        
        # 更新缺失统计（每条记录只计一次）
        if rain_actual_missing_in_record:
            missing_stats['rain_actual'] += 1
        if rain_forecast_missing_in_record:
            missing_stats['rain_forecast'] += 1
        
        # 添加降雨变化率特征
        if len(rain_actual_data) > 1:
            rain_actual_data = rain_actual_data.sort_values('监测日期')
            rain_diff = rain_actual_data['雨量'].diff().dropna()
            feat_dict['rain_change_rate'] = rain_diff.mean() if len(rain_diff) > 0 else 0
        else:
            feat_dict['rain_change_rate'] = 0
        
        # 添加组合特征
        feat_dict['water_rain_ratio'] = feat_dict.get('tide_24h_mean', 0) / (feat_dict['rain_actual_total'] + 1e-5)
        feat_dict['flow_rain_ratio'] = feat_dict.get('flow_mean', 0) / (feat_dict['rain_actual_total'] + 1e-5)
        
        # 添加时间敏感性特征
        feat_dict['is_rush_hour'] = 1 if 7 <= base_time.hour <= 9 or 17 <= base_time.hour <= 19 else 0

        # 水位工况特征
        water_status_data = df_water_status[
            (df_water_status['监测日期'] >= start_time_24h) & 
            (df_water_status['监测日期'] <= base_time)
        ]
        
        water_status_missing = 0
        if len(water_status_data) > 0:
            feat_dict['water_status_mean'] = water_status_data['水位'].mean()
            feat_dict['water_status_max'] = water_status_data['水位'].max()
            feat_dict['water_status_min'] = water_status_data['水位'].min()
            feat_dict['water_status_range'] = feat_dict['water_status_max'] - feat_dict['water_status_min']
            
            if len(water_status_data) > 1:
                times = (water_status_data['监测日期'] - water_status_data['监测日期'].min()).dt.total_seconds().values
                slope, _, _, _, _ = safe_linregress(times, water_status_data['水位'])
                feat_dict['water_status_slope'] = slope
            else:
                feat_dict['water_status_slope'] = 0
        else:
            missing_stats['water_status'] += 1
            water_status_missing = 1
            feat_dict.update({f'water_status_{stat}': 0 for stat in ['mean', 'max', 'min', 'range', 'slope']})
        
        feat_dict['water_status_missing'] = water_status_missing
        
        # 新增：24/28孔专属特征
        feat_dict['is_large_gate_pattern'] = 0
        # 检查是否符合大孔数开启模式
        if len(large_gate_history) > 0:
            # 历史大孔数开启的平均时长
            avg_large_duration = large_gate_history['开闸时长'].mean()
            feat_dict['large_gate_avg_duration'] = avg_large_duration
            
            # 历史大孔数开启的时间模式（小时）
            large_gate_hours = large_gate_history['开闸时间'].dt.hour
            current_hour = base_time.hour
            feat_dict['hour_similar_to_large_gate'] = 1 if current_hour in large_gate_hours.values else 0
            
            # 潮位条件与大孔数开启的历史模式比较
            current_tide_mean = feat_dict.get('tide_24h_mean', 0)
            historical_tide_mean = large_gate_history['关联潮位均值'].mean() if '关联潮位均值' in large_gate_history else current_tide_mean
            feat_dict['tide_similar_to_large_gate'] = abs(current_tide_mean - historical_tide_mean)
            
            # 降雨条件相似度
            current_rain = feat_dict.get('rain_actual_total', 0)
            historical_rain = large_gate_history['关联降雨量'].mean() if '关联降雨量' in large_gate_history else current_rain
            feat_dict['rain_similar_to_large_gate'] = abs(current_rain - historical_rain)
            
            # 如果是24/28孔样本，标记为重点学习样本
            if row['开闸孔数'] in [24, 28]:
                feat_dict['is_large_gate_pattern'] = 1
                feat_dict['large_gate_priority'] = 2.0  # 给予更高权重
            else:
                feat_dict['large_gate_priority'] = 1.0

        features.append(feat_dict)
    
    # 打印缺失值统计
    print("\n特征缺失统计:")
    for key, count in missing_stats.items():
        print(f"{key}: {count} 条记录缺失 (占总记录数 {count/len(df_order)*100:.2f}%)")
    return pd.DataFrame(features), np.array(targets)

def save_features(feat_dir="features/features_0822", prefix="00"):
    # 加载调令信息数据
    # if prefix == "11":
    #     # 加载两个数据源
    #     df_order = pd.read_csv(
    #         f"imports/07_processed_orders.csv",
    #         parse_dates=['SIGNTM', '开闸时间']
    #     )
    #     train_df_order = pd.read_csv(
    #         f"imports/00_processed_orders.csv",
    #         parse_dates=['SIGNTM', '开闸时间']
    #     )
        
    #     # 计算需要采样的数量（确保为整数）
    #     sample_size = int(len(train_df_order) * 0.05)
    #     # 从训练数据中随机采样，避免使用不适合DataFrame的random.choice
    #     sampled_data = train_df_order.sample(n=sample_size, random_state=42)  # 固定随机种子，保证可复现性
        
    #     # 合并数据（修正原代码中错误的+运算）
    #     df_order = pd.concat([df_order, sampled_data], ignore_index=True)
    # else:
    #     # 加载对应前缀的数据源
    #     df_order = pd.read_csv(
    #         f"imports/{prefix}_processed_orders.csv",
    #         parse_dates=['SIGNTM', '开闸时间']
    #     )

    df_order = pd.read_csv(
        f"imports/{prefix}_processed_orders.csv",
        parse_dates=['SIGNTM', '开闸时间']
    )
    del df_order['gate_bin']
    df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status = load_data()
    X, y = extract_features(df_order, df_water_level, df_flow, df_rain_actual, df_rain_forecast, df_water_status)
    
    # 保存特征和目标
    os.makedirs(feat_dir, exist_ok=True)
    X.to_csv(f'{feat_dir}/{prefix}_features.csv', index=False)
    np.save(f'{feat_dir}/{prefix}_target.npy', y)
    
    # 保存缺失值处理信息
    missing_info = pd.DataFrame({
        'feature': X.columns,
        'missing_count': X.isnull().sum(),
        'missing_percent': X.isnull().mean() * 100
    })

    missing_info.to_csv(f'{feat_dir}/{prefix}_feature_missing_info.csv', index=False)
    print(f"\n特征数据已保存: {prefix}_features.csv 和 {prefix}_target.npy")
    print(f"缺失值统计已保存: {prefix}_feature_missing_info.csv")
    print(f"目标变量维度: {y.shape} (开闸时间, 开闸时长, 开闸孔数, 目标水位, 联合时长-孔数, 联合时长-孔数-log)")
    
    # 为目标变量拟合模型
    if len(X) > 0 and len(y) > 0:
        vis_dir = f"{feat_dir}/visualizations/{prefix}"
        os.makedirs(vis_dir, exist_ok=True)
        fit_models_for_targets(X, y, vis_dir)


if __name__ == '__main__':
    # 训练集
    save_features(prefix="00")
    # 测试集 7+8月
    save_features(prefix="07")
    # 测试集 7+8+随机月
    save_features(prefix="11")
