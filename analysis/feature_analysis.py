import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体 - 解决中文乱码问题# 设置中文字体 - 解决中文乱码问题
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置绘图风格
sns.set(style="whitegrid")

def load_data(prefix='00', feat_dir="features_0823"):
    X = pd.read_csv(f'{feat_dir}/{prefix}_features.csv')
    y = np.load(f'{feat_dir}/{prefix}_target.npy')
    return X, y

def analyze_features(X, y):
    # 目标变量名称
    target_names = ['开闸时间(小时)', '开闸时长', '开闸孔数', '目标水位', '时长孔数乘积', '时长孔数乘积(对数)']
    
    # 首先处理特征中的缺失值（如果有）
    X.fillna(0, inplace=True)
    
    # 1. 特征分布检查
    print("特征分布情况：")
    print(X.describe())
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制一些特征的分布图
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    # 选择部分特征绘制直方图
    fig, axes = plt.subplots(5, 5, figsize=(20, 15))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols[:25]):
        sns.histplot(X[col], ax=axes[i], kde=True)
        axes[i].set_title(col)
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    
    # 为每个目标变量创建分析目录
    os.makedirs('target_analysis', exist_ok=True)
    
    # 对每个目标变量进行分析
    for i, target_name in enumerate(target_names):
        print(f"\n=== 分析目标变量: {target_name} ===")
        
        # 当前目标变量
        current_target = y[:, i]
        
        # 2. 特征与目标的相关性分析
        corr_with_target = X[numeric_cols].corrwith(pd.Series(current_target))
        corr_with_target.sort_values(ascending=False, inplace=True)
        print(f"\n特征与目标 '{target_name}' 的相关性：")
        print(corr_with_target.head(10))  # 只显示前10个
        plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
        plt.rcParams['axes.unicode_minus'] = False
        # 绘制相关性条形图
        plt.figure(figsize=(12, 10))
        corr_with_target.head(20).plot(kind='barh')
        plt.title(f'与"{target_name}"相关性最高的20个特征')
        plt.tight_layout()
        plt.savefig(f'target_analysis/correlation_{target_name}.png')
        plt.close()
        
        # 3. 特征重要性分析（使用随机森林）
        X_train, X_test, y_train, y_test = train_test_split(
            X[numeric_cols], current_target, test_size=0.2, random_state=42
        )
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        print(f"随机森林模型性能 ({target_name}): R2={r2_score(y_test, y_pred):.4f}, MSE={mean_squared_error(y_test, y_pred):.4f}")
        
        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': rf.feature_importances_
        })
        feature_importance.sort_values('importance', ascending=False, inplace=True)
        print(f"\n特征重要性排序 ({target_name}):")
        print(feature_importance.head(20))  # 只显示前20个
        plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
        plt.rcParams['axes.unicode_minus'] = False
        # 绘制特征重要性条形图
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title(f'对"{target_name}"最重要的20个特征')
        plt.tight_layout()
        plt.savefig(f'target_analysis/importance_{target_name}.png')
        plt.close()
        
        # 4. 保存详细结果到CSV文件
        # 合并相关性和重要性
        analysis_result = pd.DataFrame({
            'feature': numeric_cols,
            'correlation': X[numeric_cols].corrwith(pd.Series(current_target)),
            'importance': rf.feature_importances_
        })
        analysis_result.sort_values('importance', ascending=False, inplace=True)
        analysis_result.to_csv(f'target_analysis/feature_analysis_{target_name}.csv', index=False, encoding='utf-8-sig')
        
        # 5. 绘制特征与目标变量的散点图（仅对前5个最重要特征）
        plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
        plt.rcParams['axes.unicode_minus'] = False
        top_features = feature_importance.head(5)['feature'].values
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for j, feature in enumerate(top_features[:6]):  # 最多6个图
            if j < len(axes):
                axes[j].scatter(X[feature], current_target, alpha=0.5)
                axes[j].set_xlabel(feature, )
                axes[j].set_ylabel(target_name,  )
                axes[j].set_title(f'{feature} vs {target_name}',  )
        
        # 移除多余的子图
        for j in range(len(top_features), len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.savefig(f'target_analysis/scatter_{target_name}.png')
        plt.close()
    
    # 6. 目标变量之间的关系分析
    print("\n=== 目标变量之间的关系分析 ===")
    target_df = pd.DataFrame(y, columns=target_names)
    target_corr = target_df.corr()
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 10))
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('目标变量之间的相关性',  )
    plt.tight_layout()
    plt.savefig('target_analysis/target_correlation.png')
    plt.close()
    
    print("目标变量相关性矩阵:")
    print(target_corr)
    
    # 保存目标变量相关性矩阵
    target_corr.to_csv('target_analysis/target_correlation_matrix.csv', encoding='utf-8-sig')

if __name__ == '__main__':
    X, y = load_data('00')
    analyze_features(X, y)