import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import os
from datetime import datetime

# 设置字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

# 模型配置
MODEL_CONFIGS = {
    'lr': {
        'model_path': 'outputs/kz_dura_lr_optimized/pkls/optimized_lr_model.pkl',
        'encoder_path': 'outputs/kz_dura_lr_optimized/pkls/lr_label_encoder.pkl',
        'name': '逻辑回归'
    },
    'mlp': {
        'model_path': 'outputs/kz_dura_class7_mlp/pkls/mlp_classification_model.pkl',
        'encoder_path': 'outputs/kz_dura_class7_mlp/pkls/mlp_label_encoder.pkl',
        'name': 'MLP'
    },
    'rf': {
        'model_path': 'outputs/kz_dura/pkls/rf_optimized_model.pkl',
        'encoder_path': 'outputs/kz_dura/pkls/rf_label_encoder.pkl',
        'name': '随机森林'
    },
    'svm': {
        'model_path': 'outputs/kz_dura_svm_class7/pkls/svm_classification_model.pkl',
        'encoder_path': 'outputs/kz_dura_svm_class7/pkls/svm_label_encoder.pkl',
        'name': 'SVM'
    },
    'xgb': {
        'model_path': 'outputs/kz_dura_xgb_class7/pkls/xgb_classification_model.pkl',
        'encoder_path': 'outputs/kz_dura_xgb_class7/pkls/xgb_label_encoder.pkl',
        'name': 'XGBoost'
    }
}

def create_evaluation_directories():
    """创建评估输出目录"""
    for model_type in MODEL_CONFIGS.keys():
        os.makedirs(f'outputs/kz_dura/eval_{model_type}/plots', exist_ok=True)
        os.makedirs(f'outputs/kz_dura/eval_{model_type}/results', exist_ok=True)

def load_and_preprocess_test_data(test_data_path, test_target_path):
    """加载并预处理测试数据"""
    X_test = pd.read_csv(test_data_path)
    y_test_full = np.load(test_target_path)
    y_test_duration = y_test_full[:, 1].astype(int)  # 开闸时长
    
    # 检查并处理日期列
    if 'date' in X_test.columns:
        dates_test = X_test['date']
    else:
        # 如果没有日期列，创建虚拟日期
        dates_test = pd.Series([f"test_{i}" for i in range(len(X_test))])
    
    return X_test, dates_test, y_test_duration

def analyze_class7_performance(y_true, y_pred, label_encoder, dataset_name, model_name):
    """专门分析类别7（长时间开闸）的表现"""
    print(f"\n=== {model_name} - {dataset_name} - 类别7详细分析 ===")
    
    # 检查类别7是否存在
    if 7 not in label_encoder.classes_:
        print("警告: 测试集中不存在类别7")
        return None
    
    class7_encoded = label_encoder.transform([7])[0]
    class7_mask = (y_true == class7_encoded)
    
    if np.sum(class7_mask) == 0:
        print("警告: 测试集中没有类别7的样本")
        return None
    
    # 类别7的指标
    class7_accuracy = accuracy_score(y_true[class7_mask], y_pred[class7_mask])
    class7_precision = precision_score(y_true[class7_mask], y_pred[class7_mask], average='binary', pos_label=class7_encoded)
    class7_recall = recall_score(y_true[class7_mask], y_pred[class7_mask], average='binary', pos_label=class7_encoded)
    class7_f1 = f1_score(y_true[class7_mask], y_pred[class7_mask], average='binary', pos_label=class7_encoded)
    
    print(f"类别7样本数量: {np.sum(class7_mask)}")
    print(f"类别7准确率: {class7_accuracy:.4f}")
    print(f"类别7精确率: {class7_precision:.4f}")
    print(f"类别7召回率: {class7_recall:.4f}")
    print(f"类别7 F1分数: {class7_f1:.4f}")
    
    # 分析类别7的误分类情况
    class7_pred = y_pred[class7_mask]
    misclassified = class7_pred != y_true[class7_mask]
    
    if np.any(misclassified):
        misclassified_classes = class7_pred[misclassified]
        misclassified_counts = pd.Series(misclassified_classes).value_counts()
        print("\n类别7被误分类为:")
        for cls, count in misclassified_counts.items():
            original_duration = label_encoder.inverse_transform([cls])[0]
            print(f"  类别 {original_duration}: {count} 次")
    
    return {
        'sample_count': np.sum(class7_mask),
        'accuracy': class7_accuracy,
        'precision': class7_precision,
        'recall': class7_recall,
        'f1': class7_f1
    }

def create_enhanced_confusion_matrix(y_true, y_pred, label_encoder, dataset_name, model_name, model_type):
    """创建增强的混淆矩阵可视化"""
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建热力图
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                annot_kws={'size': 10})
    plt.xlabel('预测时长类别', fontsize=12)
    plt.ylabel('真实时长类别', fontsize=12)
    plt.title(f'{model_name} - {dataset_name} - 开闸时长分类混淆矩阵', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'outputs/kz_dura/eval_{model_type}/plots/{dataset_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_performance_metrics(y_true, y_pred, label_encoder, dataset_name, model_name, model_type):
    """绘制每个类别的性能指标"""
    classes = label_encoder.classes_
    class_metrics = []
    
    for cls in classes:
        cls_encoded = label_encoder.transform([cls])[0]
        mask = (y_true == cls_encoded)
        
        if np.sum(mask) > 0:
            accuracy = accuracy_score(y_true[mask], y_pred[mask])
            precision = precision_score(y_true, y_pred, average=None, labels=[cls_encoded])[0]
            recall = recall_score(y_true, y_pred, average=None, labels=[cls_encoded])[0]
            f1 = f1_score(y_true, y_pred, average=None, labels=[cls_encoded])[0]
            
            class_metrics.append({
                'class': cls,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'sample_count': np.sum(mask)
            })
    
    metrics_df = pd.DataFrame(class_metrics)
    
    # 绘制多指标对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['准确率', '精确率', '召回率', 'F1分数']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        bars = ax.bar(metrics_df['class'].astype(str), metrics_df[metric])
        ax.set_xlabel('开闸时长类别')
        ax.set_ylabel(title)
        ax.set_title(f'{title}按类别分布')
        ax.tick_params(axis='x', rotation=45)
        
        # 在柱子上添加数值标注
        for bar, value in zip(bars, metrics_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'{model_name} - {dataset_name} - 类别性能指标', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'outputs/kz_dura/eval_{model_type}/plots/{dataset_name}_class_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics_df

def evaluate_single_model(model_type, test_data_path, test_target_path, dataset_name):
    """评估单个模型"""
    config = MODEL_CONFIGS[model_type]
    
    print(f"\n{'='*60}")
    print(f"评估 {config['name']} 模型 - 数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 加载测试数据
    X_test, dates_test, y_test_duration = load_and_preprocess_test_data(test_data_path, test_target_path)
    
    # 加载模型和标签编码器
    try:
        model = joblib.load(config['model_path'])
        label_encoder = joblib.load(config['encoder_path'])
        print(f"✅ 模型加载成功: {config['model_path']}")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None, None
    
    # 编码目标变量
    try:
        y_test_encoded = label_encoder.transform(y_test_duration)
    except ValueError as e:
        print(f"❌ 标签编码失败: {str(e)}")
        print("尝试重新拟合标签编码器...")
        label_encoder = LabelEncoder()
        y_test_encoded = label_encoder.fit_transform(y_test_duration)
    
    # 预测
    y_pred_encoded = model.predict(X_test)
    y_pred_duration = label_encoder.inverse_transform(y_pred_encoded)
    
    # 基础评估指标
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    f1_weighted = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
    f1_macro = f1_score(y_test_encoded, y_pred_encoded, average='macro')
    precision_weighted = precision_score(y_test_encoded, y_pred_encoded, average='weighted')
    recall_weighted = recall_score(y_test_encoded, y_pred_encoded, average='weighted')
    
    print(f"\n📊 {config['name']}模型评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"加权F1分数: {f1_weighted:.4f}")
    print(f"宏平均F1分数: {f1_macro:.4f}")
    print(f"加权精确率: {precision_weighted:.4f}")
    print(f"加权召回率: {recall_weighted:.4f}")
    
    # 详细分类报告
    test_classes = np.unique(y_test_encoded)
    test_target_names = [str(cls) for cls in label_encoder.inverse_transform(test_classes)]
    
    print(f"\n📋 详细分类报告:")
    print(classification_report(y_test_encoded, y_pred_encoded, 
                              labels=test_classes,
                              target_names=test_target_names))
    
    # 类别7专项分析
    class7_results = analyze_class7_performance(y_test_encoded, y_pred_encoded, label_encoder, dataset_name, config['name'])
    
    # 可视化
    create_enhanced_confusion_matrix(y_test_encoded, y_pred_encoded, label_encoder, dataset_name, config['name'], model_type)
    metrics_df = plot_class_performance_metrics(y_test_encoded, y_pred_encoded, label_encoder, dataset_name, config['name'], model_type)
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'date': dates_test.values,
        'true_duration': y_test_duration,
        'pred_duration': y_pred_duration,
        'correct': (y_test_duration == y_pred_duration)
    })
    
    # 保存详细结果
    results_df.to_csv(f'outputs/kz_dura/eval_{model_type}/results/{dataset_name}_detailed_predictions.csv', index=False)
    metrics_df.to_csv(f'outputs/kz_dura/eval_{model_type}/results/{dataset_name}_class_metrics.csv', index=False)
    
    # 保存评估摘要
    summary = {
        'model': config['name'],
        'dataset': dataset_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'total_samples': len(results_df),
        'correct_predictions': results_df['correct'].sum(),
        'class7_accuracy': class7_results['accuracy'] if class7_results else None,
        'class7_f1': class7_results['f1'] if class7_results else None
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f'outputs/kz_dura/eval_{model_type}/results/{dataset_name}_evaluation_summary.csv', index=False)
    
    print(f"\n✅ {config['name']}模型评估完成!")
    print(f"📁 样本数量: {len(results_df)}")
    print(f"🎯 总体准确率: {accuracy:.4f}")
    
    return summary

def evaluate_all_models(test_data_path='features/features_0822/07_features.csv', 
                       test_target_path='features/features_0822/07_target.npy',
                       dataset_name='07'):
    """评估所有模型"""
    
    # 创建输出目录
    create_evaluation_directories()
    
    all_results = []
    
    for model_type in MODEL_CONFIGS.keys():
        try:
            result = evaluate_single_model(model_type, test_data_path, test_target_path, dataset_name)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ 评估 {MODEL_CONFIGS[model_type]['name']} 模型时出错: {str(e)}")
    
    # 生成比较报告
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        comparison_df.to_csv(f'outputs/kz_dura/all_models_comparison_{dataset_name}.csv', index=False)
        
        # 绘制比较图
        plt.figure(figsize=(12, 8))
        x = range(len(comparison_df))
        width = 0.2
        
        plt.bar([i - width for i in x], comparison_df['accuracy'], width, label='准确率', alpha=0.8)
        plt.bar(x, comparison_df['f1_weighted'], width, label='加权F1', alpha=0.8)
        plt.bar([i + width for i in x], comparison_df['f1_macro'], width, label='宏平均F1', alpha=0.8)
        
        plt.xlabel('模型')
        plt.ylabel('分数')
        plt.title(f'不同模型在{dataset_name}数据集上的性能比较')
        plt.xticks(x, comparison_df['model'])
        plt.legend()
        plt.ylim(0, 1)
        
        # 添加数值标注
        for i, (acc, f1w, f1m) in enumerate(zip(comparison_df['accuracy'], comparison_df['f1_weighted'], comparison_df['f1_macro'])):
            plt.text(i - width, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
            plt.text(i, f1w + 0.01, f'{f1w:.3f}', ha='center', va='bottom', fontsize=8)
            plt.text(i + width, f1m + 0.01, f'{f1m:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'outputs/kz_dura/models_comparison_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 所有模型评估完成! 比较结果已保存")
        print(f"🏆 最佳准确率模型: {comparison_df.loc[comparison_df['accuracy'].idxmax(), 'model']} ({comparison_df['accuracy'].max():.4f})")
        print(f"🏆 最佳F1分数模型: {comparison_df.loc[comparison_df['f1_weighted'].idxmax(), 'model']} ({comparison_df['f1_weighted'].max():.4f})")
    
    return all_results

def compare_multiple_datasets():
    """比较多个数据集的评估结果"""
    datasets = [
        ('07', 'features/features_0822/07_features.csv', 'features/features_0822/07_target.npy'),
        ('11', 'features/features_0822/11_features.csv', 'features/features_0822/11_target.npy')
    ]
    
    all_comparisons = []
    
    for dataset_name, data_path, target_path in datasets:
        print(f"\n{'='*60}")
        print(f"评估数据集: {dataset_name}")
        print(f"{'='*60}")
        
        results = evaluate_all_models(data_path, target_path, dataset_name)
        if results:
            for result in results:
                result['dataset'] = dataset_name
            all_comparisons.extend(results)
    
    # 保存跨数据集比较结果
    if all_comparisons:
        cross_comparison_df = pd.DataFrame(all_comparisons)
        cross_comparison_df.to_csv('outputs/kz_dura/cross_dataset_comparison.csv', index=False)
        
        # 绘制跨数据集比较图
        pivot_df = cross_comparison_df.pivot_table(index='model', columns='dataset', values='accuracy')
        
        plt.figure(figsize=(10, 6))
        pivot_df.plot(kind='bar', figsize=(12, 6))
        plt.title('不同模型在不同数据集上的准确率比较')
        plt.ylabel('准确率')
        plt.xticks(rotation=45)
        plt.legend(title='数据集')
        plt.tight_layout()
        plt.savefig('outputs/kz_dura/cross_dataset_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    # 比较多个数据集
    compare_multiple_datasets()
    
    # 单独评估07号数据集（详细分析）
    print(f"\n{'='*60}")
    print("详细评估07号数据集")
    print(f"{'='*60}")
    evaluate_all_models(dataset_name='07_detailed')