import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. 加载数据并训练模型（以乳腺癌分类为例）
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练一个随机森林模型（复杂模型示例）
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. 初始化SHAP解释器
explainer = shap.TreeExplainer(model)  # 针对树模型的优化解释器
shap_values = explainer.shap_values(X_test)  # 计算测试集样本的SHAP值

# 3. 可视化单个样本的预测过程（特征贡献）
sample_idx = 0  # 选择第一个测试样本
plt.figure(figsize=(10, 6))
shap.bar_plot(
    shap_values[1][sample_idx],  # 类别1的SHAP值（正值表示推动预测为类别1）
    feature_names=data.feature_names,
    # title=f"样本{sample_idx}的预测解释（真实标签：{y_test[sample_idx]}，预测标签：{model.predict([X_test[sample_idx]])[0]}）"
)
plt.tight_layout()
plt.show()

# 4. 可视化多个样本的特征影响（蜂群图）
shap.summary_plot(shap_values[1], X_test, feature_names=data.feature_names, plot_type="beeswarm")

plt.savefig("shap_explanation.png")    