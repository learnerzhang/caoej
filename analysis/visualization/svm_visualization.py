import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 1. 生成二维示例数据（二分类）
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6, n_features=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 训练线性SVM模型
svm_model = SVC(kernel='linear', C=1.0)  # 线性核
svm_model.fit(X_train, y_train)

# 3. 生成网格数据，用于绘制决策边界
h = 0.02  # 网格步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 4. 预测网格点的类别（用于绘制决策边界）
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 5. 可视化
plt.figure(figsize=(10, 6))

# 绘制决策边界和分类区域
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

# 绘制训练样本
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', label='训练样本')

# 绘制支持向量（用特殊标记突出）
plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], 
           s=100, linewidth=1, facecolors='none', edgecolors='red', label='支持向量')

# 随机选择一个测试样本，展示预测过程
test_sample = X_test[0]
pred_label = svm_model.predict([test_sample])[0]
plt.scatter(test_sample[0], test_sample[1], s=200, marker='*', 
           c='yellow', edgecolors='black', label=f'测试样本（预测：{pred_label}）')

# 添加标签和标题
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('SVM决策边界、支持向量及预测示例')
plt.legend()
plt.savefig('svm_visualization.png')
# plt.show()
