import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（与evaluate.py保持一致）
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('正弦函数示例', fontsize=16)
plt.xlabel('X轴', fontsize=14)
plt.ylabel('Y轴', fontsize=14)
plt.grid(True)
plt.savefig('simple_sine_plot.png')
plt.show()

import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

# 数据
categories = ['苹果', '香蕉', '橙子', '葡萄', '西瓜']
values = [25, 40, 30, 35, 50]

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(categories, values, color=['red', 'yellow', 'orange', 'purple', 'green'])
plt.title('水果销量', fontsize=16)
plt.xlabel('水果种类', fontsize=14)
plt.ylabel('销量(公斤)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('fruit_sales.png')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

# 生成随机数据
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100)

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7)
plt.title('散点图示例', fontsize=16)
plt.xlabel('自变量', fontsize=14)
plt.ylabel('因变量', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加趋势线
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", linewidth=2)

plt.savefig('scatter_plot.png')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
x = np.linspace(0, 10, 100)

# 创建2x2的子图
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('多函数图形示例', fontsize=16)

# 子图1: 正弦函数
axs[0, 0].plot(x, np.sin(x), 'r-')
axs[0, 0].set_title('正弦函数')
axs[0, 0].grid(True)

# 子图2: 余弦函数
axs[0, 1].plot(x, np.cos(x), 'b-')
axs[0, 1].set_title('余弦函数')
axs[0, 1].grid(True)

# 子图3: 指数函数
axs[1, 0].plot(x, np.exp(x/5), 'g-')
axs[1, 0].set_title('指数函数')
axs[1, 0].grid(True)

# 子图4: 对数函数
axs[1, 1].plot(x[1:], np.log(x[1:]), 'm-')
axs[1, 1].set_title('对数函数')
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('multiple_plots.png')
plt.show()