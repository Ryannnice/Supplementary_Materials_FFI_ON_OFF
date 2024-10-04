import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def read_values(file_path, num_lines=20):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        values = [float(line.strip()) for line in lines[:num_lines]]
    return values

# 获取当前文件夹路径
current_directory = os.path.dirname(os.path.abspath(__file__))

# 文件名列表
filenames = ['ave1.txt', 'ave2.txt', 'aveo1.txt', 'aveo2.txt']
data = pd.DataFrame()

# 读取每个文件的前20行数据并存储在 DataFrame 中
for filename in filenames:
    file_path = os.path.join(current_directory, filename)
    values = read_values(file_path)  # 仅读取前20行数据
    index = list(range(0, 20, 1))  # 从0开始，每2递增，直到20
    temp_df = pd.DataFrame({'index': index, 'time': values, 'file': filename})
    data = pd.concat([data, temp_df], ignore_index=True)

sns.set_theme(style="whitegrid")

# 创建图像和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))

# 定义填充区域函数
def fill_between_curves(ax, x, y1, y2, color, alpha=0.3):
    ax.fill_between(x, y1, y2, interpolate=True, color=color, alpha=alpha)

# 填充第一个子图的区域
subset1 = data[data['file'].isin(['ave1.txt', 'aveo1.txt'])]
index_range1 = subset1[subset1['file'] == 'ave1.txt']['index']
ax1.plot(index_range1, subset1[subset1['file'] == 'ave1.txt']['time'], label='LGMD1', color='#3596B5', linewidth=3 )
ax1.plot(index_range1, subset1[subset1['file'] == 'aveo1.txt']['time'], label='oLGMD1', color='#296073', linewidth=3)
fill_between_curves(ax1, index_range1, subset1[subset1['file'] == 'ave1.txt']['time'], subset1[subset1['file'] == 'aveo1.txt']['time'], color='#3498db', alpha=0.4)

# 设置第一个子图的标题和轴标签
ax1.set_title('Efficiency for oLGMD1', fontsize=25)
ax1.set_xlabel('Index of Collision Event', fontsize=25)
ax1.set_ylabel('Average Processing Time\n of 100 Frames / ms  ', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=17)  # 设置刻度字体大小
ax1.set_ylim(0, subset1['time'].max() * 1.05)

ax1.set_xticks(np.arange(0, 20, 2))
ax1.grid(which='both', linestyle='--', linewidth=0.5)

ax1.legend(fontsize=20)  # 调整图例字体大小

# 填充第二个子图的区域
subset2 = data[data['file'].isin(['ave2.txt', 'aveo2.txt'])]
index_range2 = subset2[subset2['file'] == 'ave2.txt']['index']
ax2.plot(index_range2, subset2[subset2['file'] == 'ave2.txt']['time'], label='LGMD2', color='#C493FF', linewidth=3)
ax2.plot(index_range2, subset2[subset2['file'] == 'aveo2.txt']['time'], label='oLGMD2', color='#845EC2', linewidth=3)
fill_between_curves(ax2, index_range2, subset2[subset2['file'] == 'ave2.txt']['time'], subset2[subset2['file'] == 'aveo2.txt']['time'], color='#9b59b6', alpha=0.4)

# 设置第二个子图的标题和轴标签
ax2.set_title('Efficiency for oLGMD2', fontsize=25)
ax2.set_xlabel('Index of Collision Event', fontsize=25)
ax2.tick_params(axis='both', which='major', labelsize=17)  # 设置刻度字体大小
ax2.set_ylim(0, subset1['time'].max() * 1.05)

ax2.set_xticks(np.arange(0, 20, 2))
ax2.grid(which='both', linestyle='--', linewidth=0.5)

ax2.legend(fontsize=20)  # 调整图例字体大小

# 去掉右侧子图的纵坐标刻度值
ax2.set_yticklabels([])

# 调整布局，以防重叠
plt.tight_layout()

# 显示图表
plt.show()
