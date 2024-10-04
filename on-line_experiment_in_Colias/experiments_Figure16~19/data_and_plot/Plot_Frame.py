import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def read_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = lines[2:]  # 删除前两行数据
        values = [float(line.strip()) for line in lines]
        # 如果数据不足100行，用 NaN 填充
        if len(values) < 100:
            values.extend([float('NaN')] * (100 - len(values)))
    return values[:100]  # 只保留前100个数据

def read_folder(folder_path, num_files=20):
    data = pd.DataFrame()
    for i in range(num_files):
        file_name = f'{i+1}.txt'  # 更新文件名格式
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        values = read_values(file_path)
        data[file_name] = values
    return data

# 文件夹列表
folder_names = ['1', '2', 'o1', 'o2']
data_all = {}
current_directory = os.path.abspath('.')

print(f"Current directory: {current_directory}")

for folder in folder_names:
    folder_path = os.path.join(current_directory, folder)  # 更新文件夹路径
    print(f"Reading data from folder: {folder_path}")
    data_all[folder] = read_folder(folder_path)

# 计算统计特征
stats = {}
for folder in folder_names:
    mean_values = data_all[folder].mean(axis=1)/10000
    stats[folder] = {'mean': mean_values}

# 绘制图像
fig, axs = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)

colors = {'1': '#3596B5', '2': '#C493FF', 'o1': '#296073', 'o2': '#845EC2'}

# 左子图，展示1和o1文件夹
ax_left = axs[0]
x_1 = np.arange(len(stats['1']['mean']))
y_1 = stats['1']['mean']
spl_1 = make_interp_spline(x_1, y_1, k=3)
x_smooth_1 = np.linspace(x_1.min(), x_1.max(), 300)
y_smooth_1 = spl_1(x_smooth_1)

x_o1 = np.arange(len(stats['o1']['mean']))
y_o1 = stats['o1']['mean']
spl_o1 = make_interp_spline(x_o1, y_o1, k=3)
x_smooth_o1 = np.linspace(x_o1.min(), x_o1.max(), 300)
y_smooth_o1 = spl_o1(x_smooth_o1)

ax_left.plot(x_smooth_1, y_smooth_1, color='#3596B5', linewidth=3, label='LGMD1')
ax_left.plot(x_smooth_o1, y_smooth_o1, color='#296073', linewidth=3, label='oLGMD1')
ax_left.fill_between(x_smooth_1, y_smooth_1, y_smooth_o1, where=(y_smooth_1 > y_smooth_o1), color='#3498db', alpha=0.4)
ax_left.fill_between(x_smooth_1, y_smooth_1, y_smooth_o1, where=(y_smooth_1 <= y_smooth_o1), color='#3498db', alpha=0.4)

ax_left.set_xlabel("Frames in Collision Events", fontsize=25)  # 设置横坐标标签字体大小
ax_left.set_ylabel("Average Processing Time\n in 20 Collision Events / ms", fontsize=19)  # 设置纵坐标标签字体大小
ax_left.tick_params(axis='both', which='major', labelsize=16)  # 设置刻度字体大小
ax_left.set_ylim(0, None)  # 设置纵坐标从0开始
ax_left.legend(loc='lower left', fontsize=20)  # 调整图例字体大小
ax_left.set_title('Efficiency for oLGMD1', fontsize=25)  # 设置子图标题字体大小

# 右子图，展示2和o2文件夹
ax_right = axs[1]
x_2 = np.arange(len(stats['2']['mean']))
y_2 = stats['2']['mean']
spl_2 = make_interp_spline(x_2, y_2, k=3)
x_smooth_2 = np.linspace(x_2.min(), x_2.max(), 300)
y_smooth_2 = spl_2(x_smooth_2)

x_o2 = np.arange(len(stats['o2']['mean']))
y_o2 = stats['o2']['mean']
spl_o2 = make_interp_spline(x_o2, y_o2, k=3)
x_smooth_o2 = np.linspace(x_o2.min(), x_o2.max(), 300)
y_smooth_o2 = spl_o2(x_smooth_o2)

ax_right.plot(x_smooth_2, y_smooth_2, color='#C493FF', linewidth=3, label='LGMD2')
ax_right.plot(x_smooth_o2, y_smooth_o2, color='#845EC2', linewidth=3, label='oLGMD2')
ax_right.fill_between(x_smooth_2, y_smooth_2, y_smooth_o2, where=(y_smooth_2 > y_smooth_o2), color='#9b59b6', alpha=0.4)
ax_right.fill_between(x_smooth_2, y_smooth_2, y_smooth_o2, where=(y_smooth_2 <= y_smooth_o2), color='#9b59b6', alpha=0.4)

ax_right.set_xlabel("Frames in Collision Events", fontsize=25)  # 设置横坐标标签字体大小
ax_right.tick_params(axis='both', which='major', labelsize=16)  # 设置刻度字体大小
ax_right.legend(loc='lower left', fontsize=20)  # 调整图例字体大小
ax_right.set_title('Efficiency for oLGMD2', fontsize=25)  # 设置子图标题字体大小

ax_right.set_ylim(0, None)  # 设置纵坐标从0开始

plt.tight_layout()
plt.show()
