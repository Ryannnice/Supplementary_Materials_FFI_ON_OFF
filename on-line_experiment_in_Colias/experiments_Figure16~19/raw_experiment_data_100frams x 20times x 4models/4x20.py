import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_values(file_path, num_lines=20):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        values = [float(line.strip()) for line in lines[:num_lines]]
    return values

# 获取当前文件夹路径
current_directory = os.path.dirname(os.path.abspath(__file__))

# 文件名列表
filenames = ['ave1.txt', 'ave2.txt', 'ave3.txt', 'ave4.txt']
data = pd.DataFrame()

# 读取每个文件的前20行数据并存储在 DataFrame 中
for filename in filenames:
    file_path = os.path.join(current_directory, filename)
    values = read_values(file_path)
    temp_df = pd.DataFrame({'time': values, 'file': filename})
    data = pd.concat([data, temp_df], ignore_index=True)

# 设置 seaborn 白色网格主题
sns.set_theme(style="whitegrid")

# 绘制密度图
palette = sns.color_palette("viridis", len(filenames))
for i, filename in enumerate(filenames):
    subset = data[data['file'] == filename]
    sns.kdeplot(data=subset, x="time", fill=True, common_norm=False, alpha=0.6, color=palette[i], label=filename)

# 添加轴标签和图例
plt.xlabel("Time/ms")
plt.ylabel("Density")
plt.legend(title='Dendity of running time in normal case')

# 显示图表
plt.show()
