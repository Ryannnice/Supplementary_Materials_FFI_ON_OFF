import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 去除前两行，并将每行数据转换为浮点数
        values = [float(line.strip()) for line in lines[2:]]
    return values

# 获取当前文件夹路径
current_directory = os.path.dirname(os.path.abspath(__file__))
all_values = []

for filename in os.listdir(current_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(current_directory, filename)
        values = read_values(file_path)
        all_values.extend(values)

# 将数据转换为 DataFrame
data = pd.DataFrame({'time': all_values})

# 设置 seaborn 白色网格主题
sns.set_theme(style="whitegrid")

# 绘制密度图
p = sns.kdeplot(data=data, x="time", fill=True, common_norm=False, alpha=0.6, palette="viridis")

# 设置 x 轴范围
plt.xlim(min(all_values), max(all_values))

# 添加轴标签
plt.xlabel("Time")
plt.ylabel("Density")

# 显示图表
plt.show()
