import os
import pandas as pd
import numpy as np

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

def write_mean_values_to_file(mean_values, file_name):
    with open(file_name, 'w') as file:
        for value in mean_values:
            file.write(f"{value}\n")

# 文件夹列表
folder_names = ['1', '2', 'o1', 'o2']
data_all = {}
current_directory = os.path.abspath('.')

print(f"Current directory: {current_directory}")

for folder in folder_names:
    folder_path = os.path.join(current_directory, 'DATA_PLOT', folder)  # 更新文件夹路径
    print(f"Reading data from folder: {folder_path}")
    data_all[folder] = read_folder(folder_path)

# 计算统计特征
stats = {}
for folder in folder_names:
    mean_values = data_all[folder].mean(axis=1)
    stats[folder] = {'mean': mean_values}

# 输出到文本文件
output_files = {'1': 'FRAME_ave1.txt', '2': 'FRAME_ave2.txt', 'o1': 'FRAME_aveo1.txt', 'o2': 'FRAME_aveo2.txt'}
for folder in folder_names:
    output_file = output_files[folder]
    write_mean_values_to_file(stats[folder]['mean'], output_file)
    print(f"Mean values for folder {folder} written to {output_file}")
