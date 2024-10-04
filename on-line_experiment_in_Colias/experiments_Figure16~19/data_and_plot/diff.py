import os

def read_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        values = [float(line.strip()) for line in lines]
    return values

def write_values(values, file_path):
    with open(file_path, 'w') as file:
        for value in values:
            file.write(f"{value}\n")

# 文件路径
current_directory = os.path.abspath('.')
file1_path = os.path.join(current_directory, 'DATA_PLOT', 'FRAME_ave1.txt')
file2_path = os.path.join(current_directory, 'DATA_PLOT', 'FRAME_aveo1.txt')
output_file_path = os.path.join(current_directory, 'FRAME_diff_1.txt')

# 读取文件数据
values1 = read_values(file1_path)
values2 = read_values(file2_path)

# 计算差值
diff_values = [val1 - val2 for val1, val2 in zip(values1, values2)]

# 将差值写入输出文件
write_values(diff_values, output_file_path)

print(f"Differences written to {output_file_path}")
