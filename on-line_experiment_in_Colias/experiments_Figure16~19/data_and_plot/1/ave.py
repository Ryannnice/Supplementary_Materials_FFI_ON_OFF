import os

def calculate_average(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 去除前两行，并将每行数据转换为浮点数
        values = [float(line.strip()) for line in lines[2:]]
        average = sum(values) / len(values)
    return average

# 获取当前文件夹路径
current_directory = os.path.dirname(os.path.abspath(__file__))
average_times = {}

for filename in os.listdir(current_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(current_directory, filename)
        average = calculate_average(file_path)
        average_times[filename] = average

# 输出每个文件的平均值
for filename, average in average_times.items():
#    print(f'{filename}: {average}')
    print(f'{average/10000}')
