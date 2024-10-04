import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

def calculate_a(file_path):
    print(f"Processing file: {file_path}")
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 找到文件中的所有数值并转换为浮点数
    values = [float(line.strip()) for line in lines]
    
    # 计算文件中所有行值的最大值
    max_value = max(values)
    
    # 计算 a 的值
    values_sum = sum(values[:100])  # 只计算前 100 行的总和
    a_value = (100 * max_value - values_sum) / (100 * max_value)
    
    # 保留小数点后两位
    a_value_rounded = round(a_value, 2)
    
    return a_value_rounded

def generate_er_file(folder_path):
    output_file_path_1 = os.path.join(folder_path, 'ER.txt')
    output_file_path_2 = os.path.join(folder_path, 'ER_CORRECT.txt')

    # 如果文件存在，删除它
    if os.path.exists(output_file_path_1):
        os.remove(output_file_path_1)
    if os.path.exists(output_file_path_2):
        os.remove(output_file_path_2)
    
    # 获取文件夹中的所有 .txt 文件，排除 correct_rate.txt
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt') and f != 'correct_rate.txt']
    
    data = []
    
    # 计算每个文件的 a 值并存储到 data 列表中
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        file_base_name = os.path.splitext(file_name)[0]
        
        a_value = calculate_a(file_path)
        
        data.append((file_base_name, a_value))
    
    # 对 data 列表按每行的第一个数据（文件名）从小到大排序
    data.sort(key=lambda x: x[0])
    
    # 将排序后的结果写入到 ER.txt 文件中，保留两位小数
    with open(output_file_path_1, 'w') as output_file:
        for item in data:
            output_file.write(f"{item[0]} {item[1]:.2f}\n")

def merge_files(er_file, correct_rate_file, output_file):
    with open(er_file, 'r') as er_f, open(correct_rate_file, 'r') as cr_f, open(output_file, 'w') as out_f:
        # 逐行读取ER.txt和correct_rate.txt文件
        for er_line, cr_line in zip(er_f, cr_f):
            # 移除每行末尾的换行符并拼接
            merged_line = er_line.strip() + ' ' + cr_line.strip() + '\n'
            # 写入到输出文件
            out_f.write(merged_line)

def process_files_in_folder(folder_path):
    # 列出文件夹中的所有.txt文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    all_data = []
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # 处理每行：去除空行，并提取每行中的第二个数值
        processed_lines = []
        for idx, line in enumerate(lines):
            if line.strip():
                parts = line.split()
                if len(parts) == 2:
                    new_line = [float(parts[1])]
                    processed_lines.append(new_line)
        
        all_data.append(processed_lines)
    
    # 转置数据以便逐行处理
    all_data_transposed = list(map(list, zip(*all_data)))
    
    # 计算每行的平均值
    averages = []
    for row in all_data_transposed:
        row_array = np.array(row)
        mean_value = np.mean(row_array, axis=0)
        averages.append(mean_value)
    
    return averages

def plot_data(file_path):
    x_values = []
    left_y_values = []
    right_y_values = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                x_values.append(float(parts[0]))
                left_y_values.append(float(parts[1]))
                right_y_values.append(float(parts[2]))
    


    # 创建图形和坐标轴
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制左侧纵坐标轴的数据
    ax1.plot(x_values, left_y_values, marker='d', linestyle='-', linewidth=3, color='#4A62FF', label='Efficiency Rate', markersize=5)
    ax1.set_xlabel('Activation Threshold in Colias',fontsize='20')
    ax1.set_ylabel('Average Efficiency Ratio',fontsize='20')
    
    # 创建右侧纵坐标轴并绘制数据
    ax2 = ax1.twinx()
    ax2.plot(x_values, right_y_values, marker='d', linestyle='-', linewidth=3, color='#FF0000', label='Correct Rate', markersize=5)
    ax2.set_ylabel('Correct Rate',fontsize='20')
    
    # 设置左侧和右侧纵坐标轴刻度的颜色
    ax1.tick_params(axis='y', colors='#4A62FF', labelsize='13')  # 左
    ax2.tick_params(axis='y', colors='#FF0000', labelsize='13')   # 右
    
    # 添加图例
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    
    # 格式化左侧和右侧纵坐标轴的刻度为百分比
    def percentage_formatter(x, pos):
        return f'{x:.0%}'  # {:.1%} 显示一位小数
    
    ax1.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax2.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    # 横坐标步长
    ax1.xaxis.set_major_locator(MultipleLocator(0.25))



    '''
    # 标记  两条线相距很小的部分 的 [右侧]
    threshold = 0.06  # 设置一个阈值，小于此值的区域将被标记
    for i in range(len(x_values)):
        if i < len(x_values) - 1:
            if abs(2.778*left_y_values[i] - right_y_values[i]) < threshold:
                ax1.axvspan(x_values[max(0, i-1)], x_values[min(len(x_values)-1, i+1)], color='#006100', alpha=0.2)  # alpha 设置透明度
    '''

    '''
    # 标记  两条线相距很小的部分 的 [两侧]
    threshold = 0.06  # 设置一个阈值，小于此值的区域将被标记
    half_x_step = (x_values[1] - x_values[0]) / 2  # 计算横坐标步长的一半
    
    for i in range(len(x_values)):
        if i < len(x_values) - 1:
            if abs(2.778 * left_y_values[i] - right_y_values[i]) < threshold:
                left_match = abs(2.778 * left_y_values[i-1] - right_y_values[i-1]) < threshold if i > 0 else False
                right_match = abs(2.778 * left_y_values[i+1] - right_y_values[i+1]) < threshold if i < len(x_values) - 1 else False

                if left_match and right_match:
                    ax1.axvspan(x_values[i] - half_x_step, x_values[i] + half_x_step, color='#006100', alpha=0.4)  # alpha 设置透明度为原来的两倍
                else:
                    ax1.axvspan(x_values[i] - half_x_step, x_values[i] + half_x_step, color='#006100', alpha=0.2)  # alpha 设置透明度
    '''


    '''
    #        [竖线对齐刻度]
    threshold = 0.06  # 设置一个阈值，小于此值的区域将被标记
    for i in range(len(x_values)):
        if abs(2.778 * left_y_values[i] - right_y_values[i]) < threshold:
            if i > 0:
                ax1.axvspan(x_values[i-1], x_values[i], color='#006100', alpha=0.2)
            if i < len(x_values) - 1:
                ax1.axvspan(x_values[i], x_values[i+1], color='#006100', alpha=0.2)
    '''


    
    # 在图中标记左侧坐标大于0.2且右侧坐标大于0.95的部分，完整区域，无竖线分隔
    half_x_step = (x_values[1] - x_values[0]) / 2  # 计算横坐标步长的一半

    i = 0
    while i < len(x_values) - 1:
        if left_y_values[i] > 0.2 and right_y_values[i] > 0.9:
            start_idx = i
            while i < len(x_values) - 1 and left_y_values[i] > 0.2 and right_y_values[i] > 0.9:
                i += 1
            end_idx = i

            # 标记完整区域
            ax1.axvspan(x_values[start_idx] - half_x_step, x_values[end_idx] - 0.25  + half_x_step, color='#006100', alpha=0.3)
        else:
            i += 1
    



    
    # 设置横坐标刻度的位置和标签
    
    ticks = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    labels = ['0', '0.25', '0.50', '0.75', '1.00', '1.25', '1.50', '1.75', '2.00', '2.25', '2.50', '2.75', '3.00']
    
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels, fontsize=13)
    

    # 设置刻度为 '2.00' 的标签颜色为绿色
    for label in ax1.get_xticklabels():
        if label.get_text() == '1.00' or label.get_text() == '1.25' or label.get_text() == '1.50' :
            label.set_color('green')
        elif label.get_text() == '0.75' or label.get_text() == '1.75' :
            label.set_color('green')


    
    # 显示图形
    plt.show()


def main():
    base_folder = 'ER_Plot/o1'
    sub_folders = ['0', '0.25', '0.5', '0.75', '1', '1.25', '1.5', '1.75', '2', '2.25', '2.5', '2.75', '3']
    
    for sub_folder in sub_folders:
        folder_path = os.path.join(base_folder, sub_folder)
        if os.path.exists(folder_path):
            print(f"Processing folder: {folder_path}")
            averages = process_files_in_folder(folder_path)
            
            # 将平均值写入输出文件
            output_file_path = os.path.join(base_folder, f"{sub_folder}.txt")
            with open(output_file_path, 'w') as output_file:
                for avg in averages:
                    output_file.write(f"{avg[0]:.2f}\n")
        else:
            print(f"Folder not found: {folder_path}")

    # 生成 ER.txt 文件
    generate_er_file(base_folder)
    
    er_file_path = os.path.join(base_folder, 'ER.txt')
    correct_rate_file_path = os.path.join(base_folder, 'correct_rate.txt')
    output_file_path = os.path.join(base_folder, 'ER_CORRECT.txt')
    
    # 合并文件
    merge_files(er_file_path, correct_rate_file_path, output_file_path)
    
    print(f"Merged data saved to {output_file_path}")

    # 绘制数据
    plot_data(output_file_path)

if __name__ == "__main__":
    main()
