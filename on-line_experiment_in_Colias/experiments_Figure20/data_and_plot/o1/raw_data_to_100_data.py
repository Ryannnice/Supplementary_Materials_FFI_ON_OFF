import os

def process_files_in_folder(folder_path):
    # 列出文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # 删除前两行
        lines = lines[2:]
        
        # 处理每行：去除空行，并删除每行中的第二个数值
        processed_lines = []
        for idx, line in enumerate(lines):
            if (idx % 2 == 0) or (line.strip() != ""):
                parts = line.split()
                if len(parts) == 3:
                    new_line = f"{parts[0]} {parts[2]}\n"
                    processed_lines.append(new_line)
        
        # 写回文件
        with open(file_path, 'w') as file:
            file.writelines(processed_lines)

# 主函数，遍历 o1 文件夹中的所有子文件夹
def main():
    base_folder = 'ER_Plot/o1'
    sub_folders = ['2']
    # sub_folders = ['0', '0.25', '0.5', '0.75', '1', '1.25', '1.5', '1.75', '2', '2.25', '2.5', '2.75', '3']

    for sub_folder in sub_folders:
        folder_path = os.path.join(base_folder, sub_folder)
        if os.path.exists(folder_path):
            print(f"Processing folder: {folder_path}")
            process_files_in_folder(folder_path)
        else:
            print(f"Folder not found: {folder_path}")

if __name__ == "__main__":
    main()
