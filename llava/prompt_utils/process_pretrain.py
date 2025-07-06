# import debugpy; debugpy.connect(('10.140.12.33', 5678))
from preprocessing import add_axis_save_image
import os
from PIL import Image
import threading
from tqdm import tqdm
num_threads = 16  # 设置线程数，这里设置为50或者你的线程池大小

def run(all_files):
    # 创建线程列表
    threads = []

    # 分配任务到线程
    for i in range(num_threads):
        file_slice = all_files[i::num_threads]  # 将文件列表分割成多个部分
        thread = threading.Thread(target=process_files, args=(file_slice,))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print("多线程所有文件处理完成。")

def process_files(files):
    for file_path in tqdm(files):
        process_file(file_path)  # 处理单个文件


def process_file(file_path):
    # 这里添加处理文件的代码
    image = Image.open(file_path).convert('RGB')
    
    image_folder_axis = file_path.replace("data", "data_axis")
    os.makedirs(os.path.dirname(image_folder_axis), exist_ok=True)
    image = add_axis_save_image(image, image_folder_axis)


if __name__ == "__main__":
    root_dir = '/mnt/petrelfs/sunqiao/tangwei/codes/LLaVA/playground/data/LLaVA-Pretrain/images'

    # 用于存储所有非隐藏目录中文件路径的列表
    all_files = []

    # 遍历目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 先对dirnames进行排序，这样隐藏目录（通常在最后）会放在一起，便于跳过
        dirnames.sort()
        
        # 忽略隐藏目录，隐藏目录名以点'.'开头
        non_hidden_dirs = [d for d in dirnames if not d.startswith('.')]
        
        # 更新dirnames为非隐藏目录列表
        dirnames[:] = non_hidden_dirs
        
        # 遍历文件
        for filename in filenames:
            # 跳过隐藏文件
            if not filename.startswith('.'):
                # 连接dirpath和filename以获取完整文件路径
                file_path = os.path.join(dirpath, filename)
                all_files.append(file_path)
    # print()
    # # 打印所有非隐藏文件的路径
    run(all_files)
    print("all done!")
    # for file_path in tqdm(all_files):
    #     # print(file_path)
    #     image = Image.open(file_path).convert('RGB')
        
    #     image_folder_axis = file_path.replace("data", "data_axis")
    #     os.makedirs(os.path.dirname(image_folder_axis), exist_ok=True)
    #     add_axis_save_image(image, image_folder_axis)