import os
import random
import shutil
from pathlib import Path
import glob

# 工程根目录（本脚本位于 Train 子目录中，图片源/采样输出等路径统一以此为基准）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def sample_images_randomly(source_folder, target_folder, sampling_ratio=0.1):
    """
    按照一定比例随机采样目录及其子目录下的图片，并复制到目标文件夹
    
    Args:
        source_folder (str): 源图片文件夹路径
        target_folder (str): 目标文件夹路径
        sampling_ratio (float): 采样比例，默认为0.1(10%)
    """
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
    
    # 创建目标文件夹
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件路径
    all_images = []
    for extension in image_extensions:
        # 递归搜索所有子目录中的图片
        pattern = os.path.join(source_folder, '**', extension)
        all_images.extend(glob.glob(pattern, recursive=True))
    
    if not all_images:
        print("未找到任何图片文件")
        return
    
    # 计算需要采样的数量
    total_images = len(all_images)
    sample_count = max(1, int(total_images * sampling_ratio))  # 至少采样1张图片
    
    print(f"总共找到 {total_images} 张图片")
    print(f"按照 {sampling_ratio*100}% 的比例采样，将采样 {sample_count} 张图片")
    
    # 随机采样
    sampled_images = random.sample(all_images, sample_count)
    
    # 复制采样后的图片到目标文件夹
    for i, image_path in enumerate(sampled_images):
        # 获取文件名
        filename = os.path.basename(image_path)
        
        # 构造目标路径
        target_path = os.path.join(target_folder, filename)
        
        # 如果目标文件已存在，则添加序号避免重名
        counter = 1
        original_target_path = target_path
        while os.path.exists(target_path):
            name, ext = os.path.splitext(original_target_path)
            target_path = f"{name}_{counter}{ext}"
            counter += 1
        
        # 复制文件
        shutil.copy2(image_path, target_path)
        
        # 显示进度
        if (i + 1) % 10 == 0 or (i + 1) == sample_count:
            print(f"已复制 {i + 1}/{sample_count} 张图片")
    
    print(f"采样完成！共复制 {sample_count} 张图片到 {target_folder}")

def sample_images_with_subdirs(source_folder, target_folder, sampling_ratio=0.1):
    """
    按照一定比例随机采样目录及其子目录下的图片，并保持原有子目录结构
    
    Args:
        source_folder (str): 源图片文件夹路径
        target_folder (str): 目标文件夹路径
        sampling_ratio (float): 采样比例，默认为0.1(10%)
    """
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
    
    # 获取所有图片文件路径
    all_images = []
    for extension in image_extensions:
        # 递归搜索所有子目录中的图片
        pattern = os.path.join(source_folder, '**', extension)
        all_images.extend(glob.glob(pattern, recursive=True))
    
    if not all_images:
        print("未找到任何图片文件")
        return
    
    # 计算需要采样的数量
    total_images = len(all_images)
    sample_count = max(1, int(total_images * sampling_ratio))  # 至少采样1张图片
    
    print(f"总共找到 {total_images} 张图片")
    print(f"按照 {sampling_ratio*100}% 的比例采样，将采样 {sample_count} 张图片")
    
    # 随机采样
    sampled_images = random.sample(all_images, sample_count)
    
    # 复制采样后的图片到目标文件夹，保持子目录结构
    for i, image_path in enumerate(sampled_images):
        # 计算相对于源文件夹的相对路径
        relative_path = os.path.relpath(image_path, source_folder)
        
        # 构造目标路径
        target_path = os.path.join(target_folder, relative_path)
        
        # 创建目标目录（如果不存在）
        target_dir = os.path.dirname(target_path)
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        shutil.copy2(image_path, target_path)
        
        # 显示进度
        if (i + 1) % 10 == 0 or (i + 1) == sample_count:
            print(f"已复制 {i + 1}/{sample_count} 张图片")
    
    print(f"采样完成！共复制 {sample_count} 张图片到 {target_folder}，保持原有目录结构")

if __name__ == "__main__":
    # 设置源文件夹和目标文件夹路径
    source_folder = "images"      # 修改为你的源图片文件夹路径
    target_folder = "sampled_images"  # 修改为你想保存采样图片的文件夹路径
    
    # 设置采样比例（例如0.2表示采样20%的图片）
    sampling_ratio = 0.02  # 10%
    
    # 方式1：将所有采样图片复制到同一个文件夹中
    print("方式1：将所有采样图片复制到同一个文件夹中")
    sample_images_randomly(source_folder, target_folder, sampling_ratio)
    
    # 方式2：保持原有子目录结构
    # print("方式2：保持原有子目录结构")
    # sample_images_with_subdirs(source_folder, target_folder + "_with_structure", sampling_ratio)