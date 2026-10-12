import cv2
import os
from pathlib import Path

# 工程根目录（本脚本位于 Train 子目录中，视频源/帧输出等路径统一以此为基准）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def extract_frames_from_videos(input_folder, output_folder):
    """
    从输入文件夹中的所有视频中提取帧并保存为jpg图片到输出文件夹
    
    Args:
        input_folder (str): 包含视频文件的输入文件夹路径
        output_folder (str): 保存提取帧的输出文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # 检查是否为文件且扩展名是视频格式
        if os.path.isfile(file_path) and Path(filename).suffix.lower() in video_extensions:
            print(f"正在处理视频: {filename}")
            
            # 创建视频特定的输出子文件夹
            video_name = Path(filename).stem
            video_output_folder = os.path.join(output_folder, video_name)
            Path(video_output_folder).mkdir(parents=True, exist_ok=True)
            
            # 打开视频文件
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                print(f"无法打开视频文件: {filename}")
                continue
            
            frame_count = 0
            success = True
            
            # 逐帧读取视频
            while success:
                success, frame = cap.read()
                
                if success:
                    # 生成帧的文件名
                    frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(video_output_folder, frame_filename)
                    
                    # 保存帧为jpg图片
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1
                    
                    # 每处理100帧显示一次进度
                    if frame_count % 100 == 0:
                        print(f"已提取 {frame_count} 帧 from {filename}")
            
            # 释放视频捕获对象
            cap.release()
            print(f"完成处理 {filename}，共提取 {frame_count} 帧\n")

if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = "videos"  # 修改为你的视频文件夹路径
    output_folder = "images"  # 修改为你想保存帧的文件夹路径
    
    # 调用函数提取帧
    extract_frames_from_videos(input_folder, output_folder)