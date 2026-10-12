import cv2
import os
from pathlib import Path
import glob

# 工程根目录（本脚本位于 Train 子目录中，视频源/采样输出等路径统一以此为基准）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def sample_videos_by_millisecond_interval(input_folder, output_folder, interval_ms=500):
    """
    按毫秒间隔采样视频帧
    
    Args:
        input_folder (str): 包含视频文件的输入文件夹路径
        output_folder (str): 输出文件夹路径
        interval_ms (int): 采样间隔（毫秒），默认500毫秒（0.5秒）
    """
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 支持的视频格式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    
    # 查找所有视频文件（包括子目录）
    video_files = []
    for extension in video_extensions:
        pattern = os.path.join(input_folder, '**', extension)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    if not video_files:
        print(f"在 {input_folder} 及其子目录中未找到任何视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"采样间隔: {interval_ms} 毫秒")
    
    total_extracted = 0
    
    # 处理每个视频文件
    for i, video_file in enumerate(video_files):
        try:
            print(f"\n处理视频 ({i+1}/{len(video_files)}): {os.path.basename(video_file)}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_file)
            
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_file}")
                continue
            
            # 获取视频帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps <= 0:
                print(f"视频文件帧率无效: {video_file}")
                cap.release()
                continue
            
            # 计算帧间隔（以帧为单位）
            frame_interval = max(1, int(fps * interval_ms / 1000.0))
            
            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                print(f"视频文件无有效帧: {video_file}")
                cap.release()
                continue
            
            # 获取视频文件名（不含扩展名）
            video_name = Path(video_file).stem
            
            extracted_count = 0
            
            # 按帧间隔采样
            frame_index = 0
            while frame_index < total_frames:
                # 设置视频位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                
                # 读取帧
                ret, frame = cap.read()
                
                if ret:
                    # 获取当前帧的时间戳（毫秒）
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    
                    # 生成输出文件名
                    output_filename = f"{video_name}_time_{int(timestamp_ms)}ms_frame_{frame_index:06d}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # 保存帧为jpg图片
                    cv2.imwrite(output_path, frame)
                    extracted_count += 1
                    
                    # 每处理10帧显示一次进度
                    if extracted_count % 10 == 0:
                        print(f"已从 {video_name} 提取 {extracted_count} 帧 (时间: {timestamp_ms/1000:.2f}秒)")
                
                # 移动到下一个采样点
                frame_index += frame_interval
            
            # 释放视频捕获对象
            cap.release()
            
            print(f"完成处理 {video_name}，共提取 {extracted_count} 帧")
            total_extracted += extracted_count
            
        except Exception as e:
            print(f"处理视频 {video_file} 时出错: {str(e)}")
    
    print(f"\n处理完成！总共从 {len(video_files)} 个视频中提取了 {total_extracted} 帧图片")
    print(f"图片保存在: {output_folder}")

def sample_videos_by_frame_count(input_folder, output_folder, num_frames=50):
    """
    按固定帧数等距采样视频帧
    
    Args:
        input_folder (str): 包含视频文件的输入文件夹路径
        output_folder (str): 输出文件夹路径
        num_frames (int): 每个视频需要采样的帧数
    """
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 支持的视频格式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    
    # 查找所有视频文件（包括子目录）
    video_files = []
    for extension in video_extensions:
        pattern = os.path.join(input_folder, '**', extension)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    if not video_files:
        print(f"在 {input_folder} 及其子目录中未找到任何视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"每个视频采样帧数: {num_frames}")
    
    total_extracted = 0
    
    # 处理每个视频文件
    for i, video_file in enumerate(video_files):
        try:
            print(f"\n处理视频 ({i+1}/{len(video_files)}): {os.path.basename(video_file)}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_file)
            
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_file}")
                continue
            
            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                print(f"视频文件无效或无帧: {video_file}")
                cap.release()
                continue
            
            # 计算采样间隔
            interval = max(1, total_frames // num_frames)
            
            # 获取视频文件名（不含扩展名）
            video_name = Path(video_file).stem
            
            extracted_count = 0
            
            # 等距采样帧
            for frame_idx in range(0, min(total_frames, num_frames * interval), interval):
                # 设置视频位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # 读取帧
                ret, frame = cap.read()
                
                if ret:
                    # 获取当前帧的时间戳（毫秒）
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    
                    # 生成输出文件名
                    output_filename = f"{video_name}_time_{int(timestamp_ms)}ms_frame_{frame_idx:06d}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # 保存帧为jpg图片
                    cv2.imwrite(output_path, frame)
                    extracted_count += 1
                    
                    # 每处理10帧显示一次进度
                    if extracted_count % 10 == 0:
                        print(f"已从 {video_name} 提取 {extracted_count} 帧 (时间: {timestamp_ms/1000:.2f}秒)")
            
            # 释放视频捕获对象
            cap.release()
            
            print(f"完成处理 {video_name}，共提取 {extracted_count} 帧")
            total_extracted += extracted_count
            
        except Exception as e:
            print(f"处理视频 {video_file} 时出错: {str(e)}")
    
    print(f"\n处理完成！总共从 {len(video_files)} 个视频中提取了 {total_extracted} 帧图片")
    print(f"图片保存在: {output_folder}")

if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = "videos"           # 修改为你的视频文件夹路径
    output_folder = "sampled_images"  # 修改为你想保存帧的文件夹路径
    
    # 按毫秒间隔采样（例如每500毫秒采样一帧）
    print("按毫秒间隔采样视频帧")
    sample_videos_by_millisecond_interval(input_folder, output_folder, interval_ms=500)
    
    # 如果需要按固定帧数采样，可以使用下面的方法
    # print("\n按固定帧数采样视频帧")
    # sample_videos_by_frame_count(input_folder, output_folder, num_frames=50)