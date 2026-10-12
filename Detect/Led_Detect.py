##############################   相机实时采集/本地视频文件 进行白灯检测跟踪    #############################
##############################   能够显示检测帧率 绘制坐标曲线 保存跟踪视频    #############################
#############################   通过阈值检测白灯并实时通过串口发送目标位置   #############################
# YYZ 2025-04-01

# ---- 子目录模块路径引导 ----
# 相机 SDK 封装、串口通信、轨迹预测模块分别位于 Detect 的 Camera / Communication / Predict 子文件夹中，
# 这里先把它们加入模块搜索路径，后续按模块名导入才能生效。
# 注意：重命名这些子文件夹时，需同步修改此处与 .vscode/settings.json 的 extraPaths。
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _sub in ("Camera", "Communication", "Predict"):
    _sub_dir = os.path.join(_SCRIPT_DIR, _sub)
    if _sub_dir not in sys.path:
        sys.path.insert(0, _sub_dir)

from SCI_Write import sci_write                  # 串口发送函数
from Initial_SCi import initialize_serial_port   # 串口初始化函数

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import threading
import queue
import atexit

# 如果需要使用相机 SDK 的采集，请使用 CCD_Track 中的 CameraCapture（已实现 get_frame）
try:
    from CCD_Track import CameraCapture
except Exception:
    CameraCapture = None

# 工程根目录（本脚本位于 Detect 子目录中，视频/结果等路径统一以此为基准）
# 切换到该目录后，脚本内所有相对路径的行为与之前"在工程根目录下运行"完全一致
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# 全局统计，用于在程序退出时打印最终平均 FPS
_LAST_FRAME_COUNT = 0
_LAST_PROCESSING_TIMES = []

def _print_exit_stats():
    try: 
        if _LAST_PROCESSING_TIMES:
            total_processing_time = sum(_LAST_PROCESSING_TIMES)
            avg_processing_time = total_processing_time / len(_LAST_PROCESSING_TIMES)
            overall_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            print(f"\n最终统计 - 已处理帧数：{_LAST_FRAME_COUNT}, 平均每帧：{avg_processing_time*1000:.2f}ms, 平均帧率：{overall_fps:.2f} FPS")
        else:
            print("\n最终统计：未处理任何帧，无法计算平均帧率")
    except Exception as e:
        print(f"打印最终统计时出错：{e}")

atexit.register(_print_exit_stats)


# ============================================================
#        串口通信开关（单独测试识别定位功能时置 False）
# ============================================================
# True : 打开 COM6 并按协议发送目标坐标 (u, v)
# False: 完全不使用串口（不打开、不发送），只做白灯识别定位 / 显示 / 画坐标曲线
ENABLE_SERIAL = False  #False


# ============================================================
#        输出模式设置（显示 / 保存视频 / 坐标曲线 / 串口发送）
# ============================================================
# 相机出厂分辨率
CAM_FULL_WIDTH  = 640
CAM_FULL_HEIGHT = 480

# 输出模式选择：'640x480' | '256x256' | '128x128'
# 显示画面、保存视频、坐标曲线、串口发送的 (u, v) 全部使用该模式的尺寸与坐标范围：
#   '640x480' -> u: 0~639, v: 0~479, 图像中心 (320, 240)
#   '256x256' -> u: 0~255, v: 0~255, 图像中心 (128, 128)
#   '128x128' -> u: 0~127, v: 0~127, 图像中心 ( 64,  64)
# 只需修改这一行即可切换工作范围
OUTPUT_MODE = '640x480'

_MODE_SIZES = {
    '640x480': (CAM_FULL_WIDTH, CAM_FULL_HEIGHT),
    '256x256': (256, 256),
    '128x128': (128, 128),
}

if OUTPUT_MODE not in _MODE_SIZES:
    raise ValueError(f"不支持的 OUTPUT_MODE={OUTPUT_MODE}，可选：{list(_MODE_SIZES.keys())}")

# 输出图像尺寸（显示 / 保存 / 坐标基准）
OUT_WIDTH, OUT_HEIGHT = _MODE_SIZES[OUTPUT_MODE]

# 输出图像中心点坐标（上/上位机脱靶量参考原点）
CENTER_U = OUT_WIDTH  // 2
CENTER_V = OUT_HEIGHT // 2

# 中心输出区域左上角在原始整帧中的偏移（640x480 模式为 0,0）
ROI_LEFT = (CAM_FULL_WIDTH  - OUT_WIDTH)  // 2
ROI_TOP  = (CAM_FULL_HEIGHT - OUT_HEIGHT) // 2

# ---------- UI 自适应（随输出模式自动缩放，避免小图遮挡目标） ----------
UI_SCALE    = OUT_WIDTH / float(CAM_FULL_WIDTH)   # 640->1.0, 256->0.4, 128->0.2
FONT_SCALE  = max(0.4, 0.5 * UI_SCALE)            # 检测标注字号
INFO_SCALE  = max(0.4, 1.0 * UI_SCALE)            # 帧号 / FPS 字号
LINE_THICK  = max(1, int(round(2 * UI_SCALE)))    # 线宽
DOT_RADIUS  = max(2, int(round(5 * UI_SCALE)))    # 质心圆半径
SHORT_LABEL = UI_SCALE < 0.5                      # 小图使用短标签，避免文字超出画面

# ---------- 白灯检测面积阈值（像素） ----------
# 中心裁剪不改变像素尺度，灯斑面积不变，故 min_area 保持基准值；
# max_area 上限自动收敛到输出图像的总像素数。
MIN_AREA = 50
MAX_AREA = min(50000, OUT_WIDTH * OUT_HEIGHT)


def crop_to_output(frame):
    """
    从整帧图像中截取中心 OUT_WIDTH x OUT_HEIGHT 区域。
    若图像尺寸已与输出尺寸一致（如 640x480 模式），则原样返回。
    """
    h, w = frame.shape[:2]
    if w == OUT_WIDTH and h == OUT_HEIGHT:
        return frame
    left = max(0, (w - OUT_WIDTH) // 2)
    top  = max(0, (h - OUT_HEIGHT) // 2)
    return frame[top:top + OUT_HEIGHT, left:left + OUT_WIDTH]


class WhiteLightDetector:
    """
    白灯阈值检测器
    通过亮度/颜色阈值分割检测白灯目标，并提取质心坐标
    """
    
    def __init__(self, brightness_threshold=200, min_area=50, max_area=50000, debug=False):
        """
        初始化白灯检测器
        
        Args:
            brightness_threshold (int): 亮度阈值，高于此值的像素视为白灯候选 (0-255)
            min_area (int): 最小目标面积（像素），过滤噪声
            max_area (int): 最大目标面积（像素），过滤过大区域
            debug (bool): 是否启用调试模式
        """
        self.brightness_threshold = brightness_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.debug = debug
    
    def detect(self, image):
        """
        检测白灯目标，返回边界框和质心坐标
        
        Args:
            image: 输入图像 (BGR 格式)
            
        Returns:
            detections: 列表，每个元素为 (x1, y1, x2, y2, confidence, class_id, center_u, center_v)
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 阈值分割 - 提取高亮度区域
            _, binary = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)
            
            # 形态学操作 - 去除噪声
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                
                # 过滤面积过小或过大的轮廓
                if area < self.min_area or area > self.max_area:
                    continue
                
                # 计算外接矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算质心（使用图像矩）
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_u = int(M["m10"] / M["m00"])
                    center_v = int(M["m01"] / M["m00"])
                else:
                    # 如果矩为 0，使用矩形中心
                    center_u = x + w // 2
                    center_v = y + h // 2
                
                # 置信度可以用面积归一化表示
                confidence = min(1.0, area / 1000)  # 简单归一化
                
                # 添加到检测结果 (格式与原有代码兼容)
                # (x1, y1, x2, y2, confidence, class_id, center_u, center_v)
                detections.append((x, y, x + w, y + h, confidence, 0, center_u, center_v))
            
            return detections
            
        except Exception as e:
            if self.debug:
                print(f"检测错误：{e}")
                import traceback
                traceback.print_exc()
            return []
    
    def draw_detections(self, image, detections, class_names=None, frame_info=None):
        """
        在图像上绘制检测结果和坐标信息
        """
        image_copy = image.copy()
        
        for i, (x1, y1, x2, y2, confidence, class_id, center_u, center_v) in enumerate(detections):
            # if self.debug:
            print(f"  检测 {i+1}: 置信度={confidence:.3f}, 位置=({x1},{y1},{x2},{y2}), 质心=({center_u},{center_v})")

            # 绘制边界框
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICK)
            
            # 绘制质心点
            cv2.circle(image_copy, (center_u, center_v), DOT_RADIUS, (0, 0, 255), -1)
            
            # 准备标签文本（小图使用短标签）
            label = f"WL {confidence:.2f}" if SHORT_LABEL else f"White Light: {confidence:.2f}"
            
            # 坐标信息文本
            coord_text = f"({center_u}, {center_v})"
            
            # 绘制标签背景（宽高按实际文字自适应，保证两行文字都不溢出）
            (label_w, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
            )
            (coord_w, _), _ = cv2.getTextSize(
                coord_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
            )
            text_width = max(label_w, coord_w)
            line_h = text_height + baseline + 2
            cv2.rectangle(
                image_copy,
                (x1, y1 - 2 * line_h),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # 绘制标签文字（上行）
            cv2.putText(
                image_copy,
                label,
                (x1, y1 - line_h - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 0, 0),
                1
            )
            
            # 绘制坐标文字（下行）
            cv2.putText(
                image_copy,
                coord_text,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 0, 0),
                1
            )
        
        # 在图像左上角显示帧信息（字号/位置随输出模式自适应）
        if frame_info:
            if SHORT_LABEL:
                info_text = f"F:{frame_info['frame_count']} D:{len(detections)}"
            else:
                info_text = f"Frame: {frame_info['frame_count']}, Detections: {len(detections)}"
            cv2.putText(image_copy, info_text, (3, max(10, int(round(30 * UI_SCALE)))),
                        cv2.FONT_HERSHEY_SIMPLEX, INFO_SCALE, (255, 255, 255), LINE_THICK)
        
        return image_copy, detections

class CoordinateTracker:
    """
    坐标跟踪器，用于记录和绘制坐标轨迹
    """
    def __init__(self, image_width=640, image_height=480):
        self.image_width = image_width
        self.image_height = image_height
        self.u_coordinates = []  # 存储 u 坐标历史
        self.v_coordinates = []  # 存储 v 坐标历史
        self.frame_numbers = []  # 存储帧号
        self.detection_status = []  # 存储检测状态 (1=检测到，0=未检测到)
        
    def add_coordinates(self, frame_count, detections):
        """
        添加当前帧的坐标信息
        检测到目标记录实际坐标，未检测到记录 (0,0)
        """
        if detections:
            # 检测到目标，取第一个检测目标的坐标
            center_u = detections[0][6]  # 质心 u 坐标
            center_v = detections[0][7]  # 质心 v 坐标
            self.detection_status.append(1)
        else:
            # 未检测到目标，记录 (0,0)
            center_u = 0
            center_v = 0
            self.detection_status.append(0)
        
        self.u_coordinates.append(center_u)
        self.v_coordinates.append(center_v)
        self.frame_numbers.append(frame_count)
    
    def plot_coordinates(self, save_path=None):
        """
        绘制坐标曲线图
        """
        if not self.u_coordinates:
            print("没有坐标数据可绘制")
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制 u 坐标曲线
        ax1.plot(self.frame_numbers, self.u_coordinates, 'b-', linewidth=2, label='u coordinate')
        # 标记未检测到的帧
        undetected_frames = [f for f, s in zip(self.frame_numbers, self.detection_status) if s == 0]
        if undetected_frames:
            ax1.scatter(undetected_frames, [0]*len(undetected_frames), c='red', s=10, label='undetected')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('u Coordinate (pixels)')
        ax1.set_title('Target u Coordinate Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, self.image_width)
        
        # 绘制 v 坐标曲线
        ax2.plot(self.frame_numbers, self.v_coordinates, 'r-', linewidth=2, label='v coordinate')
        if undetected_frames:
            ax2.scatter(undetected_frames, [0]*len(undetected_frames), c='red', s=10, label='undetected')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('v Coordinate (pixels)')
        ax2.set_title('Target v Coordinate Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, self.image_height)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"坐标曲线图已保存至：{save_path}")
        
        plt.show()
    
    def save_coordinates_to_file(self, file_path):
        """
        将坐标数据保存到文件
        """
        with open(file_path, 'w') as f:
            f.write("Frame,u_coordinate,v_coordinate,detection_status\n")
            for frame, u, v, status in zip(self.frame_numbers, self.u_coordinates, self.v_coordinates, self.detection_status):
                f.write(f"{frame},{u},{v},{status}\n")
        print(f"坐标数据已保存至：{file_path}")
        
        plt.show()
    
    def save_coordinates_to_file(self, file_path):
        """
        将坐标数据保存到文件
        """
        with open(file_path, 'w') as f:
            f.write("Frame,u_coordinate,v_coordinate\n")
            for frame, u, v in zip(self.frame_numbers, self.u_coordinates, self.v_coordinates):
                f.write(f"{frame},{u},{v}\n")
        print(f"坐标数据已保存至：{file_path}")


def process_video(input_source, output_path=None, brightness_threshold=100, debug=False, max_frames=None):
    """
    处理视频文件或相机实时流并进行白灯检测，增加坐标跟踪功能。

    Args:
        input_source: 视频文件路径或摄像头索引（int）或字符串 'camera'（等价于索引 0）。
        output_path: 可选的输出视频路径。
        brightness_threshold: 亮度阈值 (0-255)。
        debug: 是否打印调试信息。
        max_frames: 最大处理帧数。
    """
    # 支持三种输入来源：
    #  1) 整数 或 'camera' -> 使用 OpenCV 的 VideoCapture
    #  2) 字符串 'sdk' 或 'camera_sdk' -> 使用厂商 SDK（CCD_Track.CameraCapture）
    #  3) 其他字符串 -> 视作视频文件路径
    
    # ! 串口初始化函数调用（ENABLE_SERIAL=False 时跳过，便于单独测试识别定位）
    serial_port = None
    if ENABLE_SERIAL:
        serial_port = initialize_serial_port(port='COM6', baudrate=115200)
    
    # 串口状态检查
    if ENABLE_SERIAL and not serial_port:
        print("✗ 串口初始化失败，请检查 COM6 是否可用（或把 ENABLE_SERIAL 改为 False 单独测试识别定位）")
        return
    if ENABLE_SERIAL and not serial_port.is_open:
        print("✗ 串口 COM6 未成功打开")
        return
    
    if ENABLE_SERIAL:
        print(f"✓ 串口 COM6 初始化成功，波特率={serial_port.baudrate}")
    else:
        print("! 串口通信已关闭（ENABLE_SERIAL = False），仅进行白灯识别定位与显示")

    is_sdk = False
    if isinstance(input_source, str) and input_source.lower() in ('sdk', 'camera_sdk'):
        if CameraCapture is None:
            print("错误：未能导入 SDK CameraCapture 类，请确保 CCD_Track.py 可用。")
            return
        # 使用 SDK 采集
        cam = CameraCapture()
        ret = cam.enum_devices()
        if ret != 0:
            print("错误：枚举 SDK 设备失败")
            return
        ret = cam.open_first_device()
        if ret != 0:
            print("错误：打开 SDK 设备失败")
            return
        cam.set_continuous_mode()
        try:
            ret = cam.obj_cam_operation.obj_cam.MV_CC_StartGrabbing()
        except Exception as e:
            print(f"错误：无法调用 SDK StartGrabbing: {e}")
            return
        if ret != 0:
            print(f"错误：SDK StartGrabbing 返回错误：{hex(ret) if isinstance(ret,int) else ret}")
            return
        time.sleep(0.1)
        cam.is_grabbing = True
        is_sdk = True
        is_camera = True
        cap = cam
    # 统一处理非 SDK 的输入源
    is_camera = False
    if is_sdk:
        pass
    elif isinstance(input_source, int) or (isinstance(input_source, str) and input_source.lower() == 'camera'):
        cam_index = int(input_source) if isinstance(input_source, int) else 0
        is_camera = True
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 (index={cam_index})")
            return
    else:
        # 当作文件路径处理
        video_path = str(input_source)
        try:
            video_path_expanded = str(Path(video_path).expanduser())
        except Exception:
            video_path_expanded = video_path

        abs_path = os.path.abspath(video_path_expanded)
        exists = os.path.exists(abs_path)
        if not exists:
            print(f"错误：视频文件不存在 {video_path}")
            return
        cap = cv2.VideoCapture(abs_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {abs_path}")
            return
    
    try:
        # 初始化白灯检测器
        detector = WhiteLightDetector(brightness_threshold=brightness_threshold,
                                      min_area=MIN_AREA, max_area=MAX_AREA, debug=debug)
    except Exception as e:
        print(f"初始化检测器失败：{e}")
        return
    
    # 初始化坐标跟踪器（曲线图坐标范围随 OUTPUT_MODE 变化）
    coord_tracker = CoordinateTracker(image_width=OUT_WIDTH, image_height=OUT_HEIGHT)

    # 获取视频/摄像头信息
    fps = 0
    width = 0
    height = 0
    total_frames = -1
    if is_sdk:
        first_frame = cap.get_frame()
        if first_frame is None:
            print("错误：无法从 SDK 获取第一帧")
            try:
                cap.obj_cam_operation.obj_cam.MV_CC_StopGrabbing()
                cap.close_device()
            except Exception:
                pass
            return
        first_frame = crop_to_output(first_frame)   # ← 按 OUTPUT_MODE 截取中心区域
        height, width = first_frame.shape[:2]       # ← 自动变为输出尺寸
        total_frames = -1
        sdk_frame_ok = 0
        sdk_frame_fail = 0
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            fps = int(fps) if fps > 0 else 0
        except:
            fps = 0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 输出尺寸由 OUTPUT_MODE 决定（与 crop_to_output 保持一致）
        width  = min(OUT_WIDTH,  src_w)
        height = min(OUT_HEIGHT, src_h)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else -1

    print("输入源信息:")
    print(f"  分辨率：{width}x{height}")
    if is_sdk:
        print("  源：相机 SDK 实时流")
    elif is_camera:
        print(f"  源：摄像头 (OpenCV), FPS 属性：{fps}")
    else:
        print(f"  源：视频文件，FPS: {fps}, 总帧数：{total_frames}")
    
    # 初始化视频写入器（如果指定了输出路径）
    video_writer = None
    if output_path:
        writer_fps = fps if (isinstance(fps, (int, float)) and fps > 0) else 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, writer_fps, (width, height))
        print(f"已初始化 VideoWriter: path={output_path}, writer_fps={writer_fps}, size=({width},{height})")
    
    frame_count = 0
    total_detections = 0
    frame_intervals = []  # ← 修改：使用帧间隔时间计算实际帧率
    print("开始处理视频 (线程化推理模式)...")

    # 线程与队列：采集线程 -> 推理线程 -> 主线程显示
    frame_queue = queue.Queue(maxsize=4)
    result_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    def read_frame_loop():
        nonlocal sdk_frame_ok, sdk_frame_fail, frame_count
        if is_sdk and 'first_frame' in locals() and first_frame is not None:
            try:
                frame_queue.put(first_frame, timeout=0.5)
            except Exception:
                pass

        while not stop_event.is_set():
            try:
                if isinstance(max_frames, int) and max_frames > 0 and frame_count >= max_frames:
                    if debug:
                        print(f"read_frame_loop: 达到 max_frames={max_frames}, 停止读取")
                    stop_event.set()
                    break

                if is_sdk:
                    frame = cap.get_frame()
                    ok = frame is not None
                    if ok:
                        sdk_frame_ok += 1
                        frame = crop_to_output(frame)   # ← 按 OUTPUT_MODE 截取中心区域
                    else:
                        sdk_frame_fail += 1
                        time.sleep(0.005)
                        continue
                else:
                    ret, frame = cap.read()
                    ok = ret and (frame is not None)
                    if not ok:
                        stop_event.set()
                        break
                    frame = crop_to_output(frame)       # ← 按 OUTPUT_MODE 截取中心区域

                try:
                    frame_queue.put(frame, timeout=0.2)
                except queue.Full:
                    pass
            except Exception:
                time.sleep(0.01)

    def infer_loop():
        nonlocal frame_count, total_detections
        global _LAST_FRAME_COUNT, _LAST_PROCESSING_TIMES
        
        last_frame_time = time.time()  # ← 记录上一帧时间
        
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_count += 1
            if isinstance(max_frames, int) and max_frames > 0 and frame_count >= max_frames:
                if debug:
                    print(f"infer_loop: 达到最大帧数 {max_frames}，停止采集")
                stop_event.set()
                break
            
            # ← 修改：计算帧间隔时间（实际帧率）
            current_time = time.time()
            frame_interval = current_time - last_frame_time
            last_frame_time = current_time
            if frame_interval > 0:
                frame_intervals.append(frame_interval)
            
            start_time = time.time()
            try:
                detections = detector.detect(frame)
            except Exception as e:
                if debug:
                    print(f"推理错误：{e}")
                detections = []

            total_detections += len(detections)
            end_time = time.time()
            processing_time = end_time - start_time
            
            try:
                _LAST_FRAME_COUNT = frame_count
                _LAST_PROCESSING_TIMES = frame_intervals.copy()  # ← 使用帧间隔时间
            except Exception:
                pass

            frame_info = {'frame_count': frame_count}
            try:
                result_frame, detections_with_coords = detector.draw_detections(frame, detections, frame_info=frame_info)
            except Exception as e:
                if debug:
                    print(f"绘制检测错误：{e}")
                result_frame = frame.copy()
                detections_with_coords = []

            # 记录坐标
            if detections_with_coords:
                coord_tracker.add_coordinates(frame_count, detections_with_coords)

            # ! 调用串口发送函数，发送目标坐标（u，v），及控制字 control
            # 只有检测到目标时才发送；未检测到目标则不发送（与 UAV_Detect.py 一致）
            if serial_port and serial_port.is_open and len(detections_with_coords) > 0:
                # 提取第一个目标的中心坐标
                center_u = detections_with_coords[0][6]  # u 坐标
                center_v = detections_with_coords[0][7]  # v 坐标

                # 发送坐标数据，控制字设为 1
                sci_write(serial_port, center_u, center_v, 1)

            try:
                result_queue.put((result_frame, detections_with_coords, processing_time), timeout=0.5)
            except queue.Full:
                pass

    # 启动线程
    reader_thread = threading.Thread(target=read_frame_loop, daemon=True)
    infer_thread = threading.Thread(target=infer_loop, daemon=True)
    reader_thread.start()
    infer_thread.start()

    last_result = None
    try:
        while not stop_event.is_set():
            try:
                result_frame, detections_with_coords, processing_time = result_queue.get(timeout=0.5)
                last_result = (result_frame, detections_with_coords, processing_time)
            except queue.Empty:
                pass

            if last_result is not None:
                result_frame, detections_with_coords, processing_time = last_result
                current_fps = 1.0 / processing_time if processing_time > 0 else 0
                fps_text = f"FPS: {current_fps:.2f}"
                # 字号/位置随输出模式自适应
                cv2.putText(result_frame, fps_text, (3, max(20, int(round(70 * UI_SCALE)))),
                            cv2.FONT_HERSHEY_SIMPLEX, INFO_SCALE, (0, 255, 0), LINE_THICK)
                cv2.imshow('White Light Detection', result_frame)
                if video_writer:
                    video_writer.write(result_frame)
            else:
                time.sleep(0.01)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break

            if isinstance(max_frames, int) and max_frames > 0 and frame_count >= max_frames:
                if debug:
                    print(f"主循环检测到已处理帧数 {frame_count} >= max_frames {max_frames}, 退出")
                stop_event.set()
                break

    finally:
        stop_event.set()
        reader_thread.join(timeout=1)
        infer_thread.join(timeout=1)
        
        #! 串口关闭
        if serial_port:
            try:
                serial_port.close()
                print("串口连接已关闭")
            except Exception as e:
                print(f"关闭串口时出错：{e}")
        
        # ← 修改：使用帧间隔时间计算实际帧率
        if frame_intervals:
            total_time = sum(frame_intervals)
            avg_frame_interval = total_time / len(frame_intervals)
            actual_fps = 1.0 / avg_frame_interval if avg_frame_interval > 0 else 0
            min_frame_interval = min(frame_intervals)
            max_frame_interval = max(frame_intervals)
            min_fps = 1.0 / max_frame_interval if max_frame_interval > 0 else 0
            max_fps = 1.0 / min_frame_interval if min_frame_interval > 0 else 0
            
            print("\n" + "=" * 40)
            print("目标识别定位统计:")
            print("=" * 40)
            print(f"  总处理帧数：{frame_count}")
            print(f"  总处理时间：{total_time:.2f}s")
            print(f"  平均帧间隔：{avg_frame_interval*1000:.2f}ms")
            print(f"  目标识别定位帧率：{actual_fps:.2f} FPS")
            print(f"  帧率范围：{min_fps:.2f} ~ {max_fps:.2f} FPS")
            print("=" * 40)

    # 释放资源
    if is_sdk:
        try:
            cap.stop_grabbing()
            cap.close_device()
        except Exception:
            pass
    else:
        cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"视频处理完成！总共处理 {frame_count} 帧，检测到 {total_detections} 个目标")
    if output_path:
        print(f"结果视频已保存至：{output_path}")
    
    # 绘制坐标曲线（统一保存到工程根目录的 result 文件夹）
    if coord_tracker.u_coordinates:
        result_dir = PROJECT_ROOT / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        coord_tracker.plot_coordinates(str(result_dir / "coordinate_plot.png"))
        coord_tracker.save_coordinates_to_file(str(result_dir / "coordinates_data.csv"))
    else:
        print("没有检测到目标，无法生成坐标曲线")


def main():
    """
    主函数
    """
    # 输入源设置
    # input_source ='SDK'  # SDK 使用内置摄像头，或 'camera', 或 视频文件路径   'SDK'为相机实时采集 0 为电脑摄像头
    input_source = r"D:\Bit study\Visual\CCD_Detect\videos\Circle.mp4"  # 视频文件路径
    # output_path = r"D:\Bit study\Visual\CCD_Detect\videos\Circle_Detect_CCS128.mp4"   # None 表示不保存输出视频
    output_path = None   # None 表示不保存输出视频
    
    # 白灯检测参数
    brightness_threshold = 180  # 亮度阈值 (0-255)，越高检测越严格
    
    # 最大处理帧数
    max_frames = 500
    
    print(f"运行设置：input_source={input_source}, OUTPUT_MODE={OUTPUT_MODE} "
          f"(输出 {OUT_WIDTH}x{OUT_HEIGHT}, 中心=({CENTER_U},{CENTER_V})), "
          f"brightness_threshold={brightness_threshold}, max_frames={max_frames}")
    print("=" * 60)

    # 处理输入源
    process_video(
        input_source=input_source,
        output_path=output_path,
        brightness_threshold=brightness_threshold,
        debug=False,  # 关闭调试模式
        max_frames=max_frames,
    )


if __name__ == "__main__":
    main()



