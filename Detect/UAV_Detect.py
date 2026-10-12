##############################   相机实时采集/本地视频文件 进行目标检测跟踪    #############################
##############################   能够显示检测帧率 绘制坐标曲线 保存跟踪视频    #############################
#############################   但未实时通过串口发送目标位置   #############################
# YYZ 2025-12-10

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


#  todo 串口初始化及发送函数调用
from SCI_Write import sci_write                  # 714行处调用
from Initial_SCi import initialize_serial_port   # 512行处调用
# 828行处关闭串口


import cv2
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import threading
import queue
import atexit
from datetime import datetime
# 如果需要使用相机SDK的采集，请使用 CCD_Track 中的 CameraCapture（已实现 get_frame）
try:
    from CCD_Track import CameraCapture
except Exception:
    CameraCapture = None

# 工程根目录（本脚本位于 Detect 子目录中，模型/视频/结果等路径统一以此为基准）
# 切换到该目录后，脚本内所有相对路径的行为与之前"在工程根目录下运行"完全一致
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# 全局统计，用于在程序退出时打印最终平均 FPS（即便发生早期返回）
_LAST_FRAME_COUNT = 0
_LAST_PROCESSING_TIMES = []

def _print_exit_stats():
    try:
        if _LAST_PROCESSING_TIMES:
            total_processing_time = sum(_LAST_PROCESSING_TIMES)
            avg_processing_time = total_processing_time / len(_LAST_PROCESSING_TIMES)
            overall_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            print(f"\n最终统计 - 已处理帧数: {_LAST_FRAME_COUNT}, 平均每帧: {avg_processing_time*1000:.2f}ms, 平均帧率: {overall_fps:.2f} FPS")
        else:
            print("\n最终统计: 未处理任何帧，无法计算平均帧率")
    except Exception as e:
        print(f"打印最终统计时出错: {e}")

atexit.register(_print_exit_stats)

# 串口通信开关：False 时不初始化或发送串口数据
ENABLE_SERIAL = False  # False

# 相机出厂分辨率及输出模式
CAM_FULL_WIDTH = 640
CAM_FULL_HEIGHT = 480
OUTPUT_MODE = '640x480'

_MODE_SIZES = {
    '640x480': (CAM_FULL_WIDTH, CAM_FULL_HEIGHT),
    '256x256': (256, 256),
    '128x128': (128, 128),
}

if OUTPUT_MODE not in _MODE_SIZES:
    raise ValueError(f"不支持的 OUTPUT_MODE={OUTPUT_MODE}，可选：{list(_MODE_SIZES.keys())}")

OUT_WIDTH, OUT_HEIGHT = _MODE_SIZES[OUTPUT_MODE]
CENTER_U = OUT_WIDTH // 2
CENTER_V = OUT_HEIGHT // 2


def crop_to_output(frame):
    """从图像中心裁剪当前输出模式对应的区域。"""
    height, width = frame.shape[:2]
    if width == OUT_WIDTH and height == OUT_HEIGHT:
        return frame
    left = max(0, (width - OUT_WIDTH) // 2)
    top = max(0, (height - OUT_HEIGHT) // 2)
    return frame[top:top + OUT_HEIGHT, left:left + OUT_WIDTH]

class YOLOv8_ONNX_Detector:
    """
    使用ONNX模型进行YOLOv8目标检测
    """
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, debug=False):
        """
        初始化检测器
        
        Args:
            model_path (str): ONNX模型路径
            conf_threshold (float): 置信度阈值
            iou_threshold (float): IOU阈值（用于NMS）
            debug (bool): 是否启用调试模式
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.debug = debug
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 获取输入形状
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
    
    def preprocess(self, image):
        """
        预处理图像
        """
        if self.debug:
            print(f"原始图像形状: {image.shape}")
        
        # 调整图像大小
        img = cv2.resize(image, (self.input_width, self.input_height))
        
        # 转换颜色空间 BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0,1]
        img = img.astype(np.float32) / 255.0
        
        # 转换为CHW格式
        img = img.transpose(2, 0, 1)
        
        # 添加批次维度
        img = np.expand_dims(img, axis=0)
        
        if self.debug:
            print(f"预处理后图像形状: {img.shape}")
        
        return img
    
    def postprocess(self, outputs, original_shape):
        """
        后处理检测结果 (适配单类别检测模型)
        """
        try:
            if self.debug:
                print(f"模型输出形状: {outputs.shape}")
                if outputs.size > 0:
                    print(f"输出值范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
            
            # 解析输出
            predictions = np.squeeze(outputs).T
            
            if self.debug:
                print(f"预测结果形状: {predictions.shape}")
            
            # 确保predictions不是空的
            if predictions.size == 0:
                if self.debug:
                    print("警告: 模型输出为空")
                return []
            
            # 检查预测结果格式
            if predictions.ndim != 2:
                if self.debug:
                    print(f"警告: 预测结果维度异常: {predictions.ndim}")
                return []
            
            # 根据列数判断模型类型
            num_cols = predictions.shape[1]
            if self.debug:
                print(f"预测结果列数: {num_cols}")
            
            if num_cols < 5:
                if self.debug:
                    print(f"警告: 预测结果列数不足: {num_cols} < 5")
                return []
            
            # 处理不同格式的输出
            if num_cols == 5:
                # 单类别检测格式: x, y, w, h, conf
                boxes = predictions[:, :4]  # x, y, w, h
                confidences = predictions[:, 4]
                # 单类别，所有检测都属于同一类别（类别0）
                class_scores = np.ones_like(confidences)  # 假设类别置信度为1
                class_ids = np.zeros_like(confidences, dtype=int)  # 所有都是类别0
                final_confidences = confidences  # 直接使用检测置信度
            elif num_cols == 6:
                # 两类别检测格式: x, y, w, h, conf, class_conf
                boxes = predictions[:, :4]  # x, y, w, h
                confidences = predictions[:, 4]
                class_scores = predictions[:, 5]
                class_ids = np.zeros_like(confidences, dtype=int)  # 假设只有一个类别
                final_confidences = confidences * class_scores
            else:
                # 多类别检测格式: x, y, w, h, conf, class_probs...
                boxes = predictions[:, :4]  # x, y, w, h
                confidences = predictions[:, 4]
                class_probs = predictions[:, 5:]
                
                # 计算类别置信度，添加安全检查
                if class_probs.ndim > 1 and class_probs.shape[1] > 0:
                    class_scores = np.max(class_probs, axis=1)
                    class_ids = np.argmax(class_probs, axis=1)
                else:
                    class_scores = np.ones(confidences.shape[0]) if confidences.ndim > 0 else np.array([1])
                    class_ids = np.zeros(confidences.shape[0]) if confidences.ndim > 0 else np.array([0])
                
                final_confidences = confidences * class_scores
            
            if self.debug:
                print(f"边界框形状: {boxes.shape}")
                print(f"置信度形状: {confidences.shape}")
                print(f"最终置信度范围: [{final_confidences.min():.4f}, {final_confidences.max():.4f}]")
                print(f"置信度阈值: {self.conf_threshold}")
            
            # 过滤低置信度的检测
            indices = np.where(final_confidences > self.conf_threshold)[0]
            
            if self.debug:
                print(f"高于阈值的检测数量: {len(indices)}")
            
            # 如果没有满足置信度阈值的检测结果，直接返回空列表
            if len(indices) == 0:
                if self.debug:
                    print("没有检测到满足置信度阈值的目标")
                return []
            
            valid_boxes = []
            valid_confidences = []
            valid_class_ids = []
            
            orig_h, orig_w = original_shape[:2]
            input_h, input_w = self.input_height, self.input_width
            
            for i in indices:
                # 确保索引有效
                if i >= len(boxes):
                    continue
                    
                # 将边界框从模型输入尺寸转换回原始图像尺寸
                x, y, w, h = boxes[i]
                
                # 转换为中心坐标+宽高格式到左上右下坐标格式
                x1 = (x - w / 2) / input_w * orig_w
                y1 = (y - h / 2) / input_h * orig_h
                x2 = (x + w / 2) / input_w * orig_w
                y2 = (y + h / 2) / input_h * orig_h
                
                # 确保边界框在图像范围内
                x1 = max(0, min(orig_w, x1))
                y1 = max(0, min(orig_h, y1))
                x2 = max(0, min(orig_w, x2))
                y2 = max(0, min(orig_h, y2))
                
                # 只有有效的边界框才添加
                if x2 > x1 and y2 > y1:
                    valid_boxes.append([x1, y1, x2, y2])
                    valid_confidences.append(float(final_confidences[i]))
                    valid_class_ids.append(int(class_ids[i]))
            
            if self.debug:
                print(f"有效边界框数量: {len(valid_boxes)}")
            
            # 如果处理后没有有效框，返回空列表
            if len(valid_boxes) == 0:
                if self.debug:
                    print("没有有效的边界框")
                return []
            
            # 应用非极大值抑制(NMS)
            try:
                # 转换为正确的格式
                boxes_array = np.array(valid_boxes, dtype=np.float32)
                scores_array = np.array(valid_confidences, dtype=np.float32)
                
                if self.debug:
                    print(f"NMS输入 - 边界框: {boxes_array.shape}, 置信度: {scores_array.shape}")
                
                # 确保数组不为空且格式正确
                if (boxes_array.size > 0 and scores_array.size > 0 and 
                    boxes_array.ndim == 2 and scores_array.ndim == 1 and
                    len(boxes_array) == len(scores_array) and len(boxes_array) > 0):
                    
                    # 调用NMS，添加额外的安全检查
                    indices_nms = cv2.dnn.NMSBoxes(
                        boxes_array.tolist(),
                        scores_array.tolist(),
                        self.conf_threshold,
                        self.iou_threshold
                    )
                    
                    if self.debug:
                        print(f"NMS输出: {indices_nms}")
                    
                    # 安全处理NMS返回的结果
                    if indices_nms is not None and len(indices_nms) > 0:
                        # 标准化indices格式
                        if isinstance(indices_nms, np.ndarray):
                            if indices_nms.size > 0:
                                if indices_nms.ndim > 1:
                                    indices_nms = indices_nms.flatten()
                                indices_nms = indices_nms.tolist()
                        elif not isinstance(indices_nms, (list, tuple)):
                            indices_nms = [indices_nms]
                        
                        # 构建最终检测结果
                        detections = []
                        for i in indices_nms:
                            idx = int(i) if not isinstance(i, (list, np.ndarray)) else (int(i[0]) if len(i) > 0 else 0)
                            if 0 <= idx < len(valid_boxes):
                                x1, y1, x2, y2 = valid_boxes[idx]
                                confidence = valid_confidences[idx]
                                class_id = valid_class_ids[idx]
                                detections.append((int(x1), int(y1), int(x2), int(y2), float(confidence), int(class_id)))
                        
                        if self.debug:
                            print(f"NMS后保留的检测数量: {len(detections)}")
                        
                        if detections:
                            return detections
            except Exception as nms_error:
                if self.debug:
                    print(f"NMS处理错误: {nms_error}")
                # NMS失败时继续执行，使用所有检测结果
            
            # 如果NMS失败或没有返回有效结果，返回所有满足置信度阈值的检测框
            detections = []
            for i in range(len(valid_boxes)):
                x1, y1, x2, y2 = valid_boxes[i]
                confidence = valid_confidences[i]
                class_id = valid_class_ids[i]
                detections.append((int(x1), int(y1), int(x2), int(y2), float(confidence), int(class_id)))
            
            if self.debug:
                print(f"最终返回的检测数量: {len(detections)}")
            
            return detections
            
        except Exception as e:
            if self.debug:
                print(f"后处理错误: {e}")
                import traceback
                traceback.print_exc()
            return []
    
    def detect(self, image):
        """
        对单张图像进行检测
        """
        try:
            if self.debug:
                print(f"输入图像形状: {image.shape}")
            
            # 预处理
            input_tensor = self.preprocess(image)
            
            # 推理
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )[0]
            
            if self.debug:
                print(f"模型输出形状: {outputs.shape}")
            
            # 后处理
            detections = self.postprocess(outputs, image.shape)
            
            if self.debug:
                print(f"检测到的目标数量: {len(detections)}")
            
            return detections
        except Exception as e:
            if self.debug:
                print(f"检测错误: {e}")
            return []
    
    def draw_detections(self, image, detections, class_names=None, frame_info=None):
        """
        在图像上绘制检测结果和坐标信息
        """
        image_copy = image.copy()
        
        if self.debug:
            print(f"绘制检测结果，数量: {len(detections)}")
        
        for i, (x1, y1, x2, y2, confidence, class_id) in enumerate(detections):
            if self.debug:
                print(f"  检测 {i+1}: 类别={class_id}, 置信度={confidence:.3f}, 位置=({x1},{y1},{x2},{y2})")
            
            # 计算中心点坐标 (u, v)
            center_u = (x1 + x2) // 2
            center_v = (y1 + y2) // 2
            
            # 绘制边界框
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(image_copy, (center_u, center_v), 5, (0, 0, 255), -1)
            
            # 准备标签文本
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {confidence:.2f}"
            else:
                label = f"Class {class_id}: {confidence:.2f}"
            
            # 坐标信息文本
            coord_text = f"({center_u}, {center_v})"
            
            # 绘制标签背景
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                image_copy,
                (x1, y1 - text_height - baseline - 20),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # 绘制标签文字
            cv2.putText(
                image_copy,
                label,
                (x1, y1 - baseline - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            
            # 绘制坐标文字
            cv2.putText(
                image_copy,
                coord_text,
                (x1, y1 - baseline + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # 在图像左上角显示帧信息
        if frame_info:
            info_text = f"Frame: {frame_info['frame_count']}, Detections: {len(detections)}"
            cv2.putText(image_copy, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image_copy, [(x1, y1, x2, y2, conf, cls_id, (x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2, conf, cls_id in detections]

class CoordinateTracker:
    """
    坐标跟踪器，用于记录和绘制坐标轨迹
    """
    def __init__(self, image_width=1920, image_height=1080):
        self.image_width = image_width
        self.image_height = image_height
        self.u_coordinates = []  # 存储u坐标历史
        self.v_coordinates = []  # 存储v坐标历史
        self.frame_numbers = []  # 存储帧号
        
    def add_coordinates(self, frame_count, detections):
        """
        添加当前帧的坐标信息
        """
        if detections:
            # 只取第一个检测目标的坐标（如果有多个目标，可以根据需要修改）
            center_u = detections[0][6]  # 中心点u坐标
            center_v = detections[0][7]  # 中心点v坐标
            
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
        
        # 绘制u坐标曲线
        ax1.plot(self.frame_numbers, self.u_coordinates, 'b-', linewidth=2, label='u coordinate')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('u Coordinate (pixels)')
        ax1.set_title('Target u Coordinate Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, self.image_width)
        
        # 绘制v坐标曲线
        ax2.plot(self.frame_numbers, self.v_coordinates, 'r-', linewidth=2, label='v coordinate')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('v Coordinate (pixels)')
        ax2.set_title('Target v Coordinate Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, self.image_height)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"坐标曲线图已保存至: {save_path}")
        
        plt.show()
    
    def save_coordinates_to_file(self, file_path):
        """
        将坐标数据保存到文件
        """
        with open(file_path, 'w') as f:
            f.write("Frame,u_coordinate,v_coordinate\n")
            for frame, u, v in zip(self.frame_numbers, self.u_coordinates, self.v_coordinates):
                f.write(f"{frame},{u},{v}\n")
        print(f"坐标数据已保存至: {file_path}")


def process_video(input_source, model_path, output_path=None, conf_threshold=0.1, debug=False, max_frames=None):
    """
    处理视频文件或相机实时流并进行目标检测，增加坐标跟踪功能。

    Args:
        input_source: 视频文件路径或摄像头索引（int）或字符串 'camera'（等价于索引 0）。
        model_path: ONNX 模型路径。
        output_path: 可选的输出视频路径。
        conf_threshold: 置信度阈值。
        debug: 是否打印调试信息。
    """
    # 支持三种输入来源：
    #  1) 整数 或 'camera' -> 使用 OpenCV 的 VideoCapture
    #  2) 字符串 'sdk' 或 'camera_sdk' -> 使用厂商 SDK（`CCD_Track.CameraCapture`）
    #  3) 其他字符串 -> 视作视频文件路径
    # 优先判断 SDK

# ! 串口初始化函数调用
    serial_port = None
    if ENABLE_SERIAL:
        serial_port = initialize_serial_port(port='COM5', baudrate=115200)
        if not serial_port:
            print("串口初始化失败，请检查 COM5 是否可用（或把 ENABLE_SERIAL 改为 False）")
            return
        if not serial_port.is_open:
            print("串口 COM5 未成功打开")
            return

    is_sdk = False
    if isinstance(input_source, str) and input_source.lower() in ('sdk', 'camera_sdk'):
        if CameraCapture is None:
            print("错误: 未能导入 SDK CameraCapture 类，请确保 `CCD_Track.py` 可用。")
            return
        # 使用 SDK 采集
        cam = CameraCapture()
        ret = cam.enum_devices()
        if ret != 0:
            print("错误: 枚举 SDK 设备失败")
            return
        ret = cam.open_first_device()
        if ret != 0:
            print("错误: 打开 SDK 设备失败")
            return
        cam.set_continuous_mode()
        # 不使用 CameraOperation.Start_grabbing()（该方法会启动显示线程并消耗帧）
        # 直接调用底层 SDK 的 StartGrabbing，然后使用 CameraCapture.get_frame() 拉取帧
        try:
            ret = cam.obj_cam_operation.obj_cam.MV_CC_StartGrabbing()
        except Exception as e:
            print(f"错误: 无法调用 SDK StartGrabbing: {e}")
            return
        if ret != 0:
            print(f"错误: SDK StartGrabbing 返回错误: {hex(ret) if isinstance(ret,int) else ret}")
            return
        # 等待短时间以让相机开始采集
        time.sleep(0.1)
        # 标记为正在采集（但不启动显示线程）
        cam.is_grabbing = True
        is_sdk = True
        is_camera = True
        cap = cam
    # 统一处理非 SDK 的输入源
    is_camera = False
    if is_sdk:
        # 已经用 SDK 打开并设置了 cap
        pass
    elif isinstance(input_source, int) or (isinstance(input_source, str) and input_source.lower() == 'camera'):
        cam_index = int(input_source) if isinstance(input_source, int) else 0
        is_camera = True
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 (index={cam_index})")
            return
    else:
        # 当作文件路径处理（支持相对/绝对路径和 ~ 展开）
        video_path = str(input_source)
        # 展开用户路径并获取绝对路径
        try:
            video_path_expanded = str(Path(video_path).expanduser())
        except Exception:
            video_path_expanded = video_path

        abs_path = os.path.abspath(video_path_expanded)
        exists = os.path.exists(abs_path)
        if not exists:
            print(f"错误: 视频文件不存在 {video_path}\n  解析后路径: {abs_path}\n  当前工作目录: {os.getcwd()}\n  os.path.exists: {exists}")
            return
        cap = cv2.VideoCapture(abs_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {abs_path}")
            return
    
    try:
        # 初始化检测器
        detector = YOLOv8_ONNX_Detector(model_path, conf_threshold=conf_threshold, debug=debug)
    except Exception as e:
        print(f"初始化检测器失败: {e}")
        return
    
    # 初始化坐标跟踪器
    coord_tracker = CoordinateTracker(image_width=OUT_WIDTH, image_height=OUT_HEIGHT)

    # 获取视频/摄像头信息
    fps = 0
    width = 0
    height = 0
    total_frames = -1
    if is_sdk:
        # 对于 SDK，我们通过尝试取一帧来获取分辨率
        first_frame = cap.get_frame()
        if first_frame is None:
            print("错误: 无法从 SDK 获取第一帧")
            # 清理并返回
            try:
                cap.obj_cam_operation.obj_cam.MV_CC_StopGrabbing()
                cap.close_device()
            except Exception:
                pass
            return
        first_frame = crop_to_output(first_frame)
        height, width = first_frame.shape[:2]
        total_frames = -1
        # 将 first_frame 放回处理循环：我们会先处理该帧
        # SDK 调试计数器
        sdk_frame_ok = 0
        sdk_frame_fail = 0
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            fps = int(fps) if fps > 0 else 0
        except:
            fps = 0
        src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = min(OUT_WIDTH, src_width)
        height = min(OUT_HEIGHT, src_height)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else -1

    # 初始化视频写入器（如果指定了输出路径）
    video_writer = None
    if output_path:
        # 如果未能从源获得有效 FPS（例如 SDK 实时流），使用合理的默认 FPS
        writer_fps = fps if (isinstance(fps, (int, float)) and fps > 0) else 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, writer_fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    processing_times = []  # 用于存储每帧处理时间
    # 线程与队列：采集线程 -> 推理线程 -> 主线程显示
    frame_queue = queue.Queue(maxsize=4)
    result_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    def read_frame_loop():
        nonlocal sdk_frame_ok, sdk_frame_fail, frame_count
        # 如果我们在之前从 SDK 获取了 first_frame，把它先放入队列
        if is_sdk and 'first_frame' in locals() and first_frame is not None:
            try:
                frame_queue.put(first_frame, timeout=0.5)
            except Exception:
                pass

        while not stop_event.is_set():
            try:
                # 如果达到最大帧数，提前停止采集
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
                    else:
                        sdk_frame_fail += 1
                        time.sleep(0.005)
                        continue
                    frame = crop_to_output(frame)
                else:
                    ret, frame = cap.read()
                    ok = ret and (frame is not None)
                    if not ok:
                        # 视频读完或摄像头异常，停止
                        stop_event.set()
                        break
                    frame = crop_to_output(frame)

                try:
                    frame_queue.put(frame, timeout=0.2)
                except queue.Full:
                    # 队列已满，丢弃帧以保持实时性
                    pass
            except Exception:
                time.sleep(0.01)

    def infer_loop():
        nonlocal frame_count, total_detections
        global _LAST_FRAME_COUNT, _LAST_PROCESSING_TIMES
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_count += 1
            # 如果指定了最大帧数，达到后停止整个流程
            if isinstance(max_frames, int) and max_frames > 0 and frame_count >= max_frames:
                if debug:
                    print(f"infer_loop: 达到最大帧数 {max_frames}，停止采集")
                stop_event.set()
                break
            start_time = time.time()
            try:
                detections = detector.detect(frame)
            except Exception as e:
                if debug:
                    print(f"推理错误: {e}")
                detections = []

            total_detections += len(detections)
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            # 同步到模块级全局统计，供程序退出时打印
            try:
                _LAST_FRAME_COUNT = frame_count
                _LAST_PROCESSING_TIMES = processing_times.copy()
            except Exception:
                pass

            frame_info = {'frame_count': frame_count}
            try:
                result_frame, detections_with_coords = detector.draw_detections(frame, detections, frame_info=frame_info)
            except Exception as e:
                if debug:
                    print(f"绘制检测错误: {e}")
                result_frame = frame.copy()
                detections_with_coords = []

            # 记录坐标
            if detections_with_coords:
                coord_tracker.add_coordinates(frame_count, detections_with_coords)
                center_u = detections_with_coords[0][6]
                center_v = detections_with_coords[0][7]
                print(f"({center_u}, {center_v})", flush=True)

            # ! 调用串口发送函数，发送目标坐标（u，v），及控制字control;
            # 通过串口发送第一个检测目标的坐标
                if serial_port and serial_port.is_open and len(detections_with_coords) > 0:
                    # 发送坐标数据，控制字设为1（可根据需要调整）
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
                cv2.putText(result_frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('YOLOv8 Detection', result_frame)
                if video_writer:
                    video_writer.write(result_frame)
            else:
                # 没有结果时短暂等待
                time.sleep(0.01)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break

            # 如果指定 max_frames 并且已达到，主动退出
            if isinstance(max_frames, int) and max_frames > 0 and frame_count >= max_frames:
                if debug:
                    print(f"主循环检测到已处理帧数 {frame_count} >= max_frames {max_frames}, 退出")
                stop_event.set()
                break

    finally:
        stop_event.set()
        reader_thread.join(timeout=1)
        infer_thread.join(timeout=1)
        
    
    # 计算总体统计信息
    if processing_times:
        total_processing_time = sum(processing_times)
        avg_processing_time = total_processing_time / len(processing_times)
        overall_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        min_processing_time = min(processing_times)
        max_processing_time = max(processing_times)
        min_fps = 1.0 / max_processing_time if max_processing_time > 0 else 0
        max_fps = 1.0 / min_processing_time if min_processing_time > 0 else 0
        
        print(f"\n处理性能统计:")
        print(f"  总处理时间: {total_processing_time:.2f}s")
        print(f"  平均处理时间: {avg_processing_time*1000:.2f}ms")
        print(f"  最小处理时间: {min_processing_time*1000:.2f}ms (对应最高帧率 {max_fps:.2f} FPS)")
        print(f"  最大处理时间: {max_processing_time*1000:.2f}ms (对应最低帧率 {min_fps:.2f} FPS)")
        print(f"  平均帧率: {overall_fps:.2f} FPS")
    

# ! 关闭串口
    stop_event.set()
    reader_thread.join(timeout=1)
    infer_thread.join(timeout=1)
    #  关闭串口连接
    if serial_port:
        try:
            serial_port.close()
            print("串口连接已关闭")
        except Exception as e:
            print(f"关闭串口时出错: {e}")
            

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
    
    print(f"视频处理完成! 总共处理 {frame_count} 帧，检测到 {total_detections} 个目标")
    if output_path:
        print(f"结果视频已保存至: {output_path}")
    
    # 绘制坐标曲线（统一保存到工程根目录的 result 文件夹）
    if coord_tracker.u_coordinates:
        result_dir = PROJECT_ROOT / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        coord_tracker.plot_coordinates(str(result_dir / f"Data-{timestamp}.png"))
        coord_tracker.save_coordinates_to_file(str(result_dir / f"Data-{timestamp}.csv"))
    else:
        print("没有检测到目标，无法生成坐标曲线")

      

def main():
    """
    主函数
    """
    # 设置路径 / 输入源
    model_path ="yolo_CCDUAV_training/CCDUAV_detection/weights/best.onnx"

    
    # # 将输入源改为摄像头（0），也可以改为视频文件路径字符串
    input_source ='SDK'  # 使用内置摄像头，或 'camera', 或 视频文件路径   'SDK'为相机实时采集 0为电脑摄像头
    output_path = None  # 实时显示时通常不需要保存输出，如果需要可设置为文件路径（r"D:...videos\Track.mp4"）； None表示不保存

    # # # ######输入源 改为本地视频文件#####
    # input_source = r"D:\Bit study\Visual\CCD_Detect\videos\V1.mp4"  # 使用内置摄像头，或 'camera'，或 视频文件路径   'SDK'为相机实时采集 0为电脑摄像头
    # output_path =  r"D:\Bit study\Visual\CCD_Detect\videos\UAV_CCD2.mp4"  # 实时显示时通常不需要保存输出，如果需要可设置为文件路径
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    # 在这里设置最大处理帧数（若为 None 则不限制）
    max_frames = 1000  # 例如设置为 500 表示处理 500 帧后退出：max_frames = None 表示无限制
        # 处理输入源（摄像头或视频文件）
    process_video(
        input_source=input_source,
        model_path=model_path,
        output_path=output_path,
        conf_threshold=0.1,  # 降低阈值以增加检测敏感度
        debug=False,
        max_frames=max_frames,
    )

if __name__ == "__main__":
    main()



