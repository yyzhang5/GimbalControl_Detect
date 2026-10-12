# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

from CamOperation_class import CameraOperation
from MvCameraControl_class import *
from MvErrorDefine_const import *
from CameraParams_header import *

# 工程根目录（本脚本位于 Detect/Camera 子目录中，模型等路径统一以此为基准）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)


class YOLOv8_Detector:
    """
    使用OpenCV DNN模块加载YOLOv8模型进行目标检测
    """

    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, debug=False):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.debug = debug

        try:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            print(f"模型加载成功: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e

        self.input_size = 640

    def preprocess(self, image):
        self.orig_height, self.orig_width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            1/255.0,
            (self.input_size, self.input_size),
            swapRB=True,
            crop=False
        )
        return blob

    def postprocess(self, outputs):
        try:
            if isinstance(outputs, list):
                predictions = np.squeeze(outputs[0])
            else:
                predictions = np.squeeze(outputs)

            if predictions.size == 0:
                if self.debug:
                    print("警告: 模型输出为空")
                return []

            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.T

            boxes = predictions[:, :4]
            confidences = predictions[:, 4]

            indices = np.where(confidences > self.conf_threshold)[0]
            if len(indices) == 0:
                if self.debug:
                    print("没有检测到满足置信度阈值的目标")
                return []

            valid_boxes = []
            valid_confidences = []

            for i in indices:
                if i >= len(boxes):
                    continue
                x, y, w, h = boxes[i]
                x1 = (x - w / 2) / self.input_size * self.orig_width
                y1 = (y - h / 2) / self.input_size * self.orig_height
                width = w / self.input_size * self.orig_width
                height = h / self.input_size * self.orig_height
                x1, y1, x2, y2 = int(x1), int(y1), int(x1 + width), int(y1 + height)
                if x2 > x1 and y2 > y1:
                    valid_boxes.append([x1, y1, x2, y2])
                    valid_confidences.append(float(confidences[i]))

            if len(valid_boxes) == 0:
                if self.debug:
                    print("没有有效的边界框")
                return []

            # NMS
            boxes_array = np.array(valid_boxes, dtype=np.float32)
            scores_array = np.array(valid_confidences, dtype=np.float32)
            boxes_for_nms = [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in valid_boxes]
            indices_nms = cv2.dnn.NMSBoxes(boxes_for_nms, scores_array.tolist(), self.conf_threshold, self.iou_threshold)

            if indices_nms is not None and len(indices_nms) > 0:
                detections = []
                for i in indices_nms:
                    idx = int(i) if not isinstance(i, (list, np.ndarray)) else int(i[0])
                    if 0 <= idx < len(valid_boxes):
                        x1, y1, x2, y2 = valid_boxes[idx]
                        confidence = valid_confidences[idx]
                        detections.append((int(x1), int(y1), int(x2), int(y2), float(confidence), 0))
                return detections

            # 若NMS失败，返回所有检测框
            return [(int(x1), int(y1), int(x2), int(y2), float(confidence), 0) for x1, y1, x2, y2, confidence in zip(*valid_boxes, valid_confidences)]
        except Exception as e:
            if self.debug:
                print(f"后处理错误: {e}")
            return []

    def detect(self, image):
        try:
            if self.debug:
                print(f"输入图像形状: {image.shape}")
            blob = self.preprocess(image)
            self.net.setInput(blob)
            outputs = self.net.forward()
            if self.debug:
                print(f"模型输出形状: {outputs.shape}")
            detections = self.postprocess(outputs)
            if self.debug:
                print(f"检测到的目标数量: {len(detections)}")
            return detections
        except Exception as e:
            if self.debug:
                print(f"检测错误: {e}")
            return []

    def draw_detections(self, image, detections, class_names=None, frame_info=None):
        image_copy = image.copy()
        for i, (x1, y1, x2, y2, confidence, class_id) in enumerate(detections):
            center_u = (x1 + x2) // 2
            center_v = (y1 + y2) // 2
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image_copy, (center_u, center_v), 5, (0, 0, 255), -1)
            label = f"Target: {confidence:.2f}"
            coord_text = f"({center_u}, {center_v})"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image_copy, (x1, y1 - text_height - baseline - 20), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(image_copy, label, (x1, y1 - baseline - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(image_copy, coord_text, (x1, y1 - baseline + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if frame_info:
            info_text = f"Frame: {frame_info['frame_count']}, Detections: {len(detections)}"
            cv2.putText(image_copy, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image_copy, [(x1, y1, x2, y2, conf, cls_id, (x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2, conf, cls_id in detections]


class CoordinateTracker:
    def __init__(self, max_points=1000):
        self.u_coordinates = deque(maxlen=max_points)
        self.v_coordinates = deque(maxlen=max_points)
        self.frame_numbers = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)

    def add_coordinates(self, frame_count, timestamp, detections):
        if detections:
            center_u = detections[0][6]
            center_v = detections[0][7]
            self.u_coordinates.append(center_u)
            self.v_coordinates.append(center_v)
            self.frame_numbers.append(frame_count)
            self.timestamps.append(timestamp)

    def plot_coordinates(self, save_path=None):
        if not self.u_coordinates:
            print("没有坐标数据可绘制")
            return
        frames = np.array(self.frame_numbers)
        u_coords = np.array(self.u_coordinates)
        v_coords = np.array(self.v_coordinates)
        timestamps = np.array(self.timestamps)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(frames, u_coords, 'b-', linewidth=2, label='u coordinate')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('u Coordinate (pixels)')
        ax1.set_title('Target u Coordinate Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax2.plot(frames, v_coords, 'r-', linewidth=2, label='v coordinate')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('v Coordinate (pixels)')
        ax2.set_title('Target v Coordinate Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"坐标曲线图已保存至: {save_path}")
        plt.show()

    def save_coordinates_to_file(self, file_path):
        with open(file_path, 'w') as f:
            f.write("Frame,Time,u_coordinate,v_coordinate\n")
            for frame, timestamp, u, v in zip(self.frame_numbers, self.timestamps, self.u_coordinates, self.v_coordinates):
                f.write(f"{frame},{timestamp:.6f},{u},{v}\n")
        print(f"坐标数据已保存至: {file_path}")


class CameraCapture:
    def __init__(self):
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.cam = MvCamera()
        self.obj_cam_operation = None
        self.is_open = False
        self.is_grabbing = False

    def enum_devices(self):
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        n_layer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE)
        ret = MvCamera.MV_CC_EnumDevices(n_layer_type, self.deviceList)
        # 增加诊断输出，便于定位枚举失败的原因
        try:
            print(f"MV_CC_EnumDevices 返回码: {ret}")
        except Exception:
            pass
        if ret != 0:
            try:
                print(f"枚举设备失败! 错误码: {hex(ret) if isinstance(ret,int) else ret}")
            except Exception:
                print(f"枚举设备失败! 返回码: {ret}")
            return ret
        # 当调用成功但未找到设备时，打印更明确的诊断信息
        try:
            devnum = getattr(self.deviceList, 'nDeviceNum', None)
        except Exception:
            devnum = None
        if not devnum:
            print("未找到设备 (deviceList.nDeviceNum == 0)。请检查：\n  - 相机是否已连接和供电\n  - USB/GIGE 连接是否正确\  - 驱动/SDK 是否安装并配置正确")
            # 也打印设备列表结构以便调试
            try:
                print(f"deviceList 内容: nDeviceNum={self.deviceList.nDeviceNum}")
            except Exception:
                pass
            return -1
        print(f"找到 {devnum} 个设备!")
        return 0

    def open_first_device(self):
        if self.deviceList.nDeviceNum <= 0:
            print("没有可使用的设备")
            return -1
        self.obj_cam_operation = CameraOperation(self.cam, self.deviceList, 0)
        ret = self.obj_cam_operation.Open_device()
        if ret != 0:
            print(f"打开设备失败，错误码: {hex(ret)}")
            return ret
        self.is_open = True
        print("设备已打开")
        return 0

    def set_continuous_mode(self):
        if not self.obj_cam_operation:
            return -1
        ret = self.obj_cam_operation.Set_trigger_mode(False)
        if ret != 0:
            print(f"设置连续模式失败，错误码: {hex(ret)}")
            return ret
        print("已设置为连续采集模式")
        return 0

    def start_grabbing(self, hwnd=None):
        if not self.obj_cam_operation or not self.is_open:
            return -1
        ret = self.obj_cam_operation.Start_grabbing(hwnd)
        if ret != 0:
            print(f"开始采集失败，错误码: {hex(ret)}")
            return ret
        self.is_grabbing = True
        print("开始图像采集")
        return 0

    def stop_grabbing(self):
        if not self.obj_cam_operation or not self.is_grabbing:
            return 0
        try:
            ret = self.obj_cam_operation.Stop_grabbing()
            if ret != 0:
                print(f"停止采集失败，错误码: {hex(ret)}")
                return ret
            self.is_grabbing = False
            print("已停止图像采集")
            return 0
        except Exception as e:
            print(f"停止采集时发生异常: {e}")
            self.is_grabbing = False
            return -1

    def close_device(self):
        if not self.obj_cam_operation:
            return 0
        try:
            self.obj_cam_operation.Close_device()
            print("设备已关闭")
        except Exception as e:
            print(f"关闭设备时发生异常: {e}")
            return -1
        self.is_open = False
        self.is_grabbing = False
        return 0

    def get_frame(self):
        if not self.is_grabbing or not self.obj_cam_operation:
            return None
        stOutFrame = MV_FRAME_OUT()
        ret = self.obj_cam_operation.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret != 0:
            print(f"[CameraCapture.get_frame] MV_CC_GetImageBuffer 返回: {hex(ret) if isinstance(ret,int) else ret}")
            return None

        try:
            frame_len = stOutFrame.stFrameInfo.nFrameLen
            height = stOutFrame.stFrameInfo.nHeight
            width = stOutFrame.stFrameInfo.nWidth
            pixel_type = stOutFrame.stFrameInfo.enPixelType

            pData = (c_ubyte * frame_len)()
            memmove(pData, stOutFrame.pBufAddr, frame_len)
            frameArray = np.frombuffer(pData, dtype=np.uint8)
        except Exception as e:
            try:
                self.obj_cam_operation.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            except Exception:
                pass
            print(f"[CameraCapture.get_frame] 数据拷贝失败: {e}")
            return None

        # 释放 SDK 缓冲
        try:
            self.obj_cam_operation.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
        except Exception:
            pass

        try:
            # 常见像素格式处理
            if pixel_type in (PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed):
                frame = frameArray.reshape((height, width, 3))
                return frame

            # Bayer 转换
            if pixel_type in (PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8):
                gray = frameArray.reshape((height, width))
                mapping = {
                    PixelType_Gvsp_BayerRG8: cv2.COLOR_BAYER_RG2BGR,
                    PixelType_Gvsp_BayerGR8: cv2.COLOR_BAYER_GR2BGR,
                    PixelType_Gvsp_BayerGB8: cv2.COLOR_BAYER_GB2BGR,
                    PixelType_Gvsp_BayerBG8: cv2.COLOR_BAYER_BG2BGR,
                }
                code = mapping.get(pixel_type, cv2.COLOR_BAYER_RG2BGR)
                frame = cv2.cvtColor(gray, code)
                return frame

            # Mono8 -> BGR
            if pixel_type == PixelType_Gvsp_Mono8:
                gray = frameArray.reshape((height, width))
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                return frame

            # YUV422 -> BGR
            # PFNC 约定： PixelType_Gvsp_YUV422_Packed     (0x0210001F / 34603039) 字节序 = U Y V Y
            #            PixelType_Gvsp_YUV422_YUYV_Packed (0x02100032 / 34603058) 字节序 = Y U Y V
            # 本相机实测输出 0x0210001F，按 YUYV 解会整幅发绿（亮度取自色度字节），必须用 UYVY
            if pixel_type in (PixelType_Gvsp_YUV422_Packed, PixelType_Gvsp_YUV422_YUYV_Packed):
                yuv = frameArray[:height * width * 2].reshape((height, width, 2))
                if pixel_type == PixelType_Gvsp_YUV422_Packed:
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_UYVY)
                else:
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)
                return frame

            # 尝试解码压缩格式（如 JPEG）
            frame = cv2.imdecode(frameArray, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"[CameraCapture.get_frame] 解码/转换失败: {e}")
            return None


class CameraTrackingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.camera = CameraCapture()
        self.detector = None
        self.coord_tracker = CoordinateTracker()

        # 视频录制相关
        self.video_writer = None
        self.is_recording = False
        self.output_video_path = "camera_tracked_output.mp4"

        # 性能统计
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        self.start_time = time.time()

        self.init_ui()
        self.init_camera()
        self.init_detector()
        
        # 自动开始采集
        self.start_capture_auto()

    def init_ui(self):
        self.setWindowTitle("实时相机目标跟踪系统")
        self.setGeometry(100, 100, 1000, 700)

        # 主布局
        main_layout = QVBoxLayout()

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("初始化中...")
        main_layout.addWidget(self.image_label)

        # 坐标显示区域
        self.coord_label = QLabel()
        self.coord_label.setText("坐标信息: 无目标检测到")
        self.coord_label.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(self.coord_label)

        self.setLayout(main_layout)

        # 定时器用于定期更新显示
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)

        # 定时器用于定期保存坐标数据
        self.coord_timer = QTimer()
        self.coord_timer.timeout.connect(self.save_coordinates_periodically)
        self.coord_timer.start(5000)  # 每5秒保存一次坐标数据

    def init_camera(self):
        # 初始化SDK
        MvCamera.MV_CC_Initialize()

        # 枚举设备
        ret = self.camera.enum_devices()
        if ret != 0:
            self.image_label.setText("枚举设备失败")
            return

        # 打开第一个设备
        ret = self.camera.open_first_device()
        if ret != 0:
            self.image_label.setText("打开设备失败")
            return

        # 设置连续采集模式
        self.camera.set_continuous_mode()

        print("相机初始化完成")

    def init_detector(self):
        # 初始化检测器
        model_path = "yolo_CCDUAV_training/CCDUAV_detection/weights/best.onnx"
        try:
            self.detector = YOLOv8_Detector(model_path, conf_threshold=0.1, debug=False)
            print("目标检测器初始化完成")
        except Exception as e:
            print(f"初始化检测器失败: {e}")
            self.image_label.setText("检测器初始化失败")

    def start_capture_auto(self):
        # 自动开始采集
        ret = self.camera.start_grabbing(0)
        if ret == 0:
            self.frame_count = 0
            self.start_time = time.time()
            self.display_timer.start(30)  # 约33fps
            print("开始连续采集...")
        else:
            print("开始采集失败")

    def update_display(self):
        # 获取相机帧
        frame = self.camera.get_frame()
        if frame is None:
            return

        self.frame_count += 1
        current_time = time.time() - self.start_time

        # 记录开始处理时间
        start_process_time = time.time()

        # 进行目标检测
        detections = self.detector.detect(frame)

        # 记录处理结束时间
        end_process_time = time.time()
        process_time = end_process_time - start_process_time
        self.processing_times.append(process_time)

        # 绘制检测结果
        frame_info = {'frame_count': self.frame_count}
        result_frame, detections_with_coords = self.detector.draw_detections(frame, detections, frame_info=frame_info)

        # 更新坐标显示
        if detections_with_coords:
            coord_text = "检测到目标:\n"
            for i, detection in enumerate(detections_with_coords):
                x1, y1, x2, y2, conf, cls_id, center_u, center_v = detection
                coord_text += f"  目标 {i+1}: 坐标({center_u}, {center_v}), 置信度 {conf:.2f}\n"
            self.coord_label.setText(coord_text)
        else:
            self.coord_label.setText("坐标信息: 无目标检测到")

        # 记录坐标数据
        if detections_with_coords:
            self.coord_tracker.add_coordinates(self.frame_count, current_time, detections_with_coords)

        # 转换颜色空间 (如果是BGR格式)
        if len(result_frame.shape) == 3 and result_frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = result_frame

        # 确保数组是连续的
        rgb_frame = np.ascontiguousarray(rgb_frame)

        # 转换为QImage
        h, w = rgb_frame.shape[:2]
        bytes_per_line = 3 * w

        if len(rgb_frame.shape) == 3:
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            q_img = QImage(rgb_frame.data, w, h, w, QImage.Format_Grayscale8)

        # 检查QImage是否有效
        if q_img.isNull():
            print(f"QImage创建失败！尺寸: {w}x{h}, 字节每行: {bytes_per_line}")
            return

        # 缩放图像以适应显示区域
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # 显示图像
        self.image_label.setPixmap(scaled_pixmap)

        # 如果正在录制，则写入视频
        if self.is_recording and self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_video_path,
                fourcc,
                30.0,
                (w, h)
            )

        if self.is_recording and self.video_writer:
            self.video_writer.write(result_frame)

    def save_coordinates_periodically(self):
        # 定期保存坐标数据到文件
        if len(self.coord_tracker.u_coordinates) > 0:
            self.coord_tracker.save_coordinates_to_file("tracking_coordinates_temp.csv")

    def plot_coordinates(self):
        # 绘制并保存坐标曲线
        if len(self.coord_tracker.u_coordinates) > 0:
            self.coord_tracker.plot_coordinates("tracking_coordinates.png")
            self.coord_tracker.save_coordinates_to_file("tracking_coordinates.csv")
            print("坐标曲线已绘制并保存")
        else:
            print("没有足够的坐标数据绘制曲线")

    def show_performance_stats(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            avg_fps = self.frame_count / elapsed_time
            print(f"性能统计:")
            print(f"  总帧数: {self.frame_count}")
            print(f"  总耗时: {elapsed_time:.2f} 秒")
            print(f"  平均帧率: {avg_fps:.2f} FPS")

            if len(self.processing_times) > 0:
                avg_proc_time = sum(self.processing_times) / len(self.processing_times)
                print(f"  平均处理时间: {avg_proc_time*1000:.2f} ms")

    def close_application(self):
        self.display_timer.stop()
        if self.is_recording and self.video_writer:
            self.video_writer.release()
        self.camera.stop_grabbing()
        self.camera.close_device()
        MvCamera.MV_CC_Finalize()
        self.show_performance_stats()
        self.plot_coordinates()
        self.close()

    def closeEvent(self, event):
        self.close_application()
        event.accept()


def main():
    app = QApplication(sys.argv)
    camera_app = CameraTrackingApp()
    camera_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()