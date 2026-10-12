# CCD_Detect

基于海康机器人（Hikrobot MVS）工业相机的目标识别定位系统，为云台控制回路提供目标偏差量。
系统实时采集图像，检测目标并提取像素坐标，可选通过串口按固定协议将坐标下发给下位机（DSP）。
另提供白灯阈值检测版本，用于对照调试与算法验证。

## 目录结构

| 路径 | 说明 |
|---|---|
| `Detect/` | **识别定位主程序**（运行入口） |
| `Detect/Led_Detect.py` | 白灯检测：亮度阈值分割 + 轮廓质心提取，不依赖模型 |
| `Detect/UAV_Detect.py` | 无人机检测：ONNX 模型推理 + NMS |
| `Detect/Camera/` | 相机 SDK 驱动、采集封装 `CCD_Track.py`、参数调试工具 `BasicDemo.py` |
| `Detect/Communication/` | 串口初始化与数据发送（`Initial_SCi.py`、`SCI_Write.py`） |
| `Detect/Predict/` | 轨迹预测模块（Kalman，当前未接入主流程） |
| `Train/` | 数据集制作、模型训练与评估 |
| `Markdown/` | 项目文档与配图 |
| `CCD_Images/` | 人工标注源（labelme 标注 + 原始帧） |
| 其他数据目录 | `videos/`、`sampled_images/`、`CCD_UAVdataset/`、`model/`、`yolo_CCDUAV_training/`、`result/`，多数未纳入版本控制，详见 `.gitignore` |

## 环境依赖

- Python 3.12（开发环境实测 3.12.3）
- 海康机器人 MVS 工业相机客户端（提供 `MvCameraControl.dll`，并需设置环境变量 `MVCAM_COMMON_RUNENV`）

识别定位所需：

```bash
pip install opencv-python numpy onnxruntime matplotlib pyserial PyQt5
```

训练与数据集制作额外需要：

```bash
pip install ultralytics torch torchvision scikit-learn
```

## 快速开始

### 识别定位

```bash
python Detect/UAV_Detect.py    # 无人机检测
python Detect/Led_Detect.py    # 白灯检测
```

脚本启动时会自动将工作目录切换到工程根目录，因此从任意位置运行都能正确解析模型、视频与输出路径。

两个入口的常用配置均在各文件顶部或 `main()` 内：

| 配置项 | 说明 |
|---|---|
| `ENABLE_SERIAL` | 置 `True` 才会打开串口并下发坐标；单独测试识别定位时保持 `False` |
| `OUTPUT_MODE` | `'640x480'` / `'256x256'` / `'128x128'`，决定显示画面、保存视频与坐标取值范围 |
| `input_source` | `'SDK'` 使用厂商相机；`'camera'` 或 `0` 使用本机摄像头；也可填视频文件路径 |
| `max_frames` | 处理多少帧后自动退出，`None` 表示不限制 |

串口参数：`UAV_Detect.py` 使用 COM5，`Led_Detect.py` 使用 COM6，波特率均为 115200。

### 数据集制作与训练

```bash
python Train/video_sample_image.py   # 从 videos/ 按时间间隔抽帧到 sampled_images/
# 使用 labelme 标注图片，标注结果放入 CCD_Images/
python Train/lebelme2YOLOlabel.py    # 将 labelme 标注转换为 YOLO 数据集 CCD_UAVdataset/
python Train/train.py                # 训练，使用 model/yolo11n.pt 作为预训练权重
python Train/eval.py                 # 模型评估与批量测试
```

## 串口协议

目标坐标按固定 12 字节帧下发，多字节字段为小端序：

| 字节 | 内容 |
|---|---|
| 0–1 | `AA 55` 包头 |
| 2 | `control`（uint8）：`1` = 检测到目标，`0` = 未检测到 |
| 3–6 | `data1`：目标 u 坐标（float32） |
| 7–10 | `data2`：目标 v 坐标（float32） |
| 11 | `0x0D` 包尾 |

## 输出

识别定位结束后，坐标数据与曲线图统一保存到工程根目录 `result/`：

| 程序 | 输出文件 |
|---|---|
| `Detect/UAV_Detect.py` | `Data-YYYYMMDD_HHMMSS.csv` 与 `Data-YYYYMMDD_HHMMSS.png`，共用同一时间戳，每次运行生成一组 |
| `Detect/Led_Detect.py` | `coordinates_data.csv` 与 `coordinate_plot.png`，使用固定文件名 |

## 关于仓库中缺失的内容

为控制体积，以下内容未纳入版本控制（见 `.gitignore`）：

- `videos/`（约 331 MB）、`sampled_images/`、`CCD_UAVdataset/`：素材与派生数据集，可由脚本重新生成
- `model/`：YOLO 预训练权重，可从 [Ultralytics](https://github.com/ultralytics/ultralytics) 下载
- `yolo_CCDUAV_training/`：训练产物。`Train/train.py` 在训练结束时会自动导出 ONNX，
  其生成的 `yolo_CCDUAV_training/CCDUAV_detection/weights/best.onnx` **是 `UAV_Detect.py` 运行所必需**，
  clone 后需先执行训练生成该文件

## 文档索引

- [`Markdown/README_TRAIN.md`](Markdown/README_TRAIN.md) — 数据集制作与模型训练
- [`Markdown/README_CAMERA.md`](Markdown/README_CAMERA.md) — 相机驱动说明与设备参数调试
- [`Markdown/README_TRACK.md`](Markdown/README_TRACK.md) — 识别定位程序的详细用法

