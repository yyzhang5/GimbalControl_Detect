# README — README_TRACK.md
目标检测及跟踪

## 目的
- 说明如何运行 `Detect/UAV_Detect.py`、如何切换输入源（相机 SDK / OpenCV 摄像头 / 本地视频文件）、如何设置停止帧数 `max_frames`、以及输出文件与坐标曲线的配置位置和调试提示。

## 依赖（建议的 Python 包）
- Python 3.8+
- numpy
- opencv-python
- onnxruntime（或 `onnxruntime-gpu` 如果你想用 GPU）
- matplotlib
- PyQt5 （可选，仅当你使用项目中的 Qt 界面模块）

## 安装示例（Windows / PowerShell）
```powershell
python -m pip install --upgrade pip
python -m pip install numpy opencv-python onnxruntime matplotlib
# 如需 GPU 支持，按你的 CUDA/驱动情况改为 onnxruntime-gpu
# python -m pip install onnxruntime-gpu
```

## 主要文件
- `Detect/UAV_Detect.py`：主脚本，负责读取帧（支持 SDK / 摄像头 / 本地视频）、调用 ONNX 模型检测、绘制检测与坐标、并保存输出。当前实现支持多线程流水线：采集线程 -> 推理线程 -> 主线程显示/保存。
- `Detect/Camera/CCD_Track.py`：SDK 相机封装（`CameraCapture.get_frame()`），当输入源为 `sdk` 时由 `UAV_Detect.py` 调用。
- 依赖按职责分目录：`Detect/Camera/`（相机 SDK 驱动、采集封装、参数调试工具）、`Detect/Communication/`（串口通信）、`Detect/Predict/`（Kalman 轨迹预测）。
  > 主程序靠 `sys.path` 引导导入这些子目录的模块；若重命名子文件夹，需同步修改主程序内的引导列表与 `.vscode/settings.json` 的 `python.analysis.extraPaths`。
- 模型文件：例如 `yolo_CCDUAV_training/CCDUAV_detection/weights/best.onnx`（可替换为你自己的 ONNX 模型）。
- 脚本顶部的 `ENABLE_SERIAL` 控制串口通信（置 `False` 可完全不使用串口）；`OUTPUT_MODE` 选择输出范围（`640x480` / `256x256` / `128x128`）。
  > `Detect/Led_Detect.py` 是白灯阈值检测版本，拥有相同的这两个开关。

## 如何设置输入源（几种方式）
1) 在 `main()` 中设置（快速、直接）
   - 打开 `Detect/UAV_Detect.py`，在 `main()` 内修改 `input_source`。
   - 可选值：
     - `input_source = 'SDK'` 或 `'sdk'`：使用厂商 SDK（需 `CCD_Track.CameraCapture` 可用）。
     - `input_source = 0` 或 `input_source = 'camera'`：使用本机摄像头（OpenCV）。
     - `input_source = r"D:\path\to\your_video.mp4"`：使用本地视频文件（推荐使用绝对路径或 `r"..."` 原始字符串）。

   示例（在 `main()` 中）
   ```python
   model_path = "yolo_CCDUAV_training/CCDUAV_detection/weights/best.onnx"
   input_source = r"D:\Videos\uav_sample.mp4"
   output_path = r"D:\Videos\uav_sample_out.mp4"  # 可选
   max_frames = 500  # 可选：处理多少帧后退出（None 表示不限制）
   debug = True
   ```

2) 在其他脚本中直接调用 `process_video()`
   - 函数签名：
     ```python
     process_video(input_source, model_path, output_path=None, conf_threshold=0.1, debug=False, max_frames=None)
     ```
   - `input_source` 的取值同上（int / 'camera' / 'sdk' / 文件路径字符串）。

## 命令行运行（目前脚本没有 argparse，直接运行 `UAV_Detect.py`）
```powershell
cd "d:\Bit study\Visual\CCD_Detect"
python Detect\UAV_Detect.py
```
脚本启动时会自动把工作目录切换到工程根目录，因此从任意位置启动都能正确解析模型、视频与输出路径。

## 如何设置 `max_frames`
- 当前实现：在 `main()` 中有一行 `max_frames = 500`（或 `None`），将其改为你需要的帧数即可。示例：`max_frames = 300` 表示处理 300 帧后自动退出。
- 行为说明：当达到 `max_frames` 时，采集线程/推理线程会触发停止事件并整洁退出，随后脚本会打印最终统计并保存坐标曲线（如有）。

## 坐标输出位置
- 运行结束后，识别定位得到的坐标数据与坐标曲线统一保存到工程根目录的 `result` 文件夹，两者共用同一个时间戳：`Data-YYYYMMDD_HHMMSS.csv` 与 `Data-YYYYMMDD_HHMMSS.png`。
- 要更改保存位置或命名：编辑 `Detect/UAV_Detect.py` 中 `process_video()` 末尾的 `result_dir` 与 `timestamp` 相关语句。
- `Detect/Led_Detect.py` 的输出同样写入 `result` 文件夹，文件名为 `coordinates_data.csv` 与 `coordinate_plot.png`。

## v输出保存位置与编码注意事项
- 视频输出：若 `output_path` 被设置，脚本会创建 `cv2.VideoWriter` 并写入处理图像。视频编码器使用 `mp4v`（`.mp4`）。
- SDK 输入注意：SDK（实时相机）通常不提供帧率元数据；脚本在创建 `VideoWriter` 时会在没有有效 `fps` 时使用默认 `25 FPS`，以避免只写入最后一帧或文件损坏的问题。

## 调试提示与常见问题
- 无法打开视频文件 / 报 `视频文件不存在`：
  - 确认路径是否拼写正确（建议使用绝对路径，或在字符串前加 `r`）。
  - 程序会打印解析后的绝对路径与当前工作目录以作诊断。
- SDK 无法枚举或打开设备：
  - 确认 SDK 的 DLL/运行时已按厂商要求放置并配置环境变量（例如 `MVCAM_COMMON_RUNENV`）。
  - 在 `Detect/Camera/CCD_Track.py` 中查看 `enum_devices()` / `open_first_device()` 的返回值与日志。
- ONNXRuntime provider 问题：
  - 若没有 GPU provider，ONNXRuntime 会回退到 CPU；若你期望使用 GPU，请安装与系统 CUDA/驱动匹配的 `onnxruntime-gpu`。
- 性能与丢帧：
  - 通过调整 `frame_queue` 的 `maxsize`（在 `Detect/UAV_Detect.py` 中）可以控制队列深度与丢帧/延迟权衡：更大的队列能缓存更多帧（延迟增加），更小队列丢帧多但延迟小。

## 如何测试并调试（建议步骤）
1. 使用本地短视频文件做快速回归：把 `input_source` 指向较短的视频，`max_frames` 设置为 200 以下，这样能快速看到完整输出文件与坐标文件。
2. 开启 `debug=True`：会在控制台打印更多信息（推理时间、帧拉取成功/失败计数、初始化 VideoWriter 信息等）。
3. 若输出视频只有一帧或文件损坏：检查程序启动时打印的 `已初始化 VideoWriter` 行，确保 `writer_fps` 与分辨率正确；若不行，尝试将 `output_path` 改为 `.avi` 并使用 `XVID` 编码以排除编码器兼容性问题：
   ```python
   fourcc = cv2.VideoWriter_fourcc(*'XVID')
   output_path = r"D:\Videos\out.avi"
   ```

## 可选改进
- 为 `Detect/UAV_Detect.py` 添加 `argparse`（命令行参数）：`--source`, `--model`, `--output`, `--conf`, `--debug`, `--max-frames`。
- 让坐标 CSV / 图像路径通过参数或配置文件传入，而不是写死在脚本末尾。