import os
from pathlib import Path
from ultralytics import YOLO
import torch

# 工程根目录（本脚本位于 Train 子目录中，数据集/模型/训练输出等路径统一以此为基准）
# 切换到该目录后，ultralytics 对 dataset.yaml 内相对 path 的解析行为与之前完全一致
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)



# 在Windows中增加页面文件大小:
# 1. 右键"此电脑" -> "属性"
# 2. 点击"高级系统设置"
# 3. 在"性能"区域点击"设置"
# 4. 转到"高级"选项卡，点击"虚拟内存"下的"更改"
# 5. 取消勾选"自动管理所有驱动器的分页文件大小"
# 6. 选择自定义大小，建议初始大小为物理内存的1.5倍，最大值为物理内存的3倍


def train_yolo_model(dataset_yaml, model_name, epochs=100, imgsz=640, 
                     batch_size=16, project="yolo_training", name="exp"):
    """
    使用YOLOv8训练模型
    
    Args:
        dataset_yaml (str): 数据集配置文件路径 (.yaml)
        model_name (str): 预训练模型名称或路径
        epochs (int): 训练轮数
        imgsz (int): 输入图像大小
        batch_size (int): 批次大小
        project (str): 训练项目目录
        name (str): 实验名称
    
    Returns:
        results: 训练结果对象
    """
    try:
        # 检查CUDA是否可用
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        # 加载预训练模型
        print(f"加载模型: {model_name}")
        model = YOLO(model_name)
        
        # 开始训练
        print("开始训练...")
        results = model.train(
            data=dataset_yaml,      # 数据集配置文件
            epochs=epochs,          # 训练轮数
            imgsz=imgsz,            # 图像大小
            batch=batch_size,       # 批次大小
            device=device,          # 设备
            project=project,        # 项目目录
            name=name,              # 实验名称
            exist_ok=True,          # 如果实验名称存在则覆盖
            verbose=True            # 显示详细信息
        )
        
        return results
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        return None

def validate_model(model_path, dataset_yaml, imgsz=640):
    """
    验证训练好的模型
    
    Args:
        model_path (str): 训练好的模型路径
        dataset_yaml (str): 数据集配置文件路径
        imgsz (int): 输入图像大小
    
    Returns:
        results: 验证结果对象
    """
    try:
        print(f"加载模型进行验证: {model_path}")
        model = YOLO(model_path)
        
        print("开始验证...")
        results = model.val(
            data=dataset_yaml,
            imgsz=imgsz,
            verbose=True
        )
        
        return results
    except Exception as e:
        print(f"验证过程中出现错误: {str(e)}")
        return None

def predict_with_model(model_path, source, conf_threshold=0.25, save=True, show=False):
    """
    使用训练好的模型进行预测
    
    Args:
        model_path (str): 训练好的模型路径
        source (str): 预测源（图像/视频/目录路径）
        conf_threshold (float): 置信度阈值
        save (bool): 是否保存结果
        show (bool): 是否显示结果
    
    Returns:
        results: 预测结果列表
    """
    try:
        print(f"加载模型进行预测: {model_path}")
        model = YOLO(model_path)
        
        print(f"对 {source} 进行预测...")
        results = model.predict(
            source=source,
            conf=conf_threshold,
            save=save,
            show=show,
            verbose=True
        )
        
        return results
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        return None

def export_model(model_path, format="onnx"):
    """
    导出模型为其他格式
    
    Args:
        model_path (str): 训练好的模型路径
        format (str): 导出格式 (onnx, torchscript, etc.)
    
    Returns:
        export_path: 导出文件路径
    """
    try:
        print(f"导出模型: {model_path} 为 {format} 格式")
        model = YOLO(model_path)
        
        export_path = model.export(format=format)
        print(f"模型已导出至: {export_path}")
        
        return export_path
    except Exception as e:
        print(f"导出过程中出现错误: {str(e)}")
        return None

def setup_training_environment():
    """
    设置训练环境，检查依赖项
    """
    print("检查训练环境...")
    
    # 检查ultralytics是否安装
    try:
        import ultralytics
        print(f"Ultraalytics 版本: {ultralytics.__version__}")
    except ImportError:
        print("错误: 未安装 ultralytics，请运行: pip install ultralytics")
        return False
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA 不可用，将使用 CPU 训练（较慢）")
    
    return True

def main():
    """
    主训练流程
    """
    # 1. 设置训练环境
    if not setup_training_environment():
        return
    
    # 2. 定义训练参数
    dataset_yaml = "CCD_UAVdataset/dataset.yaml"  # 数据集配置文件
    model_name = "model/yolo11n.pt"             # 预训练模型
    epochs = 100                                # 训练轮数
    imgsz = 320                                 # 图像大小
    batch_size = 1                             # 批次大小
    project = "yolo_CCDUAV_training"                   # 项目目录
    exp_name = "CCDUAV_detection"                  # 实验名称
    
    # 3. 检查数据集配置文件是否存在
    if not os.path.exists(dataset_yaml):
        print(f"错误: 数据集配置文件不存在 {dataset_yaml}")
        print("请先运行数据转换脚本生成YOLO数据集")
        return
    
    # 4. 开始训练
    print("=" * 50)
    print("开始YOLO训练")
    print("=" * 50)
    
    train_results = train_yolo_model(
        dataset_yaml=dataset_yaml,
        model_name=model_name,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        project=project,
        name=exp_name
    )
    
    if train_results is None:
        print("训练失败")
        return
    
    # 5. 获取最佳模型路径
    best_model_path = os.path.join(project, exp_name, "weights", "best.pt")
    last_model_path = os.path.join(project, exp_name, "weights", "last.pt")
    
    print(f"训练完成!")
    print(f"最佳模型: {best_model_path}")
    print(f"最后模型: {last_model_path}")
    
    # 6. 验证最佳模型
    print("\n" + "=" * 50)
    print("验证最佳模型")
    print("=" * 50)
    
    val_results = validate_model(best_model_path, dataset_yaml, imgsz)
    
    if val_results is not None:
        print("验证完成")
    
    # # 7. 示例预测（可选）
    # print("\n" + "=" * 50)
    # print("示例预测")
    # print("=" * 50)
    
    # # 检查是否有测试图像
    # test_images_dir = "yolo_dataset/images/val"  # 使用验证集作为示例
    # if os.path.exists(test_images_dir):
    #     test_images = list(Path(test_images_dir).glob("*.[jJ][pP][gG]"))[:3]  # 取前3张jpg图片
    #     if test_images:
    #         predict_with_model(
    #             model_path=best_model_path,
    #             source=str(test_images[0]),  # 预测第一张图片
    #             conf_threshold=0.25,
    #             save=True,
    #             show=False
    #         )
    
    # 8. 导出模型（可选）
    print("\n" + "=" * 50)
    print("导出模型")
    print("=" * 50)
    
    export_model(best_model_path, "onnx")

if __name__ == "__main__":
    main()



