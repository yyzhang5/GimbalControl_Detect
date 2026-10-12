import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 工程根目录（本脚本位于 Train 子目录中，模型/数据集等路径统一以此为基准）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def load_model(model_path):
    """
    加载训练好的YOLO模型
    
    Args:
        model_path (str): 模型文件路径
    
    Returns:
        YOLO model: 加载的模型对象
    """
    try:
        print(f"加载模型: {model_path}")
        model = YOLO(model_path)
        print("模型加载成功")
        return model
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None

def predict_single_image(model, image_path, conf_threshold=0.25, save_result=True, show_result=False):
    """
    对单张图像进行预测
    
    Args:
        model (YOLO): 加载的模型对象
        image_path (str): 图像文件路径
        conf_threshold (float): 置信度阈值
        save_result (bool): 是否保存结果图像
        show_result (bool): 是否显示结果图像
    
    Returns:
        list: 预测结果
    """
    try:
        print(f"处理图像: {image_path}")
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return None
        
        # 进行预测
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            save=save_result,
            show=show_result,
            verbose=True
        )
        
        # 显示预测信息
        result = results[0]  # 获取第一个（也是唯一一个）结果
        boxes = result.boxes
        
        if boxes is not None:
            print(f"检测到 {len(boxes)} 个目标:")
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                print(f"  目标 {i+1}: 类别={class_id}, 置信度={confidence:.3f}, "
                      f"边界框=({xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f})")
        else:
            print("未检测到任何目标")
        
        return results
        
    except Exception as e:
        print(f"图像预测失败 {image_path}: {str(e)}")
        return None

def predict_image_directory(model, images_dir, conf_threshold=0.25, save_results=True):
    """
    对图像目录中的所有图像进行预测
    
    Args:
        model (YOLO): 加载的模型对象
        images_dir (str): 图像目录路径
        conf_threshold (float): 置信度阈值
        save_results (bool): 是否保存结果图像
    
    Returns:
        dict: 所有预测结果的统计信息
    """
    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # 查找所有图像文件
    image_files = []
    for extension in image_extensions:
        pattern = os.path.join(images_dir, extension)
        image_files.extend(Path(images_dir).glob(extension))
    
    if not image_files:
        print(f"在 {images_dir} 中未找到任何图像文件")
        return {}
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 统计信息
    total_images = len(image_files)
    total_detections = 0
    detection_per_image = []
    
    # 处理每个图像文件
    for i, image_file in enumerate(image_files):
        print(f"\n处理图像 ({i+1}/{total_images}): {image_file.name}")
        
        results = predict_single_image(
            model, 
            str(image_file), 
            conf_threshold=conf_threshold, 
            save_result=save_results, 
            show_result=False
        )
        
        if results is not None:
            # 统计检测结果
            boxes = results[0].boxes
            detection_count = len(boxes) if boxes is not None else 0
            total_detections += detection_count
            detection_per_image.append(detection_count)
            
            print(f"  该图像检测到 {detection_count} 个目标")
    
    # 计算统计信息
    stats = {
        'total_images': total_images,
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / total_images if total_images > 0 else 0,
        'max_detections_in_single_image': max(detection_per_image) if detection_per_image else 0,
        'min_detections_in_single_image': min(detection_per_image) if detection_per_image else 0
    }
    
    print(f"\n预测统计:")
    print(f"  总图像数: {stats['total_images']}")
    print(f"  总检测数: {stats['total_detections']}")
    print(f"  平均每图检测数: {stats['avg_detections_per_image']:.2f}")
    print(f"  单图最多检测数: {stats['max_detections_in_single_image']}")
    print(f"  单图最少检测数: {stats['min_detections_in_single_image']}")
    
    return stats

def evaluate_model(model, dataset_yaml, imgsz=640):
    """
    对模型进行正式评估
    
    Args:
        model (YOLO): 加载的模型对象
        dataset_yaml (str): 数据集配置文件路径
        imgsz (int): 图像大小
    
    Returns:
        dict: 评估结果
    """
    try:
        print("开始模型评估...")
        
        # 检查数据集配置文件
        if not os.path.exists(dataset_yaml):
            print(f"数据集配置文件不存在: {dataset_yaml}")
            return None
        
        # 进行验证
        results = model.val(
            data=dataset_yaml,
            imgsz=imgsz,
            verbose=True
        )
        
        # 显示评估结果
        print("\n评估结果:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  mAP75: {results.box.map75:.4f}")
        
        return results
        
    except Exception as e:
        print(f"模型评估失败: {str(e)}")
        return None

def create_visualization(model, image_path, conf_threshold=0.25):
    """
    创建检测结果的可视化图像
    
    Args:
        model (YOLO): 加载的模型对象
        image_path (str): 图像文件路径
        conf_threshold (float): 置信度阈值
    """
    try:
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 进行预测
        results = model.predict(source=image_path, conf=conf_threshold, verbose=False)
        result = results[0]
        
        # 在图像上绘制检测框
        if result.boxes is not None:
            # 获取检测框信息
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            # 绘制每个检测框
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签
                label = f"Class {int(cls_id)}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存可视化结果
        output_path = f"{Path(image_path).stem}_detection.jpg"
        cv2.imwrite(output_path, image)
        print(f"可视化结果已保存: {output_path}")
        
    except Exception as e:
        print(f"可视化创建失败: {str(e)}")

def main():
    """
    主测试流程
    """
    print("YOLO模型测试工具")
    print("=" * 50)
    
    # 1. 设置模型路径和测试参数
    model_path = "yolo_CCDUAV_training/CCDUAV_detection/weights/best.pt"    #"yolo_training/uva_detection/weights/best.pt"  # 训练好的模型路径
    test_images_dir ="CCD_UAVdataset/images/val"   #"yolo_dataset/images/val"  # 测试图像目录  
    dataset_yaml = "CCD_UAVdataset/dataset.yaml"   # 数据集配置文件
    conf_threshold = 0.25  # 置信度阈值
    
    # 2. 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请确保已经完成模型训练")
        return
    
    # 3. 加载模型
    model = load_model(model_path)
    if model is None:
        return
    
    # 4. 单张图像测试
    print("\n" + "=" * 50)
    print("单张图像测试")
    print("=" * 50)
    
    # 查找一张测试图像
    test_image = None
    if os.path.exists(test_images_dir):
        image_files = list(Path(test_images_dir).glob("*.jpg"))[:1]
        if image_files:
            test_image = str(image_files[0])
    
    if test_image:
        predict_single_image(model, test_image, conf_threshold, save_result=True, show_result=False)
        create_visualization(model, test_image, conf_threshold)
    else:
        print("未找到测试图像")
    
    # 5. 批量图像测试
    print("\n" + "=" * 50)
    print("批量图像测试")
    print("=" * 50)
    
    if os.path.exists(test_images_dir):
        predict_image_directory(model, test_images_dir, conf_threshold, save_results=True)
    else:
        print(f"测试图像目录不存在: {test_images_dir}")
    
    # 6. 模型评估（如果有数据集配置文件）
    print("\n" + "=" * 50)
    print("模型评估")
    print("=" * 50)
    
    if os.path.exists(dataset_yaml):
        evaluate_model(model, dataset_yaml)
    else:
        print(f"数据集配置文件不存在: {dataset_yaml}")
        print("跳过正式评估")

if __name__ == "__main__":
    main()