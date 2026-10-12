import json
import os
from pathlib import Path
import cv2
import shutil
from sklearn.model_selection import train_test_split

# 工程根目录（本脚本位于 Train 子目录中，标注源/数据集输出等路径统一以此为基准）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def convert_labelme_to_yolo(labelme_json_path, yolo_labels_dir, class_mapping):
    """
    将单个LabelMe JSON文件转换为YOLO格式
    
    Args:
        labelme_json_path (str): LabelMe JSON文件路径
        yolo_labels_dir (str): YOLO标签文件输出目录
        class_mapping (dict): 类别名称到ID的映射字典
    
    Returns:
        bool: 转换是否成功
    """
    try:
        # 读取LabelMe JSON文件
        with open(labelme_json_path, 'r') as f:
            labelme_data = json.load(f)
        
        # 获取图像尺寸
        img_width = labelme_data['imageWidth']
        img_height = labelme_data['imageHeight']
        
        # 获取图像文件名（不含扩展名）
        image_name = Path(labelme_json_path).stem
        
        # 创建YOLO标签文件路径
        yolo_label_path = os.path.join(yolo_labels_dir, f"{image_name}.txt")
        
        yolo_annotations = []
        
        # 处理每个形状
        for shape in labelme_data['shapes']:
            label = shape['label']
            
            # 检查标签是否在类别列表中
            if label not in class_mapping:
                print(f"警告: 标签 '{label}' 不在类别列表中，跳过该标注")
                continue
            
            class_id = class_mapping[label]
            
            # 获取形状类型和坐标
            shape_type = shape['shape_type']
            points = shape['points']
            
            if shape_type == 'rectangle':
                # 处理矩形标注
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # 转换为YOLO格式 (center_x, center_y, width, height) 归一化到0-1
                center_x = ((x1 + x2) / 2) / img_width
                center_y = ((y1 + y2) / 2) / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            elif shape_type == 'polygon':
                # 对于多边形，计算边界框
                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 转换为YOLO格式
                center_x = ((x_min + x_max) / 2) / img_width
                center_y = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # 保存YOLO标签文件
        with open(yolo_label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        return True
    except Exception as e:
        print(f"转换失败 {labelme_json_path}: {str(e)}")
        return False

def collect_all_classes(labelme_dir):
    """
    收集所有JSON文件中的类别
    
    Args:
        labelme_dir (str): 包含LabelMe JSON文件的目录
    
    Returns:
        set: 所有类别的集合
    """
    classes = set()
    json_files = list(Path(labelme_dir).glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                for shape in data['shapes']:
                    classes.add(shape['label'])
        except Exception as e:
            print(f"读取文件失败 {json_file}: {str(e)}")
    
    return sorted(list(classes))

def create_yolo_dataset_structure(dataset_root):
    """
    创建YOLO数据集的标准目录结构
    
    Args:
        dataset_root (str): 数据集根目录
    
    Returns:
        dict: 各目录路径字典
    """
    dirs = {
        'root': dataset_root,
        'images': {
            'train': os.path.join(dataset_root, "images", "train"),
            'val': os.path.join(dataset_root, "images", "val")
        },
        'labels': {
            'train': os.path.join(dataset_root, "labels", "train"),
            'val': os.path.join(dataset_root, "labels", "val")
        }
    }
    
    # 创建所有必要目录
    for dir_path in [dirs['images']['train'], dirs['images']['val'], 
                     dirs['labels']['train'], dirs['labels']['val']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    return dirs

def copy_image_files(labelme_dir, image_dirs, json_files):
    """
    复制图像文件到对应的目录
    
    Args:
        labelme_dir (str): LabelMe JSON文件所在目录
        image_dirs (dict): 图像文件目标目录
        json_files (list): JSON文件列表
    
    Returns:
        list: 成功复制的图像文件名列表
    """
    copied_images = []
    
    for json_file in json_files:
        try:
            # 读取JSON文件获取图像路径
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            image_path = data.get('imagePath')
            if not image_path:
                print(f"警告: {json_file} 中未找到imagePath字段")
                continue
            
            # 构建源图像文件路径
            src_image_path = os.path.join(labelme_dir, image_path)
            
            # 检查图像文件是否存在
            if not os.path.exists(src_image_path):
                print(f"警告: 图像文件不存在 {src_image_path}")
                continue
            
            # 确定目标目录（与JSON文件相同的名字）
            image_name = Path(json_file).stem
            image_ext = Path(image_path).suffix
            dest_image_name = f"{image_name}{image_ext}"
            
            # 复制图像文件
            dest_image_path = os.path.join(image_dirs, dest_image_name)
            shutil.copy2(src_image_path, dest_image_path)
            copied_images.append(dest_image_name)
            
        except Exception as e:
            print(f"复制图像文件失败 {json_file}: {str(e)}")
    
    return copied_images

def create_yaml_config(dataset_root, class_names, yaml_path):
    """
    创建YOLO训练配置文件(.yaml)
    
    Args:
        dataset_root (str): 数据集根目录
        class_names (list): 类别名称列表
        yaml_path (str): YAML配置文件路径
    """
    yaml_content = f"""path: {dataset_root}
train: images/train
val: images/val

names:
"""
    
    for i, class_name in enumerate(class_names):
        yaml_content += f"  {i}: {class_name}\n"
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"创建YOLO配置文件: {yaml_path}")

def batch_convert_to_yolo_dataset(labelme_dir, dataset_root, train_ratio=0.8):
    """
    批量将LabelMe数据集转换为完整的YOLO数据集
    
    Args:
        labelme_dir (str): 包含LabelMe JSON和图像文件的目录
        dataset_root (str): YOLO数据集输出根目录
        train_ratio (float): 训练集比例（默认0.8）
    """
    print("开始转换LabelMe数据集为YOLO格式...")
    
    # 1. 收集所有类别
    print("收集所有类别...")
    class_names = collect_all_classes(labelme_dir)
    if not class_names:
        print("未找到任何标注类别")
        return
    
    print(f"找到 {len(class_names)} 个类别: {class_names}")
    
    # 2. 创建类别映射
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    
    # 3. 创建YOLO数据集目录结构
    print("创建YOLO数据集目录结构...")
    dirs = create_yolo_dataset_structure(dataset_root)
    
    # 4. 查找所有JSON文件
    json_files = list(Path(labelme_dir).glob("*.json"))
    if not json_files:
        print("未找到任何JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 5. 分割训练集和验证集
    train_files, val_files = train_test_split(json_files, train_size=train_ratio, random_state=42)
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    # 6. 处理训练集
    print("处理训练集...")
    train_converted = 0
    for json_file in train_files:
        if convert_labelme_to_yolo(str(json_file), dirs['labels']['train'], class_mapping):
            train_converted += 1
    
    # 复制训练集图像文件
    train_images_copied = copy_image_files(labelme_dir, dirs['images']['train'], train_files)
    print(f"训练集转换完成: {train_converted}/{len(train_files)} 个标签文件, {len(train_images_copied)} 个图像文件")
    
    # 7. 处理验证集
    print("处理验证集...")
    val_converted = 0
    for json_file in val_files:
        if convert_labelme_to_yolo(str(json_file), dirs['labels']['val'], class_mapping):
            val_converted += 1
    
    # 复制验证集图像文件
    val_images_copied = copy_image_files(labelme_dir, dirs['images']['val'], val_files)
    print(f"验证集转换完成: {val_converted}/{len(val_files)} 个标签文件, {len(val_images_copied)} 个图像文件")
    
    # 8. 创建类别文件
    classes_txt_path = os.path.join(dataset_root, "classes.txt")
    with open(classes_txt_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"创建类别文件: {classes_txt_path}")
    
    # 9. 创建YOLO配置文件
    yaml_path = os.path.join(dataset_root, "dataset.yaml")
    create_yaml_config(dataset_root, class_names, yaml_path)
    
    print("\n转换完成!")
    print(f"数据集位置: {dataset_root}")
    print(f"类别: {class_names}")
    print(f"训练集标签: {train_converted} 个")
    print(f"验证集标签: {val_converted} 个")
    print(f"训练集图像: {len(train_images_copied)} 个")
    print(f"验证集图像: {len(val_images_copied)} 个")

if __name__ == "__main__":
    # 配置参数
    labelme_directory = "CCD_Images"    #"sampled_images"  # 包含JSON和图像文件的目录
    yolo_dataset_root = "CCD_UAVdataset"    # "yolo_dataset" YOLO数据集输出目录
    train_split_ratio = 0.8               # 训练集比例
    
    # 执行转换
    batch_convert_to_yolo_dataset(
        labelme_dir=labelme_directory,
        dataset_root=yolo_dataset_root,
        train_ratio=train_split_ratio
    )