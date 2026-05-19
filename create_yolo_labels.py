#!/usr/bin/env python3
"""
为COU水下数据集创建YOLO格式的标签
"""

import os
import subprocess
import sys

# COU水下数据集的24个类别
COU_CLASS_NAMES = [
    "plastic_bottle",
    "plastic_bag", 
    "fishing_net",
    "rope",
    "can",
    "glass_bottle",
    "tire",
    "metal_scrap",
    "wood",
    "cloth",
    "diver",
    "diving_mask",
    "diving_fins",
    "oxygen_tank",
    "underwater_camera",
    "auv",
    "rov",
    "underwater_drone",
    "sonar",
    "underwater_sensor",
    "ship_wreck",
    "anchor",
    "propeller",
    "underwater_structure"
]

def create_yolo_labels():
    """为训练集、验证集和测试集创建YOLO标签"""
    
    # 创建标签目录
    labels_dir = "data/coco/labels"
    os.makedirs(labels_dir, exist_ok=True)
    
    # 为训练集、验证集、测试集分别创建标签目录
    train_labels_dir = os.path.join(labels_dir, "train")
    val_labels_dir = os.path.join(labels_dir, "val")
    test_labels_dir = os.path.join(labels_dir, "test")
    
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    
    print("开始转换COCO标注为YOLO格式...")
    print(f"类别数量: {len(COU_CLASS_NAMES)}")
    print(f"类别: {COU_CLASS_NAMES}")
    
    # 转换训练集
    print("\n转换训练集标注...")
    train_cmd = [
        sys.executable, "convert_coco_to_yolo.py",
        "--coco_json", "data/coco/train_annotations.json",
        "--images_dir", "data/coco/images",
        "--output_dir", train_labels_dir,
        "--class_names"
    ] + COU_CLASS_NAMES
    
    print(f"执行命令: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    print("标准输出:", result.stdout)
    if result.stderr:
        print("标准错误:", result.stderr)
    
    # 转换验证集
    print("\n转换验证集标注...")
    val_cmd = [
        sys.executable, "convert_coco_to_yolo.py",
        "--coco_json", "data/coco/val_annotations.json",
        "--images_dir", "data/coco/images",
        "--output_dir", val_labels_dir,
        "--class_names"
    ] + COU_CLASS_NAMES
    
    print(f"执行命令: {' '.join(val_cmd)}")
    result = subprocess.run(val_cmd, capture_output=True, text=True)
    print("标准输出:", result.stdout)
    if result.stderr:
        print("标准错误:", result.stderr)
    
    # 转换测试集
    print("\n转换测试集标注...")
    test_cmd = [
        sys.executable, "convert_coco_to_yolo.py",
        "--coco_json", "data/coco/test_annotations.json",
        "--images_dir", "data/coco/images",
        "--output_dir", test_labels_dir,
        "--class_names"
    ] + COU_CLASS_NAMES
    
    print(f"执行命令: {' '.join(test_cmd)}")
    result = subprocess.run(test_cmd, capture_output=True, text=True)
    print("标准输出:", result.stdout)
    if result.stderr:
        print("标准错误:", result.stderr)
    
    # 创建YOLOv8所需的数据集结构
    print("\n创建YOLOv8数据集结构...")
    
    # 创建符号链接或复制图像到labels目录的父目录
    # YOLOv8期望的结构是：
    # dataset/
    #   images/
    #     train/
    #     val/
    #     test/
    #   labels/
    #     train/
    #     val/
    #     test/
    
    # 创建images目录结构
    images_dir = "data/coco/images"
    yolo_images_train = "data/coco/images/train"
    yolo_images_val = "data/coco/images/val"
    yolo_images_test = "data/coco/images/test"
    
    os.makedirs(yolo_images_train, exist_ok=True)
    os.makedirs(yolo_images_val, exist_ok=True)
    os.makedirs(yolo_images_test, exist_ok=True)
    
    # 读取图像列表文件
    def copy_images_from_list(list_file, target_dir):
        """根据列表文件复制图像到目标目录"""
        if not os.path.exists(list_file):
            print(f"警告: 列表文件不存在: {list_file}")
            return 0
        
        with open(list_file, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        copied = 0
        for image_path in image_paths:
            src_path = os.path.join("data/coco", image_path)
            if os.path.exists(src_path):
                # 获取文件名
                filename = os.path.basename(image_path)
                dst_path = os.path.join(target_dir, filename)
                
                # 如果是Windows，使用复制；如果是Unix，使用符号链接
                try:
                    if os.name == 'nt':  # Windows
                        import shutil
                        shutil.copy2(src_path, dst_path)
                    else:  # Unix/Linux/Mac
                        os.symlink(os.path.abspath(src_path), dst_path)
                    copied += 1
                except Exception as e:
                    print(f"复制/链接 {src_path} 到 {dst_path} 时出错: {e}")
        
        return copied
    
    print("复制训练图像...")
    train_copied = copy_images_from_list("data/coco/train.txt", yolo_images_train)
    print(f"复制了 {train_copied} 张训练图像")
    
    print("复制验证图像...")
    val_copied = copy_images_from_list("data/coco/val.txt", yolo_images_val)
    print(f"复制了 {val_copied} 张验证图像")
    
    print("复制测试图像...")
    test_copied = copy_images_from_list("data/coco/test.txt", yolo_images_test)
    print(f"复制了 {test_copied} 张测试图像")
    
    # 更新coco.yaml文件以使用新的结构
    print("\n更新coco.yaml文件...")
    yaml_content = f"""# COU水下目标检测数据集
# 24类人造物

path: data/coco/  # 数据集根目录
train: images/train  # 训练图像目录
val: images/val      # 验证图像目录
test: images/test    # 测试图像目录

# 类别数量
nc: {len(COU_CLASS_NAMES)}

# 类别名称
names:
"""
    for i, name in enumerate(COU_CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_path = "data/coco/coco.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"更新了 {yaml_path}")
    
    print("\n✅ YOLO标签创建完成！")
    print(f"训练标签: {train_labels_dir}")
    print(f"验证标签: {val_labels_dir}")
    print(f"测试标签: {test_labels_dir}")
    print(f"训练图像: {yolo_images_train}")
    print(f"验证图像: {yolo_images_val}")
    print(f"测试图像: {yolo_images_test}")

if __name__ == "__main__":
    create_yolo_labels()
