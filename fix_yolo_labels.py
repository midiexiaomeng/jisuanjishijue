#!/usr/bin/env python3
"""
修复YOLO标签文件格式的脚本
将多边形点格式转换为YOLOv8边界框格式
"""

import os
import glob
import numpy as np
from pathlib import Path

def convert_polygon_to_bbox(polygon_points):
    """
    将多边形点转换为边界框 [x_min, y_min, x_max, y_max]
    假设输入是归一化坐标 (0-1)
    """
    if len(polygon_points) < 2:
        return None
    
    # 将点列表转换为numpy数组
    points = np.array(polygon_points).reshape(-1, 2)
    
    # 找到边界框
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    
    # 计算中心点和宽高
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # 确保值在有效范围内
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return [x_center, y_center, width, height]

def fix_label_file(label_path):
    """
    修复单个标签文件
    """
    try:
        with open(label_path, 'r') as f:
            content = f.read().strip()
        
        if not content:
            print(f"警告: {label_path} 为空文件")
            return False
        
        # 尝试解析内容
        numbers = list(map(float, content.split()))
        
        if len(numbers) < 5:
            print(f"警告: {label_path} 数据不足")
            return False
        
        # 第一个数字应该是类别ID
        class_id = int(numbers[0])
        
        # 剩下的应该是坐标点
        coords = numbers[1:]
        
        if len(coords) % 2 != 0:
            print(f"警告: {label_path} 坐标点数量不是偶数")
            return False
        
        # 将坐标点分组为(x, y)对
        polygon_points = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                polygon_points.append([coords[i], coords[i + 1]])
        
        # 转换为边界框
        bbox = convert_polygon_to_bbox(polygon_points)
        
        if bbox is None:
            print(f"警告: {label_path} 无法转换为边界框")
            return False
        
        # 写入修复后的标签
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        return True
        
    except Exception as e:
        print(f"错误处理文件 {label_path}: {e}")
        return False

def main():
    # 修复训练集标签
    train_labels_dir = Path("data/YOLO/labels/train")
    val_labels_dir = Path("data/YOLO/labels/val")
    test_labels_dir = Path("data/YOLO/labels/test")
    
    directories = [train_labels_dir, val_labels_dir, test_labels_dir]
    
    total_fixed = 0
    total_files = 0
    
    for label_dir in directories:
        if not label_dir.exists():
            print(f"目录不存在: {label_dir}")
            continue
        
        label_files = list(label_dir.glob("*.txt"))
        print(f"处理目录: {label_dir}, 找到 {len(label_files)} 个文件")
        
        for label_file in label_files:
            total_files += 1
            if fix_label_file(label_file):
                total_fixed += 1
    
    print(f"\n修复完成!")
    print(f"总共处理文件: {total_files}")
    print(f"成功修复文件: {total_fixed}")
    
    # 创建一个简单的测试标签文件来验证格式
    test_file = Path("data/YOLO/labels/test_fixed.txt")
    with open(test_file, 'w') as f:
        f.write("0 0.5 0.5 0.2 0.2\n")  # 示例格式
    
    print(f"\n创建了测试文件: {test_file}")
    print("YOLOv8标签格式应为: class_id x_center y_center width height")

if __name__ == "__main__":
    main()
