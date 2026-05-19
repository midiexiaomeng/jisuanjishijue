#!/usr/bin/env python3
"""
将COCO格式的标注转换为YOLO格式
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np

def convert_coco_to_yolo(coco_json_path, images_dir, output_labels_dir, class_names=None):
    """
    将COCO格式的标注转换为YOLO格式
    
    Args:
        coco_json_path: COCO格式的JSON标注文件路径
        images_dir: 图像目录路径
        output_labels_dir: 输出YOLO标签目录路径
        class_names: 类别名称列表，如果为None则从COCO文件中读取
        
    Returns:
        类别映射字典
    """
    print(f"读取COCO标注文件: {coco_json_path}")
    
    # 读取COCO标注文件
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # 解析类别信息
    categories = coco_data.get('categories', [])
    if class_names is None:
        # 从COCO文件中提取类别
        class_names = []
        category_id_to_name = {}
        category_id_to_yolo_id = {}
        
        for category in categories:
            category_id = category['id']
            category_name = category['name']
            category_id_to_name[category_id] = category_name
            class_names.append(category_name)
        
        # 为每个类别分配YOLO ID（从0开始）
        for idx, (category_id, category_name) in enumerate(category_id_to_name.items()):
            category_id_to_yolo_id[category_id] = idx
    else:
        # 使用提供的类别名称
        category_id_to_yolo_id = {}
        # 这里需要根据实际情况映射，假设COCO类别ID与提供的类别名称顺序一致
        for idx, category in enumerate(categories):
            if idx < len(class_names):
                category_id_to_yolo_id[category['id']] = idx
    
    print(f"找到 {len(categories)} 个类别")
    print(f"类别名称: {class_names}")
    
    # 创建图像ID到文件名的映射
    image_id_to_info = {}
    for image_info in coco_data.get('images', []):
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        image_id_to_info[image_id] = {
            'file_name': file_name,
            'width': width,
            'height': height
        }
    
    print(f"找到 {len(image_id_to_info)} 张图像")
    
    # 按图像分组标注
    image_annotations = {}
    for annotation in coco_data.get('annotations', []):
        image_id = annotation['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        
        # 获取边界框信息
        bbox = annotation['bbox']  # [x, y, width, height]
        category_id = annotation['category_id']
        
        # 转换为YOLO格式
        image_info = image_id_to_info.get(image_id)
        if image_info:
            width = image_info['width']
            height = image_info['height']
            
            # 计算YOLO格式的边界框（归一化坐标）
            x_center = (bbox[0] + bbox[2] / 2) / width
            y_center = (bbox[1] + bbox[3] / 2) / height
            bbox_width = bbox[2] / width
            bbox_height = bbox[3] / height
            
            # 获取YOLO类别ID
            yolo_class_id = category_id_to_yolo_id.get(category_id, -1)
            if yolo_class_id >= 0:
                image_annotations[image_id].append([
                    yolo_class_id,
                    x_center,
                    y_center,
                    bbox_width,
                    bbox_height
                ])
    
    print(f"处理了 {len(image_annotations)} 张图像的标注")
    
    # 写入YOLO格式的标签文件
    labels_created = 0
    for image_id, annotations in image_annotations.items():
        image_info = image_id_to_info.get(image_id)
        if not image_info:
            continue
        
        # 获取图像文件名（不带扩展名）
        image_file_name = Path(image_info['file_name']).stem
        label_file_path = os.path.join(output_labels_dir, f"{image_file_name}.txt")
        
        # 写入标签文件
        with open(label_file_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                line = f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n"
                f.write(line)
        
        labels_created += 1
    
    print(f"创建了 {labels_created} 个YOLO标签文件")
    
    # 保存类别映射
    class_map_path = os.path.join(output_labels_dir, "classes.txt")
    with open(class_map_path, 'w', encoding='utf-8') as f:
        for idx, class_name in enumerate(class_names):
            f.write(f"{class_name}\n")
    
    print(f"类别映射已保存到: {class_map_path}")
    
    return class_names

def main():
    parser = argparse.ArgumentParser(description='将COCO格式转换为YOLO格式')
    parser.add_argument('--coco_json', type=str, required=True,
                       help='COCO格式的JSON标注文件路径')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='图像目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出YOLO标签目录路径')
    parser.add_argument('--class_names', type=str, nargs='+',
                       help='类别名称列表（可选）')
    
    args = parser.parse_args()
    
    # 转换标注
    class_names = convert_coco_to_yolo(
        args.coco_json,
        args.images_dir,
        args.output_dir,
        args.class_names
    )
    
    print(f"\n转换完成！")
    print(f"YOLO标签保存在: {args.output_dir}")
    print(f"类别数量: {len(class_names)}")
    print(f"类别: {class_names}")

if __name__ == "__main__":
    main()
