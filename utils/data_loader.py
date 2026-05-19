"""
数据加载器工具模块
为Faster R-CNN、SSD、RetinaNet、EfficientDet等模型创建统一的数据加载器
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
import yaml
from PIL import Image
import numpy as np


class COUUnderwaterDataset(Dataset):
    """COU水下目标检测数据集适配器"""
    
    def __init__(self, data_yaml_path, split='train', transform=None):
        """
        初始化数据集
        
        Args:
            data_yaml_path: YAML配置文件路径
            split: 数据集分割 ('train', 'val', 'test')
            transform: 数据转换
        """
        # 使用绝对路径避免相对路径问题
        abs_data_yaml_path = os.path.abspath(data_yaml_path)
        
        # 检查文件是否存在且可读
        if not os.path.exists(abs_data_yaml_path):
            raise FileNotFoundError(f"数据配置文件不存在: {abs_data_yaml_path}")
        
        if not os.access(abs_data_yaml_path, os.R_OK):
            raise PermissionError(f"无法读取数据配置文件（权限被拒绝）: {abs_data_yaml_path}")
        
        with open(abs_data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # 获取数据集路径
        base_dir = os.path.dirname(abs_data_yaml_path)
        
        # 获取指定分割的图像目录
        split_dir = data_config.get(f'{split}', '')
        if split_dir:
            # YOLO格式：split_dir是相对路径，如 'images/train'
            self.image_dir = os.path.join(base_dir, split_dir)
        else:
            # 默认使用images目录
            self.image_dir = os.path.join(base_dir, 'images')
        
        # 获取图像文件列表
        if os.path.exists(self.image_dir):
            self.image_files = [f for f in os.listdir(self.image_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        else:
            self.image_files = []
            print(f"警告: 图像目录不存在: {self.image_dir}")
        
        # 获取类别信息
        self.classes = data_config.get('names', [])
        self.num_classes = len(self.classes)
        
        # 标签目录：根据split确定对应的标签目录
        if split_dir:
            # 从split_dir中提取目录名，如从'images/train'提取'train'
            split_name = os.path.basename(split_dir)
            self.label_dir = os.path.join(base_dir, 'labels', split_name)
        else:
            self.label_dir = os.path.join(base_dir, 'labels')
        
        self.transform = transform
        self.split = split
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 获取原始图像尺寸
        original_width, original_height = image.size
        
        # 加载标签
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:
                    # 尝试解析标签数据
                    numbers = content.split()
                    
                    if len(numbers) >= 5:
                        # 检查是否是标准YOLO格式（每5个数字一组）
                        if len(numbers) % 5 == 0:
                            # 标准YOLO格式: class_id x_center y_center width height (归一化坐标0-1)
                            for i in range(0, len(numbers), 5):
                                try:
                                    class_id = int(numbers[i])
                                    x_center_norm = float(numbers[i+1])
                                    y_center_norm = float(numbers[i+2])
                                    width_norm = float(numbers[i+3])
                                    height_norm = float(numbers[i+4])
                                    
                                    # 将归一化坐标转换为绝对坐标
                                    x_center = x_center_norm * original_width
                                    y_center = y_center_norm * original_height
                                    width = width_norm * original_width
                                    height = height_norm * original_height
                                    
                                    # 转换为Faster R-CNN格式: x_min, y_min, x_max, y_max
                                    x_min = x_center - width / 2
                                    y_min = y_center - height / 2
                                    x_max = x_center + width / 2
                                    y_max = y_center + height / 2
                                    
                                    # 确保坐标在图像范围内
                                    x_min = max(0, min(x_min, original_width - 1))
                                    y_min = max(0, min(y_min, original_height - 1))
                                    x_max = max(0, min(x_max, original_width - 1))
                                    y_max = max(0, min(y_max, original_height - 1))
                                    
                                    # 确保宽度和高度为正数
                                    if x_max > x_min and y_max > y_min:
                                        boxes.append([x_min, y_min, x_max, y_max])
                                        labels.append(class_id + 1)  # Faster R-CNN需要类别ID从1开始（0是背景）
                                except (ValueError, IndexError):
                                    continue
                        else:
                            # 可能是多边形数据: class_id x1 y1 x2 y2 x3 y3 ...
                            # 从多边形数据计算边界框
                            i = 0
                            while i < len(numbers):
                                try:
                                    class_id = int(numbers[i])
                                    i += 1
                                    
                                    # 收集坐标
                                    coords = []
                                    while i < len(numbers) and '.' in numbers[i]:
                                        try:
                                            x_norm = float(numbers[i])
                                            y_norm = float(numbers[i+1])
                                            # 将归一化坐标转换为绝对坐标
                                            x = x_norm * original_width
                                            y = y_norm * original_height
                                            coords.extend([x, y])
                                            i += 2
                                        except (ValueError, IndexError):
                                            break
                                    
                                    # 从坐标计算边界框
                                    if len(coords) >= 4:  # 至少需要2个点
                                        x_coords = coords[0::2]  # 所有x坐标
                                        y_coords = coords[1::2]  # 所有y坐标
                                        
                                        x_min = min(x_coords)
                                        x_max = max(x_coords)
                                        y_min = min(y_coords)
                                        y_max = max(y_coords)
                                        
                                        # 确保坐标在图像范围内
                                        x_min = max(0, min(x_min, original_width - 1))
                                        y_min = max(0, min(y_min, original_height - 1))
                                        x_max = max(0, min(x_max, original_width - 1))
                                        y_max = max(0, min(y_max, original_height - 1))
                                        
                                        # 确保宽度和高度为正
                                        if x_max > x_min and y_max > y_min:
                                            boxes.append([x_min, y_min, x_max, y_max])
                                            labels.append(class_id + 1)  # Faster R-CNN需要类别ID从1开始
                                except (ValueError, IndexError):
                                    # 跳过无法解析的部分
                                    i += 1
        
        # 转换为张量
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # 应用转换
        if self.transform:
            # 保存原始图像用于计算缩放比例
            original_image = image.copy()
            image = self.transform(image)
            
            # 获取转换后的图像尺寸
            if hasattr(self.transform, 'transforms'):
                # 查找Resize转换
                for transform in self.transform.transforms:
                    if isinstance(transform, transforms.Resize):
                        target_size = transform.size
                        if isinstance(target_size, int):
                            target_width = target_size
                            target_height = target_size
                        else:
                            target_width, target_height = target_size
                        
                        # 计算缩放比例
                        width_scale = target_width / original_width
                        height_scale = target_height / original_height
                        
                        # 调整边界框坐标
                        if len(boxes) > 0:
                            boxes[:, 0] *= width_scale
                            boxes[:, 1] *= height_scale
                            boxes[:, 2] *= width_scale
                            boxes[:, 3] *= height_scale
                        
                        break
        
        # 返回图像、目标字典和图像路径
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.int64)
        }
        
        return image, target, img_path


def collate_fn(batch):
    """自定义collate函数，用于处理COUUnderwaterDataset返回的数据"""
    images = []
    targets = []
    
    for item in batch:
        if len(item) == 3:
            # 返回格式: (image, target, img_path)
            image, target, _ = item
        elif len(item) == 2:
            # 返回格式: (image, target)
            image, target = item
        else:
            # 未知格式，跳过
            continue
            
        images.append(image)
        targets.append(target)
    
    return images, targets


def create_data_loaders(data_yaml_path, batch_size=8, num_workers=2, pin_memory=True, prefetch_factor=2):
    """
    创建训练和验证数据加载器
    
    Args:
        data_yaml_path: YAML配置文件路径
        batch_size: 批次大小
        num_workers: 数据加载工作线程数
        pin_memory: 是否使用固定内存加速数据传输
        prefetch_factor: 数据预取因子，默认2以提前加载数据
        
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 数据转换 - SSD模型期望300x300输入
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = COUUnderwaterDataset(data_yaml_path, split='train', transform=transform)
    val_dataset = COUUnderwaterDataset(data_yaml_path, split='val', transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def get_dataset_info(data_yaml_path):
    """
    获取数据集信息
    
    Args:
        data_yaml_path: YAML配置文件路径
        
    Returns:
        dict: 包含数据集信息的字典
    """
    # 使用绝对路径避免相对路径问题
    abs_data_yaml_path = os.path.abspath(data_yaml_path)
    
    # 检查文件是否存在且可读
    if not os.path.exists(abs_data_yaml_path):
        raise FileNotFoundError(f"数据配置文件不存在: {abs_data_yaml_path}")
    
    if not os.access(abs_data_yaml_path, os.R_OK):
        raise PermissionError(f"无法读取数据配置文件（权限被拒绝）: {abs_data_yaml_path}")
    
    with open(abs_data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    return {
        'num_classes': len(data_config.get('names', [])),
        'class_names': data_config.get('names', []),
        'train_images': data_config.get('train', ''),
        'val_images': data_config.get('val', ''),
        'test_images': data_config.get('test', ''),
        'nc': data_config.get('nc', 0)
    }


if __name__ == '__main__':
    # 测试数据加载器
    data_path = 'data/YOLO/dataset.yaml'
    print(f"测试数据加载器: {data_path}")
    
    try:
        train_loader, val_loader = create_data_loaders(data_path, batch_size=2)
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        # 获取一个批次
        images, targets = next(iter(train_loader))
        print(f"批次图像数量: {len(images)}")
        print(f"图像形状: {images[0].shape}")
        print(f"目标数量: {len(targets)}")
        print(f"第一个目标的boxes形状: {targets[0]['boxes'].shape}")
        
        dataset_info = get_dataset_info(data_path)
        print(f"数据集类别数: {dataset_info['num_classes']}")
        print(f"类别名称: {dataset_info['class_names']}")
        
        print("数据加载器测试成功!")
        
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
