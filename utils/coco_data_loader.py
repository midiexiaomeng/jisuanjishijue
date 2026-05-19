"""
COCO格式数据加载器工具模块
为Faster R-CNN、SSD、RetinaNet、EfficientDet等模型创建COCO格式的数据加载器
"""

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class COCOUnderwaterDataset(Dataset):
    """COCO格式水下目标检测数据集适配器"""
    
    def __init__(self, data_yaml_path, split='train', transform=None):
        """
        初始化COCO格式数据集
        
        Args:
            data_yaml_path: YAML配置文件路径
            split: 数据集分割 ('train', 'val', 'test')
            transform: 数据转换
        """
        import yaml
        
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # 获取数据集根目录
        base_dir = data_config.get('path', os.path.dirname(data_yaml_path))
        
        # 获取指定分割的标注文件路径
        if split == 'train':
            annotation_file = data_config.get('train_annotations', 'train_annotations.json')
        elif split == 'val':
            annotation_file = data_config.get('val_annotations', 'val_annotations.json')
        elif split == 'test':
            annotation_file = data_config.get('test_annotations', 'test_annotations.json')
        else:
            raise ValueError(f"不支持的split类型: {split}")
        
        # 构建完整的标注文件路径
        annotation_path = os.path.join(base_dir, annotation_file)
        
        # 加载COCO标注
        with open(annotation_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # 创建图像ID到图像信息的映射
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        
        # 创建图像ID到标注列表的映射
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)
        
        # 创建类别ID到类别名称的映射
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # 获取类别信息
        self.classes = data_config.get('names', [])
        self.num_classes = len(self.classes)
        
        # 图像目录
        self.image_dir = os.path.join(base_dir, 'images')
        
        # 所有图像ID列表
        self.image_ids = list(self.image_id_to_info.keys())
        
        self.transform = transform
        self.split = split
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # 获取图像ID
        image_id = self.image_ids[idx]
        
        # 获取图像信息
        img_info = self.image_id_to_info[image_id]
        img_filename = img_info['file_name']
        
        # 加载图像
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        
        # 获取该图像的所有标注
        annotations = self.image_id_to_annotations.get(image_id, [])
        
        boxes = []
        labels = []
        
        for ann in annotations:
            # COCO格式: bbox = [x_min, y_min, width, height]
            bbox = ann['bbox']
            
            # 确保bbox格式正确
            if len(bbox) == 4:
                x_min, y_min, width, height = bbox
                
                # 确保宽度和高度为正
                if width > 0 and height > 0:
                    boxes.append([x_min, y_min, width, height])
                    labels.append(ann['category_id'])
        
        # 转换为张量
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        # 返回图像、目标字典和图像路径
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': (boxes[:, 2] * boxes[:, 3]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.int64)
        }
        
        return image, target, img_path


def coco_collate_fn(batch):
    """自定义collate函数，用于处理COCOUnderwaterDataset返回的数据"""
    images = []
    targets = []
    
    for image, target, _ in batch:  # 忽略image_paths
        images.append(image)
        targets.append(target)
    
    return images, targets


def create_coco_data_loaders(data_yaml_path, batch_size=8, num_workers=2, pin_memory=True, prefetch_factor=2):
    """
    创建COCO格式的训练和验证数据加载器
    
    Args:
        data_yaml_path: YAML配置文件路径
        batch_size: 批次大小
        num_workers: 数据加载工作线程数
        pin_memory: 是否使用固定内存加速数据传输
        prefetch_factor: 数据预取因子，默认2以提前加载数据
        
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = COCOUnderwaterDataset(data_yaml_path, split='train', transform=transform)
    val_dataset = COCOUnderwaterDataset(data_yaml_path, split='val', transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=coco_collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=coco_collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def get_coco_dataset_info(data_yaml_path):
    """
    获取COCO数据集信息
    
    Args:
        data_yaml_path: YAML配置文件路径
        
    Returns:
        dict: 包含数据集信息的字典
    """
    import yaml
    
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    return {
        'num_classes': len(data_config.get('names', [])),
        'class_names': data_config.get('names', []),
        'path': data_config.get('path', ''),
        'train_annotations': data_config.get('train_annotations', 'train_annotations.json'),
        'val_annotations': data_config.get('val_annotations', 'val_annotations.json'),
        'test_annotations': data_config.get('test_annotations', 'test_annotations.json'),
        'nc': data_config.get('nc', 0)
    }


if __name__ == '__main__':
    # 测试COCO数据加载器
    data_path = 'data/coco/coco.yaml'
    print(f"测试COCO数据加载器: {data_path}")
    
    try:
        train_loader, val_loader = create_coco_data_loaders(data_path, batch_size=2)
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        # 获取一个批次
        images, targets = next(iter(train_loader))
        print(f"批次图像数量: {len(images)}")
        print(f"图像形状: {images[0].shape}")
        print(f"目标数量: {len(targets)}")
        if len(targets) > 0:
            print(f"第一个目标的boxes形状: {targets[0]['boxes'].shape}")
            print(f"第一个目标的labels: {targets[0]['labels']}")
        
        dataset_info = get_coco_dataset_info(data_path)
        print(f"数据集类别数: {dataset_info['num_classes']}")
        print(f"类别名称: {dataset_info['class_names']}")
        
        print("COCO数据加载器测试成功!")
        
    except Exception as e:
        print(f"COCO数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
