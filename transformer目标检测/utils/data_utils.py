import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET

class COCODataset(Dataset):
    """COCO数据集加载器"""
    
    def __init__(self, data_dir, split='train', transform=None, image_size=(640, 640)):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # 加载标注文件
        annotation_file = os.path.join(data_dir, f'annotations/instances_{split}2017.json')
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # 创建图像ID到图像信息的映射
        self.image_info = {img['id']: img for img in self.coco_data['images']}
        
        # 创建类别ID到类别名称的映射
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # 创建图像ID到标注的映射
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        self.image_ids = list(self.image_info.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_info[image_id]
        
        # 加载图像
        image_path = os.path.join(self.data_dir, f'{self.split}2017', image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取标注
        annotations = self.image_annotations.get(image_id, [])
        
        # 提取边界框和类别
        boxes = []
        labels = []
        for ann in annotations:
            # COCO格式: [x, y, width, height]
            bbox = ann['bbox']
            # 转换为 [x_min, y_min, x_max, y_max]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else []
            )
            image = transformed['image']
            if len(boxes) > 0:
                boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.tensor(transformed['labels'], dtype=torch.long)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'orig_size': torch.tensor([image_info['height'], image_info['width']])
        }
        
        return image, target

def get_transforms(train=True, image_size=(640, 640)):
    """获取数据增强变换"""
    if train:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.MotionBlur(blur_limit=3),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def collate_fn(batch):
    """自定义批次处理函数"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return torch.stack(images), targets

def get_data_loaders(config):
    """获取数据加载器"""
    train_transform = get_transforms(train=True, image_size=config.image_size)
    val_transform = get_transforms(train=False, image_size=config.image_size)
    
    train_dataset = COCODataset(
        config.data_dir, 
        split='train', 
        transform=train_transform,
        image_size=config.image_size
    )
    
    val_dataset = COCODataset(
        config.data_dir, 
        split='val', 
        transform=val_transform,
        image_size=config.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
