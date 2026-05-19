# 水下目标检测项目详细技术文档

## 项目概述

本项目是一个完整的水下目标检测系统，集成了YOLOv8和Faster R-CNN两种主流目标检测算法，并提供了基于PyQt5的图形用户界面。系统专门针对水下环境优化，支持COU（Coral Underwater）数据集，包含24类人造物目标检测。

## 目录结构详解

```
d:/计算机组成原理文件/计算机视觉/目标检测/
├── main_gui.py                    # GUI主程序入口
├── train_yolov8.py               # YOLOv8训练脚本
├── train_faster_rcnn_optimized.py # 优化后的Faster R-CNN训练脚本
├── requirements.txt              # 项目依赖
├── README.md                     # 项目说明
├── TRAINING_FIXES_AND_TOOLS_GUIDE.md # 训练修复和工具指南
├── check_dataset.py              # 数据集检查工具
├── convert_coco_to_yolo.py       # COCO到YOLO格式转换
├── fix_yolo_labels.py            # YOLO标签修复工具
├── optimize_faster_rcnn_training.py # Faster R-CNN训练优化
├── process_manager.py            # 进程管理工具
├── pause_process.py              # 进程暂停工具
├── resume_process.py             # 进程恢复工具
├── test_*.py                     # 各种测试脚本
├── yolo11n.pt                    # YOLO11预训练权重
├── yolov8n.pt                    # YOLOv8预训练权重
├── checkpoints/                  # 模型检查点目录
│   ├── yolov8/                   # YOLOv8检查点
│   ├── faster r-cnn/             # Faster R-CNN检查点
│   ├── faster_rcnn/              # Faster R-CNN检查点（备用）
│   ├── faster_rcnn_test/         # Faster R-CNN测试检查点
│   ├── efficientdet/             # EfficientDet检查点
│   └── retinanet/                # RetinaNet检查点
├── config/                       # 配置文件目录
│   ├── training_config.json      # 训练配置
│   └── optimized_training_config.json # 优化训练配置
├── data/                         # 数据集目录
│   ├── coco/                     # COCO格式数据集
│   │   ├── coco.yaml             # COCO数据集配置
│   │   ├── train_annotations.json # 训练标注
│   │   ├── val_annotations.json  # 验证标注
│   │   ├── test_annotations.json # 测试标注
│   │   ├── train.txt             # 训练集文件列表
│   │   ├── val.txt               # 验证集文件列表
│   │   ├── test.txt              # 测试集文件列表
│   │   ├── images/               # 图像文件
│   │   └── labels/               # 标签文件
│   └── YOLO/                     # YOLO格式数据集
│       ├── dataset.yaml          # YOLO数据集配置
│       ├── images/               # 图像文件
│       └── labels/               # 标签文件
├── gui/                          # GUI界面代码
│   ├── main_window.py           # 主窗口实现
│   └── main_window_backup.py    # 主窗口备份
├── models/                       # 模型实现
│   ├── yolov8_model.py          # YOLOv8模型封装
│   ├── faster_rcnn_model.py     # Faster R-CNN模型封装
│   ├── faster_rcnn_new_fixed.py # 修复的Faster R-CNN模型
│   ├── faster_rcnn_new.py       # 新版Faster R-CNN模型
│   ├── efficientdet_model.py    # EfficientDet模型
│   ├── ssd_model.py             # SSD模型
│   └── retinanet_model.py       # RetinaNet模型
├── utils/                        # 工具函数
│   ├── data_loader.py           # 通用数据加载器
│   └── coco_data_loader.py      # COCO数据加载器
├── logs/                         # 日志目录
├── results/                      # 训练结果
│   ├── yolov8_training_results.json # YOLOv8训练结果
│   └── faster r-cnn_training_results.json # Faster R-CNN训练结果
└── runs/                         # Ultralytics运行目录
    ├── detect/                   # 检测结果
    └── detect_test/              # 测试检测结果
```

## 1. YOLOv8模型详细实现

### 1.1 模型架构

YOLOv8（You Only Look Once version 8）是Ultralytics开发的最新一代实时目标检测器，采用单阶段检测架构，具有以下特点：

1. **骨干网络（Backbone）**：CSPDarknet53架构，包含Cross Stage Partial连接
2. **颈部网络（Neck）**：PAN-FPN（Path Aggregation Network + Feature Pyramid Network）
3. **检测头（Head）**：解耦头，分离分类和回归任务
4. **损失函数**：TaskAlignedAssigner + Distribution Focal Loss

### 1.2 代码实现：models/yolov8_model.py

```python
import torch
import numpy as np
from ultralytics import YOLO
import os
import json
from typing import Dict, List, Tuple, Optional, Union
import cv2

class YOLOv8Detector:
    """
    YOLOv8目标检测器封装类
    专门针对水下目标检测优化，支持COU数据集（24类人造物）
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cuda'):
        """
        初始化YOLOv8检测器
        
        参数:
            model_path: 模型权重路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.model = None
        self.class_names = []
        self.model_path = model_path
        
        # COU水下数据集类别（24类人造物）
        self.cou_classes = [
            'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
            'train', 'tvmonitor', 'bird', 'book', 'building', 
            'clock', 'cup'
        ]
    
    def load_model(self, model_path: Optional[str] = None):
        """
        加载YOLOv8模型
        
        参数:
            model_path: 可选的模型权重路径
        """
        if model_path is None:
            model_path = self.model_path
            
        print(f"加载YOLOv8模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 设置设备
        if self.device == 'cuda':
            self.model.to('cuda')
            
        print(f"模型加载完成，设备: {self.device}")
    
    def train(self, data_yaml: str, epochs: int = 50, batch_size: int = 16, 
              imgsz: int = 640, save_dir: str = 'runs/detect/train'):
        """
        训练YOLOv8模型
        
        参数:
            data_yaml: 数据YAML文件路径
            epochs: 训练轮数
            batch_size: 批次大小
            imgsz: 图像尺寸
            save_dir: 保存目录
            
        返回:
            dict: 训练结果
        """
        if self.model is None:
            self.load_model()
        
        print(f"开始YOLOv8训练...")
        print(f"数据配置: {data_yaml}")
        print(f"训练轮数: {epochs}, 批次大小: {batch_size}, 图像尺寸: {imgsz}")
        
        # 训练模型
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=os.path.dirname(save_dir),
            name=os.path.basename(save_dir),
            save=True,
            save_period=5,
            device=self.device,
            workers=4,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        # 获取训练结果
        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        
        # 保存训练结果到JSON文件
        results_file = os.path.join(save_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"训练完成，结果保存到: {results_file}")
        return metrics
    
    def detect(self, image_path: Union[str, np.ndarray], 
               conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        执行目标检测
        
        参数:
            image_path: 图像路径或numpy数组
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        返回:
            tuple: (检测结果图像, 检测框列表, 置信度列表, 类别列表)
        """
        if self.model is None:
            self.load_model()
        
        # 执行检测
        results = self.model(
            image_path, 
            conf=conf_threshold, 
            iou=iou_threshold,
            device=self.device
        )
        
        # 解析结果
        detections = []
        confidences = []
        class_ids = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append([x1, y1, x2, y2])
                    confidences.append(float(conf))
                    class_ids.append(cls_id)
        
        # 获取带标注的图像
        annotated_image = results[0].plot() if len(results) > 0 else None
        
        return annotated_image, detections, confidences, class_ids
    
    def evaluate(self, data_yaml: str, batch_size: int = 16, imgsz: int = 640):
        """
        评估模型性能
        
        参数:
            data_yaml: 数据YAML文件路径
            batch_size: 批次大小
            imgsz: 图像尺寸
            
        返回:
            dict: 评估指标
        """
        if self.model is None:
            self.load_model()
        
        print(f"开始模型评估...")
        
        # 执行评估
        metrics = self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            save_json=True,
            save_hybrid=True,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=True,
            dnn=False,
            plots=True
        )
        
        # 获取评估结果
        eval_results = {
            'mAP50': metrics.box.map50 if hasattr(metrics.box, 'map50') else 0.0,
            'mAP50-95': metrics.box.map if hasattr(metrics.box, 'map') else 0.0,
            'precision': metrics.box.mp if hasattr(metrics.box, 'mp') else 0.0,
            'recall': metrics.box.mr if hasattr(metrics.box, 'mr') else 0.0,
            'f1_score': 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-16)
        }
        
        print(f"评估完成:")
        print(f"  mAP@0.5: {eval_results['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {eval_results['mAP50-95']:.4f}")
        print(f"  精度: {eval_results['precision']:.4f}")
        print(f"  召回率: {eval_results['recall']:.4f}")
        print(f"  F1分数: {eval_results['f1_score']:.4f}")
        
        return eval_results
    
    def export(self, format: str = 'onnx', imgsz: int = 640):
        """
        导出模型到指定格式
        
        参数:
            format: 导出格式 ('onnx', 'torchscript', 'tflite', 'coreml')
            imgsz: 图像尺寸
        """
        if self.model is None:
            self.load_model()
        
        print(f"导出模型为 {format.upper()} 格式...")
        
        # 导出模型
        export_path = self.model.export(
            format=format,
            imgsz=imgsz,
            device=self.device,
            half=True,
            simplify=True,
            opset=12
        )
        
        print(f"模型导出完成: {export_path}")
        return export_path
```

### 1.3 训练脚本：train_yolov8.py

```python
import json
import os
import sys
from models.yolov8_model import YOLOv8Detector

def create_data_yaml(config):
    """
    创建YOLO数据YAML文件
    
    参数:
        config: 训练配置字典
        
    返回:
        str: YAML文件路径
    """
    data_yaml = {
        'path': config['data_path'],
        'train': 'train.txt',
        'val': 'val.txt',
        'test': 'test.txt',
        'nc': config['num_classes'],
        'names': config['class_names']
    }
    
    # 创建YAML文件
    yaml_path = os.path.join(config['data_path'], 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"test: {data_yaml['test']}\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write(f"names: {data_yaml['names']}\n")
    
    return yaml_path

def train_yolov8(config_path='config/training_config.json'):
    """
    主训练函数
    
    参数:
        config_path: 配置文件路径
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=" * 50)
    print("YOLOv8训练配置:")
    print(f"  数据路径: {config['data_path']}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  图像尺寸: {config['imgsz']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  设备: {config['device']}")
    print("=" * 50)
    
    # 创建数据YAML文件
    data_yaml = create_data_yaml(config)
    print(f"创建数据YAML文件: {data_yaml}")
    
    # 初始化检测器
    detector = YOLOv8Detector(
        model_path=config.get('model_path', 'yolov8n.pt'),
        device=config['device']
    )
    
    # 训练模型
    results = detector.train(
        data_yaml=data_yaml,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        imgsz=config['imgsz'],
        save_dir=config['save_dir']
    )
    
    # 保存训练结果
    results_file = os.path.join(config['save_dir'], 'yolov8_training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"训练完成，结果保存到: {results_file}")
    return results

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/training_config.json'
    train_yolov8(config_path)
```

## 2. Faster R-CNN模型详细实现

### 2.1 模型架构

Faster R-CNN（Region-based Convolutional Neural Networks）是两阶段目标检测器，具有以下特点：

1. **骨干网络（Backbone）**：ResNet18/50，提取图像特征
2. **区域提议网络（RPN）**：生成候选区域（Region Proposals）
3. **RoI池化（RoI Pooling）**：将不同大小的候选区域转换为固定大小的特征图
4. **检测头（Detection Head）**：分类
