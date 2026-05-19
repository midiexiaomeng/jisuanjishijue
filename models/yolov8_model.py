"""
YOLOv8模型实现
基于Ultralytics YOLOv8的水下目标检测模型
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import os


class YOLOv8Detector:
    """YOLOv8目标检测器"""
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        """
        初始化YOLOv8检测器
        
        Args:
            model_path: 预训练模型路径，如果为None则使用默认模型
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # 使用预训练的YOLOv8n模型
            self.model = YOLO('yolov8n.pt')
        
        # 设置设备
        self.model.to(self.device)
        
        # COU水下数据集类别（24类人造物）- 来自data/coco/coco.yaml文件
        self.cou_class_names = [
            'plastic_bottle', 'plastic_bag', 'fishing_net', 'rope', 'can', 'glass_bottle', 'tire', 'metal_scrap',
            'wood', 'cloth', 'diver', 'diving_mask', 'diving_fins', 'oxygen_tank', 'underwater_camera', 'auv',
            'rov', 'underwater_drone', 'sonar', 'underwater_sensor', 'ship_wreck', 'anchor', 'propeller', 'underwater_structure'
        ]
        
        # 默认使用COU类别
        self.class_names = self.cou_class_names
        
        # 颜色映射
        self.colors = self._generate_colors(len(self.class_names))
        
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """生成用于可视化的颜色"""
        np.random.seed(42)
        colors = []
        for i in range(num_classes):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def train(self, 
              data_yaml: str,
              epochs: int = 100,
              batch_size: int = 16,
              img_size: int = 640,
              save_dir: str = 'checkpoints/yolov8/',
              project: str = 'runs/detect',
              name: str = 'train'):
        """
        训练YOLOv8模型
        
        Args:
            data_yaml: 数据集YAML文件路径
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 图像尺寸
            save_dir: 保存目录
            project: 项目目录
            name: 训练名称
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练参数 - 使用有效的YOLOv8参数
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': self.device,
            'project': project,
            'name': name,
            'save': True,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'degrees': 0.0,
            'translate': 0.1,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }
        
        # 开始训练
        results = self.model.train(**train_args)
        
        return results
    
    def detect(self, 
               image: np.ndarray,
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45,
               max_det: int = 300) -> Dict[str, Any]:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            max_det: 最大检测数量
            
        Returns:
            检测结果字典
        """
        # 执行推理
        results = self.model(image, 
                            conf=conf_threshold,
                            iou=iou_threshold,
                            max_det=max_det,
                            device=self.device)
        
        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 获取置信度和类别
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # 确保class_id在有效范围内
                    if class_id >= len(self.class_names):
                        class_name = f'class_{class_id}'
                    else:
                        class_name = self.class_names[class_id]
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        return {
            'detections': detections,
            'original_image': image,
            'processed_image': self._draw_detections(image, detections)
        }
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """在图像上绘制检测结果"""
        result_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            confidence = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # 确保class_id在有效范围内
            if class_id >= len(self.colors):
                color = (0, 255, 0)  # 默认绿色
            else:
                color = self.colors[class_id]
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f'{class_name}: {confidence:.2f}'
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result_image, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            cv2.putText(result_image, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def evaluate(self, 
                 data_yaml: str,
                 batch_size: int = 16,
                 img_size: int = 640) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            data_yaml: 数据集YAML文件路径
            batch_size: 批次大小
            img_size: 图像尺寸
            
        Returns:
            评估指标字典
        """
        # 执行评估
        metrics = self.model.val(data=data_yaml,
                                batch=batch_size,
                                imgsz=img_size,
                                device=self.device,
                                save_json=True,
                                save_hybrid=True,
                                conf=0.001,
                                iou=0.6,
                                max_det=300,
                                half=True,
                                dnn=False,
                                plots=True)
        
        # 提取关键指标
        results = {
            'map50': metrics.box.map50,
            'map': metrics.box.map,
            'precision': metrics.box.p,
            'recall': metrics.box.r,
            'f1_score': 2 * (metrics.box.p * metrics.box.r) / (metrics.box.p + metrics.box.r + 1e-16),
            'metrics': metrics
        }
        
        return results
    
    def export(self, format: str = 'onnx', imgsz: int = 640) -> str:
        """
        导出模型
        
        Args:
            format: 导出格式 ('onnx', 'torchscript', 'tflite', 'coreml', 'saved_model')
            imgsz: 图像尺寸
            
        Returns:
            导出模型路径
        """
        # 支持的格式
        supported_formats = ['onnx', 'torchscript', 'tflite', 'coreml', 'saved_model']
        if format not in supported_formats:
            raise ValueError(f"不支持的格式: {format}. 支持的格式: {supported_formats}")
        
        # 导出模型
        export_path = self.model.export(format=format, imgsz=imgsz)
        
        return export_path
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'colors': self.colors
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint.get('class_names', self.class_names)
        self.colors = checkpoint.get('colors', self.colors)


def create_data_yaml(data_dir: str, save_path: str = 'data/YOLO/dataset.yaml'):
    """
    创建YOLOv8训练所需的YAML文件
    
    Args:
        data_dir: 数据目录
        save_path: 保存路径
    """
    yaml_content = f"""# COU水下目标检测数据集
# 24类人造物

path: {data_dir}  # 数据集根目录
train: images/train  # 训练图像目录
val: images/val      # 验证图像目录
test: images/test    # 测试图像目录

# 类别数量
nc: 24

# 类别名称
names: ['plastic_bottle', 'plastic_bag', 'fishing_net', 'rope', 'can', 'glass_bottle', 'tire', 'metal_scrap', 'wood', 'cloth', 'diver', 'diving_mask', 'diving_fins', 'oxygen_tank', 'underwater_camera', 'auv', 'rov', 'underwater_drone', 'sonar', 'underwater_sensor', 'ship_wreck', 'anchor', 'propeller', 'underwater_structure']
"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    return save_path


if __name__ == "__main__":
    # 测试代码
    detector = YOLOv8Detector()
    print("YOLOv8检测器初始化成功")
    
    # 创建数据YAML文件
    yaml_path = create_data_yaml('data/YOLO', 'data/YOLO/dataset.yaml')
    print(f"数据YAML文件创建成功: {yaml_path}")
    
    # 测试检测
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(test_image)
    print(f"检测完成，找到 {len(results['detections'])} 个目标")
