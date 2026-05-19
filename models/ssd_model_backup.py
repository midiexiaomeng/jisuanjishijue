import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.ssd import SSD, SSDClassificationHead
from torchvision.models.detection import _utils as det_utils
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import cv2
import os
import sys
import multiprocessing

# 在Windows上设置多进程启动方法为'spawn'，避免fork相关问题
if sys.platform == 'win32':
    multiprocessing.set_start_method('spawn', force=True)


class SSDDetector:
    """SSD (Single Shot MultiBox Detector) 目标检测模型"""
    
    def __init__(self, num_classes: int = 25, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化SSD检测器
        
        Args:
            num_classes: 类别数量（包括背景），COU数据集为24类+背景=25
            device: 设备类型
        """
        self.device = device
        self.num_classes = num_classes
        
        # COU数据集类别（24类人造物）
        self.cou_class_names = [
            'plastic_bottle', 'plastic_bag', 'fishing_net', 'rope', 'can', 
            'glass_bottle', 'tire', 'metal_scrap', 'wood', 'cloth',
            'diver', 'diving_mask', 'diving_fins', 'oxygen_tank', 'underwater_camera',
            'auv', 'rov', 'underwater_drone', 'sonar', 'underwater_sensor',
            'ship_wreck', 'anchor', 'propeller', 'underwater_structure'
        ]
        
        # 默认使用COU类别
        self.class_names = self.cou_class_names
        
        # 颜色映射
        self.colors = self._generate_colors(len(self.class_names))
        
        # 初始化模型
        self.model = None
        self._init_model()
        
        # 训练状态
        self.is_trained = False
        self.current_epoch = 0
        self.total_epochs = 0
        
    def _init_model(self):
        """初始化SSD模型"""
        try:
            # 加载预训练的MobileNetV3骨干网络
            backbone = torchvision.models.mobilenet_v3_large(pretrained=True)
            
            # 获取骨干网络的特征提取器
            backbone = backbone.features
            
            # 获取骨干网络的输出通道数
            backbone_out_channels = det_utils.retrieve_out_channels(backbone, (300, 300))
            
            # 定义锚点生成器
            anchor_generator = torchvision.models.detection.ssd.DefaultBoxGenerator(
                aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                min_ratio=0.2,
                max_ratio=0.9
            )
            
            # 获取每个位置的锚点数量列表
            num_anchors = anchor_generator.num_anchors_per_location()
            
            # 扩展输出通道数列表以匹配锚点生成器的特征层数量
            # 对于MobileNetV3，我们需要将单个输出通道数扩展为与锚点数量列表长度相同
            if isinstance(backbone_out_channels, list) and len(backbone_out_channels) == 1:
                backbone.out_channels = backbone_out_channels * len(num_anchors)
            else:
                backbone.out_channels = backbone_out_channels
            
            # 定义锚点生成器和头
            num_anchors = anchor_generator.num_anchors_per_location()
            
            # 使用torchvision内置的head构造器创建完整的head（包含分类头和回归头）
            self.model = SSD(
                backbone=backbone,
                anchor_generator=anchor_generator,
                size=(300, 300),
                num_classes=self.num_classes,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                score_thresh=0.01,
                nms_thresh=0.45,
                detections_per_img=200,
                topk_candidates=400
            )
            
            # 移动到设备
            self.model.to(self.device)
            
            print(f"SSD模型初始化完成，使用设备: {self.device}")
            
        except Exception as e:
            print(f"初始化SSD模型失败: {e}")
            # 回退到简单的模型
            self._init_simple_model()
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """生成用于可视化的颜色"""
        np.random.seed(42)
        colors = []
        for i in range(num_classes):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def _init_simple_model(self):
        """初始化简单的SSD模型（回退方案）"""
        try:
            # 加载预训练的SSD模型
            self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
            
            # 替换分类头以适应我们的类别数量
            in_channels = det_utils.retrieve_out_channels(self.model.backbone, (300, 300))
            num_anchors = self.model.anchor_generator.num_anchors_per_location()
            
            # 创建新的分类头
            self.model.head.classification_head = SSDClassificationHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=self.num_classes
            )
            
            # 移动到设备
            self.model.to(self.device)
            
            print(f"简单SSD模型初始化完成，使用设备: {self.device}")
            
        except Exception as e:
            print(f"初始化简单SSD模型失败: {e}")
            raise
    
    def load_pretrained(self, model_path: str):
        """
        加载预训练权重
        
        Args:
            model_path: 模型权重文件路径
        """
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                print(f"从 {model_path} 加载预训练权重成功")
                self.is_trained = True
                
                # 加载训练状态
                if 'epoch' in checkpoint:
                    self.current_epoch = checkpoint['epoch']
                if 'total_epochs' in checkpoint:
                    self.total_epochs = checkpoint['total_epochs']
                    
            else:
                print(f"模型文件 {model_path} 不存在，使用随机初始化权重")
                
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    def _detect_dataset_type(self, data_yaml_path):
        """
        检测数据集类型（YOLO或COCO）
        
        Args:
            data_yaml_path: 数据YAML文件路径
            
        Returns:
            str: 数据集类型 ('yolo' 或 'coco')
        """
        import yaml
        
        try:
            # 使用绝对路径避免相对路径问题
            abs_path = os.path.abspath(data_yaml_path)
            
            # 检查文件是否存在且可读
            if not os.path.exists(abs_path):
                print(f"警告: 数据文件不存在: {abs_path}")
                return 'yolo'
            
            # 检查文件权限
            if not os.access(abs_path, os.R_OK):
                print(f"警告: 无法读取数据文件（权限被拒绝）: {abs_path}")
                return 'yolo'
            
            with open(abs_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            # 检查是否包含COCO特定的字段
            if 'train_annotations' in data_config or 'val_annotations' in data_config:
                return 'coco'
            
            # 检查路径是否包含'coco'关键字
            if 'coco' in data_yaml_path.lower():
                return 'coco'
            
            # 默认返回'yolo'
            return 'yolo'
            
        except Exception as e:
            print(f"检测数据集类型失败: {e}")
            # 默认返回'yolo'
            return 'yolo'
    
    def train(self, train_loader=None, val_loader=None, epochs: int = 10, 
              learning_rate: float = 0.001, save_dir: str = 'checkpoints/ssd',
              data_path: str = None, batch_size: int = 8, num_workers: int = 2, pin_memory: bool = True, 
              use_amp: bool = True, gradient_accumulation_steps: int = 1):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器（如果为None，则使用data_path创建）
            val_loader: 验证数据加载器（可选，如果为None且data_path不为None，则从data_path创建）
            epochs: 训练轮数
            learning_rate: 学习率
            save_dir: 保存目录
            data_path: 数据YAML文件路径（如果train_loader为None则必需）
            batch_size: 批次大小（如果使用data_path）
            num_workers: 数据加载工作线程数（如果使用data_path）
            pin_memory: 是否使用固定内存加速数据传输
            use_amp: 是否使用自动混合精度训练
            gradient_accumulation_steps: 梯度累积步数，用于在有限GPU内存下模拟更大的批次大小
            
        Returns:
            dict: 训练历史记录
        """
        # 如果提供了data_path但没有提供train_loader，则创建数据加载器
        if train_loader is None:
            if data_path is None:
                raise ValueError("必须提供train_loader或data_path参数")
            
            # 检测数据集类型并选择相应的数据加载器
            try:
                # 首先尝试导入YOLO数据加载器
                from utils.data_loader import create_data_loaders as create_yolo_data_loaders
                # 然后尝试导入COCO数据加载器
                from utils.coco_data_loader import create_coco_data_loaders as create_coco_data_loaders
                
                # 检测数据集类型
                dataset_type = self._detect_dataset_type(data_path)
                print(f"检测到数据集类型: {dataset_type}")
                
                if dataset_type == 'coco':
                    # 使用COCO数据加载器
                    train_loader, val_loader = create_coco_data_loaders(
                        data_path, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        pin_memory=pin_memory
                    )
                    print(f"从 {data_path} 创建COCO数据加载器成功")
                else:
                    # 默认使用YOLO数据加载器
                    train_loader, val_loader = create_yolo_data_loaders(
                        data_path, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        pin_memory=pin_memory
                    )
                    print(f"从 {data_path} 创建YOLO数据加载器成功")
                    
            except ImportError as e:
                print(f"导入数据加载器失败: {e}")
                print("请确保utils/data_loader.py和utils/coco_data_loader.py文件存在")
                raise
            except Exception as e:
                print(f"创建数据加载器失败: {e}")
                raise
        
        # 如果val_loader为None且没有从data_path创建，则设置为None
        if val_loader is None:
            print("警告: 未提供验证数据加载器，将只进行训练")
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置模型为训练模式
        self.model.train()
        
        # 定义优化器
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        
        # 学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # 初始化混合精度训练
        if use_amp and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            print("启用自动混合精度训练")
        else:
            scaler = None
            print("禁用自动混合精度训练")
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mAP': [],
            'val_mAP': []
        }
        
        print(f"开始训练SSD模型，共{epochs}轮...")
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            self.total_epochs = epochs
            
            # 训练阶段
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                # 移动到设备
                images = [img.to(self.device, non_blocking=True) for img in images]
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                # 前向传播
                if use_amp and scaler is not None:
                    with torch.amp.autocast('cuda'):
                        loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        # 梯度累积需要将损失除以累积步数
                        losses = losses / gradient_accumulation_steps
                else:
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    # 梯度累积需要将损失除以累积步数
                    losses = losses / gradient_accumulation_steps
                
                # 反向传播
                if use_amp and scaler is not None:
                    scaler.scale(losses).backward()
                else:
                    losses.backward()
                
                # 只在累积到指定步数时更新权重
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True更高效地释放内存
                
                # 记录损失（使用原始损失值，不除以累积步数）
                train_loss += losses.item() * gradient_accumulation_steps
                train_batches += 1
                
                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                          f"Loss: {losses.item() * gradient_accumulation_steps:.4f}")
            
            # 计算平均训练损失
            avg_train_loss = train_loss / max(train_batches, 1)
            history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            if val_loader is not None:
                val_loss, val_mAP = self.evaluate(val_loader, use_amp=use_amp)
                history['val_loss'].append(val_loss)
                history['val_mAP'].append(val_mAP)
                
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val mAP: {val_mAP:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
            
            # 更新学习率
            lr_scheduler.step()
            
            # 保存检查点
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                checkpoint_path = os.path.join(save_dir, f'ssd_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'total_epochs': epochs
                }, checkpoint_path)
                print(f"检查点已保存到 {checkpoint_path}")
        
        # 训练完成
        self.is_trained = True
        print("SSD模型训练完成")
        
        return history
    
    def evaluate(self, data_loader, use_amp: bool = True):
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            use_amp: 是否使用自动混合精度训练
            
        Returns:
            tuple: (平均损失, mAP分数)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_batches = 0
        
        # 用于计算mAP的列表
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                # 移动到设备
                images = [img.to(self.device, non_blocking=True) for img in images]
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                # 前向传播 - 评估模式下需要区分损失计算和预测
                if use_amp and torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        # 训练模式下调用model(images, targets)返回损失字典
                        # 评估模式下调用model(images, targets)返回预测结果
                        # 因此需要先保存当前模式，切换到训练模式计算损失，再切换回评估模式
                        original_mode = self.model.training
                        self.model.train()
                        loss_dict = self.model(images, targets)
                        self.model.eval()
                        losses = sum(loss for loss in loss_dict.values())
                else:
                    # 训练模式下调用model(images, targets)返回损失字典
                    # 评估模式下调用model(images, targets)返回预测结果
                    # 因此需要先保存当前模式，切换到训练模式计算损失，再切换回评估模式
                    original_mode = self.model.training
                    self.model.train()
                    loss_dict = self.model(images, targets)
                    self.model.eval()
                    losses = sum(loss for loss in loss_dict.values())
                
                # 记录损失
                total_loss += losses.item()
                total_batches += 1
                
                # 获取预测结果用于mAP计算
                if use_amp and torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        predictions = self.model(images)
                else:
                    predictions = self.model(images)
                
                # 收集预测和目标
                for i in range(len(predictions)):
                    pred_boxes = predictions[i]['boxes'].cpu().numpy()
                    pred_scores = predictions[i]['scores'].cpu().numpy()
                    pred_labels = predictions[i]['labels'].cpu().numpy()
                    
                    target_boxes = targets[i]['boxes'].cpu().numpy()
                    target_labels = targets[i]['labels'].cpu().numpy()
                    
                    all_predictions.append({
                        'boxes': pred_boxes,
                        'scores': pred_scores,
                        'labels': pred_labels
                    })
                    
                    all_targets.append({
                        'boxes': target_boxes,
                        'labels': target_labels
                    })
        
        # 计算平均损失
        avg_loss = total_loss / max(total_batches, 1)
        
        # 计算mAP（简化版本）
        mAP = self._calculate_simple_map(all_predictions, all_targets)
        
        # 恢复训练模式
        self.model.train()
        
        return avg_loss, mAP
    
    def _calculate_simple_map(self, predictions, targets, iou_threshold: float = 0.5):
        """
        计算简化的mAP
        
        Args:
            predictions: 预测结果列表
            targets: 目标列表
            iou_threshold: IoU阈值
            
        Returns:
            float: mAP分数
        """
        if not predictions or not targets:
            return 0.0
        
        # 简化的mAP计算
        total_precision = 0.0
        total_recall = 0.0
        num_samples = len(predictions)
        
        for i in range(num_samples):
            pred = predictions[i]
            target = targets[i]
            
            if len(pred['boxes']) == 0 or len(target['boxes']) == 0:
                continue
            
            # 计算每个预测的IoU
            tp = 0
            fp = 0
            fn = 0
            
            for j, pred_box in enumerate(pred['boxes']):
                max_iou = 0.0
                matched = False
                
                for k, target_box in enumerate(target['boxes']):
                    iou = self._calculate_iou(pred_box, target_box)
                    if iou > max_iou:
                        max_iou = iou
                
                if max_iou >= iou_threshold:
                    tp += 1
                else:
                    fp += 1
            
            fn = max(0, len(target['boxes']) - tp)
            
            # 计算精度和召回率
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            
            total_precision += precision
            total_recall += recall
        
        avg_precision = total_precision / max(num_samples, 1)
        avg_recall = total_recall / max(num_samples, 1)
        
        # 计算F1分数作为简化的mAP
        if avg_precision + avg_recall > 0:
            f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            f1_score = 0.0
        
        return f1_score
    
    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU
        
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            float: IoU值
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 计算交集面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # 计算IoU
        iou = intersection / union if union > 0 else 0.0
        
        return iou
    
    def detect(self, image, conf_threshold: float = 0.5, draw_boxes: bool = False, iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像（numpy数组或PIL图像）
            conf_threshold: 置信度阈值
            draw_boxes: 是否绘制边界框
            iou_threshold: IoU阈值，用于非极大值抑制
            
        Returns:
            dict: 检测结果，包含检测列表和处理后的图像
        """
        self.model.eval()
        
        # 保存原始图像用于绘制
        original_image = None
        if isinstance(image, np.ndarray):
            original_image = image.copy()
            # OpenCV BGR转RGB用于模型输入
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            # 转换为PIL图像
            from PIL import Image
            image = Image.fromarray(image_rgb)
        else:
            # 如果是PIL图像，转换为numpy数组用于绘制
            original_image = np.array(image)
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        
        # 转换为张量
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((300, 300)),
            torchvision.transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 临时修改模型的NMS阈值
            original_nms_thresh = self.model.nms_thresh
            self.model.nms_thresh = iou_threshold
            
            # 进行预测
            predictions = self.model(image_tensor)
            
            # 恢复原始NMS阈值
            self.model.nms_thresh = original_nms_thresh
        
        # 解析预测结果
        detections = []
        pred = predictions[0]
        
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] >= conf_threshold:
                # 获取类别索引（减1因为0是背景）
                class_idx = int(labels[i]) - 1
                
                # 确保索引在有效范围内
                if 0 <= class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    class_name = f"未知类别_{class_idx}"
                
                # 边界框格式转换（从300x300缩放到原始尺寸）
                bbox = boxes[i].tolist()
                
                # 计算原始图像尺寸与模型输入尺寸的比例
                if original_image is not None:
                    h, w = original_image.shape[:2]
                    scale_x = w / 300.0
                    scale_y = h / 300.0
                    
                    # 缩放边界框坐标到原始图像尺寸
                    bbox = [
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ]
                
                detections.append({
                    'bbox': bbox,
                    'confidence': float(scores[i]),
                    'class_id': class_idx,
                    'class_name': class_name
                })
        
        # 绘制边界框
        processed_image = None
        if draw_boxes and original_image is not None:
            processed_image = self._draw_detections(original_image, detections)
        elif original_image is not None:
            processed_image = original_image.copy()
        
        return {
            'detections': detections,
            'original_image': original_image,
            'processed_image': processed_image
        }
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """在图像上绘制检测结果"""
        result_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # 确保class_id在有效范围内
            if class_id >= len(self.colors):
                color = (0, 255, 0)  # 默认绿色
            else:
                color = self.colors[class_id]
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
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
    
    def save_model(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        try:
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'colors': self.colors,
                'is_trained': self.is_trained,
                'total_epochs': self.total_epochs
            }, save_path)
            print(f"模型已保存到 {save_path}")
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    def get_class_names(self):
        """
        获取类别名称列表
        
        Returns:
            list: 类别名称列表
        """
        return self.class_names
    
    def get_num_classes(self):
        """
        获取类别数量（不包括背景）
        
        Returns:
            int: 类别数量
        """
        return len(self.class_names)
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            dict: 模型信息
        """
        return {
            'model_type': 'SSD',
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'is_trained': self.is_trained,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'device': self.device
        }
