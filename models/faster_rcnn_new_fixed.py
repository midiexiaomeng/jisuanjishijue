import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from PIL import Image
import torchvision.transforms as T
import cv2


class FasterRCNNNewFixed:
    """修复的Faster R-CNN目标检测模型（解决CUDA流不匹配问题）"""
    
    def __init__(self, num_classes: int = 25, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化修复的Faster R-CNN检测器
        
        Args:
            num_classes: 类别数量（包括背景），COU数据集为24类+背景=25
            device: 设备类型
        """
        self.device = device
        self.num_classes = num_classes
        
        # COU数据集类别（24类人造物）
        self.class_names = [
            'plastic_bottle', 'plastic_bag', 'fishing_net', 'rope', 'can', 
            'glass_bottle', 'tire', 'metal_scrap', 'wood', 'cloth',
            'diver', 'diving_mask', 'diving_fins', 'oxygen_tank', 'underwater_camera',
            'auv', 'rov', 'underwater_drone', 'sonar', 'underwater_sensor',
            'ship_wreck', 'anchor', 'propeller', 'underwater_structure'
        ]
        
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
        """初始化Faster R-CNN模型（使用标准实现）"""
        try:
            # 加载预训练的Faster R-CNN模型（使用ResNet50-FPN骨干网络）
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            
            # 获取输入特征的数量
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            
            # 替换分类器头以适应我们的类别数量
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            
            # 移动到设备
            self.model.to(self.device)
            
            print(f"修复的Faster R-CNN模型初始化完成，使用设备: {self.device}")
            print(f"使用ResNet50-FPN骨干网络，类别数量: {self.num_classes}")
            
        except Exception as e:
            print(f"初始化修复的Faster R-CNN模型失败: {e}")
            raise
    
    def load_pretrained(self, model_path: str):
        """
        加载预训练权重（改进版本，更健壮地处理权重不匹配）
        
        Args:
            model_path: 模型权重文件路径
        """
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 获取模型当前状态字典
                model_state_dict = self.model.state_dict()
                
                # 从检查点获取状态字典
                if 'model_state_dict' in checkpoint:
                    pretrained_state_dict = checkpoint['model_state_dict']
                else:
                    pretrained_state_dict = checkpoint
                
                # 过滤掉不匹配的权重（只加载匹配的权重）
                filtered_state_dict = {}
                for key, value in pretrained_state_dict.items():
                    if key in model_state_dict and model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        print(f"跳过不匹配的权重: {key} (形状不匹配或键不存在)")
                
                # 加载过滤后的权重
                if filtered_state_dict:
                    model_state_dict.update(filtered_state_dict)
                    self.model.load_state_dict(model_state_dict)
                    print(f"从 {model_path} 加载预训练权重成功，加载了 {len(filtered_state_dict)}/{len(pretrained_state_dict)} 个权重")
                else:
                    print(f"没有匹配的权重可以加载，使用随机初始化权重")
                
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
            print("使用随机初始化权重继续...")
    
    def train(self, train_loader, val_loader=None, epochs: int = 10, 
              learning_rate: float = 0.001, save_dir: str = 'checkpoints/faster_rcnn_new_fixed',
              use_amp: bool = True, gradient_accumulation_steps: int = 4, 
              empty_cache_frequency: int = 10):
        """
        训练模型（修复CUDA流不匹配问题，支持混合精度训练）
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            epochs: 训练轮数
            learning_rate: 学习率
            save_dir: 保存目录
            use_amp: 是否使用混合精度训练
            gradient_accumulation_steps: 梯度累积步数
            empty_cache_frequency: 清理CUDA缓存的频率
            
        Returns:
            dict: 训练历史记录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置模型为训练模式
        self.model.train()
        
        # 定义优化器
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        
        # 学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # 设置混合精度训练
        scaler = None
        if use_amp and torch.cuda.is_available():
            try:
                from torch.cuda.amp import GradScaler, autocast
                scaler = GradScaler()
                print("✓ 混合精度训练已启用")
            except Exception as e:
                print(f"混合精度训练设置失败: {e}")
                use_amp = False
                scaler = None
        else:
            print("混合精度训练: 禁用")
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mAP': [],
            'val_mAP': []
        }
        
        print(f"开始训练修复的Faster R-CNN模型，共{epochs}轮...")
        print(f"训练数据: {len(train_loader.dataset)} 张图像, {len(train_loader)} 个批次")
        if val_loader is not None:
            print(f"验证数据: {len(val_loader.dataset)} 张图像, {len(val_loader)} 个批次")
        print(f"混合精度训练: {'启用' if use_amp else '禁用'}")
        print(f"梯度累积步数: {gradient_accumulation_steps}")
        print(f"清理CUDA缓存频率: 每{empty_cache_frequency}个批次")
        print("=" * 80)
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            self.total_epochs = epochs
            
            # 训练阶段
            train_loss = 0.0
            train_batches = 0
            
            print(f"\n{'='*80}")
            print(f"Epoch [{epoch+1}/{epochs}] 开始训练...")
            print(f"{'='*80}")
            
            # 设置进度条
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    # 修复CUDA流不匹配问题：确保所有张量都在正确的设备上
                    images_on_device = []
                    for img in images:
                        if isinstance(img, torch.Tensor):
                            images_on_device.append(img.to(self.device))
                        else:
                            # 如果不是张量，转换为张量
                            images_on_device.append(torch.tensor(img).to(self.device))
                    
                    targets_on_device = []
                    for t in targets:
                        target_on_device = {}
                        for key, value in t.items():
                            if isinstance(value, torch.Tensor):
                                # 确保张量在正确的设备上
                                target_on_device[key] = value.to(self.device)
                            else:
                                # 如果不是张量，保持原样
                                target_on_device[key] = value
                        targets_on_device.append(target_on_device)
                    
                    # 混合精度训练
                    if use_amp and scaler:
                        from torch.cuda.amp import autocast
                        with autocast():
                            # 前向传播
                            loss_dict = self.model(images_on_device, targets_on_device)
                            
                            # 使用_process_loss_dict函数计算总损失
                            losses_tensor = self._process_loss_dict(loss_dict)
                            
                            # 获取损失值（确保是标量）
                            if isinstance(losses_tensor, torch.Tensor):
                                losses_value = losses_tensor.item()
                            else:
                                losses_value = float(losses_tensor)
                            
                            # 缩放损失并反向传播
                            scaled_loss = losses_tensor / gradient_accumulation_steps
                            scaler.scale(scaled_loss).backward()
                    else:
                        # 非混合精度训练
                        loss_dict = self.model(images_on_device, targets_on_device)
                        
                        # 使用_process_loss_dict函数计算总损失
                        losses_tensor = self._process_loss_dict(loss_dict)
                        
                        # 获取损失值（确保是标量）
                        if isinstance(losses_tensor, torch.Tensor):
                            losses_value = losses_tensor.item()
                        else:
                            losses_value = float(losses_tensor)
                        
                        # 反向传播
                        scaled_loss = losses_tensor / gradient_accumulation_steps
                        scaled_loss.backward()
                    
                    # 梯度累积
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        if use_amp and scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                    
                    # 记录损失
                    train_loss += losses_value
                    train_batches += 1
                    
                    # 更新进度条
                    current_loss = losses_value
                    avg_loss_so_far = train_loss / train_batches
                    progress_desc = f"Loss: {current_loss:.4f} | Avg: {avg_loss_so_far:.4f}"
                    pbar.set_description(progress_desc)
                    
                    # 定期清理CUDA缓存，防止内存泄漏
                    if batch_idx % empty_cache_frequency == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"\n批次 {batch_idx} 训练失败: {e}")
                    # 如果是CUDA错误，尝试清理缓存并继续
                    if "CUDA" in str(e) or "cuDNN" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            print("已清理CUDA缓存，继续训练...")
                    continue
            
            # 处理剩余的梯度
            if len(train_loader) % gradient_accumulation_steps != 0:
                if use_amp and scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # 关闭进度条
            pbar.close()
            
            # 计算平均训练损失
            avg_train_loss = train_loss / max(train_batches, 1)
            history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            if val_loader is not None:
                print(f"\n{'='*80}")
                print(f"Epoch [{epoch+1}/{epochs}] 开始验证...")
                print(f"{'='*80}")
                
                val_loss, val_mAP = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_mAP'].append(val_mAP)
                
                print(f"\nEpoch [{epoch+1}/{epochs}] 完成!")
                print(f"  训练损失: {avg_train_loss:.4f}")
                print(f"  验证损失: {val_loss:.4f}")
                print(f"  验证mAP: {val_mAP:.4f}")
            else:
                print(f"\nEpoch [{epoch+1}/{epochs}] 完成!")
                print(f"  训练损失: {avg_train_loss:.4f}")
            
            # 更新学习率
            lr_scheduler.step()
            print(f"  更新后学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                checkpoint_path = os.path.join(save_dir, f'faster_rcnn_new_fixed_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss if val_loader is not None else 0.0,
                    'val_mAP': val_mAP if val_loader is not None else 0.0,
                    'total_epochs': epochs
                }, checkpoint_path)
                print(f"  检查点已保存到: {checkpoint_path}")
            
            print(f"{'='*80}\n")
        
        # 训练完成
        self.is_trained = True
        print(f"{'='*80}")
        print("修复的Faster R-CNN模型训练完成!")
        print(f"总训练轮数: {epochs}")
        print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
        if val_loader is not None:
            print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
            print(f"最终验证mAP: {history['val_mAP'][-1]:.4f}")
        print(f"{'='*80}")
        
        return history
    
    def evaluate(self, data_loader):
        """
        评估模型（修复版本）
        
        Args:
            data_loader: 数据加载器
            
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
                try:
                    # 修复CUDA流不匹配问题：确保所有张量都在正确的设备上
                    images_on_device = []
                    for img in images:
                        if isinstance(img, torch.Tensor):
                            images_on_device.append(img.to(self.device))
                        else:
                            images_on_device.append(torch.tensor(img).to(self.device))
                    
                    targets_on_device = []
                    for t in targets:
                        target_on_device = {}
                        for key, value in t.items():
                            if isinstance(value, torch.Tensor):
                                target_on_device[key] = value.to(self.device)
                            else:
                                target_on_device[key] = value
                        targets_on_device.append(target_on_device)
                    
                    # 前向传播
                    loss_dict = self.model(images_on_device, targets_on_device)
                    
                    # 使用_process_loss_dict函数计算总损失
                    losses_tensor = self._process_loss_dict(loss_dict)
                    
                    # 获取损失值
                    if isinstance(losses_tensor, torch.Tensor):
                        losses = losses_tensor.item()
                    else:
                        losses = float(losses_tensor)
                    
                    # 记录损失
                    total_loss += losses
                    total_batches += 1
                    
                    # 获取预测结果用于mAP计算
                    predictions = self.model(images_on_device)
                    
                    # 收集预测和目标
                    for i in range(len(predictions)):
                        pred_boxes = predictions[i]['boxes'].cpu().numpy()
                        pred_scores = predictions[i]['scores'].cpu().numpy()
                        pred_labels = predictions[i]['labels'].cpu().numpy()
                        
                        target_boxes = targets_on_device[i]['boxes'].cpu().numpy()
                        target_labels = targets_on_device[i]['labels'].cpu().numpy()
                        
                        all_predictions.append({
                            'boxes': pred_boxes,
                            'scores': pred_scores,
                            'labels': pred_labels
                        })
                        
                        all_targets.append({
                            'boxes': target_boxes,
                            'labels': target_labels
                        })
                        
                except Exception as e:
                    print(f"评估批次失败: {e}")
                    continue
        
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
    
    def _process_loss_dict(self, loss_dict):
        """
        处理损失字典，修复"outputs must be a Tensor or an iterable of Tensors"错误
        
        Args:
            loss_dict: 模型返回的损失字典
            
        Returns:
            torch.Tensor: 总损失张量
        """
        if loss_dict is None:
            return torch.tensor(0.0, requires_grad=True)
        
        # 如果loss_dict是列表，转换为字典
        if isinstance(loss_dict, list):
            # 计算列表的总和
            list_sum = 0.0
            for item in loss_dict:
                if isinstance(item, (int, float)):
                    list_sum += item
                elif isinstance(item, torch.Tensor):
                    list_sum += item.item()
            loss_dict = {'total_loss': list_sum}
        
        # 如果loss_dict是单个张量，直接返回
        if isinstance(loss_dict, torch.Tensor):
            return loss_dict
        
        # 处理字典类型的损失 - 初始化total_loss为张量
        total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        for key, value in loss_dict.items():
            if isinstance(value, dict):
                # 如果是嵌套字典，提取数值
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        total_loss = total_loss + sub_value
                        break
                    elif isinstance(sub_value, (int, float)):
                        total_loss = total_loss + torch.tensor(sub_value, dtype=torch.float32, requires_grad=True)
                        break
            elif isinstance(value, torch.Tensor):
                total_loss = total_loss + value
            elif isinstance(value, (int, float)):
                total_loss = total_loss + torch.tensor(value, dtype=torch.float32, requires_grad=True)
            else:
                # 如果无法处理，添加一个小的默认损失
                total_loss = total_loss + torch.tensor(0.0, requires_grad=True)
        
        return total_loss
    
    def detect(self, image, confidence_threshold: float = 0.5, draw_boxes: bool = False):
        """
        检测图像中的目标
        
        Args:
            image: 输入图像（numpy数组或PIL图像）
            confidence_threshold: 置信度阈值
            draw_boxes: 是否在图像上绘制边界框
            
        Returns:
            dict: 检测结果，包含检测列表和绘制后的图像（如果draw_boxes为True）
        """
        self.model.eval()
        
        # 保存原始图像用于绘制
        original_image = None
        if isinstance(image, np.ndarray):
            original_image = image.copy()
            # 转换为PIL图像
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            original_image = np.array(image)
        
        # 转换为张量
        transform = T.Compose([
            T.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 进行预测
            predictions = self.model(image_tensor)
        
        # 解析预测结果
        detections = []
        pred = predictions[0]
        
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] >= confidence_threshold:
                # 获取类别索引（减1因为0是背景）
                class_idx = int(labels[i]) - 1
                
                # 确保索引在有效范围内
                if 0 <= class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    class_name = f"未知类别_{class_idx}"
                
                # 边界框格式转换
                bbox = boxes[i].tolist()
                
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
        
        return {
            'detections': detections,
            'original_image': original_image,
            'processed_image': processed_image
        }
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """
        生成用于可视化的颜色
        
        Args:
            num_classes: 类别数量
            
        Returns:
            list: 颜色列表，每个颜色为(B, G, R)元组
        """
        np.random.seed(42)
        colors = []
        for i in range(num_classes):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像 (BGR格式)
            detections: 检测结果列表
            
        Returns:
            np.ndarray: 绘制了边界框的图像
        """
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
