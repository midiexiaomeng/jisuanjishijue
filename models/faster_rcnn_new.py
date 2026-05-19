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


class FasterRCNNNew:
    """新的Faster R-CNN目标检测模型（简化版本）"""
    
    def __init__(self, num_classes: int = 25, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化新的Faster R-CNN检测器
        
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
            
            print(f"新的Faster R-CNN模型初始化完成，使用设备: {self.device}")
            print(f"使用ResNet50-FPN骨干网络，类别数量: {self.num_classes}")
            
        except Exception as e:
            print(f"初始化新的Faster R-CNN模型失败: {e}")
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
              learning_rate: float = 0.001, save_dir: str = 'checkpoints/faster_rcnn_new'):
        """
        训练模型（简化版本）
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            epochs: 训练轮数
            learning_rate: 学习率
            save_dir: 保存目录
            
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
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mAP': [],
            'val_mAP': []
        }
        
        print(f"开始训练新的Faster R-CNN模型，共{epochs}轮...")
        print(f"训练数据: {len(train_loader.dataset)} 张图像, {len(train_loader)} 个批次")
        if val_loader is not None:
            print(f"验证数据: {len(val_loader.dataset)} 张图像, {len(val_loader)} 个批次")
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
                # 移动到设备
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 前向传播
                loss_dict = self.model(images, targets)
                
                # 计算总损失（简化处理）
                losses = sum(loss for loss in loss_dict.values())
                
                # 反向传播
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                # 记录损失
                train_loss += losses.item()
                train_batches += 1
                
                # 更新进度条
                current_loss = losses.item()
                avg_loss_so_far = train_loss / train_batches
                progress_desc = f"Loss: {current_loss:.4f} | Avg: {avg_loss_so_far:.4f}"
                pbar.set_description(progress_desc)
            
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
                checkpoint_path = os.path.join(save_dir, f'faster_rcnn_new_epoch_{epoch+1}.pth')
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
        print("新的Faster R-CNN模型训练完成!")
        print(f"总训练轮数: {epochs}")
        print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
        if val_loader is not None:
            print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
            print(f"最终验证mAP: {history['val_mAP'][-1]:.4f}")
        print(f"{'='*80}")
        
        return history
    
    def evaluate(self, data_loader):
        """
        评估模型（简化版本）
        
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
                # 移动到设备
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 前向传播
                loss_dict = self.model(images, targets)
                
                # 计算总损失
                losses = sum(loss for loss in loss_dict.values())
                
                # 记录损失
                total_loss += losses.item()
                total_batches += 1
                
                # 获取预测结果用于mAP计算
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
    
    def detect(self, image, confidence_threshold: float = 0.5):
        """
        检测图像中的目标
        
        Args:
            image: 输入图像（numpy数组或PIL图像）
            confidence_threshold: 置信度阈值
            
        Returns:
            list: 检测结果，每个元素为[类别名称, 置信度, 边界框]
        """
        self.model.eval()
        
        # 转换图像格式
        if isinstance(image, np.ndarray):
            # 转换为PIL图像
            image = Image.fromarray(image)
        
        # 转换为张量
        transform = T.Compose([
            T.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 进行预测
            predictions = self.model(image_tensor)
        
        # 解析预测结果
        results = []
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
                
                results.append([class_name, float(scores[i]), bbox])
        
        return results
    
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
                'is_trained': self.is_trained,
                'total_epochs': self.total_epochs
            }, save_path)
            print(f"模型已保存到 {save_path}")
        except Exception as e:
            print(f"保存模型失败: {e}")
