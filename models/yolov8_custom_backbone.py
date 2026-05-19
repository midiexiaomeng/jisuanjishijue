"""
YOLOv8模型实现 - 包含自定义骨干网络
基于Ultralytics YOLOv8架构，但使用自定义设计的骨干网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import os


class CustomBackbone(nn.Module):
    """
    自定义骨干网络
    基于深度可分离卷积和残差连接的高效特征提取器
    """
    
    def __init__(self, img_size: int = 640, in_channels: int = 3, width_mult: float = 0.5):
        """
        初始化自定义骨干网络
        
        Args:
            img_size: 输入图像尺寸
            in_channels: 输入通道数
            width_mult: 宽度乘数，控制网络宽度
        """
        super().__init__()
        
        # 配置参数
        self.img_size = img_size
        self.in_channels = in_channels
        self.width_mult = width_mult
        
        # 定义通道数配置
        c = lambda x: int(x * width_mult)  # 宽度乘数应用函数
        
        # Stage 1: 初始卷积层 (640x640 -> 320x320)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.SiLU(inplace=True),
            
            # 深度可分离卷积块
            DepthwiseConvBlock(c(32), c(64), stride=1),
        )
        
        # Stage 2: 下采样 (320x320 -> 160x160)
        self.stage2 = nn.Sequential(
            ResidualBlock(c(64), c(128), stride=2),
            ResidualBlock(c(128), c(128), stride=1),
        )
        
        # Stage 3: 下采样 (160x160 -> 80x80)
        self.stage3 = nn.Sequential(
            ResidualBlock(c(128), c(256), stride=2),
            ResidualBlock(c(256), c(256), stride=1),
            ResidualBlock(c(256), c(256), stride=1),
            ResidualBlock(c(256), c(256), stride=1),
        )
        
        # Stage 4: 下采样 (80x80 -> 40x40)
        self.stage4 = nn.Sequential(
            ResidualBlock(c(256), c(512), stride=2),
            ResidualBlock(c(512), c(512), stride=1),
            ResidualBlock(c(512), c(512), stride=1),
            ResidualBlock(c(512), c(512), stride=1),
            ResidualBlock(c(512), c(512), stride=1),
            ResidualBlock(c(512), c(512), stride=1),
            ResidualBlock(c(512), c(512), stride=1),
            ResidualBlock(c(512), c(512), stride=1),
        )
        
        # Stage 5: 下采样 (40x40 -> 20x20)
        self.stage5 = nn.Sequential(
            ResidualBlock(c(512), c(1024), stride=2),
            ResidualBlock(c(1024), c(1024), stride=1),
            ResidualBlock(c(1024), c(1024), stride=1),
            ResidualBlock(c(1024), c(1024), stride=1),
        )
        
        # Stage 6: 下采样 (20x20 -> 10x10)
        self.stage6 = nn.Sequential(
            ResidualBlock(c(1024), c(2048), stride=2),
            ResidualBlock(c(2048), c(2048), stride=1),
        )
        
        # 输出特征通道数
        self.out_channels = [c(256), c(512), c(1024), c(2048)]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            特征图列表，对应不同尺度
        """
        # 正向传播
        x = self.stage1(x)  # 320x320
        x = self.stage2(x)  # 160x160
        f1 = self.stage3(x)  # 80x80
        f2 = self.stage4(f1)  # 40x40
        f3 = self.stage5(f2)  # 20x20
        f4 = self.stage6(f3)  # 10x10
        
        # 返回多尺度特征图
        return [f1, f2, f3, f4]


class DepthwiseConvBlock(nn.Module):
    """
    深度可分离卷积块
    由深度卷积和逐点卷积组成
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, 
            padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.pwconv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, 
            padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.pwconv(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    残差连接块
    包含两个深度可分离卷积和残差连接
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.use_shortcut = stride == 1 and in_channels == out_channels
        
        # 主分支
        self.main = nn.Sequential(
            DepthwiseConvBlock(in_channels, out_channels, stride=stride),
            DepthwiseConvBlock(out_channels, out_channels, stride=1),
        )
        
        # 残差连接（如果需要）
        if not self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_shortcut:
            return x + self.main(x)
        else:
            return self.shortcut(x) + self.main(x)


class YOLOv8Head(nn.Module):
    """
    YOLOv8检测头
    将骨干网络输出的特征图转换为检测结果
    """
    
    def __init__(self, in_channels: List[int], num_classes: int = 25, width_mult: float = 0.5):
        """
        初始化YOLOv8检测头
        
        Args:
            in_channels: 输入特征通道列表
            num_classes: 类别数量
            width_mult: 宽度乘数
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.width_mult = width_mult
        
        # 检测头配置
        nc = num_classes  # 类别数量
        no = nc + 5  # 输出通道数：x, y, w, h, conf + classes
        
        # 定义检测头
        self.detect = nn.ModuleList()
        for i in range(len(in_channels)):
            self.detect.append(nn.Sequential(
                nn.Conv2d(in_channels[i], in_channels[i] // 2, 1, 1, 0),
                nn.BatchNorm2d(in_channels[i] // 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_channels[i] // 2, no, 1, 1, 0),  # 最终检测层
            ))
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 特征图列表
            
        Returns:
            检测结果张量 [B, num_anchors, 5 + num_classes]
        """
        outputs = []
        for i, (f, head) in enumerate(zip(features, self.detect)):
            output = head(f)
            
            # 调整输出形状 [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            B, C, H, W = output.shape
            output = output.permute(0, 2, 3, 1).reshape(B, H * W, C)
            
            outputs.append(output)
        
        # 合并所有尺度的输出
        return torch.cat(outputs, dim=1)


class YOLOv8CustomBackbone(nn.Module):
    """
    完整的YOLOv8模型，使用自定义骨干网络
    """
    
    def __init__(self, num_classes: int = 25, img_size: int = 640, width_mult: float = 0.5):
        """
        初始化YOLOv8模型
        
        Args:
            num_classes: 类别数量
            img_size: 输入图像尺寸
            width_mult: 宽度乘数
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.width_mult = width_mult
        
        # 自定义骨干网络
        self.backbone = CustomBackbone(img_size=img_size, width_mult=width_mult)
        
        # YOLOv8检测头
        self.head = YOLOv8Head(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            width_mult=width_mult
        )
        
        # 初始化权重
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 手动计算SiLU的增益值，因为PyTorch早期版本不支持'silu'参数
                # SiLU的近似增益值为0.56
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, 3, H, W]
            
        Returns:
            检测结果张量 [B, num_anchors, 5 + num_classes]
        """
        # 骨干网络特征提取
        features = self.backbone(x)
        
        # 检测头前向传播
        outputs = self.head(features)
        
        return outputs
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class YOLOv8CustomDetector:
    """
    YOLOv8目标检测器 - 自定义骨干网络版本
    """
    
    def __init__(self, num_classes: int = 25, device: str = 'cuda', width_mult: float = 0.5):
        """
        初始化YOLOv8检测器
        
        Args:
            num_classes: 类别数量
            device: 设备 ('cuda' 或 'cpu')
            width_mult: 宽度乘数，控制网络宽度
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.num_classes = num_classes
        self.width_mult = width_mult
        
        # 创建模型
        self.model = YOLOv8CustomBackbone(
            num_classes=num_classes,
            width_mult=width_mult
        )
        
        # 移动到设备
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
        
        # 训练状态
        self.is_trained = False
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """
        生成用于可视化的颜色
        
        Args:
            num_classes: 类别数量
            
        Returns:
            颜色元组列表
        """
        np.random.seed(42)
        colors = []
        for i in range(num_classes):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
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
                    self.is_trained = True
                    print(f"✓ 成功加载预训练权重: {model_path}")
                else:
                    self.model.load_state_dict(checkpoint)
                    self.is_trained = True
                    print(f"✓ 成功加载预训练权重: {model_path}")
            else:
                print(f"✗ 模型路径不存在: {model_path}")
        except Exception as e:
            print(f"✗ 加载预训练权重失败: {e}")
    
    def detect(self, 
               image: np.ndarray,
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            
        Returns:
            检测结果字典
        """
        # 预处理图像
        original_h, original_w = image.shape[:2]
        resized_image = cv2.resize(image, (self.model.img_size, self.model.img_size))
        
        # 转换为RGB格式
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # 转换为张量
        tensor_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        tensor_image = tensor_image.unsqueeze(0).to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(tensor_image)
        
        # 后处理
        detections = self._postprocess(outputs, conf_threshold, iou_threshold, original_h, original_w)
        
        # 绘制检测结果
        processed_image = self._draw_detections(image, detections)
        
        return {
            'detections': detections,
            'original_image': image,
            'processed_image': processed_image
        }
    
    def _postprocess(self, outputs: torch.Tensor, conf_threshold: float, 
                    iou_threshold: float, original_h: int, original_w: int) -> List[Dict]:
        """
        后处理检测结果
        
        Args:
            outputs: 模型输出张量
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            original_h: 原始图像高度
            original_w: 原始图像宽度
            
        Returns:
            检测结果列表
        """
        detections = []
        
        # 解析输出
        B, num_anchors, C = outputs.shape
        
        # 置信度过滤
        conf_mask = outputs[..., 4] > conf_threshold
        outputs = outputs[conf_mask]
        
        if len(outputs) == 0:
            return detections
        
        # 获取边界框和类别
        boxes = outputs[..., :4]
        confs = outputs[..., 4:5]
        classes = outputs[..., 5:]
        class_ids = torch.argmax(classes, dim=1)
        class_confs = torch.max(classes, dim=1)[0].unsqueeze(1)
        
        # 合并置信度
        confs *= class_confs
        
        # 边界框坐标转换
        boxes = self._xywh2xyxy(boxes)
        
        # 非极大值抑制
        keep = self._nms(boxes, confs, iou_threshold)
        boxes = boxes[keep]
        confs = confs[keep]
        class_ids = class_ids[keep]
        
        # 缩放到原始图像尺寸
        scale_w = original_w / self.model.img_size
        scale_h = original_h / self.model.img_size
        
        # 生成检测结果
        for box, conf, class_id in zip(boxes, confs, class_ids):
            x1, y1, x2, y2 = box.cpu().numpy()
            
            # 缩放坐标
            x1 = int(x1 * scale_w)
            y1 = int(y1 * scale_h)
            x2 = int(x2 * scale_w)
            y2 = int(y2 * scale_h)
            
            # 确保坐标在有效范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_w - 1, x2)
            y2 = min(original_h - 1, y2)
            
            # 获取类别名称
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f'class_{class_id}'
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class_id': int(class_id),
                'class_name': class_name
            })
        
        return detections
    
    def _xywh2xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        将XYWH格式的边界框转换为XYXY格式
        
        Args:
            boxes: XYWH格式的边界框 [x_center, y_center, width, height]
            
        Returns:
            XYXY格式的边界框 [x_min, y_min, x_max, y_max]
        """
        x_c, y_c, w, h = boxes.unbind(1)
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, 
             iou_threshold: float) -> torch.Tensor:
        """
        非极大值抑制
        
        Args:
            boxes: 边界框张量
            scores: 置信度分数张量
            iou_threshold: IOU阈值
            
        Returns:
            保留的索引张量
        """
        if boxes.numel() == 0:
            return torch.tensor([], dtype=torch.long)
        
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)
        
        # 按置信度降序排序
        _, idx = scores.squeeze(1).sort(descending=True)
        
        keep = []
        while idx.numel() > 0:
            i = idx[0]
            keep.append(i)
            
            if idx.numel() == 1:
                break
            
            # 计算与当前框的IoU
            idx = idx[1:]
            
            # 获取当前框和其他框的坐标
            x1_i = torch.max(x1[i], x1[idx])
            y1_i = torch.max(y1[i], y1[idx])
            x2_i = torch.min(x2[i], x2[idx])
            y2_i = torch.min(y2[i], y2[idx])
            
            # 计算交集面积
            inter = torch.clamp(x2_i - x1_i, min=0) * torch.clamp(y2_i - y1_i, min=0)
            
            # 计算并集面积
            union = areas[i] + areas[idx] - inter
            
            # 计算IoU
            iou = inter / union
            
            # 保留IoU小于阈值的框
            idx = idx[iou <= iou_threshold]
        
        return torch.tensor(keep, dtype=torch.long)
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            
        Returns:
            带有检测框的图像
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
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 标签背景
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 标签文本
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image
    
    def train(self, train_loader, val_loader, epochs: int = 100, 
              learning_rate: float = 0.001, save_dir: str = 'checkpoints/yolov8_custom'):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            save_dir: 模型保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义损失函数
        criterion = self._yolo_loss
        
        # 定义优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.0005
        )
        
        # 定义学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
        
        # 训练循环
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            for i, (images, targets) in enumerate(train_loader):
                # 移动到设备
                images = images.to(self.device)
                
                # 清空梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                
                # 更新权重
                optimizer.step()
                
                running_loss += loss.item()
                
                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            # 更新学习率
            scheduler.step()
            
            # 计算平均损失
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss:.4f}")
            
            # 验证模型
            val_loss = self._validate(val_loader, criterion)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")
            
            # 保存模型
            save_path = os.path.join(save_dir, f'yolov8_custom_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, save_path)
            print(f"模型保存到: {save_path}")
        
        self.is_trained = True
        print("训练完成!")
    
    def _yolo_loss(self, predictions: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        YOLO损失函数
        
        Args:
            predictions: 模型预测
            targets: 真实标签
            
        Returns:
            总损失
        """
        # 简化的YOLO损失函数实现
        # 在实际应用中，应该使用完整的YOLOv8损失函数
        B, num_anchors, C = predictions.shape
        
        # 这里返回一个简化的损失值
        # 实际应用中需要实现完整的边界框回归损失、置信度损失和分类损失
        return torch.mean(predictions)  # 占位符，实际应用中需要替换为完整损失函数
    
    def _validate(self, val_loader, criterion) -> float:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            criterion: 损失函数
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_name': 'YOLOv8 Custom Backbone',
            'num_classes': self.num_classes,
            'img_size': self.model.img_size,
            'width_mult': self.width_mult,
            'backbone': 'Custom Depthwise Separable CNN',
            'num_parameters': self.model.get_num_params(),
            'device': self.device,
            'is_trained': self.is_trained
        }
