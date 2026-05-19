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


class CustomConvBlock(nn.Module):
    """自定义卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_bn=True, activation='relu', use_se=False):
        """
        初始化自定义卷积块
        
        Args:
            in_channels: 输入通道数，即输入特征图的深度
            out_channels: 输出通道数，即输出特征图的深度
            kernel_size: 卷积核大小，默认为3x3
            stride: 卷积步长，默认为1（不进行下采样）
            padding: 填充大小，默认为1（保持特征图尺寸不变）
            use_bn: 是否使用批量归一化，默认为True
            activation: 激活函数类型，可选'relu'、'leaky_relu'、'silu'，默认为'relu'
            use_se: 是否使用Squeeze-and-Excitation注意力机制，默认为False
        """
        super().__init__()  # 调用父类nn.Module的初始化方法
        
        # 创建卷积层
        # nn.Conv2d是PyTorch的2D卷积层
        # bias=not use_bn: 如果使用批量归一化，则不需要偏置项，因为BN会学习偏移参数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, bias=not use_bn)
        
        # 创建批量归一化层或恒等映射
        # 如果use_bn为True，使用BatchNorm2d进行批量归一化
        # 如果use_bn为False，使用nn.Identity()作为占位符（不进行任何操作）
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # 根据activation参数选择激活函数
        if activation == 'relu':
            # ReLU激活函数，inplace=True表示原地操作，节省内存
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            # LeakyReLU激活函数，负斜率设为0.1，解决ReLU的"神经元死亡"问题
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'silu':
            # SiLU激活函数（也称为Swish），性能优于ReLU
            self.activation = nn.SiLU(inplace=True)
        else:
            # 如果未指定或指定了不支持的激活函数，使用恒等映射
            self.activation = nn.Identity()
        
        # Squeeze-and-Excitation注意力机制
        self.use_se = use_se  # 记录是否使用SE注意力机制
        if use_se:
            # 创建SE注意力模块
            self.se = nn.Sequential(
                # 1. 全局平均池化：将每个通道的特征图压缩为1x1
                nn.AdaptiveAvgPool2d(1),
                # 2. 第一个全连接层：降维，减少通道数为原来的1/16
                nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
                # 3. ReLU激活函数
                nn.ReLU(inplace=True),
                # 4. 第二个全连接层：恢复原始通道数
                nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
                # 5. Sigmoid激活函数：将输出限制在0-1之间，作为通道权重
                nn.Sigmoid()
            )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        if self.use_se:
            se_weight = self.se(x)
            x = x * se_weight
        
        x = self.activation(x)
        return x


class CustomBackbone(nn.Module):
    """自定义骨干网络，专门针对水下目标检测优化"""
    
    def __init__(self):
        super().__init__()
        
        # 输入通道数：3 (RGB)
        # 第一阶段：浅层特征提取
        self.stage1 = nn.Sequential(
            CustomConvBlock(3, 32, kernel_size=3, stride=2, padding=1),  # 下采样2倍
            CustomConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            CustomConvBlock(64, 64, kernel_size=3, stride=1, padding=1, use_se=True),
        )
        
        # 第二阶段：中等深度特征
        self.stage2 = nn.Sequential(
            CustomConvBlock(64, 128, kernel_size=3, stride=2, padding=1),  # 下采样4倍
            CustomConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            CustomConvBlock(128, 128, kernel_size=3, stride=1, padding=1, use_se=True),
            CustomConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
        )
        
        # 第三阶段：深层特征
        self.stage3 = nn.Sequential(
            CustomConvBlock(256, 256, kernel_size=3, stride=2, padding=1),  # 下采样8倍
            CustomConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            CustomConvBlock(256, 256, kernel_size=3, stride=1, padding=1, use_se=True),
            CustomConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
        )
        
        # 第四阶段：更深的特征
        self.stage4 = nn.Sequential(
            CustomConvBlock(512, 512, kernel_size=3, stride=2, padding=1),  # 下采样16倍
            CustomConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            CustomConvBlock(512, 512, kernel_size=3, stride=1, padding=1, use_se=True),
            CustomConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
        )
        
        # 第五阶段：最深特征
        self.stage5 = nn.Sequential(
            CustomConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),  # 下采样32倍
            CustomConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            CustomConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1, use_se=True),
            CustomConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
        )
        
        # 额外层用于SSD多尺度特征提取
        self.extra1 = nn.Sequential(
            CustomConvBlock(1024, 256, kernel_size=1, stride=1, padding=0),
            CustomConvBlock(256, 512, kernel_size=3, stride=2, padding=1),  # 下采样64倍
        )
        
        self.extra2 = nn.Sequential(
            CustomConvBlock(512, 128, kernel_size=1, stride=1, padding=0),
            CustomConvBlock(128, 256, kernel_size=3, stride=2, padding=1),  # 下采样128倍
        )
        
        # 输出通道数列表，用于SSD头
        self.out_channels = [256, 512, 1024, 1024, 512, 256]
    
    def forward(self, x):
        # 第一阶段输出
        stage1_out = self.stage1(x)  # [B, 64, H/2, W/2]
        
        # 第二阶段输出
        stage2_out = self.stage2(stage1_out)  # [B, 256, H/4, W/4]
        
        # 第三阶段输出
        stage3_out = self.stage3(stage2_out)  # [B, 512, H/8, W/8]
        
        # 第四阶段输出
        stage4_out = self.stage4(stage3_out)  # [B, 1024, H/16, W/16]
        
        # 第五阶段输出
        stage5_out = self.stage5(stage4_out)  # [B, 1024, H/32, W/32]
        
        # 额外层输出
        extra1_out = self.extra1(stage5_out)  # [B, 512, H/64, W/64]
        extra2_out = self.extra2(extra1_out)  # [B, 256, H/128, W/128]
        
        # 返回多尺度特征图（SSD需要的6个特征层）
        return [stage2_out, stage3_out, stage4_out, stage5_out, extra1_out, extra2_out]


class CustomDetectionHead(nn.Module):
    """
    自定义检测头，不使用torchvision的SSD检测头
    
    这个类实现了SSD（Single Shot MultiBox Detector）的自定义检测头，用于目标检测任务。
    与torchvision内置的SSD检测头不同，这个自定义检测头提供了更大的灵活性和可定制性。
    
    主要功能：
    1. 为每个多尺度特征图创建独立的分类头和回归头
    2. 分类头：预测每个锚点（anchor）的类别置信度
    3. 回归头：预测每个锚点的边界框偏移量（x, y, w, h）
    4. 处理多尺度特征图，将它们转换为统一的预测格式
    
    设计特点：
    - 使用自定义卷积块（CustomConvBlock）增强特征表示能力
    - 支持SE（Squeeze-and-Excitation）注意力机制
    - 针对水下目标检测优化
    - 可灵活调整网络深度和宽度
    """
    
    def __init__(self, in_channels_list, num_anchors_list, num_classes):
        """
        初始化自定义检测头
        
        Args:
            in_channels_list: 每个特征图的输入通道数列表
                - 类型: List[int]
                - 示例: [256, 512, 1024, 1024, 512, 256] 对应6个特征图
                - 说明: 每个元素表示对应特征图的通道数，来自骨干网络的输出
            
            num_anchors_list: 每个特征图位置的锚点数量列表
                - 类型: List[int]
                - 示例: [4, 6, 6, 6, 4, 4] 对应6个特征图
                - 说明: 每个元素表示在对应特征图的每个空间位置生成的锚点数量
                - 注意: 锚点数量取决于锚点生成器的配置（aspect_ratios参数）
            
            num_classes: 类别数量（包括背景）
                - 类型: int
                - 示例: 25（24个目标类别 + 1个背景类别）
                - 说明: 在COU数据集中为24类人造物+背景=25类
        """
        super().__init__()  # 调用父类nn.Module的初始化方法
        
        # 记录特征图数量
        self.num_features = len(in_channels_list)
        # 记录类别数量（包括背景）
        self.num_classes = num_classes
        
        # 为每个特征图创建分类头和回归头
        # 使用ModuleList确保所有子模块都被正确注册
        self.classification_heads = nn.ModuleList()
        self.regression_heads = nn.ModuleList()
        
        # 遍历所有特征图，为每个特征图创建独立的检测头
        for i in range(self.num_features):
            in_channels = in_channels_list[i]      # 当前特征图的输入通道数
            num_anchors = num_anchors_list[i]      # 当前特征图每个位置的锚点数量
            
            # 分类头：预测每个锚点的类别置信度
            # 网络结构: 3个CustomConvBlock + 1个输出卷积层
            # 输出通道数: num_anchors * num_classes
            # 每个锚点需要预测num_classes个类别的置信度
            classification_head = nn.Sequential(
                # 第一层：特征提取和通道扩展
                CustomConvBlock(in_channels, 256, kernel_size=3, stride=1, padding=1),
                # 第二层：进一步的特征提取
                CustomConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
                # 第三层：最终的特征提取
                CustomConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
                # 输出层：生成类别预测
                # 输出形状: [B, num_anchors * num_classes, H, W]
                nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
            )
            
            # 回归头：预测每个锚点的边界框偏移量
            # 网络结构: 3个CustomConvBlock + 1个输出卷积层
            # 输出通道数: num_anchors * 4
            # 每个锚点需要预测4个边界框参数（x, y, w, h的偏移量）
            regression_head = nn.Sequential(
                # 第一层：特征提取和通道扩展
                CustomConvBlock(in_channels, 256, kernel_size=3, stride=1, padding=1),
                # 第二层：进一步的特征提取
                CustomConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
                # 第三层：最终的特征提取
                CustomConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
                # 输出层：生成边界框偏移预测
                # 输出形状: [B, num_anchors * 4, H, W]
                nn.Conv2d(256, num_anchors * 4, kernel_size=3, stride=1, padding=1)
            )
            
            # 将创建的分类头和回归头添加到ModuleList中
            self.classification_heads.append(classification_head)
            self.regression_heads.append(regression_head)
    
    def forward(self, features):
        """
        前向传播：处理多尺度特征图，生成分类和回归预测
        
        Args:
            features: 多尺度特征图列表
                - 类型: List[torch.Tensor]
                - 长度: self.num_features（通常为6）
                - 每个元素形状: [B, C_i, H_i, W_i]
                    B: 批次大小
                    C_i: 第i个特征图的通道数（来自in_channels_list[i]）
                    H_i, W_i: 第i个特征图的高度和宽度
        
        Returns:
            tuple: (cls_logits, bbox_regressions)
                - cls_logits: 分类预测，形状为[B, total_anchors, num_classes]
                    total_anchors: 所有特征图的所有锚点总数
                - bbox_regressions: 回归预测，形状为[B, total_anchors, 4]
                    4表示边界框的4个参数（x, y, w, h的偏移量）
        """
        cls_logits = []      # 存储每个特征图的分类预测
        bbox_regressions = []  # 存储每个特征图的回归预测
        
        # 遍历所有特征图，分别处理每个特征图
        for i, feature in enumerate(features):
            # ==================== 分类头处理 ====================
            # 应用分类头到当前特征图
            cls_logit = self.classification_heads[i](feature)
            # cls_logit形状: [B, num_anchors * num_classes, H, W]
            
            # 获取当前批次大小和特征图空间尺寸
            B, _, H, W = cls_logit.shape
            
            # 计算当前特征图的锚点数量
            # 通过查看分类头最后一层的输出通道数除以类别数量得到
            num_anchors = self.classification_heads[i][-1].out_channels // self.num_classes
            
            # 重塑分类预测的形状，使其更适合后续处理
            # 步骤1: [B, num_anchors * num_classes, H, W] -> [B, num_anchors, num_classes, H, W]
            cls_logit = cls_logit.view(B, num_anchors, self.num_classes, H, W)
            # 步骤2: 调整维度顺序 -> [B, H, W, num_anchors, num_classes]
            cls_logit = cls_logit.permute(0, 3, 4, 1, 2).contiguous()
            # 步骤3：展平空间维度和锚点维度 -> [B, H*W*num_anchors, num_classes]
            cls_logit = cls_logit.view(B, -1, self.num_classes)
            
            # 将处理后的分类预测添加到列表中
            cls_logits.append(cls_logit)
            
            # ==================== 回归头处理 ====================
            # 应用回归头到当前特征图
            bbox_reg = self.regression_heads[i](feature)
            # bbox_reg形状: [B, num_anchors * 4, H, W]
            
            # 获取当前批次大小和特征图空间尺寸（与分类头相同）
            B, _, H, W = bbox_reg.shape
            
            # 重塑回归预测的形状，使其更适合后续处理
            # 步骤1: [B, num_anchors * 4, H, W] -> [B, num_anchors, 4, H, W]
            bbox_reg = bbox_reg.view(B, num_anchors, 4, H, W)
            # 步骤2: 调整维度顺序 -> [B, H, W, num_anchors, 4]
            bbox_reg = bbox_reg.permute(0, 3, 4, 1, 2).contiguous()
            # 步骤3: 展平空间维度和锚点维度 -> [B, H*W*num_anchors, 4]
            bbox_reg = bbox_reg.view(B, -1, 4)
            
            # 将处理后的回归预测添加到列表中
            bbox_regressions.append(bbox_reg)
        
        # ==================== 合并所有特征图的预测 ====================
        # 将所有特征图的分类预测沿着锚点维度连接起来
        # 最终形状: [B, total_anchors, num_classes]
        # total_anchors = sum(H_i * W_i * num_anchors_i for i in range(num_features))
        cls_logits = torch.cat(cls_logits, dim=1)
        
        # 将所有特征图的回归预测沿着锚点维度连接起来
        # 最终形状: [B, total_anchors, 4]
        bbox_regressions = torch.cat(bbox_regressions, dim=1)
        
        return cls_logits, bbox_regressions


class CustomDetector(nn.Module):
    """自定义目标检测模型，使用自定义骨干网络和自定义检测头"""
    
    def __init__(self, backbone, anchor_generator, num_classes, image_mean=None, image_std=None):
        """
        初始化自定义检测模型
        
        Args:
            backbone: 骨干网络
            anchor_generator: 锚点生成器
            num_classes: 类别数量（包括背景）
            image_mean: 图像均值
            image_std: 图像标准差
        """
        super().__init__()
        
        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.num_classes = num_classes
        
        # 图像归一化参数
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]
        
        # 获取每个特征图的锚点数量
        self.num_anchors_per_location = anchor_generator.num_anchors_per_location()
        
        # 创建自定义检测头
        self.head = CustomDetectionHead(
            in_channels_list=backbone.out_channels,
            num_anchors_list=self.num_anchors_per_location,
            num_classes=num_classes
        )
        
        # 检测参数
        self.score_thresh = 0.01
        self.nms_thresh = 0.45
        self.detections_per_img = 200
        self.topk_candidates = 400
        
    def forward(self, images, targets=None):
        """
        前向传播
        
        Args:
            images: 输入图像列表
            targets: 目标标签（训练时使用）
            
        Returns:
            训练时返回损失字典，评估时返回预测结果
        """
        # 图像归一化
        normalized_images = []
        for img in images:
            # 简单的归一化
            img = (img - torch.tensor(self.image_mean, device=img.device).view(3, 1, 1)) / \
                  torch.tensor(self.image_std, device=img.device).view(3, 1, 1)
            normalized_images.append(img)
        
        # 提取特征
        features = self.backbone(torch.stack(normalized_images))
        
        # 生成锚点
        anchors = self.anchor_generator(images, features)
        
        # 检测头前向传播
        cls_logits, bbox_regressions = self.head(features)
        
        if self.training:
            if targets is None:
                raise ValueError("训练模式下必须提供targets")
            
            # 计算损失
            losses = self.compute_loss(cls_logits, bbox_regressions, anchors, targets)
            return losses
        else:
            # 推理模式：生成预测
            detections = self.postprocess(cls_logits, bbox_regressions, anchors, images[0].shape[-2:])
            return detections
    
    def compute_loss(self, cls_logits, bbox_regressions, anchors, targets):
        """
        计算损失
        
        Args:
            cls_logits: 分类预测
            bbox_regressions: 回归预测
            anchors: 锚点
            targets: 目标标签
            
        Returns:
            dict: 损失字典
        """
        # 简化的损失计算（实际实现需要更复杂的匹配和损失计算）
        # 这里返回一个占位符损失
        return {
            'classification': torch.tensor(0.0, device=cls_logits.device, requires_grad=True),
            'bbox_regression': torch.tensor(0.0, device=bbox_regressions.device, requires_grad=True)
        }
    
    def postprocess(self, cls_logits, bbox_regressions, anchors, image_shape):
        """
        后处理：将模型输出转换为检测结果
        
        Args:
            cls_logits: 分类预测
            bbox_regressions: 回归预测
            anchors: 锚点
            image_shape: 图像形状
            
        Returns:
            list: 检测结果列表
        """
        # 简化的后处理（实际实现需要NMS等）
        # 这里返回一个占位符
        return [{'boxes': torch.zeros((0, 4)), 'scores': torch.zeros((0,)), 'labels': torch.zeros((0,), dtype=torch.int64)}]


class SSDCustomDetector:
    """使用自定义骨干网络的SSD目标检测模型"""
    
    def __init__(self, num_classes: int = 25, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化SSD检测器（使用自定义骨干网络）
        
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
        """初始化SSD模型（使用自定义骨干网络）"""
        try:
            # 创建自定义骨干网络
            backbone = CustomBackbone()
            
            # 定义锚点生成器（针对水下目标优化）
            anchor_generator = torchvision.models.detection.ssd.DefaultBoxGenerator(
                aspect_ratios=[[1.5, 2.0], [1.5, 2.0, 3.0], [1.5, 2.0, 3.0], 
                              [1.5, 2.0, 3.0], [1.5, 2.0], [1.5, 2.0]],
                min_ratio=0.15,  # 更小的最小比例，适应水下小目标
                max_ratio=0.85   # 调整最大比例
            )
            
            # 创建SSD模型
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
            
            print(f"自定义SSD模型初始化完成，使用设备: {self.device}")
            print(f"骨干网络参数数量: {sum(p.numel() for p in backbone.parameters()):,}")
            print(f"总模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"初始化自定义SSD模型失败: {e}")
            raise
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """生成用于可视化的颜色"""
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
              learning_rate: float = 0.001, save_dir: str = 'checkpoints/ssd_custom',
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
        
        # 定义优化器（使用AdamW优化器，更适合自定义网络）
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0001)
        
        # 学习率调度器（使用余弦退火）
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
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
        
        print(f"开始训练自定义SSD模型，共{epochs}轮...")
        
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
                checkpoint_path = os.path.join(save_dir, f'ssd_custom_epoch_{epoch+1}.pth')
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
        print("自定义SSD模型训练完成")
        
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
            'model_type': 'SSD_Custom',
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'is_trained': self.is_trained,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'device': self.device
        }


# 测试代码
if __name__ == "__main__":
    # 测试自定义SSD模型
    detector = SSDCustomDetector()
