"""
自定义YOLOv8模型实现
包含自写的骨干网络、颈部网络和检测头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import os


# ==================== 自定义骨干网络组件 ====================

class ConvBlock(nn.Module):
    """自定义卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_bn=True, activation='silu', use_se=False):
        """
        初始化卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            use_bn: 是否使用批量归一化
            activation: 激活函数类型 ('silu', 'relu', 'leaky_relu')
            use_se: 是否使用Squeeze-and-Excitation注意力
        """
        super().__init__()
        
        # 卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=not use_bn)
        
        # 批量归一化
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # 激活函数
        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.activation = nn.Identity()
        
