#!/usr/bin/env python3
"""
测试Faster R-CNN模型修复后的训练功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.faster_rcnn_model import FasterRCNNDetector

def test_faster_rcnn_training():
    """测试Faster R-CNN模型训练"""
    print("=" * 80)
    print("测试Faster R-CNN模型修复后的训练功能")
    print("=" * 80)
    
    try:
        # 初始化检测器
        print("1. 初始化Faster R-CNN检测器...")
        detector = FasterRCNNDetector(num_classes=25)
        print("   检测器初始化成功")
        
        # 测试数据加载器创建
        print("\n2. 测试数据加载器创建...")
        data_path = 'data/YOLO/dataset.yaml'
        
        if os.path.exists(data_path):
            print(f"   使用数据路径: {data_path}")
            
            # 测试数据加载器参数传递
            try:
                # 导入数据加载器
                from utils.data_loader import create_data_loaders
                
                # 测试创建数据加载器（包含prefetch_factor参数）
                train_loader, val_loader = create_data_loaders(
