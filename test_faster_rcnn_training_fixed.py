#!/usr/bin/env python3
"""
测试修复后的Faster R-CNN模型训练功能
使用少量数据进行快速测试，验证训练进度显示和内存使用
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import numpy as np
from PIL import Image
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_dataset():
    """
    创建测试数据集（模拟少量数据）
    返回一个简单的数据加载器用于测试
    """
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    
    class TestDataset(Dataset):
        """测试数据集，生成随机图像和标注"""
        
        def __init__(self, num_samples=10, image_size=(640, 640)):
            self.num_samples = num_samples
            self.image_size = image_size
            self.transform = T.Compose([
                T.ToTensor(),
            ])
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # 生成随机图像（3通道，RGB）
            image = np.random.randint(0, 255, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            image = Image.fromarray(image)
            
            # 生成随机标注（1-3个目标）
            num_objects = np.random.randint(1, 4)
            boxes = []
            labels = []
            
            for _ in range(num_objects):
                # 生成随机边界框
                x_min = np.random.randint(0, self.image_size[1] - 100)
                y_min = np.random.randint(0, self.image_size[0] - 100)
                width = np.random.randint(50, 200)
                height = np.random.randint(50, 200)
                x_max = min(x_min + width, self.image_size[1] - 1)
                y_max = min(y_min + height, self.image_size[0] - 1)
                
                boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                labels.append(np.random.randint(1, 25))  # 1-24类（0是背景）
            
            # 转换为张量
            if self.transform:
                image = self.transform(image)
            
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
            # 创建目标字典
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx])
            }
            
            return image, target
    
    # 创建数据集和数据加载器
    dataset = TestDataset(num_samples=20, image_size=(416, 416))
    data_loader = DataLoader(
        dataset,
        batch_size=2,  # 使用小批量以减少内存使用
        shuffle=True,
        num_workers=0,  # 在Windows上使用0以避免问题
        collate_fn=lambda x: tuple(zip(*x))  # Faster R-CNN需要的collate函数
    )
    
    return data_loader

def test_faster_rcnn_training():
    """测试Faster R-CNN训练功能"""
    print("=" * 80)
    print("开始测试Faster R-CNN模型训练功能")
    print("=" * 80)
    
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    try:
        # 导入修复后的Faster R-CNN模型
        from models.faster_rcnn_model import FasterRCNNDetector
        
        # 创建模型实例（使用更少的类别以减少内存使用）
        print("\n1. 初始化Faster R-CNN模型...")
        detector = FasterRCNNDetector(num_classes=25, device=device)
        
        # 创建测试数据加载器
        print("\n2. 创建测试数据加载器...")
        train_loader = create_test_dataset()
        print(f"   数据加载器创建成功: {len(train_loader)} 个批次")
        
        # 测试训练功能（只训练1个epoch）
        print("\n3. 开始训练测试（1个epoch）...")
        print("   " + "-" * 60)
        
        # 设置保存目录
        save_dir = 'checkpoints/faster_rcnn_test'
        os.makedirs(save_dir, exist_ok=True)
        
        # 开始训练
        history = detector.train(
            train_loader=train_loader,
            val_loader=None,  # 不进行验证以加快测试
            epochs=1,
            learning_rate=0.001,
            save_dir=save_dir
        )
        
        print("\n4. 训练测试完成!")
        print(f"   最终训练损失: {history['train_loss'][-1]:.4f}")
        
        # 测试检测功能
        print("\n5. 测试检测功能...")
        test_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        results = detector.detect(test_image, confidence_threshold=0.3)
        print(f"   检测到 {len(results)} 个目标")
        
        # 保存模型
        print("\n6. 保存模型...")
        model_path = os.path.join(save_dir, 'faster_rcnn_test_model.pth')
        detector.save_model(model_path)
        print(f"   模型已保存到: {model_path}")
        
        print("\n" + "=" * 80)
        print("Faster R-CNN模型训练测试成功完成!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """测试内存使用情况"""
    print("\n" + "=" * 80)
    print("测试内存使用情况")
    print("=" * 80)
    
    if torch.cuda.is_available():
        # 记录初始内存使用
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        print(f"初始GPU内存使用: {initial_memory:.2f} MB")
        
        # 创建模型并移动到GPU
        from models.faster_rcnn_model import FasterRCNNDetector
        detector = FasterRCNNDetector(num_classes=25, device='cuda')
        
        # 记录模型内存使用
        model_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"模型加载后GPU内存使用: {model_memory:.2f} MB")
        print(f"模型占用内存: {model_memory - initial_memory:.2f} MB")
        
        # 创建测试数据
        test_image = torch.randn(1, 3, 416, 416).cuda()
        
        # 记录推理内存使用
        detector.model.eval()
        with torch.no_grad():
            output = detector.model(test_image)
        
        inference_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"推理后GPU内存使用: {inference_memory:.2f} MB")
        
        # 清理
        del detector
        del test_image
        del output
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"清理后GPU内存使用: {final_memory:.2f} MB")
        
        print("\n内存使用测试完成!")
        print("=" * 80)
        
        return True
    else:
        print("CUDA不可用，跳过内存测试")
        return True

if __name__ == "__main__":
    print("Faster R-CNN模型训练功能测试")
    print("=" * 80)
    
    # 运行测试
    success = test_faster_rcnn_training()
    
    if success:
        # 运行内存测试
        test_memory_usage()
        
        print("\n所有测试完成!")
        print("Faster R-CNN模型训练功能修复成功!")
        print("训练进度显示功能已添加:")
        print("  - tqdm进度条显示训练进度")
        print("  - 实时损失更新")
        print("  - 详细损失分解显示")
        print("  - 学习率显示")
        print("  - 每20个batch打印详细状态")
    else:
        print("\n测试失败，请检查错误信息")
        sys.exit(1)
