#!/usr/bin/env python3
"""
测试新的Faster R-CNN模型
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.faster_rcnn_new import FasterRCNNNew

def test_model_initialization():
    """测试模型初始化"""
    print("=" * 80)
    print("测试新的Faster R-CNN模型初始化...")
    print("=" * 80)
    
    try:
        # 创建模型实例
        model = FasterRCNNNew(num_classes=25, device='cpu')
        print("✓ 模型初始化成功")
        print(f"  设备: {model.device}")
        print(f"  类别数量: {model.num_classes}")
        print(f"  类别名称: {model.class_names[:5]}...")  # 只显示前5个类别
        
        # 检查模型是否已创建
        if model.model is not None:
            print("✓ 模型架构创建成功")
            print(f"  模型类型: {type(model.model).__name__}")
            
            # 检查模型参数
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            print(f"  总参数数量: {total_params:,}")
            print(f"  可训练参数数量: {trainable_params:,}")
        else:
            print("✗ 模型架构创建失败")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """测试模型前向传播"""
    print("\n" + "=" * 80)
    print("测试模型前向传播...")
    print("=" * 80)
    
    try:
        # 创建模型实例
        model = FasterRCNNNew(num_classes=25, device='cpu')
        
        # 创建模拟输入数据
        batch_size = 2
        image_size = 640
        
        # 创建模拟图像（随机像素值）
        images = [torch.rand(3, image_size, image_size) for _ in range(batch_size)]
        
        # 创建模拟目标（随机边界框和标签）
        targets = []
        for i in range(batch_size):
            # 随机生成1-3个边界框
            num_boxes = torch.randint(1, 4, (1,)).item()
            boxes = torch.rand(num_boxes, 4) * image_size
            boxes[:, 2:] += boxes[:, :2]  # 确保x2 > x1, y2 > y1
            
            # 随机生成标签（1-24，因为0是背景）
            labels = torch.randint(1, 25, (num_boxes,))
            
            targets.append({
                'boxes': boxes,
                'labels': labels
            })
        
        print(f"  创建了 {batch_size} 张模拟图像")
        print(f"  每张图像有 {[len(t['boxes']) for t in targets]} 个边界框")
        
        # 测试训练模式下的前向传播
        model.model.train()
        loss_dict = model.model(images, targets)
        
        print("✓ 训练模式前向传播成功")
        print(f"  损失字典包含 {len(loss_dict)} 个损失项:")
        for key, value in loss_dict.items():
            print(f"    {key}: {value.item():.4f}")
        
        # 测试评估模式下的前向传播
        model.model.eval()
        with torch.no_grad():
            predictions = model.model(images)
        
        print("✓ 评估模式前向传播成功")
        print(f"  预测结果包含 {len(predictions)} 个预测:")
        for i, pred in enumerate(predictions):
            print(f"    图像{i}: {len(pred['boxes'])} 个检测框")
        
        return True
        
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection():
    """测试目标检测功能"""
    print("\n" + "=" * 80)
    print("测试目标检测功能...")
    print("=" * 80)
    
    try:
        # 创建模型实例
        model = FasterRCNNNew(num_classes=25, device='cpu')
        
        # 创建模拟图像（随机像素值）
        image_size = 640
        # 创建形状为(高度, 宽度, 通道)的图像，这是PIL期望的格式
        image = torch.rand(image_size, image_size, 3).numpy() * 255
        image = image.astype('uint8')
        
        print(f"  创建了 {image.shape} 的模拟图像 (高度, 宽度, 通道)")
        
        # 测试检测功能
        results = model.detect(image, confidence_threshold=0.1)
        
        print(f"✓ 目标检测成功")
        print(f"  检测到 {len(results)} 个目标:")
        for i, (class_name, confidence, bbox) in enumerate(results[:3]):  # 只显示前3个
            print(f"    目标{i}: {class_name} (置信度: {confidence:.4f}), 边界框: {bbox}")
        
        if len(results) > 3:
            print(f"    ... 和 {len(results) - 3} 个其他目标")
        
        return True
        
    except Exception as e:
        print(f"✗ 目标检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_methods():
    """测试模型的其他方法"""
    print("\n" + "=" * 80)
    print("测试模型的其他方法...")
    print("=" * 80)
    
    try:
        # 创建模型实例
        model = FasterRCNNNew(num_classes=25, device='cpu')
        
        # 测试获取类别名称
        class_names = model.class_names
        print(f"✓ 获取类别名称成功: {len(class_names)} 个类别")
        
        # 测试获取类别数量
        num_classes = len(class_names)
        print(f"✓ 获取类别数量成功: {num_classes}")
        
        # 测试保存模型（模拟）
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            model.save_model(temp_path)
            print(f"✓ 保存模型成功: {temp_path}")
            
            # 测试加载预训练权重
            model.load_pretrained(temp_path)
            print(f"✓ 加载预训练权重成功")
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"✗ 模型方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试新的Faster R-CNN模型")
    print("=" * 80)
    
    # 运行所有测试
    tests = [
        ("模型初始化", test_model_initialization),
        ("前向传播", test_forward_pass),
        ("目标检测", test_detection),
        ("模型方法", test_model_methods),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"测试 '{test_name}' 发生异常: {e}")
            results.append((test_name, False))
    
    # 打印测试结果摘要
    print("\n" + "=" * 80)
    print("测试结果摘要")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！新的Faster R-CNN模型可以正常工作。")
        return True
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，需要进一步调试。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
