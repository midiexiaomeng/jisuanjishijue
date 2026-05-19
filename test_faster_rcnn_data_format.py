"""
测试Faster R-CNN数据格式和修复CUDA流不匹配问题
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import create_data_loaders
from models.faster_rcnn_new import FasterRCNNNew

def test_data_format():
    """测试数据加载器返回的数据格式"""
    print("测试数据加载器格式...")
    
    try:
        # 创建数据加载器
        data_path = 'data/YOLO/dataset.yaml'
        train_loader, val_loader = create_data_loaders(data_path, batch_size=2, num_workers=0)
        
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        # 获取一个批次
        batch = next(iter(train_loader))
        print(f"\n批次类型: {type(batch)}")
        print(f"批次长度: {len(batch)}")
        
        if len(batch) == 2:
            images, targets = batch
            print("批次格式: (images, targets)")
        elif len(batch) == 3:
            images, targets, img_paths = batch
            print("批次格式: (images, targets, img_paths)")
        else:
            print(f"未知批次格式: {batch}")
            return
        
        print(f"\n图像数量: {len(images)}")
        print(f"图像类型: {type(images[0])}")
        print(f"图像形状: {images[0].shape}")
        print(f"图像设备: {images[0].device}")
        
        print(f"\n目标数量: {len(targets)}")
        print(f"目标类型: {type(targets[0])}")
        
        # 检查第一个目标的键
        if targets:
            first_target = targets[0]
            print(f"目标键: {list(first_target.keys())}")
            
            for key, value in first_target.items():
                print(f"  {key}: 类型={type(value)}, 形状={value.shape if hasattr(value, 'shape') else 'N/A'}, 设备={value.device if hasattr(value, 'device') else 'N/A'}")
        
        # 测试设备移动
        print("\n测试设备移动...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        # 将图像移动到设备
        images_on_device = [img.to(device) for img in images]
        print(f"图像设备移动后: {images_on_device[0].device}")
        
        # 将目标移动到设备
        targets_on_device = []
        for t in targets:
            target_on_device = {}
            for key, value in t.items():
                if isinstance(value, torch.Tensor):
                    target_on_device[key] = value.to(device)
                else:
                    target_on_device[key] = value
            targets_on_device.append(target_on_device)
        
        print(f"目标设备移动后:")
        for key, value in targets_on_device[0].items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: 设备={value.device}")
        
        print("\n数据格式测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_model_training():
    """测试模型训练（简化版本）"""
    print("\n测试模型训练...")
    
    try:
        # 创建数据加载器
        data_path = 'data/YOLO/dataset.yaml'
        train_loader, val_loader = create_data_loaders(data_path, batch_size=2, num_workers=0)
        
        # 创建模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        model = FasterRCNNNew(num_classes=25, device=device)
        
        # 获取一个批次进行测试
        images, targets = next(iter(train_loader))
        
        print(f"\n测试前向传播...")
        
        # 将数据移动到设备
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # 测试前向传播
        model.model.train()
        loss_dict = model.model(images, targets)
        
        print(f"前向传播成功!")
        print(f"损失字典键: {list(loss_dict.keys())}")
        for key, value in loss_dict.items():
            print(f"  {key}: {value.item():.4f}")
        
        # 计算总损失
        losses = sum(loss for loss in loss_dict.values())
        print(f"总损失: {losses.item():.4f}")
        
        print("\n模型训练测试完成!")
        
    except Exception as e:
        print(f"模型训练测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("=" * 80)
    print("Faster R-CNN数据格式和训练测试")
    print("=" * 80)
    
    test_data_format()
    test_model_training()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
