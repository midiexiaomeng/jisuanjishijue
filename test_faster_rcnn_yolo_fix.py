"""
测试Faster R-CNN处理YOLO格式数据的修复
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from utils.data_loader import create_data_loaders
from models.faster_rcnn_model import FasterRCNNDetector

def test_yolo_data_loader():
    """测试YOLO数据加载器"""
    print("测试YOLO数据加载器...")
    
    data_path = 'data/YOLO/dataset.yaml'
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(data_path, batch_size=2, num_workers=0)
        
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        # 获取一个批次
        images, targets = next(iter(train_loader))
        print(f"批次图像数量: {len(images)}")
        print(f"图像形状: {images[0].shape}")
        
        # 检查边界框坐标
        for i, target in enumerate(targets):
            boxes = target['boxes']
            if len(boxes) > 0:
                print(f"\n目标 {i} 的边界框:")
                print(f"  边界框数量: {len(boxes)}")
                print(f"  第一个边界框: {boxes[0]}")
                print(f"  边界框范围 - x_min: {boxes[:, 0].min():.4f}, x_max: {boxes[:, 2].max():.4f}")
                print(f"  边界框范围 - y_min: {boxes[:, 1].min():.4f}, y_max: {boxes[:, 3].max():.4f}")
                
                # 检查边界框是否在合理范围内
                img_width = images[i].shape[2]  # 图像宽度
                img_height = images[i].shape[1]  # 图像高度
                print(f"  图像尺寸: {img_width}x{img_height}")
                
                # 检查边界框是否超出图像范围
                if boxes[:, 0].min() < 0 or boxes[:, 1].min() < 0:
                    print(f"  警告: 边界框坐标有负值!")
                if boxes[:, 2].max() > img_width or boxes[:, 3].max() > img_height:
                    print(f"  警告: 边界框坐标超出图像范围!")
        
        print("\nYOLO数据加载器测试成功!")
        return True
        
    except Exception as e:
        print(f"YOLO数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faster_rcnn_training():
    """测试Faster R-CNN训练"""
    print("\n测试Faster R-CNN训练...")
    
    try:
        # 创建模型
        model = FasterRCNNDetector(num_classes=25, device='cpu')
        
        # 测试训练（只训练一个批次）
        data_path = 'data/YOLO/dataset.yaml'
        
        # 使用固定数据路径
        fixed_data_dir = 'data/YOLO'
        
        print(f"使用固定数据目录: {fixed_data_dir}")
        
        # 检查数据目录是否存在
        if not os.path.exists(fixed_data_dir):
            print(f"数据目录不存在: {fixed_data_dir}")
            return False
        
        # 尝试训练一个批次
        try:
            # 创建数据加载器
            train_loader, val_loader = create_data_loaders(data_path, batch_size=2, num_workers=0)
            
            # 获取一个批次
            images, targets = next(iter(train_loader))
            
            # 移动到设备
            images = [img.to('cpu') for img in images]
            targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
            
            # 前向传播
            model.model.train()
            loss_dict = model.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            print(f"前向传播成功，损失: {losses.item():.4f}")
            print("Faster R-CNN训练测试成功!")
            return True
            
        except Exception as e:
            print(f"Faster R-CNN训练测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"创建Faster R-CNN模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("测试Faster R-CNN处理YOLO格式数据的修复")
    print("=" * 60)
    
    # 测试1: YOLO数据加载器
    test1_success = test_yolo_data_loader()
    
    # 测试2: Faster R-CNN训练
    test2_success = test_faster_rcnn_training()
    
    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"  测试1 (YOLO数据加载器): {'成功' if test1_success else '失败'}")
    print(f"  测试2 (Faster R-CNN训练): {'成功' if test2_success else '失败'}")
    
    if test1_success and test2_success:
        print("\n所有测试通过! Faster R-CNN可以处理YOLO格式数据。")
    else:
        print("\n测试失败，需要进一步修复。")

if __name__ == '__main__':
    main()
