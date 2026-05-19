"""
测试Faster R-CNN模型使用COCO数据加载器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.faster_rcnn_model import FasterRCNNDetector

def test_faster_rcnn_coco():
    """测试Faster R-CNN模型使用COCO数据集"""
    print("测试Faster R-CNN模型使用COCO数据集...")
    
    try:
        # 创建Faster R-CNN检测器
        detector = FasterRCNNDetector(num_classes=25, device='cpu')
        print("Faster R-CNN检测器创建成功")
        
        # 测试数据集类型检测
        coco_data_path = 'data/coco/coco.yaml'
        dataset_type = detector._detect_dataset_type(coco_data_path)
        print(f"数据集类型检测结果: {dataset_type}")
        
        # 测试训练方法（只创建数据加载器，不实际训练）
        print("\n测试数据加载器创建...")
        try:
            # 为了测试数据加载器创建，我们需要直接调用数据加载器创建函数
            # 而不是通过train()方法
            from utils.coco_data_loader import create_coco_data_loaders
            
            train_loader, val_loader = create_coco_data_loaders(
                coco_data_path,
                batch_size=2,
                num_workers=0
            )
            
            print("数据加载器创建成功!")
            print(f"训练集大小: {len(train_loader.dataset)}")
            print(f"验证集大小: {len(val_loader.dataset)}")
            
            # 测试一个批次
            images, targets = next(iter(train_loader))
            print(f"批次图像数量: {len(images)}")
            print(f"图像形状: {images[0].shape}")
            print(f"目标数量: {len(targets)}")
            
            if len(targets) > 0:
                print(f"第一个目标的boxes形状: {targets[0]['boxes'].shape}")
                print(f"第一个目标的labels: {targets[0]['labels']}")
            
            return True
            
        except Exception as e:
            print(f"数据加载器创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_faster_rcnn_coco()
    if success:
        print("\n测试成功! Faster R-CNN模型可以使用COCO数据加载器进行训练。")
    else:
        print("\n测试失败! 请检查错误信息。")
