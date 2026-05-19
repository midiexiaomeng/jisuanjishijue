"""
调试SSD模型的head_outputs结构
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision
from torchvision.models.detection.ssd import SSD, SSDClassificationHead
from torchvision.models.detection import _utils as det_utils

def debug_ssd_model():
    """调试SSD模型结构"""
    print("调试SSD模型结构...")
    
    # 创建简单的SSD模型
    backbone = torchvision.models.mobilenet_v3_large(pretrained=True)
    backbone = backbone.features
    backbone.out_channels = det_utils.retrieve_out_channels(backbone, (300, 300))
    
    # 定义锚点生成器
    anchor_generator = torchvision.models.detection.ssd.DefaultBoxGenerator(
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        min_ratio=0.2,
        max_ratio=0.9
    )
    
    # 定义分类头
    num_anchors = anchor_generator.num_anchors_per_location()
    classification_head = SSDClassificationHead(
        in_channels=backbone.out_channels,
        num_anchors=num_anchors,
        num_classes=25  # 24类 + 背景
    )
    
    # 创建SSD模型
    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(300, 300),
        num_classes=25,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        head=classification_head,
        score_thresh=0.01,
        nms_thresh=0.45,
        detections_per_img=200,
        topk_candidates=400
    )
    
    print(f"模型类型: {type(model)}")
    print(f"模型结构: {model}")
    
    # 检查head属性
    print(f"\n模型head属性: {model.head}")
    print(f"head类型: {type(model.head)}")
    
    # 检查head是否有classification_head属性
    if hasattr(model.head, 'classification_head'):
        print(f"head.classification_head: {model.head.classification_head}")
        print(f"head.classification_head类型: {type(model.head.classification_head)}")
    
    # 创建虚拟输入
    batch_size = 2
    dummy_images = [torch.randn(3, 300, 300) for _ in range(batch_size)]
    dummy_targets = [
        {
            'boxes': torch.tensor([[50, 50, 150, 150]]),
            'labels': torch.tensor([1]),
            'image_id': torch.tensor([0]),
            'area': torch.tensor([10000]),
            'iscrowd': torch.tensor([0])
        },
        {
            'boxes': torch.tensor([[100, 100, 200, 200]]),
            'labels': torch.tensor([2]),
            'image_id': torch.tensor([1]),
            'area': torch.tensor([10000]),
            'iscrowd': torch.tensor([0])
        }
    ]
    
    print(f"\n虚拟图像形状: {[img.shape for img in dummy_images]}")
    print(f"虚拟目标: {dummy_targets}")
    
    # 测试前向传播
    model.train()
    try:
        # 直接调用模型
        print("\n尝试直接调用模型...")
        loss_dict = model(dummy_images, dummy_targets)
        print(f"✓ 直接调用成功")
        print(f"损失字典: {loss_dict}")
    except Exception as e:
        print(f"✗ 直接调用失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试head的前向传播
    print("\n尝试调用head...")
    try:
        # 获取特征
        features = model.backbone(dummy_images[0].unsqueeze(0))
        print(f"特征类型: {type(features)}")
        
        if isinstance(features, dict):
            print(f"特征键: {list(features.keys())}")
            for k, v in features.items():
                print(f"  {k}: {v.shape}")
        elif isinstance(features, list):
            print(f"特征列表长度: {len(features)}")
            for i, f in enumerate(features):
                print(f"  特征[{i}]: {f.shape}")
        else:
            print(f"特征形状: {features.shape}")
        
        # 尝试调用head
        if hasattr(model, 'head'):
            print(f"\n尝试调用model.head...")
            head_outputs = model.head(features)
            print(f"head_outputs类型: {type(head_outputs)}")
            
            if isinstance(head_outputs, dict):
                print(f"head_outputs键: {list(head_outputs.keys())}")
                for k, v in head_outputs.items():
                    print(f"  {k}: {type(v)}, 形状: {v.shape if hasattr(v, 'shape') else 'N/A'}")
            elif isinstance(head_outputs, list):
                print(f"head_outputs列表长度: {len(head_outputs)}")
                for i, out in enumerate(head_outputs):
                    print(f"  head_outputs[{i}]: {type(out)}")
                    if isinstance(out, dict):
                        print(f"    键: {list(out.keys())}")
                    elif hasattr(out, 'shape'):
                        print(f"    形状: {out.shape}")
            elif hasattr(head_outputs, 'shape'):
                print(f"head_outputs形状: {head_outputs.shape}")
            else:
                print(f"head_outputs: {head_outputs}")
                
    except Exception as e:
        print(f"✗ 调用head失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ssd_model()
