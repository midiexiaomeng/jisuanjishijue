"""
调试SSD分类头结构
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision
from torchvision.models.detection.ssd import SSD, SSDClassificationHead
from torchvision.models.detection import _utils as det_utils

def debug_ssd_head_structure():
    """调试SSD分类头结构"""
    print("调试SSD分类头结构...")
    
    # 创建简单的SSD模型
    backbone = torchvision.models.mobilenet_v3_large(pretrained=True)
    backbone = backbone.features
    backbone.out_channels = det_utils.retrieve_out_channels(backbone, (300, 300))
    
    print(f"backbone.out_channels: {backbone.out_channels}")
    
    # 注意：backbone.out_channels 是一个列表，我们需要第一个元素
    in_channels = backbone.out_channels[0] if isinstance(backbone.out_channels, list) else backbone.out_channels
    print(f"实际输入通道数: {in_channels}")
    
    # 定义锚点生成器
    anchor_generator = torchvision.models.detection.ssd.DefaultBoxGenerator(
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        min_ratio=0.2,
        max_ratio=0.9
    )
    
    # 计算每个位置的锚点数量
    num_anchors = anchor_generator.num_anchors_per_location()
    print(f"每个位置的锚点数量: {num_anchors}")
    
    # 定义分类头
    classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=25  # 24类 + 背景
    )
    
    print(f"\n分类头结构: {classification_head}")
    print(f"分类头类型: {type(classification_head)}")
    
    # 检查module_list
    print(f"\n分类头module_list长度: {len(classification_head.module_list)}")
    for i, module in enumerate(classification_head.module_list):
        print(f"  模块[{i}]: {module}")
        print(f"    输入通道: {module.in_channels}")
        print(f"    输出通道: {module.out_channels}")
        print(f"    核大小: {module.kernel_size}")
    
    # 创建虚拟特征
    # SSD期望来自不同特征层的特征列表
    # 对于MobileNetV3，我们需要模拟不同尺度的特征
    dummy_features = [
        torch.randn(1, in_channels, 38, 38),  # 第一个特征层
        torch.randn(1, in_channels, 19, 19),  # 第二个特征层
        torch.randn(1, in_channels, 10, 10),  # 第三个特征层
        torch.randn(1, in_channels, 5, 5),    # 第四个特征层
        torch.randn(1, in_channels, 3, 3),    # 第五个特征层
        torch.randn(1, in_channels, 1, 1),    # 第六个特征层
    ]
    
    print(f"\n虚拟特征列表长度: {len(dummy_features)}")
    for i, feat in enumerate(dummy_features):
        print(f"  特征[{i}]形状: {feat.shape}")
    
    # 测试分类头的前向传播
    try:
        print("\n尝试调用分类头...")
        head_outputs = classification_head(dummy_features)
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
        print(f"✗ 调用分类头失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 检查SSD模型的head属性期望什么
    print("\n" + "="*60)
    print("检查SSD模型head属性...")
    
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
    
    print(f"模型head: {model.head}")
    print(f"模型head类型: {type(model.head)}")
    
    # 检查SSD模型的forward方法签名
    print("\n检查SSD模型forward方法...")
    import inspect
    try:
        sig = inspect.signature(model.forward)
        print(f"forward方法签名: {sig}")
    except:
        print("无法获取forward方法签名")

if __name__ == "__main__":
    debug_ssd_head_structure()
