#!/usr/bin/env python3
"""
简单的GPU利用率测试脚本
用于快速验证SSD模型的GPU训练优化效果
"""

import torch
import time
from models.ssd_model import SSDDetector

def test_gpu_utilization():
    """测试GPU利用率"""
    print("=" * 50)
    print("SSD模型GPU利用率测试")
    print("=" * 50)
    
    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("[错误] 未检测到可用的GPU")
        return
    
    device = torch.device("cuda")
    print(f"[信息] 使用GPU: {torch.cuda.get_device_name(device)}")
    print(f"[信息] GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    
    # 初始化SSD检测器
    print("\n[信息] 初始化SSD模型...")
    ssd = SSDDetector(num_classes=25, device="cuda")
    
    # 创建随机测试数据
    batch_size = 16
    print(f"\n[信息] 创建随机测试数据 (batch_size={batch_size})...")
    
    # 创建随机图像（300x300x3）
    images = [torch.randn(3, 300, 300).to(device) for _ in range(batch_size)]
    
    # 创建随机目标（每个图像有2个目标）
    targets = []
    for _ in range(batch_size):
        # 生成有效的边界框（确保x_max > x_min，y_max > y_min）
        num_boxes = 2
        boxes = torch.zeros(num_boxes, 4).to(device)
        
        # 生成x_min和y_min
        boxes[:, 0] = torch.rand(num_boxes).to(device) * 200  # x_min (0-200)
        boxes[:, 1] = torch.rand(num_boxes).to(device) * 200  # y_min (0-200)
        
        # 生成宽度和高度（确保为正）
        width = torch.rand(num_boxes).to(device) * 100 + 10  # 宽度 (10-110)
        height = torch.rand(num_boxes).to(device) * 100 + 10  # 高度 (10-110)
        
        # 计算x_max和y_max
        boxes[:, 2] = boxes[:, 0] + width  # x_max
        boxes[:, 3] = boxes[:, 1] + height  # y_max
        
        target = {
            'boxes': boxes,
            'labels': torch.randint(1, 25, (num_boxes,)).to(device)
        }
        targets.append(target)
    
    # 设置模型为训练模式
    ssd.model.train()
    
    # 测试前向传播
    print("\n[信息] 测试前向传播...")
    start_time = time.time()
    
    # 运行10次前向传播
    for i in range(10):
        with torch.amp.autocast('cuda'):
            loss_dict = ssd.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        if (i + 1) % 2 == 0:
            print(f"  前向传播 {i+1}/10 完成，损失: {losses.item():.4f}")
    
    forward_time = time.time() - start_time
    print(f"前向传播平均时间: {forward_time / 10:.4f} 秒/次")
    
    # 测试反向传播
    print("\n[信息] 测试反向传播...")
    
    # 定义优化器
    params = [p for p in ssd.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    # 初始化混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    start_time = time.time()
    
    # 运行5次反向传播
    for i in range(5):
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            loss_dict = ssd.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"  反向传播 {i+1}/5 完成，损失: {losses.item():.4f}")
    
    backward_time = time.time() - start_time
    print(f"反向传播平均时间: {backward_time / 5:.4f} 秒/次")
    
    # 测试梯度累积
    print("\n[信息] 测试梯度累积...")
    gradient_accumulation_steps = 4
    
    start_time = time.time()
    
    optimizer.zero_grad(set_to_none=True)
    
    # 运行梯度累积
    for i in range(gradient_accumulation_steps):
        with torch.amp.autocast('cuda'):
            loss_dict = ssd.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses = losses / gradient_accumulation_steps  # 除以累积步数
        
        scaler.scale(losses).backward()
        print(f"  梯度累积 {i+1}/{gradient_accumulation_steps} 完成")
    
    scaler.step(optimizer)
    scaler.update()
    
    grad_accum_time = time.time() - start_time
    print(f"梯度累积时间: {grad_accum_time:.4f} 秒")
    
    print(f"\n" + "=" * 50)
    print("测试完成!")
    print(f"=" * 50)
    print("\n[提示] 使用以下命令监控GPU利用率：")
    print("- Windows: 任务管理器 -> 性能 -> GPU")
    print("- Linux: nvidia-smi (实时监控: watch -n 1 nvidia-smi)")
    
    # 清理GPU内存
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_gpu_utilization()