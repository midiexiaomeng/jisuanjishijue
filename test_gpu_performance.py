#!/usr/bin/env python3
"""
GPU性能测试脚本
用于简单测试SSD模型的GPU计算性能，排除数据加载等因素的干扰
"""

import torch
import time
from models.ssd_model import SSDDetector

def test_gpu_performance():
    """测试SSD模型的GPU计算性能"""
    print("=" * 50)
    print("SSD模型GPU性能测试")
    print("=" * 50)
    
    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("[警告] 未检测到可用的GPU，将使用CPU进行测试")
        return
    
    device = torch.device("cuda")
    print(f"[信息] 使用GPU: {torch.cuda.get_device_name(device)}")
    print(f"[信息] GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    
    # 初始化SSD检测器
    print("\n[信息] 初始化SSD模型...")
    ssd = SSDDetector(num_classes=25, device="cuda")
    
    # 创建随机测试数据（模拟8张300x300的RGB图像）
    print("\n[信息] 创建测试数据...")
    batch_size = 8
    images = [torch.randn(3, 300, 300).to(device) for _ in range(batch_size)]
    
    # 创建模拟目标数据
    targets = []
    for _ in range(batch_size):
        # 随机生成5个目标
        num_targets = 5
        
        # 生成有效边界框（确保x1 < x2且y1 < y2）
        x1 = torch.rand(num_targets, 1).to(device) * 250  # x1范围[0, 250)
        y1 = torch.rand(num_targets, 1).to(device) * 250  # y1范围[0, 250)
        x2 = x1 + torch.rand(num_targets, 1).to(device) * 50 + 10  # x2 = x1 + [10, 60)
        y2 = y1 + torch.rand(num_targets, 1).to(device) * 50 + 10  # y2 = y1 + [10, 60)
        
        boxes = torch.cat([x1, y1, x2, y2], dim=1)  # 合并为[num_targets, 4]
        labels = torch.randint(1, 25, (num_targets,)).to(device)  # 随机类别
        targets.append({"boxes": boxes, "labels": labels})
    
    # 设置优化器
    optimizer = torch.optim.SGD(ssd.model.parameters(), lr=0.001, momentum=0.9)
    
    # 测试基础训练性能（无优化）
    print("\n[测试1] 基础训练性能（无优化）")
    ssd.model.train()
    torch.cuda.empty_cache()
    
    start_time = time.time()
    
    # 进行10次前向传播和反向传播
    for i in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss_dict = ssd.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"  迭代 {i+1}/10 完成，损失: {losses.item():.4f}")
    
    base_time = time.time() - start_time
    print(f"基础训练时间: {base_time:.4f} 秒")
    print(f"平均每轮时间: {base_time / 10:.4f} 秒")
    
    # 测试混合精度训练性能
    print("\n[测试2] 混合精度训练性能")
    scaler = torch.cuda.amp.GradScaler()
    ssd.model.train()
    torch.cuda.empty_cache()
    
    start_time = time.time()
    
    # 进行10次前向传播和反向传播（混合精度）
    for i in range(10):
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            loss_dict = ssd.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (i + 1) % 5 == 0:
            print(f"  迭代 {i+1}/10 完成，损失: {losses.item():.4f}")
    
    amp_time = time.time() - start_time
    print(f"混合精度训练时间: {amp_time:.4f} 秒")
    print(f"平均每轮时间: {amp_time / 10:.4f} 秒")
    print(f"性能提升: {(base_time / amp_time - 1) * 100:.2f}%")
    
    # 测试梯度累积性能
    print("\n[测试3] 梯度累积训练性能 (steps=4)")
    gradient_accumulation_steps = 4
    scaler = torch.cuda.amp.GradScaler()
    ssd.model.train()
    torch.cuda.empty_cache()
    
    start_time = time.time()
    
    # 进行10次前向传播和反向传播（梯度累积）
    optimizer.zero_grad(set_to_none=True)
    for i in range(10):
        with torch.amp.autocast('cuda'):
            loss_dict = ssd.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses = losses / gradient_accumulation_steps
        
        scaler.scale(losses).backward()
        
        # 每4步更新一次参数
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == 10:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        if (i + 1) % 5 == 0:
            print(f"  迭代 {i+1}/10 完成，损失: {losses.item() * gradient_accumulation_steps:.4f}")
    
    grad_accum_time = time.time() - start_time
    print(f"梯度累积训练时间: {grad_accum_time:.4f} 秒")
    print(f"平均每轮时间: {grad_accum_time / 10:.4f} 秒")
    
    print(f"\n" + "=" * 50)
    print("性能测试总结")
    print("=" * 50)
    print(f"基础训练: {base_time:.4f} 秒 (10轮)")
    print(f"混合精度: {amp_time:.4f} 秒 (10轮) - 提升 {(base_time / amp_time - 1) * 100:.2f}%")
    print(f"梯度累积: {grad_accum_time:.4f} 秒 (10轮) - 提升 {(base_time / grad_accum_time - 1) * 100:.2f}%")
    print(f"\n[提示] 使用以下命令监控GPU利用率：")
    print("- Windows: 任务管理器 -> 性能 -> GPU")
    print("- Linux: nvidia-smi (实时监控: watch -n 1 nvidia-smi)")

if __name__ == "__main__":
    test_gpu_performance()
