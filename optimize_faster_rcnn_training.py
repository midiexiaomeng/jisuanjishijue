#!/usr/bin/env python3
"""
优化Faster R-CNN训练性能
解决训练过慢和GPU占用率不稳定的问题
"""

import os
import sys
import torch
import time
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def optimize_training():
    """优化Faster R-CNN训练性能"""
    print("=" * 80)
    print("Faster R-CNN训练性能优化")
    print("=" * 80)
    
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"最大GPU内存使用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    try:
        # 导入修复的Faster R-CNN模型
        from models.faster_rcnn_new_fixed import FasterRCNNNewFixed
        
        # 导入数据加载器
        from utils.data_loader import create_data_loaders
        
        # 1. 优化数据加载器配置
        print("\n1. 优化数据加载器配置...")
        
        # 使用较小的批次大小以减少内存使用
        batch_size = 2  # 从16减少到2，因为GPU内存几乎满了
        num_workers = 0  # Windows上使用0以避免问题
        
        # 数据路径
        data_yaml_path = 'data/YOLO/dataset.yaml'
        if not os.path.exists(data_yaml_path):
            print(f"错误: YAML文件不存在: {data_yaml_path}")
            return False
        
        print(f"   批次大小: {batch_size}")
        print(f"   工作线程数: {num_workers}")
        print(f"   数据YAML文件: {data_yaml_path}")
        
        # 2. 创建优化的数据加载器
        print("\n2. 创建优化的数据加载器...")
        
        # 使用较小的图像尺寸以减少内存使用
        from torchvision import transforms
        
        # 创建自定义转换，使用较小的图像尺寸
        transform = transforms.Compose([
            transforms.Resize((416, 416)),  # 从640x640减少到416x416
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 修改create_data_loaders以接受自定义转换
        from utils.data_loader import COUUnderwaterDataset
        from torch.utils.data import DataLoader
        
        # 创建数据集
        train_dataset = COUUnderwaterDataset(data_yaml_path, split='train', transform=transform)
        val_dataset = COUUnderwaterDataset(data_yaml_path, split='val', transform=transform)
        
        # 导入正确的collate_fn函数
        from utils.data_loader import collate_fn
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if device == 'cuda' else False
        )
        
        print(f"   训练集大小: {len(train_dataset)} 图像")
        print(f"   验证集大小: {len(val_dataset)} 图像")
        print(f"   训练批次数量: {len(train_loader)}")
        
        # 3. 创建优化的模型
        print("\n3. 创建优化的Faster R-CNN模型...")
        
        # 使用较小的模型配置
        detector = FasterRCNNNewFixed(
            num_classes=25,  # 24类+背景
            device=device
        )
        
        # 4. 测试训练性能
        print("\n4. 测试训练性能...")
        
        # 设置优化器
        learning_rate = 0.001
        optimizer = torch.optim.AdamW(detector.model.parameters(), lr=learning_rate)
        
        # 训练一个epoch来测试性能
        detector.model.train()
        
        # 记录时间
        start_time = time.time()
        batch_times = []
        
        print(f"\n   开始性能测试 (1个epoch)...")
        print(f"   {'='*60}")
        
        # 只测试前几个批次
        test_batches = min(10, len(train_loader))
        
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, total=test_batches, desc="测试批次")):
            if batch_idx >= test_batches:
                break
                
            batch_start_time = time.time()
            
            try:
                # 将数据移动到设备
                images = [img.to(device) for img in images]
                
                # 准备目标
                targets_on_device = []
                for t in targets:
                    target_on_device = {}
                    for key, value in t.items():
                        if isinstance(value, torch.Tensor):
                            target_on_device[key] = value.to(device)
                        else:
                            target_on_device[key] = value
                    targets_on_device.append(target_on_device)
                
                # 前向传播
                loss_dict = detector.model(images, targets_on_device)
                
                # 计算总损失
                losses = sum(loss for loss in loss_dict.values())
                
                # 反向传播
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                # 记录批次时间
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                # 每2个批次清理一次CUDA缓存
                if batch_idx % 2 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 显示进度
                if batch_idx % 2 == 0:
                    current_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    print(f"   批次 {batch_idx}: 损失={losses.item():.4f}, 时间={batch_time:.2f}s, GPU内存={current_memory:.2f}GB")
                    
            except Exception as e:
                print(f"   批次 {batch_idx} 错误: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # 计算性能指标
        total_time = time.time() - start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        
        print(f"\n   性能测试完成!")
        print(f"   总时间: {total_time:.2f} 秒")
        print(f"   平均批次时间: {avg_batch_time:.2f} 秒")
        print(f"   测试的批次数量: {len(batch_times)}")
        
        # 5. 内存使用分析
        print("\n5. 内存使用分析...")
        
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            reserved_memory = torch.cuda.memory_reserved() / 1024**3
            
            print(f"   当前GPU内存使用: {current_memory:.2f} GB")
            print(f"   最大GPU内存使用: {max_memory:.2f} GB")
            print(f"   保留的GPU内存: {reserved_memory:.2f} GB")
            
            # 清理内存
            torch.cuda.empty_cache()
            after_clean_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"   清理后GPU内存: {after_clean_memory:.2f} GB")
        
        # 6. 优化建议
        print("\n6. 优化建议:")
        
        if avg_batch_time > 5.0:
            print(f"   ⚠ 批次时间过长 ({avg_batch_time:.2f}s)，建议:")
            print(f"     1. 进一步减少批次大小 (当前: {batch_size})")
            print(f"     2. 进一步减少图像尺寸 (当前: 416x416)")
            print(f"     3. 使用更轻量的骨干网络")
        elif avg_batch_time > 2.0:
            print(f"   ⚠ 批次时间偏长 ({avg_batch_time:.2f}s)，建议:")
            print(f"     1. 考虑使用混合精度训练")
            print(f"     2. 优化数据加载器")
        else:
            print(f"   ✓ 批次时间正常 ({avg_batch_time:.2f}s)")
        
        if torch.cuda.is_available() and max_memory > 7.0:
            print(f"   ⚠ GPU内存使用过高 ({max_memory:.2f}GB)，建议:")
            print(f"     1. 减少批次大小 (当前: {batch_size})")
            print(f"     2. 使用梯度累积")
            print(f"     3. 使用更小的模型")
        elif torch.cuda.is_available():
            print(f"   ✓ GPU内存使用正常 ({max_memory:.2f}GB)")
        
        # 7. 创建优化后的训练配置
        print("\n7. 创建优化后的训练配置...")
        
        optimized_config = {
            'batch_size': batch_size,
            'image_size': 416,
            'num_workers': num_workers,
            'learning_rate': learning_rate,
            'device': device,
            'backbone': 'resnet50',
            'use_amp': True,  # 自动混合精度
            'gradient_accumulation_steps': 4,  # 梯度累积
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'warmup_epochs': 3
        }
        
        # 保存优化配置
        config_path = 'config/optimized_training_config.json'
        os.makedirs('config', exist_ok=True)
        
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_config, f, indent=2, ensure_ascii=False)
        
        print(f"   优化配置已保存到: {config_path}")
        
        print("\n" + "=" * 80)
        print("性能优化完成!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n优化过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_precision():
    """测试混合精度训练"""
    print("\n" + "=" * 80)
    print("测试混合精度训练")
    print("=" * 80)
    
    try:
        # 检查是否支持混合精度
        if not torch.cuda.is_available():
            print("CUDA不可用，跳过混合精度测试")
            return True
        
        from torch.cuda.amp import autocast, GradScaler
        
        print("混合精度训练支持: ✓")
        
        # 简单测试混合精度
        scaler = GradScaler()
        
        # 创建测试张量
        x = torch.randn(4, 3, 416, 416).cuda()
        y = torch.randn(4, 3, 416, 416).cuda()
        
        # 测试混合精度前向传播
        with autocast():
            output = x * y
            loss = output.mean()
        
        print(f"混合精度前向传播测试: ✓")
        print(f"损失值: {loss.item():.4f}")
        
        # 测试混合精度反向传播
        scaler.scale(loss).backward()
        scaler.step(torch.optim.Adam([x], lr=0.001))
        scaler.update()
        
        print(f"混合精度反向传播测试: ✓")
        
        print("\n混合精度训练测试完成!")
        return True
        
    except Exception as e:
        print(f"混合精度测试失败: {e}")
        return False

def main():
    """主函数"""
    print("Faster R-CNN训练性能优化工具")
    print("=" * 80)
    
    # 运行优化
    success = optimize_training()
    
    if success:
        # 测试混合精度
        test_mixed_precision()
        
        print("\n优化建议总结:")
        print("1. 使用较小的批次大小 (2-4)")
        print("2. 使用较小的图像尺寸 (416x416)")
        print("3. 使用ResNet50而不是ResNet101")
        print("4. 启用混合精度训练")
        print("5. 使用梯度累积")
        print("6. 在Windows上设置num_workers=0")
        print("7. 定期清理CUDA缓存")
        print("8. 监控GPU内存使用")
        
        print("\n下一步:")
        print("1. 使用优化配置重新训练Faster R-CNN模型")
        print("2. 监控训练过程中的GPU占用率")
        print("3. 根据实际性能进一步调整参数")
    else:
        print("\n优化失败，请检查错误信息")

if __name__ == "__main__":
    main()
