#!/usr/bin/env python3
"""
GPU训练优化测试脚本
用于测试SSD模型的GPU训练优化效果，包括混合精度训练、梯度累积等功能
"""

import os
import torch
import time
from models.ssd_model import SSDDetector

def test_ssd_gpu_optimization():
    """测试SSD模型的GPU训练优化"""
    print("=" * 50)
    print("SSD模型GPU训练优化测试")
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
    
    # 准备测试数据（使用小批量进行快速测试）
    print("\n[信息] 准备测试数据...")
    
    # 使用YOLO数据集进行测试
    data_path = "data/YOLO/dataset.yaml"
    if not os.path.exists(data_path):
        print(f"[警告] 数据集配置文件不存在: {data_path}")
        print("请确保数据集配置文件路径正确")
        return
    
    # 测试不同配置下的训练性能
    test_configs = [
        {
            "name": "基础配置（无优化）",
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "use_amp": False,
            "gradient_accumulation_steps": 1
        },
        {
            "name": "优化配置1（数据加载优化）",
            "batch_size": 8,
            "num_workers": 2,
            "pin_memory": True,
            "use_amp": False,
            "gradient_accumulation_steps": 1
        },
        {
            "name": "优化配置2（混合精度训练）",
            "batch_size": 8,
            "num_workers": 2,
            "pin_memory": True,
            "use_amp": True,
            "gradient_accumulation_steps": 1
        },
        {
            "name": "优化配置3（梯度累积+混合精度）",
            "batch_size": 8,
            "num_workers": 2,
            "pin_memory": True,
            "use_amp": True,
            "gradient_accumulation_steps": 4
        }
    ]
    
    for config in test_configs:
        print(f"\n" + "=" * 50)
        print(f"测试: {config['name']}")
        print("=" * 50)
        print(f"配置: batch_size={config['batch_size']}, num_workers={config['num_workers']}, "
              f"pin_memory={config['pin_memory']}, use_amp={config['use_amp']}, "
              f"gradient_accumulation_steps={config['gradient_accumulation_steps']}")
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 开始测试训练（只训练1个epoch，10个batch）
            print("\n[信息] 开始测试训练...")
            
            # 调用train方法，但限制训练时间和batch数
            history = ssd.train(
                data_path=data_path,
                batch_size=config['batch_size'],
                epochs=1,
                learning_rate=0.001,
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory'],
                use_amp=config['use_amp'],
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                save_dir="checkpoints/test_ssd_gpu"
            )
            
            # 记录结束时间
            end_time = time.time()
            
            # 计算训练时间
            training_time = end_time - start_time
            
            print(f"\n[结果] 测试完成!")
            print(f"训练时间: {training_time:.2f} 秒")
            print(f"平均损失: {history['train_loss'][0]:.4f}")
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[错误] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            continue
    
    print(f"\n" + "=" * 50)
    print("所有测试完成")
    print("=" * 50)
    print("\n[提示] 使用以下命令监控GPU利用率：")
    print("- Windows: 任务管理器 -> 性能 -> GPU")
    print("- Linux: nvidia-smi (实时监控: watch -n 1 nvidia-smi)")

if __name__ == "__main__":
    test_ssd_gpu_optimization()
