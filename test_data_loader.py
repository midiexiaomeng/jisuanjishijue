import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import COUUnderwaterDataset
import torch
from torch.utils.data import DataLoader

def test_data_loader():
    """测试数据加载器是否正常工作"""
    print("测试数据加载器...")
    
    # 使用YOLO格式的数据集配置文件
    data_yaml_path = "data/YOLO/dataset.yaml"
    
    if not os.path.exists(data_yaml_path):
        print(f"错误: 数据集配置文件不存在: {data_yaml_path}")
        return False
    
    print(f"使用数据集配置文件: {data_yaml_path}")
    
    try:
        # 创建训练数据集
        print("创建训练数据集...")
        train_dataset = COUUnderwaterDataset(data_yaml_path, split='train')
        
        print(f"训练数据集大小: {len(train_dataset)}")
        print(f"类别数量: {train_dataset.num_classes}")
        print(f"类别名称: {train_dataset.classes}")
        
        if len(train_dataset) == 0:
            print("警告: 训练数据集为空！")
            # 检查目录结构
            print(f"图像目录: {train_dataset.image_dir}")
            print(f"标签目录: {train_dataset.label_dir}")
            
            # 列出目录内容
            if os.path.exists(train_dataset.image_dir):
                print(f"图像目录内容: {os.listdir(train_dataset.image_dir)[:10]}")
            else:
                print(f"图像目录不存在: {train_dataset.image_dir}")
            
            if os.path.exists(train_dataset.label_dir):
                print(f"标签目录内容: {os.listdir(train_dataset.label_dir)[:10]}")
            else:
                print(f"标签目录不存在: {train_dataset.label_dir}")
        
        # 创建数据加载器
        print("创建数据加载器...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0  # 在Windows上使用0避免多进程问题
        )
        
        # 测试一个批次
        print("测试一个批次的数据加载...")
        for batch_idx, (images, targets) in enumerate(train_loader):
            print(f"批次 {batch_idx}:")
            print(f"  图像形状: {images.shape}")
            print(f"  目标数量: {len(targets)}")
            
            # 显示第一个目标
            if len(targets) > 0:
                target = targets[0]
                print(f"  第一个目标 - 边界框: {target['boxes'].shape if 'boxes' in target else 'N/A'}")
                print(f"  第一个目标 - 标签: {target['labels'].shape if 'labels' in target else 'N/A'}")
            
            # 只测试一个批次
            break
        
        print("数据加载器测试完成！")
        return True
        
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loader()
    if success:
        print("\n数据加载器测试成功！")
    else:
        print("\n数据加载器测试失败！")
