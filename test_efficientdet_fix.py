#!/usr/bin/env python3
"""
测试EfficientDet修复脚本
验证：
1. 模型初始化没有torchvision警告
2. 数据加载器创建没有权限错误
"""

import sys
import os
import warnings
import torch

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_initialization():
    """测试EfficientDet模型初始化"""
    print("=" * 60)
    print("测试1: EfficientDet模型初始化")
    print("=" * 60)
    
    try:
        # 捕获警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # 导入并初始化模型
            from models.efficientdet_model import EfficientDetDetector
            
            print("正在初始化EfficientDet模型...")
            model = EfficientDetDetector()
            
            # 检查是否有torchvision相关的警告
            torchvision_warnings = []
            for warning in w:
                if 'torchvision' in str(warning.message) or 'weights' in str(warning.message):
                    torchvision_warnings.append(warning)
            
            if torchvision_warnings:
                print(f"❌ 发现 {len(torchvision_warnings)} 个torchvision警告:")
                for i, warning in enumerate(torchvision_warnings, 1):
                    print(f"  警告 {i}: {warning.category.__name__}: {warning.message}")
                return False
            else:
                print("✅ 模型初始化成功，没有torchvision警告")
                print(f"   设备: {model.device}")
                print(f"   模型类型: {type(model.model).__name__ if hasattr(model, 'model') else 'N/A'}")
                return True
                
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader_creation():
    """测试数据加载器创建"""
    print("\n" + "=" * 60)
    print("测试2: 数据加载器创建")
    print("=" * 60)
    
    try:
        # 检查YAML文件是否存在
        yaml_path = "data/YOLO/dataset.yaml"
        if not os.path.exists(yaml_path):
            print(f"⚠️  数据集配置文件不存在: {yaml_path}")
            print("   创建测试用的临时数据集配置...")
            
            # 创建临时目录结构
            os.makedirs("data/YOLO/images", exist_ok=True)
            os.makedirs("data/YOLO/labels", exist_ok=True)
            
            # 创建简单的YAML文件
            with open(yaml_path, "w") as f:
                f.write("""# 测试数据集配置
train: data/YOLO/images
val: data/YOLO/images
test: data/YOLO/images

nc: 24
names: ['AUV', 'Boat', 'Buoy', 'Cable', 'Chain', 'Container', 'Diver', 'Dock', 'Figure', 'Fish', 'Flare', 'Floating', 'Gear', 'Instrument', 'Mine', 'Net', 'Pipe', 'Platform', 'ROV', 'Reef', 'Rubble', 'Seafloor', 'Ship', 'Vehicle']
""")
            print(f"✅ 已创建测试配置文件: {yaml_path}")
        
        # 测试数据加载器
        from utils.data_loader import create_data_loaders
        
        print("正在创建数据加载器...")
        
        # 使用num_workers=0来避免权限问题
        train_loader, val_loader = create_data_loaders(
            data_yaml_path=yaml_path,
            batch_size=2,
            num_workers=0,  # 使用修复后的默认值
            prefetch_factor=2
        )
        
        print("✅ 数据加载器创建成功")
        print(f"   训练集批次数量: {len(train_loader) if train_loader else 0}")
        print(f"   验证集批次数量: {len(val_loader) if val_loader else 0}")
        
        # 尝试获取一个批次（如果数据集为空也没关系）
        if train_loader:
            try:
                batch = next(iter(train_loader))
                print(f"   批次数据形状: {type(batch)}")
                if isinstance(batch, (list, tuple)):
                    print(f"   批次包含 {len(batch)} 个元素")
            except StopIteration:
                print("   数据集为空（预期行为，因为这是测试配置）")
        
        return True
        
    except PermissionError as e:
        print(f"❌ 权限错误: {e}")
        print("   请检查data/YOLO/目录的权限")
        return False
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_training_method():
    """测试模型的train方法参数"""
    print("\n" + "=" * 60)
    print("测试3: 模型train方法参数检查")
    print("=" * 60)
    
    try:
        from models.efficientdet_model import EfficientDetDetector
        import inspect
        
        model = EfficientDetDetector()
        
        # 获取train方法的参数
        sig = inspect.signature(model.train)
        params = sig.parameters
        
        print("检查train方法的参数默认值:")
        
        # 检查num_workers参数
        if 'num_workers' in params:
            num_workers_default = params['num_workers'].default
            print(f"   num_workers 默认值: {num_workers_default}")
            
            if num_workers_default == 0:
                print("   ✅ num_workers默认值已正确设置为0（修复已应用）")
            else:
                print(f"   ❌ num_workers默认值应为0，但实际为{num_workers_default}")
                return False
        else:
            print("   ❌ train方法没有num_workers参数")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 参数检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("EfficientDet修复验证测试")
    print("=" * 60)
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Torchvision版本: {torchvision.__version__ if 'torchvision' in sys.modules else '未导入'}")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print()
    
    # 运行测试
    test1_passed = test_model_initialization()
    test2_passed = test_data_loader_creation()
    test3_passed = test_model_training_method()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"测试1 - 模型初始化: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"测试2 - 数据加载器: {'✅ 通过' if test2_passed else '❌ 失败'}")
    print(f"测试3 - 参数检查: {'✅ 通过' if test3_passed else '❌ 失败'}")
    
    total_passed = sum([test1_passed, test2_passed, test3_passed])
    total_tests = 3
    
    print(f"\n总成绩: {total_passed}/{total_tests} 通过")
    
    if total_passed == total_tests:
        print("🎉 所有测试通过！修复已成功应用。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        return 1

if __name__ == "__main__":
    # 导入torchvision用于版本检查
    import torchvision
    sys.exit(main())
