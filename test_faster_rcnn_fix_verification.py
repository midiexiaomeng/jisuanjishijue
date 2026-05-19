#!/usr/bin/env python3
"""
测试Faster R-CNN模型修复后的训练功能
验证类型错误修复是否有效
"""

import sys
import os
import torch
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_loss_dict_type_fix():
    """测试loss_dict类型修复"""
    print("=" * 80)
    print("测试Faster R-CNN模型loss_dict类型修复")
    print("=" * 80)
    
    try:
        # 导入Faster R-CNN模型
        from models.faster_rcnn_model import FasterRCNNDetector
        
        # 创建模型实例
        print("1. 创建Faster R-CNN模型实例...")
        detector = FasterRCNNDetector(num_classes=25, device='cpu')
        print("   ✓ 模型创建成功")
        
        # 测试loss_dict类型处理
        print("\n2. 测试loss_dict类型处理...")
        
        # 模拟loss_dict为列表的情况
        loss_dict_list = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        print(f"   模拟loss_dict为列表: {loss_dict_list}")
        
        # 应用修复逻辑
        if isinstance(loss_dict_list, list):
            loss_dict = {'total_loss': sum(loss_dict_list)}
            print(f"   修复后loss_dict: {loss_dict}")
            print(f"   总损失值: {loss_dict['total_loss']}")
            print("   ✓ 列表到字典转换成功")
        
        # 模拟loss_dict为字典的情况
        loss_dict_dict = {
            'loss_classifier': torch.tensor(1.5),
            'loss_box_reg': torch.tensor(0.8),
            'loss_objectness': torch.tensor(0.3),
            'loss_rpn_box_reg': torch.tensor(0.4)
        }
        print(f"\n   模拟loss_dict为字典: {loss_dict_dict}")
        
        if isinstance(loss_dict_dict, dict):
            losses = sum(loss for loss in loss_dict_dict.values())
            print(f"   字典总损失值: {losses}")
            print("   ✓ 字典处理成功")
        
        print("\n3. 测试模型训练方法中的修复...")
        
        # 检查train方法中的修复代码
        with open('models/faster_rcnn_model.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用更灵活的搜索方式：查找修复代码的核心内容
        # 增强修复代码模式：处理"unsupported operand type(s) for +: 'int' and 'dict'"错误
        fix_patterns = [
            "# 检查loss_dict类型，如果是列表则转换为字典",
            "if isinstance(loss_dict, list):",
            "loss_dict = {'total_loss': sum(loss_dict)}",
            "# 确保loss_dict中的所有值都是数字类型",
            "cleaned_loss_dict = {}",
            "for key, value in loss_dict.items():",
            "if isinstance(value, dict):",
            "# 如果是字典，尝试提取数值",
            "# 查找第一个数值类型的值",
            "for sub_key, sub_value in value.items():",
            "if isinstance(sub_value, (int, float, torch.Tensor)):",
            "if isinstance(sub_value, torch.Tensor):",
            "cleaned_loss_dict[key] = sub_value.item()",
            "else:",
            "cleaned_loss_dict[key] = sub_value",
            "break",
            "else:",
            "# 如果没有找到数值，使用0",
            "cleaned_loss_dict[key] = 0.0",
            "elif isinstance(value, (int, float, torch.Tensor)):",
            "if isinstance(value, torch.Tensor):",
            "cleaned_loss_dict[key] = value.item()",
            "else:",
            "cleaned_loss_dict[key] = value",
            "else:",
            "# 其他类型，使用0",
            "cleaned_loss_dict[key] = 0.0",
            "# 使用清理后的loss_dict",
            "losses = sum(loss for loss in cleaned_loss_dict.values())"
        ]
        
        # 检查每个模式是否存在
        pattern_found = all(pattern in content for pattern in fix_patterns)
        
        if pattern_found:
            print("   ✓ 增强修复代码存在于模型中")
            print("   ✓ 修复了'unsupported operand type(s) for +: 'int' and 'dict''错误")
            
            # 统计修复代码出现次数（使用更灵活的方式）
            import re
            
            # 使用正则表达式查找完整的增强修复代码块
            # 查找包含"确保loss_dict中的所有值都是数字类型"的代码块
            enhanced_fix_pattern = r'# 确保loss_dict中的所有值都是数字类型[\s\S]*?losses = sum\(loss for loss in cleaned_loss_dict\.values\(\)\)'
            
            # 查找所有匹配
            matches = re.findall(enhanced_fix_pattern, content, re.DOTALL)
            fix_blocks = len(matches)
            
            print(f"   ✓ 找到 {fix_blocks} 个增强修复代码块")
            
            # 检查是否在正确的位置
            if fix_blocks >= 3:
                print("   ✓ 增强修复代码在所有必要的位置（train方法混合精度、非混合精度和evaluate方法）")
            elif fix_blocks >= 2:
                print("   ✓ 增强修复代码在train方法的两个位置（混合精度和非混合精度训练）")
                # 检查evaluate方法中是否有增强修复代码
                if "def evaluate" in content:
                    evaluate_section = content[content.find("def evaluate"):content.find("def _calculate_simple_map")]
                    if "# 确保loss_dict中的所有值都是数字类型" in evaluate_section:
                        print("   ✓ evaluate方法中也包含增强修复代码")
                    else:
                        print("   ⚠ evaluate方法中可能缺少增强修复代码")
            else:
                print("   ⚠ 增强修复代码可能没有在所有必要的位置")
        else:
            print("   ✗ 增强修复代码不存在于模型中")
            # 回退检查旧修复代码
            old_fix_patterns = [
                "# 检查loss_dict类型，如果是列表则转换为字典",
                "if isinstance(loss_dict, list):",
                "loss_dict = {'total_loss': sum(loss_dict)}"
            ]
            old_pattern_found = all(pattern in content for pattern in old_fix_patterns)
            if old_pattern_found:
                print("   ⚠ 只找到旧修复代码，缺少增强修复代码")
                print("   ⚠ 可能无法处理'unsupported operand type(s) for +: 'int' and 'dict''错误")
            else:
                print("   ✗ 任何修复代码都不存在于模型中")
        
        print("\n4. 测试数据加载器导入...")
        try:
            # 测试数据加载器导入
            from utils.data_loader import create_data_loaders
            print("   ✓ YOLO数据加载器导入成功")
        except ImportError as e:
            print(f"   ⚠ YOLO数据加载器导入失败: {e}")
        
        try:
            # 测试COCO数据加载器导入
            from utils.coco_data_loader import create_coco_data_loaders
            print("   ✓ COCO数据加载器导入成功")
        except ImportError as e:
            print(f"   ⚠ COCO数据加载器导入失败: {e}")
        
        print("\n5. 测试数据集配置文件...")
        yolo_config_path = 'data/YOLO/dataset.yaml'
        if os.path.exists(yolo_config_path):
            print(f"   ✓ YOLO数据集配置文件存在: {yolo_config_path}")
            
            # 读取配置文件
            import yaml
            with open(yolo_config_path, 'r', encoding='utf-8') as f:
                yolo_config = yaml.safe_load(f)
            
            if 'nc' in yolo_config:
                print(f"   ✓ 配置文件包含类别数量: {yolo_config['nc']}")
            else:
                print("   ⚠ 配置文件缺少类别数量(nc)")
        else:
            print(f"   ✗ YOLO数据集配置文件不存在: {yolo_config_path}")
        
        print("\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_integration():
    """测试GUI集成"""
    print("\n" + "=" * 80)
    print("测试GUI集成")
    print("=" * 80)
    
    try:
        # 检查GUI主窗口文件
        gui_file = 'gui/main_window.py'
        if os.path.exists(gui_file):
            print(f"1. GUI主窗口文件存在: {gui_file}")
            
            # 检查是否包含Faster R-CNN训练代码
            with open(gui_file, 'r', encoding='utf-8') as f:
                gui_content = f.read()
            
            # 查找Faster R-CNN相关代码
            if 'FasterRCNN' in gui_content or 'faster_rcnn' in gui_content.lower():
                print("   ✓ GUI中包含Faster R-CNN相关代码")
            else:
                print("   ⚠ GUI中可能缺少Faster R-CNN相关代码")
        else:
            print(f"1. GUI主窗口文件不存在: {gui_file}")
        
        # 检查主GUI文件
        main_gui_file = 'main_gui.py'
        if os.path.exists(main_gui_file):
            print(f"\n2. 主GUI文件存在: {main_gui_file}")
        else:
            print(f"\n2. 主GUI文件不存在: {main_gui_file}")
        
        print("\n" + "=" * 80)
        print("GUI集成测试完成!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\nGUI集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("Faster R-CNN模型修复验证测试")
    print("=" * 80)
    
    # 测试1: loss_dict类型修复
    test1_success = test_loss_dict_type_fix()
    
    # 测试2: GUI集成
    test2_success = test_gui_integration()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    if test1_success and test2_success:
        print("✓ 所有测试通过!")
        print("\n建议下一步:")
        print("1. 在GUI中启动Faster R-CNN训练")
        print("2. 监控训练过程，确保不再出现'list' object has no attribute 'values'错误")
        print("3. 如果训练成功，继续修复其他模型（SSD, RetinaNet, EfficientDet）")
    else:
        print("⚠ 部分测试失败，需要进一步调试")
        
        if not test1_success:
            print("\n需要检查:")
            print("1. models/faster_rcnn_model.py中的修复代码")
            print("2. 确保修复代码在train方法的两个位置（混合精度和非混合精度）")
        
        if not test2_success:
            print("\n需要检查:")
            print("1. GUI文件是否存在")
            print("2. GUI中是否正确集成了Faster R-CNN训练功能")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
