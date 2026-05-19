#!/usr/bin/env python3
"""
YOLOv8训练功能演示
展示如何使用GUI进行YOLOv8模型训练
"""

import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_header():
    """打印标题"""
    print("=" * 70)
    print("水下目标检测系统 - YOLOv8训练功能演示")
    print("=" * 70)
    print()

def print_section(title):
    """打印章节标题"""
    print("\n" + "-" * 60)
    print(f" {title}")
    print("-" * 60)

def main():
    """主函数"""
    print_header()
    
    print("本演示展示YOLOv8训练功能的实现:")
    print("1. 训练线程创建和管理")
    print("2. 实时进度更新和状态显示")
    print("3. 训练指标可视化")
    print("4. 训练结果保存和导出")
    print()
    
    # 检查项目结构
    print_section("1. 项目结构检查")
    
    project_structure = {
        "核心文件": [
            ("gui/main_window.py", "GUI主窗口，包含训练线程"),
            ("models/yolov8_model.py", "YOLOv8模型实现"),
            ("train_yolov8.py", "独立训练脚本"),
            ("config/training_config.json", "训练配置文件")
        ],
        "数据文件": [
            ("data/coco/train_annotations.json", "训练数据标注"),
            ("data/coco/val_annotations.json", "验证数据标注"),
            ("data/coco/test_annotations.json", "测试数据标注")
        ],
        "输出目录": [
            ("checkpoints/", "模型检查点保存目录"),
            ("results/", "训练结果保存目录"),
            ("runs/detect/", "训练日志和可视化目录")
        ]
    }
    
    for category, files in project_structure.items():
        print(f"\n{category}:")
        for file_path, description in files:
            if os.path.exists(file_path) or (file_path.endswith('/') and os.path.exists(file_path.rstrip('/'))):
                status = "✓"
            else:
                status = "✗"
            print(f"  {status} {file_path:40} - {description}")
    
    # 展示训练配置
    print_section("2. 训练配置")
    
    try:
        import json
        with open('config/training_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("训练配置文件内容:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"读取配置文件失败: {e}")
    
    # 展示训练功能
    print_section("3. 训练功能演示")
    
    print("YOLOv8训练功能特点:")
    print("1. 多线程训练 - 使用QThread实现后台训练，避免GUI冻结")
    print("2. 实时进度更新 - 通过信号槽机制实时更新训练进度")
    print("3. 指标可视化 - 损失、准确率、mAP、学习率曲线实时显示")
    print("4. 训练控制 - 支持开始、停止、暂停训练")
    print("5. 结果保存 - 自动保存检查点和训练结果")
    print("6. 异常处理 - 完善的错误处理和日志记录")
    
    # 展示如何使用
    print_section("4. 如何使用")
    
    print("方法1: 通过GUI界面训练")
    print("  python main_gui.py")
    print("  1. 选择YOLOv8模型")
    print("  2. 设置训练参数（轮数、批次大小、学习率等）")
    print("  3. 点击'开始训练'按钮")
    print("  4. 在'训练可视化'标签页查看训练进度")
    print()
    
    print("方法2: 通过命令行训练")
    print("  python train_yolov8.py --config config/training_config.json")
    print("  1. 修改配置文件中的参数")
    print("  2. 运行训练脚本")
    print("  3. 查看训练结果")
    print()
    
    print("方法3: 通过Python API训练")
    print("  from models.yolov8_model import YOLOv8Detector")
    print("  detector = YOLOv8Detector()")
    print("  results = detector.train(data_yaml='data/coco/coco.yaml', epochs=100)")
    print()
    
    # 展示测试结果
    print_section("5. 功能测试结果")
    
    test_results_file = 'results/yolov8_training_results.json'
    if os.path.exists(test_results_file):
        try:
            with open(test_results_file, 'r', encoding='utf-8') as f:
                test_results = json.load(f)
            
            print("最新测试训练结果:")
            print(f"  模型类型: {test_results.get('model_type', 'N/A')}")
            print(f"  最终损失: {test_results.get('final_loss', 0):.4f}")
            print(f"  最终准确率: {test_results.get('final_accuracy', 0):.4f}")
            print(f"  最终mAP: {test_results.get('final_map', 0):.4f}")
            print(f"  训练轮数: {test_results.get('epochs_trained', 0)}")
            print(f"  训练时间: {test_results.get('training_time', 0):.1f}秒")
            print(f"  检查点路径: {test_results.get('checkpoint_path', 'N/A')}")
        except Exception as e:
            print(f"读取测试结果失败: {e}")
    else:
        print("未找到测试结果文件，运行测试脚本以生成:")
        print("  python test_yolov8_training.py")
    
    # 下一步计划
    print_section("6. 下一步计划")
    
    print("根据用户要求，已完成YOLOv8训练功能的实现。")
    print("后续可以添加:")
    print("1. 其他四个模型（Faster R-CNN, SSD, RetinaNet, EfficientDet）")
    print("2. 更高级的训练可视化（TensorBoard集成）")
    print("3. 分布式训练支持")
    print("4. 模型压缩和优化")
    print("5. 自动超参数调优")
    
    print("\n" + "=" * 70)
    print("演示完成! YOLOv8训练功能已准备就绪。")
    print("=" * 70)
    
    # 提供启动GUI的选项
    print("\n要启动GUI界面进行训练，请运行:")
    print("  python main_gui.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
