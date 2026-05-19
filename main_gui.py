#!/usr/bin/env python3
"""
水下目标检测系统 - GUI主程序入口
支持YOLOv8和其他四个模型（Faster R-CNN, SSD, RetinaNet, EfficientDet）
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """检查依赖包是否安装"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
        
    try:
        import torchvision
    except ImportError:
        missing_deps.append("torchvision")
        
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
        
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
        
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        missing_deps.append("PyQt5")
        
    try:
        import ultralytics
    except ImportError:
        missing_deps.append("ultralytics")
        
    return missing_deps

def setup_environment():
    """设置环境"""
    # 创建必要的目录
    directories = [
        'checkpoints',
        'checkpoints/yolov8',
        'checkpoints/faster_rcnn',
        'checkpoints/ssd',
        'checkpoints/retinanet',
        'checkpoints/efficientdet',
        'logs',
        'results',
        'data/coco/images',
        'data/coco/labels'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("环境设置完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='水下目标检测系统')
    parser.add_argument('--setup', action='store_true', help='设置环境并检查依赖')
    parser.add_argument('--check-deps', action='store_true', help='检查依赖包')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    if args.setup:
        print("正在设置环境...")
        setup_environment()
        missing_deps = check_dependencies()
        if missing_deps:
            print(f"缺少依赖包: {', '.join(missing_deps)}")
            print("请运行: pip install " + " ".join(missing_deps))
            return 1
        else:
            print("所有依赖包已安装")
        return 0
        
    if args.check_deps:
        missing_deps = check_dependencies()
        if missing_deps:
            print(f"缺少依赖包: {', '.join(missing_deps)}")
            print("请运行: pip install " + " ".join(missing_deps))
            return 1
        else:
            print("所有依赖包已安装")
        return 0
    
    # 检查依赖
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"错误: 缺少依赖包: {', '.join(missing_deps)}")
        print("请运行以下命令安装依赖:")
        print(f"pip install {' '.join(missing_deps)}")
        print("\n或使用 --setup 参数自动设置环境")
        return 1
    
    # 导入GUI模块
    try:
        from gui.main_window import MainWindow
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保在项目根目录下运行此程序")
        return 1
    
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格
    
    # 设置应用程序信息
    app.setApplicationName("水下目标检测系统")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("水下目标检测实验室")
    
    # 设置字体
    font = QFont("Microsoft YaHei", 10)  # 使用微软雅黑字体
    app.setFont(font)
    
    # 创建主窗口
    try:
        window = MainWindow()
        
        # 显示系统信息
        if args.debug:
            print("调试模式已启用")
            import torch  # 在调试模式下导入torch
            print(f"PyTorch版本: {torch.__version__}")
            print(f"CUDA可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU数量: {torch.cuda.device_count()}")
                print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        
        window.show()
        
        # 显示欢迎消息
        if not args.debug:
            QMessageBox.information(window, "欢迎", 
                                  "欢迎使用水下目标检测系统！\n\n"
                                  "系统功能:\n"
                                  "1. 支持YOLOv8模型（重点）\n"
                                  "2. 支持其他四个模型（Faster R-CNN, SSD, RetinaNet, EfficientDet）\n"
                                  "3. 使用COU水下目标检测数据集\n"
                                  "4. 包含GUI界面和模型评估功能\n"
                                  "5. 训练过程可视化\n\n"
                                  "请先加载图像，然后选择模型进行检测。")
        
        return app.exec_()
        
    except Exception as e:
        print(f"程序错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 显示错误对话框
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("程序错误")
        error_msg.setText(f"程序启动时发生错误: {str(e)}")
        error_msg.setDetailedText(traceback.format_exc())
        error_msg.exec_()
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
