import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.ssd_model import SSDDetector
from PIL import Image
import numpy as np

def test_detect_method():
    """测试修复后的detect方法"""
    # 创建模型实例
    model = SSDDetector(num_classes=20)
    
    # 创建测试图像
    test_image = np.ones((300, 300, 3), dtype=np.uint8) * 255
    
    try:
        # 测试使用conf_threshold参数调用detect方法
        result = model.detect(image=test_image, conf_threshold=0.5, draw_boxes=False)
        print("✓ 检测方法调用成功！参数名conf_threshold正常工作。")
        
        # 测试使用iou_threshold参数调用detect方法
        result = model.detect(image=test_image, conf_threshold=0.5, draw_boxes=False, iou_threshold=0.5)
        print("✓ 检测方法调用成功！参数名iou_threshold正常工作。")
        
        # 测试使用所有参数调用detect方法
        result = model.detect(image=test_image, conf_threshold=0.7, draw_boxes=True, iou_threshold=0.3)
        print("✓ 检测方法调用成功！所有参数组合正常工作。")
        
        return True
    except TypeError as e:
        print(f"✗ 检测方法调用失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试detect方法修复...")
    success = test_detect_method()
    if success:
        print("\n测试通过！detect方法参数不匹配问题已修复。")
    else:
        print("\n测试失败！detect方法参数问题仍存在。")
        sys.exit(1)
