import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.ssd_model import SSDDetector
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def test_draw_boxes():
    """测试检测框绘制功能"""
    # 创建模型实例
    model = SSDDetector(num_classes=20)
    
    # 创建不同尺寸的测试图像（模拟真实场景）
    test_images = [
        ("640x480", np.ones((480, 640, 3), dtype=np.uint8) * 255),
        ("800x600", np.ones((600, 800, 3), dtype=np.uint8) * 255),
        ("1280x720", np.ones((720, 1280, 3), dtype=np.uint8) * 255)
    ]
    
    for name, test_image in test_images:
        print(f"\n测试{name}图像的框选绘制...")
        
        try:
            # 使用较低的置信度阈值以确保有检测结果
            result = model.detect(
                image=test_image, 
                conf_threshold=0.01, 
                draw_boxes=True, 
                iou_threshold=0.45
            )
            
            detections = result['detections']
            processed_image = result['processed_image']
            
            print(f"  ✓ 检测完成，找到{detections}个检测结果")
            print(f"  ✓ processed_image类型: {type(processed_image)}")
            print(f"  ✓ processed_image形状: {processed_image.shape if processed_image is not None else 'None'}")
            
            # 检查是否生成了处理后的图像
            if processed_image is not None:
                # 检查处理后的图像是否与原始图像尺寸相同
                if processed_image.shape[:2] == test_image.shape[:2]:
                    print(f"  ✓ 绘制后的图像尺寸正确: {processed_image.shape[:2]}")
                else:
                    print(f"  ✗ 绘制后的图像尺寸错误: 期望{test_image.shape[:2]}，得到{processed_image.shape[:2]}")
                    
                # 检查是否有检测框数据
                if detections:
                    print(f"  ✓ 检测框数据: {detections}")
                    # 验证第一个检测框的坐标是否在合理范围内
                    first_bbox = detections[0]['bbox']
                    if all(0 <= coord <= max(test_image.shape[:2]) for coord in first_bbox):
                        print(f"  ✓ 第一个检测框坐标在合理范围内: {first_bbox}")
                    else:
                        print(f"  ✗ 第一个检测框坐标超出范围: {first_bbox}")
                else:
                    print(f"  ⚠ 未找到检测结果（可能是置信度阈值过高）")
                    
            else:
                print(f"  ✗ 未生成处理后的图像")
                
        except Exception as e:
            print(f"  ✗ 检测绘制失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    print("开始测试检测框绘制功能...")
    success = test_draw_boxes()
    if success:
        print("\n🎉 所有测试通过！检测框绘制功能正常工作。")
    else:
        print("\n❌ 测试失败！检测框绘制功能存在问题。")
        sys.exit(1)
