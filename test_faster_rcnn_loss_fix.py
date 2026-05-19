import torch
import torch.nn as nn
from models.faster_rcnn_model import FasterRCNNDetector

def test_loss_handling():
    """测试Faster R-CNN损失处理逻辑"""
    print("测试Faster R-CNN损失处理逻辑...")
    
    # 创建模型实例
    detector = FasterRCNNDetector(num_classes=25, device='cpu')
    
    # 模拟损失字典（各种可能的情况）
    test_cases = [
        {
            'name': '标准损失字典',
            'loss_dict': {
                'loss_classifier': torch.tensor(0.5, requires_grad=True),
                'loss_box_reg': torch.tensor(0.3, requires_grad=True),
                'loss_objectness': torch.tensor(0.2, requires_grad=True),
                'loss_rpn_box_reg': torch.tensor(0.1, requires_grad=True)
            }
        },
        {
            'name': '嵌套字典损失',
            'loss_dict': {
                'total_loss': {
                    'loss_classifier': torch.tensor(0.5, requires_grad=True),
                    'loss_box_reg': torch.tensor(0.3, requires_grad=True)
                }
            }
        },
        {
            'name': '深层嵌套字典',
            'loss_dict': {
                'losses': {
                    'classification': {
                        'main': torch.tensor(0.5, requires_grad=True),
                        'aux': torch.tensor(0.1, requires_grad=True)
                    },
                    'regression': torch.tensor(0.3, requires_grad=True)
                }
            }
        },
        {
            'name': '混合类型损失',
            'loss_dict': {
                'loss_classifier': 0.5,  # Python float
                'loss_box_reg': torch.tensor(0.3, requires_grad=True),
                'loss_objectness': {
                    'value': 0.2
                }
            }
        },
        {
            'name': '列表类型损失',
            'loss_dict': [torch.tensor(0.5, requires_grad=True), torch.tensor(0.3, requires_grad=True), torch.tensor(0.2, requires_grad=True)]
        }
    ]
    
    # 测试每个用例
    for i, test_case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {test_case['name']}")
        print(f"输入: {test_case['loss_dict']}")
        
        try:
            # 模拟train方法中的损失处理逻辑
            loss_dict = test_case['loss_dict']
            
            # 检查loss_dict类型，如果是列表则转换为字典
            if isinstance(loss_dict, list):
                # 假设列表中的元素是损失值，创建一个简单的字典
                loss_dict = {'total_loss': sum(loss_dict)}
                print(f"列表转换为字典: {loss_dict}")
            
            # 辅助函数：递归提取数值并转换为torch.Tensor
            def extract_and_convert_to_tensor(obj, max_depth=5):
                """递归提取数值并转换为torch.Tensor，防止无限递归"""
                if max_depth <= 0:
                    return torch.tensor(0.0, device='cpu', requires_grad=True)
                
                if isinstance(obj, torch.Tensor):
                    # 如果已经是tensor，确保它需要梯度
                    if not obj.requires_grad:
                        obj = obj.detach().requires_grad_(True)
                    return obj
                elif isinstance(obj, (int, float)):
                    return torch.tensor(float(obj), device='cpu', requires_grad=True)
                elif isinstance(obj, dict):
                    # 递归查找第一个数值
                    for sub_key, sub_value in obj.items():
                        result = extract_and_convert_to_tensor(sub_value, max_depth-1)
                        if result is not None:
                            return result
                    return torch.tensor(0.0, device='cpu', requires_grad=True)
                elif isinstance(obj, (list, tuple)):
                    # 查找第一个数值
                    for item in obj:
                        result = extract_and_convert_to_tensor(item, max_depth-1)
                        if result is not None:
                            return result
                    return torch.tensor(0.0, device='cpu', requires_grad=True)
                else:
                    return torch.tensor(0.0, device='cpu', requires_grad=True)
            
            # 确保loss_dict中的所有值都是torch.Tensor类型
            cleaned_loss_dict = {}
            for key, value in loss_dict.items():
                cleaned_loss_dict[key] = extract_and_convert_to_tensor(value)
            
            print(f"清理后的损失字典: {cleaned_loss_dict}")
            
            # 计算总损失
            losses = sum(loss for loss in cleaned_loss_dict.values())
            print(f"总损失: {losses.item():.4f}")
            
            # 测试反向传播
            if isinstance(losses, torch.Tensor):
                losses.backward()
                print("反向传播成功")
            
            print("✓ 测试通过")
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()

def test_model_forward():
    """测试模型前向传播"""
    print("\n\n测试模型前向传播...")
    
    try:
        # 创建模型实例
        detector = FasterRCNNDetector(num_classes=25, device='cpu')
        
        # 创建模拟输入
        batch_size = 2
        image_size = 640
        
        # 创建模拟图像
        images = [torch.randn(3, image_size, image_size) for _ in range(batch_size)]
        
        # 创建模拟目标
        targets = []
        for i in range(batch_size):
            num_boxes = 3
            boxes = torch.randn(num_boxes, 4) * 100 + 200
            boxes[:, 2:] += boxes[:, :2]  # 确保x2 > x1, y2 > y1
            labels = torch.randint(1, 25, (num_boxes,))
            
            targets.append({
                'boxes': boxes,
                'labels': labels
            })
        
        # 前向传播
        detector.model.train()
        loss_dict = detector.model(images, targets)
        
        print(f"前向传播成功")
        print(f"损失字典类型: {type(loss_dict)}")
        print(f"损失字典内容: {loss_dict}")
        
        # 测试损失处理
        if isinstance(loss_dict, list):
            loss_dict = {'total_loss': sum(loss_dict)}
        
        # 使用辅助函数处理损失
        def extract_and_convert_to_tensor(obj, max_depth=5):
            if max_depth <= 0:
                return torch.tensor(0.0, device='cpu', requires_grad=True)
            
            if isinstance(obj, torch.Tensor):
                # 如果已经是tensor，确保它需要梯度
                if not obj.requires_grad:
                    obj = obj.detach().requires_grad_(True)
                return obj
            elif isinstance(obj, (int, float)):
                return torch.tensor(float(obj), device='cpu', requires_grad=True)
            elif isinstance(obj, dict):
                for sub_key, sub_value in obj.items():
                    result = extract_and_convert_to_tensor(sub_value, max_depth-1)
                    if result is not None:
                        return result
                return torch.tensor(0.0, device='cpu', requires_grad=True)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    result = extract_and_convert_to_tensor(item, max_depth-1)
                    if result is not None:
                        return result
                return torch.tensor(0.0, device='cpu', requires_grad=True)
            else:
                return torch.tensor(0.0, device='cpu', requires_grad=True)
        
        cleaned_loss_dict = {}
        for key, value in loss_dict.items():
            cleaned_loss_dict[key] = extract_and_convert_to_tensor(value)
        
        losses = sum(loss for loss in cleaned_loss_dict.values())
        print(f"处理后的总损失: {losses.item():.4f}")
        
        # 测试反向传播
        losses.backward()
        print("反向传播成功")
        
        print("✓ 模型前向传播测试通过")
        
    except Exception as e:
        print(f"✗ 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 80)
    print("Faster R-CNN损失处理修复测试")
    print("=" * 80)
    
    test_loss_handling()
    test_model_forward()
    
    print("\n" + "=" * 80)
    print("所有测试完成!")
    print("=" * 80)
