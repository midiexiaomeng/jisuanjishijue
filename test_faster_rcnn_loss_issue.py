import torch

# 模拟Faster R-CNN可能返回的不同损失格式
def test_loss_formats():
    print("测试不同的损失格式...\n")
    
    # 情况1: 正常的损失字典（来自我们的测试）
    loss_dict_1 = {
        'loss_classifier': torch.tensor(3.2885),
        'loss_box_reg': torch.tensor(0.0138),
        'loss_objectness': torch.tensor(0.7249),
        'loss_rpn_box_reg': torch.tensor(0.0047)
    }
    
    # 情况2: 包含嵌套字典的损失字典（可能导致问题）
    loss_dict_2 = {
        'loss_classifier': {'value': torch.tensor(3.2885), 'weight': 1.0},
        'loss_box_reg': {'value': torch.tensor(0.0138), 'weight': 1.0},
        'loss_objectness': {'value': torch.tensor(0.7249), 'weight': 1.0},
        'loss_rpn_box_reg': {'value': torch.tensor(0.0047), 'weight': 1.0}
    }
    
    # 情况3: 包含更深层嵌套的字典
    loss_dict_3 = {
        'loss_classifier': {'details': {'value': torch.tensor(3.2885), 'components': [1.0, 2.0]}},
        'loss_box_reg': {'details': {'value': torch.tensor(0.0138), 'components': [0.5]}},
        'loss_objectness': {'details': {'value': torch.tensor(0.7249), 'components': [0.3, 0.4]}},
        'loss_rpn_box_reg': {'details': {'value': torch.tensor(0.0047), 'components': [0.1]}}
    }
    
    # 测试清理函数
    def clean_loss_dict_original(loss_dict):
        """原始清理函数（来自faster_rcnn_model.py）"""
        cleaned_loss_dict = {}
        for key, value in loss_dict.items():
            if isinstance(value, dict):
                # 如果是字典，尝试提取数值
                # 查找第一个数值类型的值
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, torch.Tensor)):
                        if isinstance(sub_value, torch.Tensor):
                            cleaned_loss_dict[key] = sub_value.item()
                        else:
                            cleaned_loss_dict[key] = sub_value
                        break
                else:
                    # 如果没有找到数值，使用0
                    cleaned_loss_dict[key] = 0.0
            elif isinstance(value, (int, float, torch.Tensor)):
                if isinstance(value, torch.Tensor):
                    cleaned_loss_dict[key] = value.item()
                else:
                    cleaned_loss_dict[key] = value
            else:
                # 其他类型，使用0
                cleaned_loss_dict[key] = 0.0
        return cleaned_loss_dict
    
    def clean_loss_dict_fixed(loss_dict):
        """修复后的清理函数"""
        cleaned_loss_dict = {}
        for key, value in loss_dict.items():
            # 递归提取数值
            extracted_value = extract_numeric_value(value)
            cleaned_loss_dict[key] = extracted_value
        return cleaned_loss_dict
    
    def extract_numeric_value(obj, max_depth=5):
        """递归提取数值，防止无限递归"""
        if max_depth <= 0:
            return 0.0
        
        if isinstance(obj, torch.Tensor):
            return obj.item()
        elif isinstance(obj, (int, float)):
            return float(obj)
        elif isinstance(obj, dict):
            # 递归查找第一个数值
            for sub_key, sub_value in obj.items():
                result = extract_numeric_value(sub_value, max_depth-1)
                if result is not None:
                    return result
            return 0.0
        elif isinstance(obj, (list, tuple)):
            # 查找第一个数值
            for item in obj:
                result = extract_numeric_value(item, max_depth-1)
                if result is not None:
                    return result
            return 0.0
        else:
            return 0.0
    
    # 测试每个情况
    test_cases = [
        ("情况1: 正常损失字典", loss_dict_1),
        ("情况2: 嵌套字典损失", loss_dict_2),
        ("情况3: 深层嵌套字典", loss_dict_3)
    ]
    
    for name, loss_dict in test_cases:
        print(f"\n{name}:")
        print(f"  原始损失字典: {loss_dict}")
        
        # 测试原始清理函数
        try:
            cleaned_original = clean_loss_dict_original(loss_dict)
            print(f"  原始清理结果: {cleaned_original}")
            
            # 尝试求和
            total_original = sum(cleaned_original.values())
            print(f"  原始求和结果: {total_original}")
        except Exception as e:
            print(f"  原始清理失败: {e}")
        
        # 测试修复后的清理函数
        try:
            cleaned_fixed = clean_loss_dict_fixed(loss_dict)
            print(f"  修复清理结果: {cleaned_fixed}")
            
            # 尝试求和
            total_fixed = sum(cleaned_fixed.values())
            print(f"  修复求和结果: {total_fixed}")
        except Exception as e:
            print(f"  修复清理失败: {e}")

if __name__ == "__main__":
    test_loss_formats()
