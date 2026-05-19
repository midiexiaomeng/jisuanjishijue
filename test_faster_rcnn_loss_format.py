import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn as nn

# 创建一个简单的Faster R-CNN模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 使用更轻量级的ResNet18骨干网络
backbone = torchvision.models.resnet18(pretrained=True)
backbone = nn.Sequential(*list(backbone.children())[:-2])
backbone.out_channels = 512

# 定义锚点生成器
anchor_generator = AnchorGenerator(
    sizes=((16, 32, 64, 128, 256),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# 创建Faster R-CNN模型
model = FasterRCNN(
    backbone=backbone,
    num_classes=25,  # 24类+背景
    rpn_anchor_generator=anchor_generator,
    min_size=416,
    max_size=640
)

model.to(device)
model.train()

# 创建一个简单的测试输入
batch_size = 2
images = [torch.randn(3, 416, 640).to(device) for _ in range(batch_size)]
targets = [
    {
        'boxes': torch.tensor([[100, 100, 200, 200], [150, 150, 250, 250]], dtype=torch.float32).to(device),
        'labels': torch.tensor([1, 2], dtype=torch.int64).to(device)
    },
    {
        'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32).to(device),
        'labels': torch.tensor([3], dtype=torch.int64).to(device)
    }
]

# 测试前向传播
print("\n测试前向传播...")
try:
    loss_dict = model(images, targets)
    print(f"loss_dict类型: {type(loss_dict)}")
    print(f"loss_dict内容: {loss_dict}")
    
    # 检查loss_dict的结构
    if isinstance(loss_dict, dict):
        print("\n损失字典键值:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value}, 类型: {type(value)}")
            if isinstance(value, dict):
                print(f"    子字典: {value}")
    elif isinstance(loss_dict, list):
        print(f"\n损失列表长度: {len(loss_dict)}")
        for i, item in enumerate(loss_dict):
            print(f"  索引{i}: {item}, 类型: {type(item)}")
    
    # 测试求和
    print("\n测试损失求和...")
    if isinstance(loss_dict, dict):
        total_loss = sum(loss_dict.values())
        print(f"总损失: {total_loss}")
    elif isinstance(loss_dict, list):
        # 检查列表中的元素类型
        for i, item in enumerate(loss_dict):
            print(f"  索引{i}的类型: {type(item)}")
            if isinstance(item, dict):
                print(f"    字典键: {list(item.keys())}")
        
        # 尝试求和
        try:
            total_loss = sum(loss_dict)
            print(f"总损失: {total_loss}")
        except Exception as e:
            print(f"求和失败: {e}")
            
except Exception as e:
    print(f"前向传播失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")
