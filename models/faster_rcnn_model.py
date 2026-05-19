import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
import os


class FasterRCNNDetector:
    """Faster R-CNN 目标检测模型"""
    
    def __init__(self, num_classes: int = 25, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化Faster R-CNN检测器
        
        Args:
            num_classes: 类别数量（包括背景），COU数据集为24类+背景=25
            device: 设备类型
        """
        self.device = device
        self.num_classes = num_classes
        
        # COU数据集类别（24类人造物）- 从coco.yaml获取
        self.cou_class_names = [
            'plastic_bottle', 'plastic_bag', 'fishing_net', 'rope', 'can', 
            'glass_bottle', 'tire', 'metal_scrap', 'wood', 'cloth',
            'diver', 'diving_mask', 'diving_fins', 'oxygen_tank', 'underwater_camera',
            'auv', 'rov', 'underwater_drone', 'sonar', 'underwater_sensor',
            'ship_wreck', 'anchor', 'propeller', 'underwater_structure'
        ]
        
        # 默认使用COU类别
        self.class_names = self.cou_class_names
        
        # 初始化模型
        self.model = None
        self._init_model()
        
        # 训练状态
        self.is_trained = False
        self.current_epoch = 0
        self.total_epochs = 0
        
    def _init_model(self):
        """初始化Faster R-CNN模型"""
        try:
            # 使用更轻量级的ResNet18骨干网络以减少内存使用
            backbone = torchvision.models.resnet18(pretrained=True)
            
            # 移除最后的全连接层和平均池化层
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            
            # 获取骨干网络的输出通道数
            backbone.out_channels = 512
            
            # 定义锚点生成器（使用更小的锚点尺寸）
            anchor_generator = AnchorGenerator(
                sizes=((16, 32, 64, 128, 256),),  # 更小的锚点尺寸
                aspect_ratios=((0.5, 1.0, 2.0),)
            )
            
            # 定义ROI对齐
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2
            )
            
            # 创建Faster R-CNN模型，使用更小的图像尺寸
            self.model = FasterRCNN(
                backbone=backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                min_size=416,  # 更小的最小尺寸
                max_size=640   # 更小的最大尺寸
            )
            
            # 移动到设备
            self.model.to(self.device)
            
            print(f"Faster R-CNN模型初始化完成，使用设备: {self.device}")
            print(f"使用ResNet18骨干网络，图像尺寸: min_size=416, max_size=640")
            
        except Exception as e:
            print(f"初始化Faster R-CNN模型失败: {e}")
            # 回退到简单的模型
            self._init_simple_model()
    
    def _init_simple_model(self):
        """初始化简单的Faster R-CNN模型（回退方案）"""
        try:
            # 加载预训练的Faster R-CNN模型
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            
            # 获取输入特征的数量
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            
            # 替换分类器头以适应我们的类别数量
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            
            # 移动到设备
            self.model.to(self.device)
            
            print(f"简单Faster R-CNN模型初始化完成，使用设备: {self.device}")
            
        except Exception as e:
            print(f"初始化简单Faster R-CNN模型失败: {e}")
            raise
    
    def load_pretrained(self, model_path: str):
        """
        加载预训练权重
        
        Args:
            model_path: 模型权重文件路径
        """
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                print(f"从 {model_path} 加载预训练权重成功")
                self.is_trained = True
                
                # 加载训练状态
                if 'epoch' in checkpoint:
                    self.current_epoch = checkpoint['epoch']
                if 'total_epochs' in checkpoint:
                    self.total_epochs = checkpoint['total_epochs']
                    
            else:
                print(f"模型文件 {model_path} 不存在，使用随机初始化权重")
                
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    def _detect_dataset_type(self, data_yaml_path):
        """
        检测数据集类型（YOLO或COCO）
        
        Args:
            data_yaml_path: 数据YAML文件路径
            
        Returns:
            str: 数据集类型 ('yolo' 或 'coco')
        """
        import yaml
        
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            # 检查是否包含COCO特定的字段
            if 'train_annotations' in data_config or 'val_annotations' in data_config:
                return 'coco'
            
            # 检查路径是否包含'coco'关键字
            if 'coco' in data_yaml_path.lower():
                return 'coco'
            
            # 默认返回'yolo'
            return 'yolo'
            
        except Exception as e:
            print(f"检测数据集类型失败: {e}")
            # 默认返回'yolo'
            return 'yolo'
    
    def _create_fixed_data_loaders(self, data_dir, batch_size=8, num_workers=2):
        """
        创建固定数据路径的数据加载器
        
        Args:
            data_dir: 数据目录路径
            batch_size: 批次大小
            num_workers: 数据加载工作线程数
            
        Returns:
            tuple: (训练数据加载器, 验证数据加载器)
        """
        import json
        from torch.utils.data import Dataset, DataLoader
        from PIL import Image
        import torchvision.transforms as T
        
        # 将相对路径转换为绝对路径
        if not os.path.isabs(data_dir):
            # 获取当前工作目录
            current_dir = os.getcwd()
            data_dir = os.path.join(current_dir, data_dir)
            print(f"将相对路径转换为绝对路径: {data_dir}")
        
        class FixedCOCODataset(Dataset):
            """固定COCO格式数据集"""
            
            def __init__(self, images_dir, annotation_file, transform=None):
                self.images_dir = images_dir
                self.transform = transform
                
                # 加载COCO标注
                with open(annotation_file, 'r') as f:
                    self.coco_data = json.load(f)
                
                # 创建图像ID到图像信息的映射
                self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
                
                # 创建图像ID到标注列表的映射
                self.image_id_to_annotations = {}
                for ann in self.coco_data['annotations']:
                    img_id = ann['image_id']
                    if img_id not in self.image_id_to_annotations:
                        self.image_id_to_annotations[img_id] = []
                    self.image_id_to_annotations[img_id].append(ann)
                
                # 创建类别ID到名称的映射
                self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
                
                # 图像ID列表
                self.image_ids = list(self.image_id_to_info.keys())
                
                print(f"加载了 {len(self.image_ids)} 张图像，{len(self.coco_data['annotations'])} 个标注")
            
            def __len__(self):
                return len(self.image_ids)
            
            def __getitem__(self, idx):
                image_id = self.image_ids[idx]
                image_info = self.image_id_to_info[image_id]
                
                # 加载图像
                image_path = os.path.join(self.images_dir, image_info['file_name'])
                image = Image.open(image_path).convert('RGB')
                
                # 获取标注
                annotations = self.image_id_to_annotations.get(image_id, [])
                
                # 准备目标字典
                boxes = []
                labels = []
                
                for ann in annotations:
                    # COCO格式: [x_min, y_min, width, height] (绝对坐标)
                    x_min, y_min, width, height = ann['bbox']
                    
                    # 确保宽度和高度为正数
                    if width <= 0 or height <= 0:
                        continue
                    
                    # 计算最大坐标
                    x_max = x_min + width
                    y_max = y_min + height
                    
                    # 确保x_max > x_min且y_max > y_min
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    
                    # 转换为浮点数张量
                    boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                    labels.append(ann['category_id'])  # COCO类别ID从1开始
                
                # 转换为张量
                if self.transform:
                    image = self.transform(image)
                
                # 如果没有标注，创建空的边界框和标签
                if len(boxes) == 0:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
                else:
                    boxes = torch.tensor(boxes, dtype=torch.float32)
                    labels = torch.tensor(labels, dtype=torch.int64)
                
                # 创建目标字典
                target = {
                    'boxes': boxes,
                    'labels': labels,
                    'image_id': torch.tensor([image_id])
                }
                
                return image, target
        
        # 定义转换
        transform = T.Compose([
            T.ToTensor(),
        ])
        
        # 检查数据目录结构
        images_dir = os.path.join(data_dir, 'images')
        train_annotation_file = os.path.join(data_dir, 'train_annotations.json')
        val_annotation_file = os.path.join(data_dir, 'val_annotations.json')
        
        # 检查路径是否存在
        print(f"检查图像目录: {images_dir}")
        print(f"检查训练标注文件: {train_annotation_file}")
        print(f"检查验证标注文件: {val_annotation_file}")
        
        if not os.path.exists(images_dir):
            raise ValueError(f"图像目录不存在: {images_dir}")
        
        # 创建训练数据集
        if os.path.exists(train_annotation_file):
            train_dataset = FixedCOCODataset(images_dir, train_annotation_file, transform=transform)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers,
                collate_fn=lambda x: tuple(zip(*x))  # Faster R-CNN需要的collate函数
            )
            print(f"创建训练数据加载器: {len(train_dataset)} 张图像")
        else:
            print(f"训练标注文件不存在: {train_annotation_file}")
            train_loader = None
        
        # 创建验证数据集
        if os.path.exists(val_annotation_file):
            val_dataset = FixedCOCODataset(images_dir, val_annotation_file, transform=transform)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                collate_fn=lambda x: tuple(zip(*x))  # Faster R-CNN需要的collate函数
            )
            print(f"创建验证数据加载器: {len(val_dataset)} 张图像")
        else:
            print(f"验证标注文件不存在: {val_annotation_file}")
            val_loader = None
        
        return train_loader, val_loader
    
    def train(self, train_loader=None, val_loader=None, epochs: int = 10, 
              learning_rate: float = 0.001, save_dir: str = 'checkpoints/faster_rcnn',
              data_path: str = None, batch_size: int = 16, num_workers: int = 8,
              use_fixed_data: bool = False, fixed_data_dir: str = None,
              prefetch_factor: int = 2):
        """
        训练模型（使用基本的训练方法，不包含混合精度训练）
        
        Args:
            train_loader: 训练数据加载器（如果为None，则使用data_path创建）
            val_loader: 验证数据加载器（可选，如果为None且data_path不为None，则从data_path创建）
            epochs: 训练轮数
            learning_rate: 学习率
            save_dir: 保存目录
            data_path: 数据YAML文件路径（如果train_loader为None则必需）
            batch_size: 批次大小（如果使用data_path），默认16以更好利用GPU
            num_workers: 数据加载工作线程数（如果使用data_path），默认8以提高数据加载效率
            use_fixed_data: 是否使用固定数据路径（不使用数据加载器）
            fixed_data_dir: 固定数据目录路径（如果use_fixed_data为True则必需）
            prefetch_factor: 数据预取因子，默认2以提前加载数据
            
        Returns:
            dict: 训练历史记录
        """
        # 如果使用固定数据路径，直接加载数据
        if use_fixed_data:
            if fixed_data_dir is None:
                raise ValueError("使用固定数据路径时，必须提供fixed_data_dir参数")
            
            print(f"使用固定数据路径: {fixed_data_dir}")
            train_loader, val_loader = self._create_fixed_data_loaders(
                fixed_data_dir, batch_size, num_workers
            )
        # 如果提供了data_path但没有提供train_loader，则创建数据加载器
        elif train_loader is None:
            if data_path is None:
                raise ValueError("必须提供train_loader或data_path参数")
            
            # 检测数据集类型并选择相应的数据加载器
            try:
                # 首先尝试导入YOLO数据加载器
                from utils.data_loader import create_data_loaders as create_yolo_data_loaders
                # 然后尝试导入COCO数据加载器
                from utils.coco_data_loader import create_coco_data_loaders as create_coco_data_loaders
                
                # 检测数据集类型
                dataset_type = self._detect_dataset_type(data_path)
                print(f"检测到数据集类型: {dataset_type}")
                
                if dataset_type == 'coco':
                    # 使用COCO数据加载器
                    train_loader, val_loader = create_coco_data_loaders(
                        data_path, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor
                    )
                    print(f"从 {data_path} 创建COCO数据加载器成功")
                else:
                    # 默认使用YOLO数据加载器
                    train_loader, val_loader = create_yolo_data_loaders(
                        data_path, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor
                    )
                    print(f"从 {data_path} 创建YOLO数据加载器成功")
                    
            except ImportError as e:
                print(f"导入数据加载器失败: {e}")
                print("请确保utils/data_loader.py和utils/coco_data_loader.py文件存在")
                raise
            except Exception as e:
                print(f"创建数据加载器失败: {e}")
                raise
        
        # 如果val_loader为None且没有从data_path创建，则设置为None
        if val_loader is None:
            print("警告: 未提供验证数据加载器，将只进行训练")
        
        # 打印训练配置
        print(f"\n{'='*80}")
        print("训练配置优化:")
        print(f"  批次大小: {batch_size} (增加以更好利用GPU)")
        print(f"  数据加载工作线程数: {num_workers} (增加以减少数据加载瓶颈)")
        print(f"  数据预取因子: {prefetch_factor} (提前加载数据)")
        print(f"  混合精度训练: 禁用 (使用基本的训练方法)")
        print(f"  设备: {self.device}")
        print(f"{'='*80}")
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置模型为训练模式
        self.model.train()
        
        # 定义优化器
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        
        # 学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # 不使用混合精度训练，仅使用基本的训练方法
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mAP': [],
            'val_mAP': []
        }
        
        print(f"开始训练Faster R-CNN模型，共{epochs}轮...")
        print(f"训练数据: {len(train_loader.dataset)} 张图像, {len(train_loader)} 个批次")
        if val_loader is not None:
            print(f"验证数据: {len(val_loader.dataset)} 张图像, {len(val_loader)} 个批次")
        print("=" * 80)
        
        # 记录GPU使用情况
        if self.device == 'cuda':
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            self.total_epochs = epochs
            
            # 训练阶段
            train_loss = 0.0
            train_batches = 0
            
            # 详细的损失记录
            epoch_loss_dict = {
                'loss_classifier': 0.0,
                'loss_box_reg': 0.0,
                'loss_objectness': 0.0,
                'loss_rpn_box_reg': 0.0
            }
            
            print(f"\n{'='*80}")
            print(f"Epoch [{epoch+1}/{epochs}] 开始训练...")
            print(f"{'='*80}")
            
            # 设置进度条
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                # 移动到设备
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 前向传播（使用基本的训练方法，不包含混合精度训练）
                loss_dict = self.model(images, targets)
                # 检查loss_dict类型，如果是列表则转换为字典
                if isinstance(loss_dict, list):
                    # 假设列表中的元素是损失值，创建一个简单的字典
                    # 确保列表中的元素都是数值类型
                    try:
                        total_loss = sum(float(loss) for loss in loss_dict)
                        loss_dict = {'total_loss': total_loss}
                    except (TypeError, ValueError):
                        # 如果无法转换为数值，使用默认值
                        loss_dict = {'total_loss': 0.0}
                
                # 辅助函数：递归提取数值并转换为torch.Tensor
                def extract_and_convert_to_tensor(obj, max_depth=5):
                    """递归提取数值并转换为torch.Tensor，防止无限递归"""
                    if max_depth <= 0:
                        return torch.tensor(0.0, device=self.device, requires_grad=True)
                    
                    if isinstance(obj, torch.Tensor):
                        # 如果已经是tensor，确保它需要梯度
                        if not obj.requires_grad:
                            obj = obj.detach().requires_grad_(True)
                        return obj
                    elif isinstance(obj, (int, float)):
                        return torch.tensor(float(obj), device=self.device, requires_grad=True)
                    elif isinstance(obj, dict):
                        # 递归查找第一个数值
                        for sub_key, sub_value in obj.items():
                            result = extract_and_convert_to_tensor(sub_value, max_depth-1)
                            if result is not None:
                                return result
                        return torch.tensor(0.0, device=self.device, requires_grad=True)
                    elif isinstance(obj, (list, tuple)):
                        # 查找第一个数值
                        for item in obj:
                            result = extract_and_convert_to_tensor(item, max_depth-1)
                            if result is not None:
                                return result
                        return torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        return torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # 确保loss_dict中的所有值都是torch.Tensor类型以支持反向传播
                cleaned_loss_dict = {}
                for key, value in loss_dict.items():
                    cleaned_loss_dict[key] = extract_and_convert_to_tensor(value)
                
                # 使用清理后的loss_dict，确保所有值都是torch.Tensor
                losses = sum(loss for loss in cleaned_loss_dict.values())
                
                # 反向传播（基本的训练方法）
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                # 记录损失
                train_loss += losses.item()
                train_batches += 1
                
                # 记录详细的损失
                for key in epoch_loss_dict.keys():
                    if key in loss_dict:
                        epoch_loss_dict[key] += loss_dict[key].item()
                
                # 实时更新进度条
                current_loss = losses.item()
                avg_loss_so_far = train_loss / train_batches
                
                # 构建进度条描述
                progress_desc = f"Loss: {current_loss:.4f} | Avg: {avg_loss_so_far:.4f}"
                
                # 显示GPU内存使用情况
                if self.device == 'cuda' and batch_idx % 10 == 0:
                    gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_memory_max = torch.cuda.max_memory_allocated(0) / 1024**3
                    progress_desc += f" | GPU: {gpu_memory:.2f}GB"
                
                if batch_idx % 5 == 0:  # 每5个batch显示一次详细损失
                    detailed_losses = []
                    for key, value in loss_dict.items():
                        detailed_losses.append(f"{key}: {value.item():.4f}")
                    progress_desc += " | " + ", ".join(detailed_losses[:2])  # 只显示前两个详细损失
                
                pbar.set_description(progress_desc)
                pbar.refresh()  # 强制刷新显示
                
            
            # 关闭进度条
            pbar.close()
            
            # 计算平均训练损失
            avg_train_loss = train_loss / max(train_batches, 1)
            history['train_loss'].append(avg_train_loss)
            
            # 计算详细的平均损失
            for key in epoch_loss_dict.keys():
                epoch_loss_dict[key] = epoch_loss_dict[key] / max(train_batches, 1)
            
            # 验证阶段
            if val_loader is not None:
                print(f"\n{'='*80}")
                print(f"Epoch [{epoch+1}/{epochs}] 开始验证...")
                print(f"{'='*80}")
                
                val_loss, val_mAP = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_mAP'].append(val_mAP)
                
                print(f"\nEpoch [{epoch+1}/{epochs}] 完成!")
                print(f"  训练损失: {avg_train_loss:.4f}")
                print(f"  验证损失: {val_loss:.4f}")
                print(f"  验证mAP: {val_mAP:.4f}")
                print(f"  详细训练损失:")
                for key, value in epoch_loss_dict.items():
                    print(f"    {key}: {value:.4f}")
            else:
                print(f"\nEpoch [{epoch+1}/{epochs}] 完成!")
                print(f"  训练损失: {avg_train_loss:.4f}")
                print(f"  详细训练损失:")
                for key, value in epoch_loss_dict.items():
                    print(f"    {key}: {value:.4f}")
            
            # 更新学习率
            lr_scheduler.step()
            print(f"  更新后学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                checkpoint_path = os.path.join(save_dir, f'faster_rcnn_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss if val_loader is not None else 0.0,
                    'val_mAP': val_mAP if val_loader is not None else 0.0,
                    'total_epochs': epochs,
                    'detailed_losses': epoch_loss_dict
                }, checkpoint_path)
                print(f"  检查点已保存到: {checkpoint_path}")
            
            print(f"{'='*80}\n")
        
        # 训练完成
        self.is_trained = True
        print(f"{'='*80}")
        print("Faster R-CNN模型训练完成!")
        print(f"总训练轮数: {epochs}")
        print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
        if val_loader is not None:
            print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
            print(f"最终验证mAP: {history['val_mAP'][-1]:.4f}")
        print(f"{'='*80}")
        
        return history
    
    def evaluate(self, data_loader):
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            tuple: (平均损失, mAP分数)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_batches = 0
        
        # 用于计算mAP的列表
        all_predictions = []
        all_targets = []
        
        # 辅助函数：递归提取数值
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
        
        with torch.no_grad():
            for images, targets in data_loader:
                # 移动到设备
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 前向传播
                loss_dict = self.model(images, targets)
                
                # 检查loss_dict类型，如果是列表则转换为字典
                if isinstance(loss_dict, list):
                    # 假设列表中的元素是损失值，创建一个简单的字典
                    loss_dict = {'total_loss': sum(loss_dict)}
                
                # 确保loss_dict中的所有值都是数字类型，评估模式下不需要保持torch.Tensor类型
                cleaned_loss_dict = {}
                for key, value in loss_dict.items():
                    cleaned_loss_dict[key] = extract_numeric_value(value)
                
                # 使用清理后的loss_dict
                losses = sum(loss for loss in cleaned_loss_dict.values())
                
                # 记录损失
                total_loss += losses
                total_batches += 1
                
                # 获取预测结果用于mAP计算
                predictions = self.model(images)
                
                # 收集预测和目标
                for i in range(len(predictions)):
                    pred_boxes = predictions[i]['boxes'].cpu().numpy()
                    pred_scores = predictions[i]['scores'].cpu().numpy()
                    pred_labels = predictions[i]['labels'].cpu().numpy()
                    
                    target_boxes = targets[i]['boxes'].cpu().numpy()
                    target_labels = targets[i]['labels'].cpu().numpy()
                    
                    all_predictions.append({
                        'boxes': pred_boxes,
                        'scores': pred_scores,
                        'labels': pred_labels
                    })
                    
                    all_targets.append({
                        'boxes': target_boxes,
                        'labels': target_labels
                    })
        
        # 计算平均损失
        avg_loss = total_loss / max(total_batches, 1)
        
        # 计算mAP（简化版本）
        mAP = self._calculate_simple_map(all_predictions, all_targets)
        
        # 恢复训练模式
        self.model.train()
        
        return avg_loss, mAP
    
    def _calculate_simple_map(self, predictions, targets, iou_threshold: float = 0.5):
        """
        计算简化的mAP
        
        Args:
            predictions: 预测结果列表
            targets: 目标列表
            iou_threshold: IoU阈值
            
        Returns:
            float: mAP分数
        """
        if not predictions or not targets:
            return 0.0
        
        # 简化的mAP计算
        total_precision = 0.0
        total_recall = 0.0
        num_samples = len(predictions)
        
        for i in range(num_samples):
            pred = predictions[i]
            target = targets[i]
            
            if len(pred['boxes']) == 0 or len(target['boxes']) == 0:
                continue
            
            # 计算每个预测的IoU
            tp = 0
            fp = 0
            fn = 0
            
            for j, pred_box in enumerate(pred['boxes']):
                max_iou = 0.0
                matched = False
                
                for k, target_box in enumerate(target['boxes']):
                    iou = self._calculate_iou(pred_box, target_box)
                    if iou > max_iou:
                        max_iou = iou
                
                if max_iou >= iou_threshold:
                    tp += 1
                else:
                    fp += 1
            
            fn = max(0, len(target['boxes']) - tp)
            
            # 计算精度和召回率
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            
            total_precision += precision
            total_recall += recall
        
        avg_precision = total_precision / max(num_samples, 1)
        avg_recall = total_recall / max(num_samples, 1)
        
        # 计算F1分数作为简化的mAP
        if avg_precision + avg_recall > 0:
            f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            f1_score = 0.0
        
        return f1_score
    
    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU
        
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            float: IoU值
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 计算交集面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # 计算IoU
        iou = intersection / union if union > 0 else 0.0
        
        return iou
    
    def detect(self, image, confidence_threshold: float = 0.5):
        """
        检测图像中的目标
        
        Args:
            image: 输入图像（numpy数组或PIL图像）
            confidence_threshold: 置信度阈值
            
        Returns:
            list: 检测结果，每个元素为[类别名称, 置信度, 边界框]
        """
        self.model.eval()
        
        # 转换图像格式
        if isinstance(image, np.ndarray):
            # OpenCV BGR转RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 转换为PIL图像
            from PIL import Image
            image = Image.fromarray(image)
        
        # 转换为张量
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 进行预测
            predictions = self.model(image_tensor)
        
        # 解析预测结果
        results = []
        pred = predictions[0]
        
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] >= confidence_threshold:
                # 获取类别索引（减1因为0是背景）
                class_idx = int(labels[i]) - 1
                
                # 确保索引在有效范围内
                if 0 <= class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    class_name = f"未知类别_{class_idx}"
                
                # 边界框格式转换
                bbox = boxes[i].tolist()
                
                results.append([class_name, float(scores[i]), bbox])
        
        return results
    
    def save_model(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        try:
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'is_trained': self.is_trained,
                'total_epochs': self.total_epochs
            }, save_path)
            print(f"模型已保存到 {save_path}")
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    def get_class_names(self):
        """
        获取类别名称列表
        
        Returns:
            list: 类别名称列表
        """
        return self.class_names
    
    def get_num_classes(self):
        """
        获取类别数量（不包括背景）
        
        Returns:
            int: 类别数量
        """
        return len(self.class_names)
