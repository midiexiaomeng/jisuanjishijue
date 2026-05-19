# Transformer目标检测实验项目

基于Transformer架构的端到端目标检测系统，使用DETR方法在COCO数据集上进行训练和评估。

## 项目概述

本项目实现了一个完整的基于Transformer的目标检测系统，包含：

- **模型架构**: DETR (Detection Transformer) 端到端检测
- **数据集**: MS COCO 2017 (80个类别)
- **训练框架**: PyTorch + Transformers库
- **评估指标**: COCO标准评估指标

## 项目结构

```
transformer_detection/
├── configs/                    # 配置文件
│   ├── __init__.py
│   └── default_config.py      # 实验配置参数
├── models/                     # 模型定义
│   ├── __init__.py
│   └── transformer_detector.py # Transformer检测模型
├── training/                   # 训练脚本
│   ├── __init__.py
│   ├── train.py               # 模型训练
│   └── evaluate.py            # 模型评估
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── data_utils.py          # 数据加载和预处理
│   └── losses.py              # 损失函数
├── data/                       # 数据集目录
├── demo.py                    # 目标检测演示
├── download_dataset.py        # 数据集下载脚本
├── run_experiment.py          # 完整实验流程
├── experiment_report.md       # 详细实验报告
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明
```

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
# 下载COCO数据集
python download_dataset.py --dataset coco

# 或下载简单测试数据集
python download_dataset.py --dataset simple
```

### 3. 运行完整实验

```bash
python run_experiment.py
```

### 4. 单独运行各组件

```bash
# 训练模型
python training/train.py

# 评估模型
python training/evaluate.py

# 运行演示
python demo.py --image path/to/image.jpg
```

## 模型架构

### DETR (Detection Transformer)

- **骨干网络**: ResNet-50
- **Transformer**: 6层编码器 + 6层解码器
- **隐藏维度**: 256
- **注意力头数**: 8
- **目标查询数**: 100

### 损失函数

- **分类损失**: 交叉熵损失
- **边界框损失**: L1损失 + GIoU损失
- **匹配算法**: 匈牙利匹配

## 实验结果

### 预期性能 (基于DETR论文)

| 指标 | 数值 |
|------|------|
| AP | 42.0 |
| AP50 | 62.4 |
| AP75 | 44.2 |
| 推理速度 | 28 FPS |

### 实际性能

*注：实际性能取决于训练完成度和硬件配置*

## 主要特性

1. **端到端检测**: 无需NMS后处理
2. **全局上下文**: Transformer提供全局信息理解
3. **并行处理**: 高效推理速度
4. **多类别检测**: 支持80个COCO类别
5. **可视化工具**: 完整的检测结果可视化

## 应用场景

- 自动驾驶
- 视频监控
- 工业检测
- 医学影像分析
- 智能安防

## 技术亮点

### 1. 端到端架构
- 直接输出预测结果
- 无需复杂的后处理流程
- 简化检测pipeline

### 2. Transformer优势
- 全局上下文理解能力
- 强大的特征表示
- 并行计算效率

### 3. 匈牙利匹配
- 解决预测-标注匹配问题
- 避免重复检测
- 优化训练过程

## 文件说明

### 核心文件

- `models/transformer_detector.py`: Transformer检测模型实现
- `training/train.py`: 完整的训练循环
- `training/evaluate.py`: 模型评估和性能分析
- `utils/data_utils.py`: COCO数据集加载器
- `utils/losses.py`: 匈牙利匹配和损失函数
- `demo.py`: 目标检测演示脚本

### 配置文件

- `configs/default_config.py`: 实验参数配置
- `requirements.txt`: 项目依赖包

### 实验文件

- `experiment_report.md`: 详细实验报告
- `run_experiment.py`: 自动化实验流程

## 依赖包

```python
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.0.0
opencv-python>=4.5.0
albumentations>=1.0.0
pycocotools>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
```

## 使用示例

### 单张图像检测

```python
python demo.py --image test_image.jpg --confidence 0.7
```

### 视频流检测

```python
python demo.py --video test_video.mp4 --output output_video.mp4
```

### 摄像头实时检测

```python
python demo.py --camera 0
```

## 训练配置

### 默认训练参数

- **批量大小**: 4
- **学习率**: 1e-4
- **优化器**: AdamW
- **训练轮数**: 300
- **权重衰减**: 1e-4

### 自定义训练

```python
python training/train.py \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --epochs 150 \
    --num_queries 100
```

## 评估指标

使用COCO标准评估指标：
- **AP**: 平均精度 (IoU=0.50:0.95)
- **AP50**: IoU=0.50时的AP
- **AP75**: IoU=0.75时的AP
- **APS/APM/APL**: 不同尺度目标的AP

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 致谢

- Facebook AI Research的DETR论文
- COCO数据集团队
- PyTorch和Transformers社区

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

---

**项目版本**: v1.0  
**最后更新**: 2025年11月10日  
**维护者**: AI助手
