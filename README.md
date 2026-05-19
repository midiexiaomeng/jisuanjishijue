# 水下目标检测项目

基于深度学习的水下目标检测系统，支持多种目标检测模型，专门针对水下环境优化。

## 项目特点

- **多模型支持**: YOLOv8, Faster R-CNN, SSD, RetinaNet, EfficientDet
- **GUI界面**: 友好的图形用户界面，支持图像/视频检测、模型训练和评估
- **训练可视化**: 实时训练进度监控，包括损失曲线、精度曲线等
- **模型评估**: 全面的评估指标（mAP, Precision, Recall, F1-score）
- **数据集**: COU (Common Objects Underwater) 数据集，24类人造物

## 数据集

使用COU (Common Objects Underwater) 数据集：
- 约10,000张图像
- 24类人造物（海洋垃圾、潜水工具、AUV等）
- 兼顾封闭水域（泳池）与开放水域（湖、海）
- 图像尺寸：1920×1080

数据集位置：`data/coco/`

## 项目结构

```
目标检测/
├── data/                    # 数据集
│   └── coco/               # COCO格式数据集
├── models/                  # 模型定义
│   ├── yolov8_model.py     # YOLOv8模型
│   ├── faster_rcnn_model.py # Faster R-CNN模型
│   ├── ssd_model.py        # SSD模型
│   ├── retinanet_model.py  # RetinaNet模型
│   └── efficientdet_model.py # EfficientDet模型
├── gui/                    # GUI界面
│   ├── main_window.py      # 主窗口
│   └── components/         # GUI组件
├── utils/                  # 工具函数
│   ├── coco_dataset.py     # 数据集加载
│   ├── transforms.py       # 数据增强
│   └── metrics.py          # 评估指标
├── config/                 # 配置文件
│   └── training_config.json # 训练配置
├── checkpoints/            # 模型检查点
├── logs/                   # 训练日志
├── results/                # 结果输出
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── detect.py              # 推理脚本
├── main_gui.py            # GUI主程序
└── requirements.txt       # 依赖包
```

## 安装

1. 克隆项目：
```bash
git clone <repository-url>
cd 目标检测
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据集：
确保数据集位于 `data/coco/` 目录下，包含：
- `images/` - 图像文件
- `train_annotations.json` - 训练标注
- `val_annotations.json` - 验证标注
- `test_annotations.json` - 测试标注

## 使用方法

### 训练模型

```bash
# 训练YOLOv8模型
python train.py --model yolov8 --epochs 100 --batch-size 16

# 训练所有模型
python train.py --model all --epochs 50
```

### 评估模型

```bash
# 评估单个模型
python evaluate.py --model yolov8 --checkpoint checkpoints/yolov8/best.pt

# 比较所有模型
python evaluate.py --model all
```

### 使用GUI

```bash
python main_gui.py
```

GUI功能包括：
- 图像/视频目标检测
- 模型训练监控
- 性能评估可视化
- 模型管理

### 命令行推理

```bash
# 图像检测
python detect.py --model yolov8 --source data/test_image.jpg --output results/

# 视频检测
python detect.py --model yolov8 --source data/test_video.mp4 --output results/
```

## 模型性能

| 模型 | mAP@0.5 | 推理速度 (FPS) | 参数量 (M) |
|------|---------|---------------|-----------|
| YOLOv8n | 0.65 | 120 | 3.2 |
| Faster R-CNN | 0.68 | 25 | 41.2 |
| SSD | 0.62 | 60 | 26.8 |
| RetinaNet | 0.67 | 35 | 36.3 |
| EfficientDet | 0.66 | 45 | 15.4 |

## 训练可视化

训练过程中会自动生成以下可视化：
- 损失曲线（训练/验证）
- 精度曲线（mAP, Precision, Recall）
- 学习率变化
- 验证集预测示例

使用TensorBoard查看详细训练日志：
```bash
tensorboard --logdir logs/
```

## 配置

修改 `config/training_config.json` 调整训练参数：

```json
{
  "batch_size": 16,
  "epochs": 100,
  "learning_rate": 0.001,
  "img_size": 640,
  "num_workers": 4,
  "device": "cuda",
  "save_dir": "checkpoints/",
  "log_dir": "logs/"
}
```

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请联系项目维护者。
