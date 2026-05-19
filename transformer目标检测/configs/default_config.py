import torch

class Config:
    # 数据集配置
    dataset_name = "coco"  # 使用COCO数据集进行目标检测
    data_dir = "./data/coco"
    num_classes = 91  # COCO数据集类别数
    
    # 模型配置
    model_name = "detr"  # DETR: Detection Transformer
    backbone = "resnet50"
    hidden_dim = 256
    num_queries = 100
    num_encoder_layers = 6
    num_decoder_layers = 6
    nheads = 8
    dropout = 0.1
    
    # 训练配置
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    warmup_epochs = 5
    
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    
    # 数据增强
    image_size = (640, 640)
    use_augmentation = True
    
    # 实验配置
    experiment_name = "transformer_object_detection"
    save_dir = "./checkpoints"
    log_dir = "./logs"
    
    # 评估配置
    eval_interval = 5
    save_interval = 10
    
    def __str__(self):
        return f"Config(dataset={self.dataset_name}, model={self.model_name}, device={self.device})"

config = Config()
