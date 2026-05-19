import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerDetector(nn.Module):
    """基于Transformer的目标检测器"""
    
    def __init__(self, config):
        super(TransformerDetector, self).__init__()
        self.config = config
        
        # 骨干网络
        self.backbone = self._build_backbone()
        
        # 特征投影
        self.input_proj = nn.Conv2d(
            self.backbone.out_channels, 
            config.hidden_dim, 
            kernel_size=1
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(config.hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nheads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nheads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_decoder_layers)
        
        # 查询嵌入
        self.query_embed = nn.Embedding(config.num_queries, config.hidden_dim)
        
        # 预测头
        self.class_embed = nn.Linear(config.hidden_dim, config.num_classes + 1)  # +1 for background
        self.bbox_embed = nn.Linear(config.hidden_dim, 4)  # [x, y, w, h]
        
        # 初始化权重
        self._init_weights()
    
    def _build_backbone(self):
        """构建骨干网络"""
        backbone = resnet50(pretrained=True)
        
        # 移除最后的全连接层和平均池化层
        modules = list(backbone.children())[:-2]
        backbone = nn.Sequential(*modules)
        
        # 添加属性以获取输出通道数
        backbone.out_channels = 2048
        
        return backbone
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 特殊初始化
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], 0.0)
    
    def forward(self, images):
        batch_size = images.shape[0]
        
        # 特征提取
        features = self.backbone(images)  # [batch_size, 2048, H/32, W/32]
        features = self.input_proj(features)  # [batch_size, hidden_dim, H/32, W/32]
        
        # 展平空间维度
        h, w = features.shape[-2:]
        features = features.flatten(2).permute(2, 0, 1)  # [H*W, batch_size, hidden_dim]
        
        # 添加位置编码
        features = self.pos_encoding(features)
        
        # 编码器处理
        memory = self.encoder(features)  # [H*W, batch_size, hidden_dim]
        
        # 查询嵌入
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # [num_queries, batch_size, hidden_dim]
        
        # 解码器处理
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, query_pos=query_embed)  # [num_queries, batch_size, hidden_dim]
        
        # 预测
        outputs_class = self.class_embed(hs)  # [num_queries, batch_size, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [num_queries, batch_size, 4]
        
        # 转置为 [batch_size, num_queries, ...]
        outputs_class = outputs_class.transpose(0, 1)
        outputs_coord = outputs_coord.transpose(0, 1)
        
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

class DETR(nn.Module):
    """DETR模型包装器"""
    
    def __init__(self, config):
        super(DETR, self).__init__()
        self.transformer = TransformerDetector(config)
        self.config = config
    
    def forward(self, images, targets=None):
        return self.transformer(images)

def build_model(config):
    """构建模型"""
    model = DETR(config)
    return model

if __name__ == "__main__":
    from configs.default_config import Config
    config = Config()
    model = build_model(config)
    
    # 测试模型
    dummy_input = torch.randn(2, 3, 640, 640)
    output = model(dummy_input)
    
    print(f"模型输出形状:")
    print(f"预测类别: {output['pred_logits'].shape}")
    print(f"预测边界框: {output['pred_boxes'].shape}")
