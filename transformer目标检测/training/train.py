import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import time
import json
from tqdm import tqdm

from models.transformer_detector import build_model
from utils.data_utils import get_data_loaders
from utils.losses import build_criterion
from configs.default_config import Config

# 可选导入wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("警告: wandb未安装，将使用本地日志记录")

class Trainer:
    """训练器类"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 创建目录
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 初始化模型
        self.model = build_model(config).to(self.device)
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = get_data_loaders(config)
        
        # 初始化损失函数
        self.criterion, self.weight_dict = build_criterion(config)
        self.criterion = self.criterion.to(self.device)
        
        # 初始化优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 初始化学习率调度器
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # 初始化WandB
        if self._init_wandb():
            print("WandB初始化成功")
        
        # 训练状态
        self.current_epoch = 0
        self.best_map = 0.0
        
    def _init_wandb(self):
        """初始化WandB"""
        if not WANDB_AVAILABLE:
            return False
        try:
            wandb.init(
                project="transformer-object-detection",
                name=self.config.experiment_name,
                config=vars(self.config)
            )
            return True
        except Exception as e:
            print(f"WandB初始化失败: {e}")
            return False
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        loss_dict = {}
        
        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # 前向传播
            outputs = self.model(images)
            
            # 计算损失
            losses = self.criterion(outputs, targets)
            
            # 加权总损失
            loss = sum(losses[k] * self.weight_dict.get(k, 1) for k in losses.keys())
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            for k, v in losses.items():
                if k not in loss_dict:
                    loss_dict[k] = 0
                loss_dict[k] += v.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        for k in loss_dict:
            loss_dict[k] /= len(self.train_loader)
        
        return avg_loss, loss_dict
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        loss_dict = {}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"验证 Epoch {self.current_epoch + 1}")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                losses = self.criterion(outputs, targets)
                
                # 加权总损失
                loss = sum(losses[k] * self.weight_dict.get(k, 1) for k in losses.keys())
                
                # 更新统计信息
                total_loss += loss.item()
                for k, v in losses.items():
                    if k not in loss_dict:
                        loss_dict[k] = 0
                    loss_dict[k] += v.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader)
        for k in loss_dict:
            loss_dict[k] /= len(self.val_loader)
        
        return avg_loss, loss_dict
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_map': self.best_map,
            'config': vars(self.config)
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.save_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到 {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_map = checkpoint.get('best_map', 0.0)
            print(f"从 epoch {self.current_epoch} 加载检查点")
            return True
        return False
    
    def train(self):
        """训练循环"""
        print(f"开始训练，设备: {self.device}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_loss_dict = self.train_epoch()
            
            # 验证
            val_loss, val_loss_dict = self.validate()
            
            # 更新学习率
            self.lr_scheduler.step()
            
            # 记录到WandB
            if WANDB_AVAILABLE and wandb.run is not None:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                
                for k, v in train_loss_dict.items():
                    log_dict[f'train/{k}'] = v
                
                for k, v in val_loss_dict.items():
                    log_dict[f'val/{k}'] = v
                
                wandb.log(log_dict)
            
            # 打印统计信息
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint()
            
            # 评估并保存最佳模型
            if (epoch + 1) % self.config.eval_interval == 0:
                # 这里可以添加mAP计算
                current_map = 0.0  # 暂时设为0，实际应该计算mAP
                
                if current_map > self.best_map:
                    self.best_map = current_map
                    self.save_checkpoint(is_best=True)
                    print(f"新的最佳mAP: {current_map:.4f}")
        
        # 训练完成
        print("训练完成!")
        self.save_checkpoint()
        
        if wandb.run is not None:
            wandb.finish()

def main():
    """主函数"""
    config = Config()
    trainer = Trainer(config)
    
    # 尝试加载现有检查点
    checkpoint_path = os.path.join(config.save_dir, 'latest.pth')
    if os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
