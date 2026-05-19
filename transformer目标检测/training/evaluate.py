import os
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.transformer_detector import build_model
from utils.data_utils import get_data_loaders
from configs.default_config import Config

class Evaluator:
    """评估器类"""
    
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.device = config.device
        
        # 加载模型
        self.model = build_model(config).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从 {checkpoint_path} 加载模型")
        else:
            print("使用随机初始化的模型")
        
        # 加载数据
        _, self.val_loader = get_data_loaders(config)
        
        # 类别名称
        self.class_names = self._get_class_names()
    
    def _get_class_names(self):
        """获取类别名称"""
        # COCO数据集类别
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        return coco_classes
    
    def evaluate_coco(self):
        """使用COCO评估指标评估模型"""
        self.model.eval()
        
        # 准备COCO格式的结果
        results = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.val_loader, desc="评估")):
                images = images.to(self.device)
                outputs = self.model(images)
                
                # 处理每个图像的输出
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    image_id = target['image_id'].item()
                    
                    # 获取预测结果
                    pred_logits = output['pred_logits'].softmax(-1)[0]  # [num_queries, num_classes+1]
                    pred_boxes = output['pred_boxes'][0]  # [num_queries, 4]
                    
                    # 过滤背景类别并应用阈值
                    scores = pred_logits[:, :-1].max(dim=1)[0]  # 排除背景
                    keep = scores > 0.05  # 置信度阈值
                    
                    if keep.sum() == 0:
                        continue
                    
                    scores = scores[keep]
                    labels = pred_logits[:, :-1][keep].argmax(dim=1)
                    boxes = pred_boxes[keep]
                    
                    # 转换为COCO格式 [x, y, width, height]
                    boxes_coco = boxes.clone()
                    boxes_coco[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
                    boxes_coco[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
                    
                    # 添加到结果
                    for score, label, box in zip(scores, labels, boxes_coco):
                        results.append({
                            'image_id': image_id,
                            'category_id': label.item() + 1,  # COCO类别从1开始
                            'bbox': box.tolist(),
                            'score': score.item()
                        })
        
        # 保存结果
        results_file = os.path.join(self.config.log_dir, 'detection_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        # COCO评估
        annotation_file = os.path.join(self.config.data_dir, 'annotations/instances_val2017.json')
        if os.path.exists(annotation_file):
            coco_gt = COCO(annotation_file)
            coco_dt = coco_gt.loadRes(results_file)
            
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # 获取评估结果
            stats = coco_eval.stats
            metrics = {
                'AP': stats[0],
                'AP50': stats[1],
                'AP75': stats[2],
                'AP_small': stats[3],
                'AP_medium': stats[4],
                'AP_large': stats[5],
                'AR1': stats[6],
                'AR10': stats[7],
                'AR100': stats[8],
                'AR_small': stats[9],
                'AR_medium': stats[10],
                'AR_large': stats[11]
            }
            
            return metrics
        else:
            print(f"标注文件不存在: {annotation_file}")
            return {}
    
    def visualize_predictions(self, num_images=5):
        """可视化预测结果"""
        self.model.eval()
        
        fig, axes = plt.subplots(num_images, 2, figsize=(15, 5*num_images))
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(self.val_loader):
                if i >= num_images:
                    break
                
                images = images.to(self.device)
                outputs = self.model(images)
                
                # 处理第一个图像
                image = images[0].cpu().permute(1, 2, 0).numpy()
                image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                image = image.astype(np.uint8)
                
                # 真实边界框
                target = targets[0]
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                # 预测边界框
                output = outputs[0]
                pred_logits = output['pred_logits'].softmax(-1)[0]
                pred_boxes = output['pred_boxes'][0].cpu().numpy()
                
                # 过滤预测
                scores = pred_logits[:, :-1].max(dim=1)[0]
                keep = scores > 0.5
                pred_scores = scores[keep].cpu().numpy()
                pred_labels = pred_logits[:, :-1][keep].argmax(dim=1).cpu().numpy()
                pred_boxes = pred_boxes[keep]
                
                # 绘制图像
                ax1, ax2 = axes[i]
                
                # 真实标注
                ax1.imshow(image)
                for box, label in zip(gt_boxes, gt_labels):
                    x1, y1, x2, y2 = box
                    ax1.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              fill=False, edgecolor='green', linewidth=2))
                    ax1.text(x1, y1, self.class_names[label], 
                            bbox=dict(boxstyle="round", fc="green", alpha=0.5))
                ax1.set_title('真实标注')
                ax1.axis('off')
                
                # 预测结果
                ax2.imshow(image)
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    x1, y1, x2, y2 = box
                    ax2.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              fill=False, edgecolor='red', linewidth=2))
                    ax2.text(x1, y1, f'{self.class_names[label]}: {score:.2f}', 
                            bbox=dict(boxstyle="round", fc="red", alpha=0.5))
                ax2.set_title('预测结果')
                ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'predictions_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, metrics):
        """生成评估报告"""
        report = {
            'experiment_name': self.config.experiment_name,
            'model_config': {
                'backbone': self.config.backbone,
                'hidden_dim': self.config.hidden_dim,
                'num_queries': self.config.num_queries,
                'num_encoder_layers': self.config.num_encoder_layers,
                'num_decoder_layers': self.config.num_decoder_layers
            },
            'evaluation_metrics': metrics,
            'dataset_info': {
                'dataset': self.config.dataset_name,
                'num_classes': self.config.num_classes,
                'val_samples': len(self.val_loader.dataset)
            }
        }
        
        # 保存报告
        report_file = os.path.join(self.config.log_dir, 'evaluation_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 生成可视化报告
        self._generate_visual_report(metrics)
        
        return report
    
    def _generate_visual_report(self, metrics):
        """生成可视化报告"""
        if not metrics:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AP指标
        ap_metrics = ['AP', 'AP50', 'AP75']
        ap_values = [metrics.get(m, 0) for m in ap_metrics]
        
        axes[0, 0].bar(ap_metrics, ap_values, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('平均精度指标')
        axes[0, 0].set_ylabel('精度')
        
        # 不同尺度的AP
        scale_metrics = ['AP_small', 'AP_medium', 'AP_large']
        scale_values = [metrics.get(m, 0) for m in scale_metrics]
        
        axes[0, 1].bar(scale_metrics, scale_values, color=['gold', 'orange', 'red'])
        axes[0, 1].set_title('不同尺度目标的AP')
        axes[0, 1].set_ylabel('精度')
        
        # AR指标
        ar_metrics = ['AR1', 'AR10', 'AR100']
        ar_values = [metrics.get(m, 0) for m in ar_metrics]
        
        axes[1, 0].bar(ar_metrics, ar_values, color=['lightblue', 'lightgreen', 'pink'])
        axes[1, 0].set_title('平均召回率指标')
        axes[1, 0].set_ylabel('召回率')
        
        # 不同尺度的AR
        scale_ar_metrics = ['AR_small', 'AR_medium', 'AR_large']
        scale_ar_values = [metrics.get(m, 0) for m in scale_ar_metrics]
        
        axes[1, 1].bar(scale_ar_metrics, scale_ar_values, color=['yellow', 'orange', 'brown'])
        axes[1, 1].set_title('不同尺度目标的AR')
        axes[1, 1].set_ylabel('召回率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'metrics_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    config = Config()
    
    # 创建评估器
    checkpoint_path = os.path.join(config.save_dir, 'best.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config.save_dir, 'latest.pth')
    
    evaluator = Evaluator(config, checkpoint_path)
    
    # 执行评估
    print("开始评估...")
    metrics = evaluator.evaluate_coco()
    
    # 生成报告
    report = evaluator.generate_report(metrics)
    
    # 可视化预测
    print("生成预测可视化...")
    evaluator.visualize_predictions(num_images=3)
    
    # 打印结果
    print("\n评估结果:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print(f"\n详细报告已保存到: {config.log_dir}")

if __name__ == "__main__":
    main()
