import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from models.transformer_detector import build_model
from configs.default_config import Config

class ObjectDetectionDemo:
    """目标检测演示类"""
    
    def __init__(self, checkpoint_path=None):
        self.config = Config()
        self.device = self.config.device
        
        # 加载模型
        self.model = build_model(self.config).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从 {checkpoint_path} 加载模型")
        else:
            print("使用随机初始化的模型")
        
        # COCO类别名称
        self.class_names = [
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
        
        self.model.eval()
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 保存原始尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 调整尺寸
        image_resized = cv2.resize(image, self.config.image_size)
        
        # 归一化
        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device), (orig_h, orig_w), image
    
    def detect_objects(self, image_tensor):
        """检测目标"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        return outputs
    
    def postprocess_detections(self, outputs, orig_size, confidence_threshold=0.5):
        """后处理检测结果"""
        pred_logits = outputs['pred_logits'][0].softmax(-1)  # [num_queries, num_classes+1]
        pred_boxes = outputs['pred_boxes'][0]  # [num_queries, 4]
        
        # 过滤背景类别
        scores = pred_logits[:, :-1].max(dim=1)[0]  # 排除背景
        labels = pred_logits[:, :-1].argmax(dim=1)
        
        # 应用置信度阈值
        keep = scores > confidence_threshold
        
        if keep.sum() == 0:
            return [], [], []
        
        scores = scores[keep].cpu().numpy()
        labels = labels[keep].cpu().numpy()
        boxes = pred_boxes[keep].cpu().numpy()
        
        # 调整边界框到原始图像尺寸
        orig_h, orig_w = orig_size
        scale_h = orig_h / self.config.image_size[0]
        scale_w = orig_w / self.config.image_size[1]
        
        boxes[:, 0] *= scale_w  # x1
        boxes[:, 1] *= scale_h  # y1
        boxes[:, 2] *= scale_w  # x2
        boxes[:, 3] *= scale_h  # y2
        
        # 确保边界框在图像范围内
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
        
        return boxes, labels, scores
    
    def visualize_detections(self, image, boxes, labels, scores, save_path=None):
        """可视化检测结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 原始图像
        ax1.imshow(image)
        ax1.set_title('原始图像')
        ax1.axis('off')
        
        # 检测结果
        ax2.imshow(image)
        
        # 为不同类别生成颜色
        colors = plt.cm.get_cmap('tab20', len(self.class_names))
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            class_name = self.class_names[label]
            color = colors(label)
            
            # 绘制边界框
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor=color, linewidth=2)
            ax2.add_patch(rect)
            
            # 添加标签
            ax2.text(x1, y1-5, f'{class_name}: {score:.2f}', 
                    bbox=dict(boxstyle="round", fc=color, alpha=0.8),
                    fontsize=8, color='white')
        
        ax2.set_title('目标检测结果')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果保存到: {save_path}")
        
        plt.show()
    
    def process_image(self, image_path, confidence_threshold=0.5, save_path=None):
        """处理单张图像"""
        print(f"处理图像: {image_path}")
        
        # 预处理
        image_tensor, orig_size, original_image = self.preprocess_image(image_path)
        
        # 检测
        outputs = self.detect_objects(image_tensor)
        
        # 后处理
        boxes, labels, scores = self.postprocess_detections(
            outputs, orig_size, confidence_threshold
        )
        
        # 打印结果
        print(f"检测到 {len(boxes)} 个目标:")
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            class_name = self.class_names[label]
            print(f"  {i+1}. {class_name}: {score:.3f} - 位置: {box.astype(int)}")
        
        # 可视化
        self.visualize_detections(original_image, boxes, labels, scores, save_path)
        
        return boxes, labels, scores
    
    def process_video(self, video_path, confidence_threshold=0.5, output_path=None):
        """处理视频"""
        print(f"处理视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建输出视频
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print("开始处理视频...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 预处理
            image_tensor, orig_size, _ = self.preprocess_image_from_array(frame_rgb)
            
            # 检测
            outputs = self.detect_objects(image_tensor)
            
            # 后处理
            boxes, labels, scores = self.postprocess_detections(
                outputs, (height, width), confidence_threshold
            )
            
            # 在帧上绘制结果
            frame_with_detections = self.draw_detections_on_frame(frame, boxes, labels, scores)
            
            # 写入输出视频
            if output_path:
                out.write(frame_with_detections)
            
            # 显示进度
            if frame_count % 30 == 0:
                print(f"已处理 {frame_count} 帧")
        
        # 释放资源
        cap.release()
        if output_path:
            out.release()
            print(f"视频处理完成，结果保存到: {output_path}")
    
    def preprocess_image_from_array(self, image_array):
        """从数组预处理图像"""
        # 保存原始尺寸
        orig_h, orig_w = image_array.shape[:2]
        
        # 调整尺寸
        image_resized = cv2.resize(image_array, self.config.image_size)
        
        # 归一化
        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device), (orig_h, orig_w), image_array
    
    def draw_detections_on_frame(self, frame, boxes, labels, scores):
        """在帧上绘制检测结果"""
        frame_with_detections = frame.copy()
        
        # 为不同类别生成颜色
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                 (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            class_name = self.class_names[label]
            color = colors[label % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签
            label_text = f'{class_name}: {score:.2f}'
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(frame_with_detections, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            
            cv2.putText(frame_with_detections, label_text,
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame_with_detections

def main():
    parser = argparse.ArgumentParser(description="目标检测演示")
    parser.add_argument("--input", type=str, required=True, help="输入图像或视频路径")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pth", 
                       help="模型检查点路径")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="置信度阈值")
    parser.add_argument("--output", type=str, help="输出路径")
    
    args = parser.parse_args()
    
    # 创建演示器
    demo = ObjectDetectionDemo(args.checkpoint)
    
    # 检查输入类型
    if args.input.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 图像处理
        demo.process_image(args.input, args.confidence, args.output)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        # 视频处理
        demo.process_video(args.input, args.confidence, args.output)
    else:
        print("不支持的输入格式，请提供图像(.png, .jpg, .jpeg)或视频(.mp4, .avi, .mov)")

if __name__ == "__main__":
    main()
