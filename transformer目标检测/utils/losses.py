import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

class HungarianMatcher(nn.Module):
    """匈牙利匹配器，用于匹配预测和真实标注"""
    
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        
        # 展平批次维度
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # 为每个批次样本准备目标
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # 计算分类成本
        cost_class = -out_prob[:, tgt_ids]
        
        # 计算L1成本
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # 计算GIoU成本
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # 最终成本矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    """集合预测损失函数"""
    
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """分类损失"""
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """边界框损失"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        
        # GIoU损失
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # 排列索引以匹配预测
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        # 排列索引以匹配目标
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def forward(self, outputs, targets):
        # 匹配预测和目标
        indices = self.matcher(outputs, targets)
        
        # 计算匹配的数量
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        losses = {}
        
        # 计算分类损失
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        
        # 计算边界框损失
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        
        return losses

def box_cxcywh_to_xyxy(x):
    """将边界框从 (center_x, center_y, width, height) 转换为 (x_min, y_min, x_max, y_max)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    """将边界框从 (x_min, y_min, x_max, y_max) 转换为 (center_x, center_y, width, height)"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_area(boxes):
    """计算边界框面积"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """计算IoU"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """计算广义IoU"""
    # 退化情况给出精确的IoU
    iou, union = box_iou(boxes1, boxes2)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    
    return iou - (area - union) / area

def build_criterion(config):
    """构建损失函数"""
    matcher = HungarianMatcher(
        cost_class=config.get('cost_class', 1),
        cost_bbox=config.get('cost_bbox', 5),
        cost_giou=config.get('cost_giou', 2)
    )
    
    weight_dict = {
        'loss_ce': config.get('weight_class', 1),
        'loss_bbox': config.get('weight_bbox', 5),
        'loss_giou': config.get('weight_giou', 2)
    }
    
    criterion = SetCriterion(
        num_classes=config.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=config.get('eos_coef', 0.1)
    )
    
    return criterion, weight_dict
