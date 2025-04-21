#评价指标计算
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from scipy.ndimage import binary_dilation, label
import cv2

class AverageMeter:
    """
    跟踪多个平均值的类
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = {}
        self.sum = {}
        self.count = {}
        self.avg = {}
    
    def update(self, val_dict):
        """
        更新平均值
        Args:
            val_dict: 包含多个值的字典
        """
        for k, v in val_dict.items():
            # 确保v是标量
            if torch.is_tensor(v):
                v = v.detach().cpu().item()
            
            if k not in self.val:
                self.val[k] = v
                self.sum[k] = v
                self.count[k] = 1
                self.avg[k] = v
            else:
                self.val[k] = v
                self.sum[k] += v
                self.count[k] += 1
                self.avg[k] = self.sum[k] / self.count[k]

def calculate_pixel_accuracy(pred, target):
    """计算像素准确率"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    
    accuracy = correct / total
    return accuracy

def calculate_iou(pred, target):
    """计算IoU(交并比)"""
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

def calculate_dice(pred, target):
    """计算Dice系数"""
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def calculate_boundary_f1(pred, target, radius=2):
    """
    计算边界F1分数
    半径参数控制匹配的宽松程度
    """
    # 确保输入是CPU上的张量或NumPy数组
    if torch.is_tensor(pred):
        pred_cpu = pred.detach().cpu()
        if pred_cpu.shape[0] > 1:  # 如果是批次，只取第一个
            pred_cpu = pred_cpu[0]
        pred_np = pred_cpu.numpy()
    else:
        pred_np = pred
    
    if torch.is_tensor(target):
        target_cpu = target.detach().cpu()
        if target_cpu.shape[0] > 1:  # 如果是批次，只取第一个
            target_cpu = target_cpu[0]
        target_np = target_cpu.numpy()
    else:
        target_np = target
    
    # 转换为二值图像
    pred_binary = (pred_np > 0.5).astype(np.uint8)
    target_binary = (target_np > 0.5).astype(np.uint8)
    
    # 使用Canny检测器提取边缘
    try:
        pred_edges = cv2.Canny(pred_binary, 0, 1)
        target_edges = cv2.Canny(target_binary, 0, 1)
    except:
        # 如果Canny失败，使用简单的方法提取边缘
        from scipy.ndimage import binary_dilation, binary_erosion
        pred_edges = pred_binary - binary_erosion(pred_binary)
        target_edges = target_binary - binary_erosion(target_binary)
    
    # 创建距离图
    pred_distances = cv2.distanceTransform(1 - pred_edges, cv2.DIST_L2, 3)
    target_distances = cv2.distanceTransform(1 - target_edges, cv2.DIST_L2, 3)
    
    # 计算精确度和召回率
    pred_match = (pred_edges > 0) & (target_distances <= radius)
    target_match = (target_edges > 0) & (pred_distances <= radius)
    
    precision = np.sum(pred_match) / (np.sum(pred_edges > 0) + 1e-6)
    recall = np.sum(target_match) / (np.sum(target_edges > 0) + 1e-6)
    
    # 计算F1分数
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    return torch.tensor(f1)

def calculate_boundary_accuracy(pred, target, tolerance=2):
    """计算边界准确度"""
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # 确保输入是二值图像
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    
    # 使用形态学操作提取边界
    from scipy.ndimage import binary_dilation
    kernel = np.ones((3,3), np.uint8)
    
    # 提取预测边界
    pred_dilated = binary_dilation(pred, kernel)
    pred_boundary = pred_dilated - pred
    
    # 提取目标边界并扩展
    target_dilated = binary_dilation(target, kernel)
    target_boundary = target_dilated - target
    target_boundary_expanded = binary_dilation(target_boundary, iterations=tolerance)
    
    # 计算边界准确度
    if pred_boundary.sum() == 0:
        return 0.0
    
    boundary_overlap = np.logical_and(pred_boundary, target_boundary_expanded).sum()
    return float(boundary_overlap) / float(pred_boundary.sum() + 1e-6)

def calculate_small_object_detection(pred, target, size_threshold=100):
    """计算小目标检测率
    Args:
        pred: 预测掩码
        target: 真实掩码
        size_threshold: 小目标面积阈值
    """
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    from scipy.ndimage import label
    
    # 标记连通区域
    target_labels, target_num = label(target)
    pred_labels, pred_num = label(pred)
    
    # 统计小目标
    small_objects_detected = 0
    small_objects_total = 0
    
    for i in range(1, target_num + 1):
        obj_size = (target_labels == i).sum()
        if obj_size <= size_threshold:
            small_objects_total += 1
            # 检查是否被检测到
            overlap = np.logical_and(pred > 0.5, target_labels == i)
            if overlap.sum() > 0.5 * obj_size:
                small_objects_detected += 1
    
    if small_objects_total == 0:
        return 1.0  # 如果没有小目标，返回1
    return small_objects_detected / small_objects_total

def calculate_precision_recall(pred, target):
    """计算精确率和召回率
    Returns:
        precision: 精确率
        recall: 召回率
        ap: 平均精确率
    """
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    precision, recall, _ = precision_recall_curve(target_flat, pred_flat)
    ap = average_precision_score(target_flat, pred_flat)
    
    # 返回当前阈值(0.5)下的精确率和召回率
    pred_binary = (pred_flat > 0.5).astype(np.bool_)
    target_flat = target_flat.astype(np.bool_)
    
    tp = np.logical_and(pred_binary, target_flat).sum()
    fp = np.logical_and(pred_binary, np.logical_not(target_flat)).sum()
    fn = np.logical_and(np.logical_not(pred_binary), target_flat).sum()
    
    precision_at_threshold = tp / (tp + fp + 1e-6)
    recall_at_threshold = tp / (tp + fn + 1e-6)
    
    return precision_at_threshold, recall_at_threshold, ap

def calculate_precision(pred, target):
    """计算精确率"""
    smooth = 1e-6
    pred = (pred > 0.5).float().view(-1)
    target = (target > 0.5).float().view(-1)
    
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision

def calculate_recall(pred, target):
    """计算召回率"""
    smooth = 1e-6
    pred = (pred > 0.5).float().view(-1)
    target = (target > 0.5).float().view(-1)
    
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall

def calculate_f1_score(pred, target):
    """计算F1分数"""
    precision = calculate_precision(pred, target)
    recall = calculate_recall(pred, target)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1

class MetricTracker:
    """跟踪多个评估指标的类"""
    def __init__(self):
        self.meters = {
            'iou': AverageMeter(),
            'dice': AverageMeter(),
            'pixel_acc': AverageMeter()
        }
    
    def reset(self):
        """重置所有指标"""
        for meter in self.meters.values():
            meter.reset()
    
    def update(self, pred, target):
        """更新所有指标
        Args:
            pred: 预测结果
            target: 真实标签
        """
        # 确保输入是张量
        if not torch.is_tensor(pred):
            pred = torch.tensor(pred)
        if not torch.is_tensor(target):
            target = torch.tensor(target)
        
        # 二值化预测结果
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        
        # 计算各项指标
        iou = calculate_iou(pred, target)
        dice = calculate_dice(pred, target)
        pixel_acc = calculate_pixel_accuracy(pred, target)
        
        # 更新指标
        self.meters['iou'].update({'iou': iou})
        self.meters['dice'].update({'dice': dice})
        self.meters['pixel_acc'].update({'pixel_acc': pixel_acc})
    
    def get_averages(self):
        """获取所有指标的平均值"""
        return {
            'iou': self.meters['iou'].avg['iou'],
            'dice': self.meters['dice'].avg['dice'],
            'pixel_acc': self.meters['pixel_acc'].avg['pixel_acc']
        }

def evaluate_model(model, val_loader, device):
    """评估模型性能，返回各项指标的平均值"""
    model.eval()
    metrics = {
        'iou': [],
        'dice': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            try:
                # 前向传播
                outputs = model(images)
                
                # 确保掩码维度正确 - 进一步增强健壮性
                if isinstance(outputs, dict):
                    pred_masks = outputs['binary_map']
                else:
                    pred_masks = outputs
                    
                # 确保pred_masks和masks有相同的形状
                if len(pred_masks.shape) == 4 and pred_masks.shape[1] == 1:
                    pred_flat = pred_masks.squeeze(1)  # (B, H, W)
                else:
                    pred_flat = pred_masks
                    
                if len(masks.shape) == 4 and masks.shape[1] == 1:
                    masks_flat = masks.squeeze(1)  # (B, H, W)
                else:
                    masks_flat = masks
                
                # 阈值处理
                pred_binary = (pred_flat > 0.5).float()
                
                # 计算各项指标并确保是CPU上的标量
                batch_iou = calculate_iou(pred_binary, masks_flat)
                batch_dice = calculate_dice(pred_binary, masks_flat)
                batch_accuracy = calculate_pixel_accuracy(pred_binary, masks_flat)
                batch_precision = calculate_precision(pred_binary, masks_flat)
                batch_recall = calculate_recall(pred_binary, masks_flat)
                batch_f1 = calculate_f1_score(pred_binary, masks_flat)
                
                # 转换为CPU上的标量
                metrics['iou'].append(batch_iou.detach().cpu().item())
                metrics['dice'].append(batch_dice.detach().cpu().item())
                metrics['accuracy'].append(batch_accuracy.detach().cpu().item())
                metrics['precision'].append(batch_precision.detach().cpu().item())
                metrics['recall'].append(batch_recall.detach().cpu().item())
                metrics['f1'].append(batch_f1.detach().cpu().item())
            except Exception as e:
                print(f"评估时出错: {e}")
                # 出错时跳过这个batch
                continue
    
    # 检查是否有有效度量值
    if not any(metrics.values()):
        print("警告: 没有有效的评估指标!")
        return {k: 0.0 for k in metrics.keys()}
    
    # 计算平均值并返回
    return {k: np.mean(v) for k, v in metrics.items() if v}

# 确保所有类和函数都在文件中定义
__all__ = [
    'AverageMeter',
    'calculate_pixel_accuracy',
    'calculate_iou',
    'calculate_dice',
    'calculate_boundary_f1',
    'calculate_boundary_accuracy',
    'calculate_small_object_detection',
    'calculate_precision_recall',
    'calculate_precision',
    'calculate_recall',
    'calculate_f1_score',
    'MetricTracker',
    'evaluate_model'
]
