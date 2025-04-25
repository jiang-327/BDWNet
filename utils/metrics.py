import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from scipy.ndimage import binary_dilation, label
import cv2
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = {}
        self.sum = {}
        self.count = {}
        self.avg = {}
    def update(self, val_dict):
        for k, v in val_dict.items():
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
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    accuracy = correct / total
    return accuracy
def calculate_iou(pred, target):
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou
def calculate_dice(pred, target):
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice
def calculate_boundary_f1(pred, target, radius=2):
    if torch.is_tensor(pred):
        pred_cpu = pred.detach().cpu()
        if pred_cpu.shape[0] > 1:
            pred_cpu = pred_cpu[0]
        pred_np = pred_cpu.numpy()
    else:
        pred_np = pred
    if torch.is_tensor(target):
        target_cpu = target.detach().cpu()
        if target_cpu.shape[0] > 1:
            target_cpu = target_cpu[0]
        target_np = target_cpu.numpy()
    else:
        target_np = target
    pred_binary = (pred_np > 0.5).astype(np.uint8)
    target_binary = (target_np > 0.5).astype(np.uint8)
    try:
        pred_edges = cv2.Canny(pred_binary, 0, 1)
        target_edges = cv2.Canny(target_binary, 0, 1)
    except:
        from scipy.ndimage import binary_dilation, binary_erosion
        pred_edges = pred_binary - binary_erosion(pred_binary)
        target_edges = target_binary - binary_erosion(target_binary)
    pred_distances = cv2.distanceTransform(1 - pred_edges, cv2.DIST_L2, 3)
    target_distances = cv2.distanceTransform(1 - target_edges, cv2.DIST_L2, 3)
    pred_match = (pred_edges > 0) & (target_distances <= radius)
    target_match = (target_edges > 0) & (pred_distances <= radius)
    precision = np.sum(pred_match) / (np.sum(pred_edges > 0) + 1e-6)
    recall = np.sum(target_match) / (np.sum(target_edges > 0) + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    return torch.tensor(f1)
def calculate_boundary_accuracy(pred, target, tolerance=2):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    from scipy.ndimage import binary_dilation
    kernel = np.ones((3,3), np.uint8)
    pred_dilated = binary_dilation(pred, kernel)
    pred_boundary = pred_dilated - pred
    target_dilated = binary_dilation(target, kernel)
    target_boundary = target_dilated - target
    target_boundary_expanded = binary_dilation(target_boundary, iterations=tolerance)
    if pred_boundary.sum() == 0:
        return 0.0
    boundary_overlap = np.logical_and(pred_boundary, target_boundary_expanded).sum()
    return float(boundary_overlap) / float(pred_boundary.sum() + 1e-6)
def calculate_small_object_detection(pred, target, size_threshold=100):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    from scipy.ndimage import label
    target_labels, target_num = label(target)
    pred_labels, pred_num = label(pred)
    small_objects_detected = 0
    small_objects_total = 0
    for i in range(1, target_num + 1):
        obj_size = (target_labels == i).sum()
        if obj_size <= size_threshold:
            small_objects_total += 1
            overlap = np.logical_and(pred > 0.5, target_labels == i)
            if overlap.sum() > 0.5 * obj_size:
                small_objects_detected += 1
    if small_objects_total == 0:
        return 1.0
    return small_objects_detected / small_objects_total
def calculate_precision_recall(pred, target):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    precision, recall, thresholds = precision_recall_curve(target_flat, pred_flat)
    ap = average_precision_score(target_flat, pred_flat)
    pred_binary = (pred_flat > 0.5).astype(np.bool_)
    target_flat = target_flat.astype(np.bool_)
    tp = np.logical_and(pred_binary, target_flat).sum()
    fp = np.logical_and(pred_binary, np.logical_not(target_flat)).sum()
    fn = np.logical_and(np.logical_not(pred_binary), target_flat).sum()
    precision_at_threshold = tp / (tp + fp + 1e-6)
    recall_at_threshold = tp / (tp + fn + 1e-6)
    return precision_at_threshold, recall_at_threshold, ap
def calculate_precision(pred, target):
    smooth = 1e-6
    pred = (pred > 0.5).float().view(-1)
    target = (target > 0.5).float().view(-1)
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision
def calculate_recall(pred, target):
    smooth = 1e-6
    pred = (pred > 0.5).float().view(-1)
    target = (target > 0.5).float().view(-1)
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall
def calculate_f1_score(pred, target):
    precision = calculate_precision(pred, target)
    recall = calculate_recall(pred, target)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1
class MetricTracker:
    def __init__(self):
        self.meters = {
            'iou': AverageMeter(),
            'dice': AverageMeter(),
            'boundary_f1': AverageMeter()
        }
    def reset(self):
        for meter in self.meters.values():
            meter.reset()
    def update(self, pred, target):
        if not torch.is_tensor(pred):
            pred = torch.tensor(pred)
        if not torch.is_tensor(target):
            target = torch.tensor(target)
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        iou = calculate_iou(pred, target)
        dice = calculate_dice(pred, target)
        self.meters['iou'].update({'iou': iou})
        self.meters['dice'].update({'dice': dice})
        try:
            if len(pred.shape) == 4 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            if len(target.shape) == 4 and target.shape[1] == 1:
                target = target.squeeze(1)
            boundary_f1 = calculate_boundary_f1(pred[0], target[0])
            self.meters['boundary_f1'].update({'boundary_f1': boundary_f1})
        except Exception as e:
            pass
    def get_averages(self):
        return {
            'iou': self.meters['iou'].avg['iou'],
            'dice': self.meters['dice'].avg['dice'],
            'boundary_f1': self.meters['boundary_f1'].avg.get('boundary_f1', 0.0)
        }
def evaluate_model(model, val_loader, device):
    model.eval()
    metrics = MetricTracker()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            try:
                outputs = model(images)
                if isinstance(outputs, dict):
                    pred_masks = outputs['binary_map']
                else:
                    pred_masks = outputs
                if len(pred_masks.shape) == 4 and pred_masks.shape[1] == 1:
                    pred_flat = pred_masks.squeeze(1)
                else:
                    pred_flat = pred_masks
                if len(masks.shape) == 4 and masks.shape[1] == 1:
                    masks_flat = masks.squeeze(1)
                else:
                    masks_flat = masks
                pred_binary = (pred_flat > 0.5).float()
                metrics.update(pred_binary, masks_flat)
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    if not any(metrics.meters.values()):
        print("Warning: No valid evaluation metrics available!")
        return {k: 0.0 for k in metrics.meters.keys()}
    return metrics.get_averages()
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