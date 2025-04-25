import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        if not pred.is_floating_point():
            pred = pred.float()
        if not target.is_floating_point():
            target = target.float()
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        if torch.isnan(pred).any():
            pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
            print("Warning: NaN values in prediction, replaced with zeros")
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        if torch.isnan(dice) or torch.isinf(dice):
            print(f"Warning: Abnormal Dice calculation result: {dice}")
            return torch.tensor(0.0, device=pred.device, requires_grad=True) + pred.sum() * 0
        return 1.0 - dice
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    def forward(self, pred, target):
        if not pred.is_floating_point():
            pred = pred.float()
        if not target.is_floating_point():
            target = target.float()
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if torch.isnan(tversky) or torch.isinf(tversky):
            print(f"Warning: Abnormal Tversky calculation result: {tversky}")
            return torch.tensor(1.0, device=pred.device, requires_grad=True) + pred.sum() * 0
        return 1.0 - tversky
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, pred, target):
        if not pred.is_floating_point():
            pred = pred.float()
        if not target.is_floating_point():
            target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal_loss.mean()
class WoodDefectLoss(nn.Module):
    def __init__(self, edge_weight=0.5, threshold_weight=0.5, dice_weight=1.0,
                 tversky_weight=0.5, focal_weight=0.5, alpha=0.3, beta=0.7):
        super().__init__()
        self.edge_weight = edge_weight
        self.threshold_weight = threshold_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta)
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
    def edge_loss(self, pred, target):
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=torch.float32, device=pred.device).reshape(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=torch.float32, device=pred.device).reshape(1, 1, 3, 3)
        if not pred.is_floating_point():
            pred = pred.float()
        if not target.is_floating_point():
            target = target.float()
        batch_size = pred.size(0)
        pred_input = pred
        target_input = target
        if pred.ndim == 4 and pred.size(1) != 1:
            pred_input = pred.mean(dim=1, keepdim=True)
        if target.ndim == 4 and target.size(1) != 1:
            target_input = target.mean(dim=1, keepdim=True)
        if pred.ndim == 3:
            pred_input = pred.unsqueeze(1)
        if target.ndim == 3:
            target_input = target.unsqueeze(1)
        try:
            pred_grads = []
            target_grads = []
            for i in range(batch_size):
                pred_sample = pred_input[i:i+1]
                target_sample = target_input[i:i+1]
                pred_grad_x = F.conv2d(pred_sample, sobel_x_kernel, padding=1)
                target_grad_x = F.conv2d(target_sample, sobel_x_kernel, padding=1)
                pred_grad_y = F.conv2d(pred_sample, sobel_y_kernel, padding=1)
                target_grad_y = F.conv2d(target_sample, sobel_y_kernel, padding=1)
                pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
                target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
                pred_grads.append(pred_grad)
                target_grads.append(target_grad)
            pred_edges = torch.cat(pred_grads, dim=0)
            target_edges = torch.cat(target_grads, dim=0)
            edge_loss = F.mse_loss(pred_edges, target_edges)
            if torch.isnan(edge_loss):
                print("Warning: Edge loss calculation resulted in NaN, using zero loss instead")
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            return edge_loss
        except Exception as e:
            print(f"Error calculating edge loss: {e}")
            return torch.tensor(0.0, device=pred.device, requires_grad=True) + pred.sum() * 0
    def threshold_loss(self, pred_threshold, pred_prob, target):
        if not pred_prob.is_floating_point():
            pred_prob = pred_prob.float()
        if not target.is_floating_point():
            target = target.float()
        try:
            pred_prob_flat = pred_prob.view(-1)
            target_flat = target.view(-1)
            if torch.isnan(pred_prob_flat).any() or torch.isnan(target_flat).any():
                print("Warning: NaN values found in threshold loss calculation")
                pred_prob_flat = torch.where(torch.isnan(pred_prob_flat), torch.zeros_like(pred_prob_flat), pred_prob_flat)
                target_flat = torch.where(torch.isnan(target_flat), torch.zeros_like(target_flat), target_flat)
            optimal_threshold = target_flat.mean()
            if isinstance(pred_threshold, (int, float)):
                threshold_loss = F.mse_loss(torch.tensor(pred_threshold, device=target.device), optimal_threshold.detach())
            elif torch.numel(pred_threshold) == 1:
                threshold_loss = F.mse_loss(pred_threshold, optimal_threshold.detach())
            else:
                threshold_loss = F.mse_loss(pred_threshold.mean(), optimal_threshold.detach())
            return threshold_loss
        except Exception as e:
            print(f"Error in threshold loss calculation: {e}")
            dummy_loss = torch.tensor(0.0, device=target.device, requires_grad=True)
            if pred_prob.requires_grad:
                dummy_loss = dummy_loss + 0 * pred_prob.sum()
            return dummy_loss
    def forward(self, outputs, targets):
        if not isinstance(outputs, dict):
            if torch.is_tensor(outputs):
                outputs = {
                    'logits': outputs,
                    'prob_map': torch.sigmoid(outputs),
                    'binary_map': (torch.sigmoid(outputs) > 0.5).float()
                }
            else:
                raise ValueError("Output must be a tensor or a dictionary containing 'logits' key")
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        logits = outputs.get('logits')
        prob_map = outputs.get('prob_map')
        binary_map = outputs.get('binary_map')
        threshold = outputs.get('threshold', torch.tensor(0.5, device=targets.device))
        if logits is not None and logits.shape != targets.shape:
            targets = F.interpolate(targets.float(), size=logits.shape[2:], mode='nearest')
        bce, dice, tversky, focal, edge, threshold_loss = 0, 0, 0, 0, 0, 0
        if logits is not None:
            try:
                bce = self.bce_loss(logits, targets)
                if torch.isnan(bce):
                    print("Warning: BCE loss resulted in NaN, recalculating")
                    bce = F.binary_cross_entropy_with_logits(
                        torch.where(torch.isnan(logits), torch.zeros_like(logits), logits),
                        targets,
                        reduction='mean'
                    )
            except Exception as e:
                print(f"Error calculating BCE loss: {e}")
                bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        if prob_map is not None:
            try:
                dice = self.dice_loss(prob_map, targets)
            except Exception as e:
                print(f"Error calculating Dice loss: {e}")
                dice = torch.tensor(0.0, device=targets.device)
                if logits is not None and logits.requires_grad:
                    dice = dice + logits.sum() * 0
        if prob_map is not None:
            try:
                tversky = self.tversky_loss(prob_map, targets)
            except Exception as e:
                print(f"Error calculating Tversky loss: {e}")
                tversky = torch.tensor(0.0, device=targets.device)
                if logits is not None and logits.requires_grad:
                    tversky = tversky + logits.sum() * 0
        if logits is not None:
            try:
                focal = self.focal_loss(logits, targets)
            except Exception as e:
                print(f"Error calculating Focal loss: {e}")
                focal = torch.tensor(0.0, device=targets.device)
                if logits is not None and logits.requires_grad:
                    focal = focal + logits.sum() * 0
        if prob_map is not None:
            try:
                edge = self.edge_loss(prob_map, targets)
            except Exception as e:
                print(f"Error calculating edge loss: {e}")
                edge = torch.tensor(0.0, device=targets.device)
                if logits is not None and logits.requires_grad:
                    edge = edge + logits.sum() * 0
        if threshold is not None and prob_map is not None:
            try:
                threshold_loss = self.threshold_loss(threshold, prob_map, targets)
            except Exception as e:
                print(f"Error calculating threshold loss: {e}")
                threshold_loss = torch.tensor(0.0, device=targets.device)
                if logits is not None and logits.requires_grad:
                    threshold_loss = threshold_loss + logits.sum() * 0
        total_loss = bce + self.dice_weight * dice + self.tversky_weight * tversky + \
                    self.focal_weight * focal + self.edge_weight * edge + \
                    self.threshold_weight * threshold_loss
        if not torch.is_tensor(total_loss) or not total_loss.requires_grad:
            print("Warning: Total loss lacks gradient, adding a gradient-carrying term")
            if logits is not None and logits.requires_grad:
                total_loss = total_loss + 0 * logits.sum()
            elif prob_map is not None and prob_map.requires_grad:
                total_loss = total_loss + 0 * prob_map.sum()
        if torch.isnan(total_loss):
            print("Warning: Total loss resulted in NaN, using BCE loss instead")
            if logits is not None and not torch.isnan(logits).all():
                total_loss = F.binary_cross_entropy_with_logits(
                    torch.where(torch.isnan(logits), torch.zeros_like(logits), logits),
                    targets,
                    reduction='mean'
                )
            else:
                print("Severe error: Unable to calculate valid loss")
                total_loss = torch.tensor(0.1, device=targets.device, requires_grad=True)
        return {
            'bce': bce,
            'dice': dice,
            'tversky': tversky,
            'focal': focal,
            'edge': edge,
            'threshold': threshold_loss,
            'total': total_loss
        }
    def compute_per_sample_loss(self, outputs, target):
        logits = outputs['logits']
        binary_logits = outputs.get('binary_logits', logits)
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        target = target.float()
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none'
        ).mean(dim=(1,2,3))
        binary_bce = F.binary_cross_entropy_with_logits(
            binary_logits, target, reduction='none'
        ).mean(dim=(1,2,3))
        return bce_loss + binary_bce