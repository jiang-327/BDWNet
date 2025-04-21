#损失函数定义
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
        """
        计算Dice损失
        Args:
            pred: 预测概率图，形状为[B, 1, H, W]
            target: 目标掩码，形状为[B, 1, H, W]
        Returns:
            Dice损失值
        """
        # 确保输入是浮点型
        if not pred.is_floating_point():
            pred = pred.float()
        if not target.is_floating_point():
            target = target.float()
        
        # 处理空批次
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # 检查是否有NaN值并处理
        if torch.isnan(pred).any():
            pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
            print("警告：预测中存在NaN值，已替换为零")
        
        # 展平预测和目标
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 检查值是否合法
        if torch.isnan(dice) or torch.isinf(dice):
            print(f"警告：Dice计算结果异常: {dice}")
            return torch.tensor(0.0, device=pred.device, requires_grad=True) + pred.sum() * 0
        
        # 返回Dice损失（1 - Dice系数）
        return 1.0 - dice

class TverskyLoss(nn.Module):
    """
    Tversky损失，对FP和FN的权重可调，更适合处理小目标和不平衡数据
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # FP权重
        self.beta = beta    # FN权重
        self.smooth = smooth
        
    def forward(self, pred, target):
        # 确保输入是浮点型
        if not pred.is_floating_point():
            pred = pred.float()
        if not target.is_floating_point():
            target = target.float()
        
        # 展平预测和目标
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算TP, FP, FN
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        # 计算Tversky指标
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # 检查值是否合法
        if torch.isnan(tversky) or torch.isinf(tversky):
            print(f"警告：Tversky计算结果异常: {tversky}")
            return torch.tensor(1.0, device=pred.device, requires_grad=True) + pred.sum() * 0
            
        # 返回Tversky损失
        return 1.0 - tversky

class FocalLoss(nn.Module):
    """Focal Loss，增强对难分样本的关注"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma  # 聚焦参数
        self.alpha = alpha  # 类别平衡参数
        
    def forward(self, pred, target):
        # 确保输入是浮点型
        if not pred.is_floating_point():
            pred = pred.float()
        if not target.is_floating_point():
            target = target.float()
            
        # 使用BCEWithLogitsLoss的零件计算BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算概率
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)
        
        # 应用Focal项
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce
        
        return focal_loss.mean()

class WoodDefectLoss(nn.Module):
    def __init__(self, edge_weight=0.5, threshold_weight=0.5, dice_weight=1.0, 
                 tversky_weight=0.5, focal_weight=0.5, alpha=0.3, beta=0.7):
        """
        木材缺陷分割的综合损失函数
        Args:
            edge_weight: 边缘损失的权重
            threshold_weight: 阈值损失的权重
            dice_weight: Dice损失的权重
            tversky_weight: Tversky损失的权重
            focal_weight: Focal损失的权重
            alpha: Tversky损失中的FP权重
            beta: Tversky损失中的FN权重
        """
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
        """计算边缘损失，使用PyTorch实现代替OpenCV"""
        # 创建Sobel核
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=pred.device).reshape(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=pred.device).reshape(1, 1, 3, 3)

        # 确保输入是浮点型
        if not pred.is_floating_point():
            pred = pred.float()
        if not target.is_floating_point():
            target = target.float()
        
        # 如有必要，调整输入维度
        batch_size = pred.size(0)
        pred_input = pred
        target_input = target
        
        if pred.ndim == 4 and pred.size(1) != 1:
            pred_input = pred.mean(dim=1, keepdim=True)  # 多通道转为单通道
        if target.ndim == 4 and target.size(1) != 1:
            target_input = target.mean(dim=1, keepdim=True)  # 多通道转为单通道
        
        if pred.ndim == 3:
            pred_input = pred.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
        if target.ndim == 3:
            target_input = target.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
            
        try:
            # 求边缘梯度
            # 对每个图像应用卷积
            pred_grads = []
            target_grads = []
            
            for i in range(batch_size):
                pred_sample = pred_input[i:i+1]  # 保留维度 [1,1,H,W]
                target_sample = target_input[i:i+1]
                
                # 计算x方向梯度
                pred_grad_x = F.conv2d(pred_sample, sobel_x_kernel, padding=1)
                target_grad_x = F.conv2d(target_sample, sobel_x_kernel, padding=1)
                
                # 计算y方向梯度
                pred_grad_y = F.conv2d(pred_sample, sobel_y_kernel, padding=1)
                target_grad_y = F.conv2d(target_sample, sobel_y_kernel, padding=1)
                
                # 计算梯度强度
                pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
                target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
                
                pred_grads.append(pred_grad)
                target_grads.append(target_grad)
                
            # 合并批次
            pred_edges = torch.cat(pred_grads, dim=0)
            target_edges = torch.cat(target_grads, dim=0)
            
            # 计算MSE损失
            edge_loss = F.mse_loss(pred_edges, target_edges)
            
            # 检查是否有NaN值
            if torch.isnan(edge_loss):
                print("警告：边缘损失计算为NaN，使用零损失代替")
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
                
            return edge_loss
            
        except Exception as e:
            print(f"边缘损失计算出错: {e}")
            # 返回虚拟损失，确保梯度流动
            return torch.tensor(0.0, device=pred.device, requires_grad=True) + pred.sum() * 0
    
    def threshold_loss(self, pred_threshold, pred_prob, target):
        """计算阈值损失"""
        # 确保输入是浮点型
        if not pred_prob.is_floating_point():
            pred_prob = pred_prob.float()
        if not target.is_floating_point():
            target = target.float()
        
        # 数据验证
        try:
            # 浮点化和展平
            pred_prob_flat = pred_prob.view(-1)
            target_flat = target.view(-1)
            
            # 检查是否有异常值
            if torch.isnan(pred_prob_flat).any() or torch.isnan(target_flat).any():
                print("警告: 阈值损失计算中发现NaN值")
                # 替换NaN值
                pred_prob_flat = torch.where(torch.isnan(pred_prob_flat), torch.zeros_like(pred_prob_flat), pred_prob_flat)
                target_flat = torch.where(torch.isnan(target_flat), torch.zeros_like(target_flat), target_flat)
            
            # 计算最优阈值 - 使用真实样本的均值作为理想阈值
            optimal_threshold = target_flat.mean()
            
            # 阈值可能是标量、向量或张量，需要适当处理
            if isinstance(pred_threshold, (int, float)):
                # 如果是Python标量，转换为张量
                threshold_loss = F.mse_loss(torch.tensor(pred_threshold, device=target.device), optimal_threshold.detach())
            elif torch.numel(pred_threshold) == 1:
                # 如果是标量张量
                threshold_loss = F.mse_loss(pred_threshold, optimal_threshold.detach())
            else:
                # 如果是张量，计算平均值与最优阈值的MSE
                threshold_loss = F.mse_loss(pred_threshold.mean(), optimal_threshold.detach())
            
            return threshold_loss
        except Exception as e:
            print(f"阈值损失计算出错: {e}")
            # 返回一个虚拟损失，确保梯度流动
            dummy_loss = torch.tensor(0.0, device=target.device, requires_grad=True)
            if pred_prob.requires_grad:
                dummy_loss = dummy_loss + 0 * pred_prob.sum()
            return dummy_loss
    
    def forward(self, outputs, targets):
        """
        计算总损失
        Args:
            outputs: 模型输出字典，包含logits, prob_map, binary_map和threshold
            targets: 目标掩码
        Returns:
            损失字典，包含各个损失项和总损失
        """
        # 输入验证和标准化
        if not isinstance(outputs, dict):
            if torch.is_tensor(outputs):
                outputs = {
                    'logits': outputs,
                    'prob_map': torch.sigmoid(outputs),
                    'binary_map': (torch.sigmoid(outputs) > 0.5).float()
                }
            else:
                raise ValueError("输出必须是张量或包含'logits'键的字典")
        
        # 确保targets是适当的形状
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)  # 添加通道维度 [B,H,W] -> [B,1,H,W]
        
        # 提取需要的输出
        logits = outputs.get('logits')
        prob_map = outputs.get('prob_map')
        binary_map = outputs.get('binary_map')
        threshold = outputs.get('threshold', torch.tensor(0.5, device=targets.device))
        
        # 处理可能存在的维度不匹配
        if logits is not None and logits.shape != targets.shape:
            # 调整targets的形状以匹配logits
            targets = F.interpolate(targets.float(), size=logits.shape[2:], mode='nearest')
        
        # 初始化损失值
        bce, dice, tversky, focal, edge, threshold_loss = 0, 0, 0, 0, 0, 0
        
        # 计算BCE损失
        if logits is not None:
            try:
                bce = self.bce_loss(logits, targets)
                # 检查BCE损失是否为NaN
                if torch.isnan(bce):
                    print("警告: BCE损失为NaN，重新计算")
                    bce = F.binary_cross_entropy_with_logits(
                        torch.where(torch.isnan(logits), torch.zeros_like(logits), logits),
                        targets,
                        reduction='mean'
                    )
            except Exception as e:
                print(f"计算BCE损失时出错: {e}")
                # 使用更简单的方法计算损失
                bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        
        # 计算Dice损失
        if prob_map is not None:
            try:
                dice = self.dice_loss(prob_map, targets)
            except Exception as e:
                print(f"计算Dice损失时出错: {e}")
                # 添加一个虚拟损失
                dice = torch.tensor(0.0, device=targets.device)
                if logits is not None and logits.requires_grad:
                    dice = dice + logits.sum() * 0
        
        # 计算Tversky损失
        if prob_map is not None:
            try:
                tversky = self.tversky_loss(prob_map, targets)
            except Exception as e:
                print(f"计算Tversky损失时出错: {e}")
                tversky = torch.tensor(0.0, device=targets.device)
                if logits is not None and logits.requires_grad:
                    tversky = tversky + logits.sum() * 0
        
        # 计算Focal损失
        if logits is not None:
            try:
                focal = self.focal_loss(logits, targets)
            except Exception as e:
                print(f"计算Focal损失时出错: {e}")
                focal = torch.tensor(0.0, device=targets.device)
                if logits is not None and logits.requires_grad:
                    focal = focal + logits.sum() * 0
        
        # 计算边缘损失
        if prob_map is not None:
            try:
                edge = self.edge_loss(prob_map, targets)
            except Exception as e:
                print(f"计算边缘损失时出错: {e}")
                edge = torch.tensor(0.0, device=targets.device)
                if logits is not None and logits.requires_grad:
                    edge = edge + logits.sum() * 0
        
        # 计算阈值损失
        if threshold is not None and prob_map is not None:
            try:
                threshold_loss = self.threshold_loss(threshold, prob_map, targets)
            except Exception as e:
                print(f"计算阈值损失时出错: {e}")
                threshold_loss = torch.tensor(0.0, device=targets.device)
                if logits is not None and logits.requires_grad:
                    threshold_loss = threshold_loss + logits.sum() * 0
        
        # 计算总损失
        total_loss = bce + self.dice_weight * dice + self.tversky_weight * tversky + \
                    self.focal_weight * focal + self.edge_weight * edge + \
                    self.threshold_weight * threshold_loss
        
        # 确保总损失有梯度
        if not torch.is_tensor(total_loss) or not total_loss.requires_grad:
            print("警告: 总损失没有梯度，添加一个带梯度的项")
            if logits is not None and logits.requires_grad:
                total_loss = total_loss + 0 * logits.sum()
            elif prob_map is not None and prob_map.requires_grad:
                total_loss = total_loss + 0 * prob_map.sum()
                
        # 检查总损失是否为NaN
        if torch.isnan(total_loss):
            print("警告: 总损失为NaN，使用BCE损失替代")
            if logits is not None and not torch.isnan(logits).all():
                total_loss = F.binary_cross_entropy_with_logits(
                    torch.where(torch.isnan(logits), torch.zeros_like(logits), logits),
                    targets,
                    reduction='mean'
                )
            else:
                print("严重错误: 无法计算有效损失")
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
        """计算每个样本的损失"""
        logits = outputs['logits']
        binary_logits = outputs.get('binary_logits', logits)
        
        # 确保维度正确
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        target = target.float()
        
        # 计算每个样本的BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none'
        ).mean(dim=(1,2,3))
        
        # 计算每个样本的二值化损失
        binary_bce = F.binary_cross_entropy_with_logits(
            binary_logits, target, reduction='none'
        ).mean(dim=(1,2,3))
        
        return bce_loss + binary_bce
