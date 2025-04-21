import os
import torch
import cv2
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

try:
    from models.network import WoodDefectDB
    from utils.dataset import WoodDefectDataset
    from utils.metrics import calculate_iou, calculate_dice, calculate_precision_recall, calculate_boundary_accuracy
    from configs.config import Config

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"导入依赖时出错: {e}")
    DEPENDENCIES_AVAILABLE = False

def check_dependencies():
    """检查所有必要的依赖是否可用"""
    if not DEPENDENCIES_AVAILABLE:
        return False
    
    # 检查CUDA可用性
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
    
    return True

def calculate_pixel_accuracy(pred, target):
    """
    计算像素准确率 (PA)
    Args:
        pred: 预测掩码 [B, H, W]
        target: 真实掩码 [B, H, W]
    Returns:
        像素准确率
    """
    correct = (pred == target).float().sum()
    total = torch.ones_like(target).sum()
    return correct / total

def calculate_class_pixel_accuracy(pred, target):
    """
    计算各类别的像素准确率
    Args:
        pred: 预测掩码 [B, H, W]，值为0或1
        target: 真实掩码 [B, H, W]，值为0或1
    Returns:
        各类别的像素准确率 [fg_acc, bg_acc]
    """
    # 前景(缺陷)像素准确率
    if target.sum() > 0:  # 避免分母为0
        fg_correct = ((pred == 1) & (target == 1)).float().sum()
        fg_total = (target == 1).float().sum()
        fg_acc = (fg_correct / fg_total).item()
    else:
        fg_acc = 1.0 if pred.sum() == 0 else 0.0
    
    # 背景像素准确率
    bg_correct = ((pred == 0) & (target == 0)).float().sum()
    bg_total = (target == 0).float().sum()
    bg_acc = (bg_correct / bg_total).item()
    
    return [fg_acc, bg_acc]

def calculate_mean_iou(pred, target, num_classes=2):
    """
    计算多类别的平均IoU (mIoU)
    Args:
        pred: 预测掩码 [B, H, W]，值为0或1
        target: 真实掩码 [B, H, W]，值为0或1
        num_classes: 类别数量，默认为2（前景和背景）
    Returns:
        mIoU: 各类别IoU的平均值
    """
    ious = []
    # 对于二分类问题（背景和前景）
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        
        if union > 0:
            iou = (intersection / union).item()
        else:
            iou = 1.0  # 如果没有该类别，则IoU为1
        
        ious.append(iou)
    
    return np.mean(ious)

def visualize_results(images, masks, predictions, prob_maps, edge_maps=None, save_path=None):
    """
    可视化测试结果
    Args:
        images: 原始图像 [B, C, H, W]
        masks: 真实掩码 [B, H, W]
        predictions: 预测掩码 [B, H, W]
        prob_maps: 概率图 [B, H, W] 或 [B, 1, H, W]
        edge_maps: 边缘图 [B, H, W] 或 [B, 1, H, W]
        save_path: 保存路径
    """
    try:
        num_samples = min(3, len(images))  # 最多显示3张图
        
        # 计算需要的列数（原始图像、真实掩码、预测掩码、概率图、边缘图(可选)、分割效果图）
        num_cols = 5 if edge_maps is None else 6
        
        # 处理单样本情况
        if num_samples == 1:
            # 创建具有单行多列的图
            fig, axes = plt.subplots(1, num_cols, figsize=(20, 5))
        else:
            fig, axes = plt.subplots(num_samples, num_cols, figsize=(20, 5*num_samples))
        
        for i in range(num_samples):
            # 获取当前样本的轴对象
            if num_samples == 1:
                current_axes = axes
            else:
                current_axes = axes[i]
            
            # 显示原始图像
            img = images[i].cpu()
            
            # 确保图像有三个通道
            if img.shape[0] == 1:  # 如果是单通道图像
                # 复制到三个通道
                img = img.repeat(3, 1, 1)
            elif img.shape[0] > 3:  # 如果通道数超过3
                img = img[:3]  # 只取前三个通道
                
            img = img.permute(1, 2, 0).numpy()
            # 归一化到[0,1]
            if img.max() > 0:
                img = (img - img.min()) / (img.max() - img.min())
            
            current_axes[0].imshow(img)
            current_axes[0].set_title('原始图像')
            current_axes[0].axis('off')
            
            # 显示真实掩码
            mask = masks[i].cpu().numpy()
            current_axes[1].imshow(mask, cmap='gray')
            current_axes[1].set_title('真实掩码')
            current_axes[1].axis('off')
            
            # 显示预测掩码
            pred = predictions[i].cpu().numpy()
            current_axes[2].imshow(pred, cmap='gray')
            current_axes[2].set_title('预测掩码')
            current_axes[2].axis('off')
            
            # 显示概率图
            prob = prob_maps[i].cpu()
            # 确保是2D数组
            if len(prob.shape) == 3 and prob.shape[0] == 1:
                prob = prob.squeeze(0)
            prob = prob.numpy()
            current_axes[3].imshow(prob, cmap='jet')
            current_axes[3].set_title('概率图')
            current_axes[3].axis('off')
            
            # 如果有边缘图，则显示
            edge_col_idx = 4
            if edge_maps is not None:
                edge = edge_maps[i].cpu()
                # 确保是2D数组
                if len(edge.shape) == 3 and edge.shape[0] == 1:
                    edge = edge.squeeze(0)
                edge = edge.numpy()
                current_axes[4].imshow(edge, cmap='jet')
                current_axes[4].set_title('边缘图')
                current_axes[4].axis('off')
                edge_col_idx = 5
            
            # 添加分割结果图(用轮廓线标记在原图上)
            contour_img = img.copy()
            # 将预测掩码转换为uint8类型以用于findContours
            pred_uint8 = (pred * 255).astype(np.uint8)
            contours, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 在原图上画出轮廓
            cv2.drawContours(contour_img, contours, -1, (1, 0, 0), 2)  # 红色轮廓，线宽2
            
            # 显示轮廓图像
            current_axes[edge_col_idx].imshow(contour_img)
            current_axes[edge_col_idx].set_title('分割轮廓标记')
            current_axes[edge_col_idx].axis('off')
        
        plt.tight_layout()
        if save_path:
            try:
                plt.savefig(save_path)
                print(f"可视化结果已保存到: {save_path}")
            except Exception as e:
                print(f"保存可视化结果失败: {e}")
            
            # 为每个样本保存单独的图像
            for i in range(num_samples):
                # 创建样本保存目录
                try:
                    sample_dir = os.path.dirname(save_path)
                    sample_dir = os.path.join(sample_dir, f'sample{i+1}')
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # 获取原始图像
                    img = images[i].cpu()
                    if img.shape[0] == 1:
                        img = img.repeat(3, 1, 1)
                    elif img.shape[0] > 3:
                        img = img[:3]
                    img = img.permute(1, 2, 0).numpy()
                    if img.max() > 0:
                        img = (img - img.min()) / (img.max() - img.min())
                    
                    # 保存原始图像
                    img_uint8 = (img * 255).astype(np.uint8)
                    orig_path = os.path.join(sample_dir, "original.png")
                    try:
                        cv2.imwrite(orig_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
                        print(f"已保存原始图像到: {orig_path}")
                    except Exception as e:
                        print(f"保存原始图像失败: {e}")
                    
                    # 保存真实掩码
                    mask = masks[i].cpu().numpy()
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask_path = os.path.join(sample_dir, "mask.png")
                    try:
                        cv2.imwrite(mask_path, mask_uint8)
                        print(f"已保存掩码到: {mask_path}")
                    except Exception as e:
                        print(f"保存掩码失败: {e}")
                    
                    # 保存预测掩码
                    pred = predictions[i].cpu().numpy()
                    pred_uint8 = (pred * 255).astype(np.uint8)
                    pred_path = os.path.join(sample_dir, "prediction.png")
                    try:
                        cv2.imwrite(pred_path, pred_uint8)
                        print(f"已保存预测掩码到: {pred_path}")
                    except Exception as e:
                        print(f"保存预测掩码失败: {e}")
                    
                    # 保存概率图（使用热力图颜色）
                    prob = prob_maps[i].cpu()
                    if len(prob.shape) == 3 and prob.shape[0] == 1:
                        prob = prob.squeeze(0)
                    prob = prob.numpy()
                    # 归一化到0-255
                    prob_norm = (prob * 255).astype(np.uint8)
                    # 应用热力图
                    prob_color = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
                    prob_path = os.path.join(sample_dir, "probability_map.png")
                    try:
                        cv2.imwrite(prob_path, prob_color)
                        print(f"已保存概率图到: {prob_path}")
                    except Exception as e:
                        print(f"保存概率图失败: {e}")
                    
                    # 如果有边缘图，则保存
                    if edge_maps is not None:
                        edge = edge_maps[i].cpu()
                        if len(edge.shape) == 3 and edge.shape[0] == 1:
                            edge = edge.squeeze(0)
                        edge = edge.numpy()
                        edge_norm = (edge * 255).astype(np.uint8)
                        edge_color = cv2.applyColorMap(edge_norm, cv2.COLORMAP_JET)
                        edge_path = os.path.join(sample_dir, "boundary_map.png")
                        try:
                            cv2.imwrite(edge_path, edge_color)
                            print(f"已保存边界图到: {edge_path}")
                        except Exception as e:
                            print(f"保存边界图失败: {e}")
                    
                    # 在原图上绘制轮廓并保存
                    contour_img = img.copy()
                    contours, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour_img_uint8 = (contour_img * 255).astype(np.uint8)
                    cv2.drawContours(contour_img_uint8, contours, -1, (255, 0, 0), 2)  # 红色轮廓
                    contour_path = os.path.join(sample_dir, "contour_result.png")
                    try:
                        cv2.imwrite(contour_path, cv2.cvtColor(contour_img_uint8, cv2.COLOR_RGB2BGR))
                        print(f"已保存轮廓图到: {contour_path}")
                    except Exception as e:
                        print(f"保存轮廓图失败: {e}")
                    
                    print(f"样本{i+1}的各图像已保存至: {sample_dir}")
                except Exception as e:
                    print(f"处理样本{i+1}时出错: {e}")
        
        plt.close()
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        # 出错时不要中断测试过程，继续执行
        plt.close()

def test(model, loader, device, save_dir=None, visualize=True, max_vis_samples=3):
    model.eval()
    
    # 初始化各项指标列表
    ious = []
    mious = []  # 新增mIoU指标
    dices = []
    pas = []  # 像素准确率列表
    mpas = []  # 平均像素准确率列表
    aps = []   # 平均精度列表
    boundary_accs = []  # 边界准确性列表
    
    # 用于FPS计算的变量
    total_time = 0
    total_frames = 0
    fps_list = []
    
    # 用于可视化的数据
    vis_images = []
    vis_masks = []
    vis_preds = []
    vis_probs = []
    vis_edges = []
    vis_thresh_maps = []  # 新增阈值图
    vis_count = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(loader, desc="测试中")):
            images = images.to(device)
            masks = masks.to(device)
            
            batch_size = images.size(0)
            total_frames += batch_size
            
            # 测量推理时间（开始）
            start_time = time.time()
            
            try:
                outputs = model(images)
                
                # 测量推理时间（结束）
                end_time = time.time()
                inference_time = end_time - start_time
                total_time += inference_time
                
                # 计算当前批次的FPS
                if inference_time > 0:
                    batch_fps = batch_size / inference_time
                    fps_list.append(batch_fps)
                    if i % 10 == 0:  # 每10个批次打印一次FPS
                        print(f"Batch {i+1} - FPS: {batch_fps:.2f}")
                    
                # 确保pred_masks和masks有正确的形状
                pred_masks = outputs.get('binary_map', None)
                if pred_masks is None:
                    print(f"警告: 模型输出中没有binary_map, 可用键：{outputs.keys()}")
                    continue
                    
                if len(pred_masks.shape) == 4 and pred_masks.shape[1] == 1:
                    pred_masks = pred_masks.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
                
                if len(masks.shape) == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
                
                # 二值化预测结果
                thresholded_masks = (pred_masks > 0.5).float()
                
                # 计算批次的PA (像素准确率)
                batch_pa = calculate_pixel_accuracy(thresholded_masks, masks).detach().cpu().item()
                pas.append(batch_pa)
                
                # 计算mPA (各类别平均准确率)
                for j in range(pred_masks.size(0)):
                    class_accs = calculate_class_pixel_accuracy(thresholded_masks[j], masks[j])
                    mpas.append(np.mean(class_accs))
                
                # 计算IoU, mIoU和Dice
                batch_iou = calculate_iou(thresholded_masks, masks).detach().cpu().item()
                ious.append(batch_iou)
                
                # 计算mIoU (新增)
                for j in range(pred_masks.size(0)):
                    batch_miou = calculate_mean_iou(thresholded_masks[j], masks[j])
                    mious.append(batch_miou)
                
                batch_dice = calculate_dice(thresholded_masks, masks).detach().cpu().item()
                dices.append(batch_dice)
                
                # 计算掩码平均精度和边界准确性
                for j in range(pred_masks.size(0)):
                    # 检查是否有正样本
                    has_positive = masks[j].sum() > 0
                    
                    if has_positive:
                        # 计算掩码平均精度 (AP)
                        precision, recall, ap = calculate_precision_recall(pred_masks[j], masks[j])
                        aps.append(float(ap))
                    else:
                        # 如果没有正样本，且预测也都是负样本（全0），则AP为1.0
                        if thresholded_masks[j].sum() == 0:
                            aps.append(1.0)
                        else:
                            # 否则AP为0（全是假阳性）
                            aps.append(0.0)
                    
                    # 计算边界准确性
                    try:
                        boundary_acc = calculate_boundary_accuracy(thresholded_masks[j], masks[j])
                        # 确保转换为Python标量
                        if isinstance(boundary_acc, torch.Tensor):
                            boundary_acc = boundary_acc.detach().cpu().item()
                        else:
                            boundary_acc = float(boundary_acc)
                        boundary_accs.append(boundary_acc)
                    except Exception as e:
                        if i == 0:  # 只在第一批次出错时打印
                            print(f"计算边界准确性时出错: {e}")

                # 收集可视化数据
                if visualize and vis_count < max_vis_samples:
                    num_to_add = min(batch_size, max_vis_samples - vis_count)
                    vis_images.extend(images[:num_to_add].detach().cpu())
                    vis_masks.extend(masks[:num_to_add].detach().cpu())
                    vis_preds.extend(thresholded_masks[:num_to_add].detach().cpu())
                    
                    # 确保prob_map存在并提取
                    if 'prob_map' in outputs:
                        vis_probs.extend(outputs['prob_map'][:num_to_add].detach().cpu())
                    else:
                        # 如果没有prob_map，使用预测掩码作为替代
                        vis_probs.extend(pred_masks[:num_to_add].detach().cpu())
                    
                    # 如果有边缘图，则收集
                    if 'edge_map' in outputs:
                        vis_edges.extend(outputs['edge_map'][:num_to_add].detach().cpu())
                        
                    # 如果有阈值图，则收集
                    print(f"模型输出键值: {outputs.keys()}")
                    if 'threshold' in outputs:
                        # 创建与图像大小相同的阈值图
                        batch_thresholds = outputs['threshold'][:num_to_add].detach().cpu()
                        print(f"检测到阈值: shape={batch_thresholds.shape}, values={batch_thresholds}")
                        
                        for j, thresh in enumerate(batch_thresholds):
                            # 获取概率图
                            prob_map = None
                            if 'prob_map' in outputs:
                                prob_map = outputs['prob_map'][j].detach().cpu()
                                if len(prob_map.shape) == 3 and prob_map.shape[0] == 1:
                                    prob_map = prob_map.squeeze(0)
                            
                            # 如果阈值是标量，我们需要生成一个有意义的阈值图
                            # 而不仅仅是一个均匀的颜色
                            if not isinstance(thresh, torch.Tensor) or thresh.numel() == 1:
                                thresh_value = float(thresh.item() if isinstance(thresh, torch.Tensor) else thresh)
                                print(f"为样本{vis_count+j+1}创建阈值图，阈值值={thresh_value:.4f}")
                                
                                if prob_map is not None:
                                    # 创建实际的阈值图，显示不同区域与阈值的关系
                                    # 而不是简单的常数图
                                    thresh_map = torch.zeros_like(prob_map)
                                    
                                    # 小于阈值的区域显示为从0到阈值的颜色
                                    below_thresh = prob_map < thresh_value
                                    if below_thresh.any():
                                        # 将小于阈值的值归一化到[0, 0.5)区间
                                        thresh_map[below_thresh] = prob_map[below_thresh] / (2 * thresh_value + 1e-8)
                                    
                                    # 大于阈值的区域显示为从阈值到1的颜色
                                    above_thresh = ~below_thresh
                                    if above_thresh.any():
                                        # 将大于阈值的值归一化到[0.5, 1]区间
                                        thresh_map[above_thresh] = 0.5 + (prob_map[above_thresh] - thresh_value) / (2 * (1 - thresh_value) + 1e-8)
                                    
                                    # 创建决策边界区域的高亮显示
                                    edge_width = 0.05
                                    near_thresh = torch.abs(prob_map - thresh_value) < edge_width
                                    if near_thresh.any():
                                        # 对接近阈值的区域增加亮度
                                        highlight = 0.7 * (1 - torch.abs(prob_map[near_thresh] - thresh_value) / edge_width)
                                        thresh_map[near_thresh] = torch.clamp(thresh_map[near_thresh] + highlight, 0, 1)
                                else:
                                    # 如果没有概率图，则使用默认尺寸
                                    h, w = 256, 256  
                                    thresh_map = torch.ones((h, w)) * thresh_value
                            else:
                                # 如果已经是图像则直接使用
                                print(f"样本{vis_count+j+1}使用已有阈值图，形状={thresh.shape}")
                                thresh_map = thresh
                                # 如果是三维张量，转为二维
                                if len(thresh_map.shape) == 3 and thresh_map.shape[0] == 1:
                                    thresh_map = thresh_map.squeeze(0)
                            
                            # 确保阈值图的值在0-1范围内
                            if thresh_map.max() > 1.0 or thresh_map.min() < 0.0:
                                print(f"警告: 阈值图值不在[0,1]范围内，将进行裁剪，min={thresh_map.min()}, max={thresh_map.max()}")
                                thresh_map = torch.clamp(thresh_map, 0.0, 1.0)
                            
                            vis_thresh_maps.append(thresh_map)
                            print(f"已添加阈值图, shape={thresh_map.shape}")
                    else:
                        # 如果模型没有输出阈值，使用默认阈值0.5创建有意义的阈值图
                        print("模型没有输出阈值，使用默认值0.5创建阈值图")
                        for j in range(num_to_add):
                            # 获取概率图
                            prob_map = None
                            if 'prob_map' in outputs:
                                prob_map = outputs['prob_map'][j].detach().cpu()
                                if len(prob_map.shape) == 3 and prob_map.shape[0] == 1:
                                    prob_map = prob_map.squeeze(0)
                            
                            if prob_map is not None:
                                # 使用默认阈值0.5创建有意义的阈值图
                                thresh_value = 0.5
                                thresh_map = torch.zeros_like(prob_map)
                                
                                # 小于阈值的区域
                                below_thresh = prob_map < thresh_value
                                if below_thresh.any():
                                    thresh_map[below_thresh] = prob_map[below_thresh] / 1.0
                                
                                # 大于阈值的区域
                                above_thresh = ~below_thresh
                                if above_thresh.any():
                                    thresh_map[above_thresh] = prob_map[above_thresh]
                                
                                # 创建决策边界区域的高亮显示
                                edge_width = 0.05
                                near_thresh = torch.abs(prob_map - thresh_value) < edge_width
                                if near_thresh.any():
                                    # 对接近阈值的区域使用不同的颜色
                                    highlight = 0.7 * (1 - torch.abs(prob_map[near_thresh] - thresh_value) / edge_width)
                                    thresh_map[near_thresh] = torch.clamp(thresh_map[near_thresh] + highlight, 0, 1)
                            else:
                                # 如果没有概率图，则使用默认尺寸
                                h, w = 256, 256
                                thresh_map = torch.ones((h, w)) * 0.5
                            
                            # 确保值在0-1范围内
                            thresh_map = torch.clamp(thresh_map, 0, 1)
                            
                            vis_thresh_maps.append(thresh_map)
                            print(f"已添加默认阈值图, shape={thresh_map.shape}")
                    
                    vis_count += num_to_add
            
            except Exception as e:
                print(f"处理批次 {i+1} 时出错: {e}")
                # 跳过这个批次，继续处理下一个
                continue

    # 计算平均指标
    mean_iou = np.mean(ious) if ious else 0.0
    mean_miou = np.mean(mious) if mious else 0.0  # 新增mIoU
    mean_dice = np.mean(dices) if dices else 0.0
    mean_pa = np.mean(pas) if pas else 0.0
    mean_mpa = np.mean(mpas) if mpas else 0.0
    mean_ap = np.mean(aps) if aps else 0.0
    mean_boundary_acc = np.mean(boundary_accs) if boundary_accs else 0.0
    
    # 计算FPS
    average_fps = total_frames / total_time if total_time > 0 else 0
    median_fps = np.median(fps_list) if fps_list else 0
    
    # 可视化测试结果
    if visualize and vis_count > 0 and save_dir:
        try:
            vis_path = os.path.join(save_dir, 'visualization.png')
            if len(vis_edges) > 0:
                visualize_results(vis_images, vis_masks, vis_preds, vis_probs, vis_edges, vis_path)
            else:
                visualize_results(vis_images, vis_masks, vis_preds, vis_probs, None, vis_path)
                
            # 额外保存阈值图（如果存在）
            if True:  # 无论如何都创建阈值图
                try:
                    thresh_dir = os.path.join(save_dir, 'threshold_maps')
                    os.makedirs(thresh_dir, exist_ok=True)
                    for i, thresh_map in enumerate(vis_thresh_maps):
                        # 确保阈值图是numpy数组
                        thresh_map_np = thresh_map.numpy()
                        
                        # 确保值在0-1范围内
                        if thresh_map_np.min() < 0 or thresh_map_np.max() > 1:
                            thresh_map_np = np.clip(thresh_map_np, 0, 1)
                        
                        # 转换为uint8格式 (0-255)
                        thresh_map_uint8 = (thresh_map_np * 255).astype(np.uint8)
                        
                        # 确保是单通道图像
                        if len(thresh_map_uint8.shape) > 2:
                            thresh_map_uint8 = thresh_map_uint8.squeeze()
                        
                        # 应用颜色映射前确保图像格式正确
                        print(f"阈值图形状: {thresh_map_uint8.shape}, 类型: {thresh_map_uint8.dtype}, 范围: [{thresh_map_uint8.min()}, {thresh_map_uint8.max()}]")
                        
                        # 应用热力图颜色映射
                        try:
                            colored_thresh = cv2.applyColorMap(thresh_map_uint8, cv2.COLORMAP_VIRIDIS)
                            
                            # 保存到全局阈值图目录
                            thresh_path = os.path.join(thresh_dir, f'threshold_map_sample{i+1}.png')
                            cv2.imwrite(thresh_path, colored_thresh)
                            print(f"已保存阈值图到 {thresh_path}")
                            
                            # 同时保存到对应的sample目录
                            sample_dir = os.path.join(save_dir, f'sample{i+1}')
                            if os.path.exists(sample_dir):
                                try:
                                    sample_thresh_path = os.path.join(sample_dir, "threshold_map.png")
                                    cv2.imwrite(sample_thresh_path, colored_thresh)
                                    print(f"已保存阈值图到 {sample_thresh_path}")
                                except Exception as e:
                                    print(f"保存到sample目录失败: {e}")
                        except Exception as e:
                            print(f"应用颜色映射失败: {e}")
                            # 如果应用颜色映射失败，直接保存原始灰度图
                            thresh_path = os.path.join(thresh_dir, f'threshold_map_sample{i+1}_gray.png')
                            cv2.imwrite(thresh_path, thresh_map_uint8)
                            print(f"已保存灰度阈值图到 {thresh_path}")
                    
                    print(f"阈值图已保存至: {thresh_dir} 和各sample目录")
                except Exception as e:
                    print(f"生成阈值图时出错: {e}")
        except Exception as e:
            print(f"创建可视化结果时出错: {e}")
    
    # 打印详细结果
    print("\n" + "="*50)
    print("测试结果:")
    print(f"IoU: {mean_iou:.4f}")
    print(f"mIoU: {mean_miou:.4f}")  # 新增mIoU
    print(f"Dice 系数: {mean_dice:.4f}")
    print(f"Pixel Accuracy (PA): {mean_pa:.4f}")
    print(f"Mean Pixel Accuracy (mPA): {mean_mpa:.4f}")
    print(f"平均精度 (AP): {mean_ap:.4f}")
    print(f"边界准确性: {mean_boundary_acc:.4f}")
    print("-"*50)
    print(f"性能分析:")
    print(f"平均 FPS: {average_fps:.2f}")
    print(f"中位数 FPS: {median_fps:.2f}")
    print(f"总推理时间: {total_time:.2f}秒")
    print("="*50)

    # 返回所有指标，包括新增的mIoU
    return {
        'iou': mean_iou,
        'miou': mean_miou,  # 新增mIoU
        'dice': mean_dice,
        'pa': mean_pa,
        'mpa': mean_mpa,
        'ap': mean_ap,
        'boundary_acc': mean_boundary_acc,
        'avg_fps': average_fps,
        'median_fps': median_fps
    }

def main():
    # 首先检查依赖
    if not check_dependencies():
        print("缺少必要的依赖，无法继续测试")
        return
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试木材缺陷分割模型')
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--visualize', action='store_true', default=True, help='是否可视化结果')
    
    args = parser.parse_args()
    
    # 如果命令行没有指定模型路径，则交互式地询问用户
    model_path = args.model_path
    if model_path is None:
        model_path = input("请输入模型文件路径 (默认为 outputs/best_model.pth): ").strip()
        if not model_path:
            model_path = "outputs/best_model.pth"
    
    config = Config()
    
    # 如果 Config 类中没有测试集路径，在这里添加
    if not hasattr(config, 'test_image_dir'):
        config.test_image_dir = os.path.join(config.data_dir, 'test/images')
        config.test_mask_dir = os.path.join(config.data_dir, 'test/masks')
    
    # 如果测试集不存在，可以临时使用验证集
    if not os.path.exists(config.test_image_dir):
        print("警告：测试集目录不存在，使用验证集代替")
        config.test_image_dir = config.val_image_dir
        config.test_mask_dir = config.val_mask_dir
    
    # 创建保存目录
    save_dir = os.path.join(config.save_dir, 'predictions')
    os.makedirs(save_dir, exist_ok=True)

    # 准备测试数据
    try:
        test_dataset = WoodDefectDataset(
            config.test_image_dir,
            config.test_mask_dir,
            is_train=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        print(f"加载了 {len(test_dataset)} 个测试样本")
    except Exception as e:
        print(f"加载测试数据集时出错: {e}")
        return

    # 设置运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    try:
        model = WoodDefectDB(pretrained=False)
        
        print(f"尝试加载模型: {model_path}")

        # 尝试加载检查点
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("从检查点字典加载模型成功")
            else:
                # 尝试直接加载模型
                model.load_state_dict(checkpoint)
                print("直接加载模型成功")
        else:
            print(f"警告: 模型文件 {model_path} 不存在，使用随机初始化权重")
        
        model = model.to(device)
        print("模型加载到设备完成，开始测试...")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 测试
    try:
        results = test(model, test_loader, device, save_dir, visualize=args.visualize, max_vis_samples=3)
        
        # 获取模型文件名（不包含扩展名）以用于保存结果
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        result_file = os.path.join(config.save_dir, f'test_results_{model_name}.txt')
        
        # 保存测试结果到文件
        with open(result_file, 'w') as f:
            f.write("测试结果\n")
            f.write(f"IoU: {results['iou']:.4f}\n")
            f.write(f"mIoU: {results['miou']:.4f}\n")  # 新增mIoU
            f.write(f"Dice 系数: {results['dice']:.4f}\n")
            f.write(f"Pixel Accuracy (PA): {results['pa']:.4f}\n")
            f.write(f"Mean Pixel Accuracy (mPA): {results['mpa']:.4f}\n")
            f.write(f"平均精度 (AP): {results['ap']:.4f}\n")
            f.write(f"边界准确性: {results['boundary_acc']:.4f}\n")
            f.write("-"*50 + "\n")
            f.write(f"平均 FPS: {results['avg_fps']:.2f}\n")
            f.write(f"中位数 FPS: {results['median_fps']:.2f}\n")
        
        print(f"测试结果已保存到: {result_file}")
        return results
    except Exception as e:
        print(f"测试过程中出错: {e}")
        return None

if __name__ == '__main__':
    result = main()
    
    if result:
        print("\n测试完成!")
        # 如果有返回结果，可以在这里做进一步处理
