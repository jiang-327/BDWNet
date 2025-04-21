import os
# 设置PyTorch内存管理选项
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # 增加内存分配上限以适应A6000显卡
# 设置环境变量以减少内存泄漏
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁止TensorFlow日志

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import gc  # 用于垃圾回收
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, 
    CosineAnnealingLR,
    OneCycleLR,
    LambdaLR
)
from sklearn.model_selection import KFold

from models.network import WoodDefectDB
from models.loss import WoodDefectLoss
from utils.dataset import (
    WoodDefectDataset, 
    get_training_augmentation,
    get_strong_augmentation
)
from utils.metrics import (
    calculate_pixel_accuracy, calculate_iou, calculate_dice,
    calculate_boundary_f1, MetricTracker, AverageMeter, evaluate_model,
    calculate_precision, calculate_recall, calculate_f1_score
)
from configs.config import Config

import logging
import sys
import time
import json

# 配置日志
def setup_logger():
    """设置日志记录器"""
    logger = logging.getLogger('wood_defect_detection')
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def print_gpu_info():
    """打印GPU信息"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"Memory Usage: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB / {torch.cuda.memory_reserved(i)/1024**3:.2f}GB")
    else:
        logger.info("No GPU available, using CPU")

def train_one_epoch(model, loader, criterion, optimizer, device, metric_tracker):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, masks in pbar:
        # 将数据移动到设备
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向传播
        outputs = model(images)
        loss_dict = criterion(outputs, masks)
        loss = loss_dict['total']
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新统计信息
        epoch_loss += loss.item()
        
        # 更新指标
        metric_tracker.update(outputs['binary_map'], masks)
    
        # 更新进度条
        current_loss = loss.item()
        current_iou = metric_tracker.meters['iou'].val['iou']
        pbar.set_postfix({
            'loss': f"{current_loss:.4f}",
            'iou': f"{current_iou:.4f}"
        })
    
    # 计算平均损失和指标
    avg_loss = epoch_loss / len(loader)
    metrics = metric_tracker.get_averages()
    metrics['loss'] = avg_loss
    
    return metrics

def validate(model, loader, criterion, device, metric_tracker):
    """验证模型性能"""
    model.eval()
    epoch_loss = 0
    pbar = tqdm(loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for images, masks in pbar:
            # 将数据移动到设备
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss_dict = criterion(outputs, masks)
            loss = loss_dict['total']
            
            # 更新统计信息
            epoch_loss += loss.item()
            
            # 更新指标
            metric_tracker.update(outputs['binary_map'], masks)
    
            # 更新进度条
            current_loss = loss.item()
            current_iou = metric_tracker.meters['iou'].val['iou']
            pbar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'iou': f"{current_iou:.4f}"
            })
    
    # 计算平均损失和指标
    avg_loss = epoch_loss / len(loader)
    metrics = metric_tracker.get_averages()
    metrics['loss'] = avg_loss
    
    return metrics

def train_with_kfold(dataset, k=3):
    """使用K折交叉验证进行训练"""
    config = Config()
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 初始化K折交叉验证
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # 跟踪所有折的性能
    fold_results = []
    
    # 对每一折进行训练
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        logger.info(f"FOLD {fold+1}/{k}")
        logger.info(f"Train Size: {len(train_ids)}, Validation Size: {len(val_ids)}")
        
        # 创建数据加载器
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=config.batch_size,
            sampler=train_subsampler,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=val_subsampler,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # 训练当前折
        fold_result = train_fold(train_loader, val_loader, config, fold)
        fold_results.append(fold_result)
    
    return fold_results

def train_fold(train_loader, val_loader, config, fold):
    """训练一个折"""
    # 初始化模型
    model = config.get_model().to(config.device)
    
    # 初始化损失函数和优化器
    criterion = config.get_loss_function()
    optimizer = config.get_optimizer(model)
    
    # 初始化学习率调度器
    scheduler = config.get_lr_scheduler(optimizer)
    
    # 初始化早停
    early_stopping = config.get_early_stopping()
    
    # 记录最佳性能
    best_val_iou = 0.0
    best_epoch = 0
    
    # 训练循环
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device,
            MetricTracker()
        )
        
        # 验证
        val_metrics = validate(
            model, val_loader, criterion, config.device,
            MetricTracker()
        )
        
        # 更新学习率
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics['iou'])
        else:
            scheduler.step()
            
            # 保存最佳模型
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_epoch = epoch
            torch.save(model.state_dict(), f"{config.save_dir}/best_model_fold{fold+1}.pth")
        
        # 打印当前性能
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # 早停检查
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
    
    logger.info(f"Best Val IoU: {best_val_iou:.4f} at epoch {best_epoch+1}")
    return {
        'fold': fold + 1,
        'best_val_iou': best_val_iou,
        'best_epoch': best_epoch + 1
    }

def main():
    """主函数"""
    # 设置配置
    config = Config()
    
    # 创建数据集
    dataset = WoodDefectDataset(
        image_dir=config.train_image_dir,
        mask_dir=config.train_mask_dir,
        transform=get_training_augmentation(config.image_size)
    )
    
    # 使用K折交叉验证训练
    results = train_with_kfold(dataset)
    
    # 保存结果
    with open(f"{config.save_dir}/training_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # 打印最终结果
    avg_iou = np.mean([r['best_val_iou'] for r in results])
    logger.info(f"Average Best Val IoU across folds: {avg_iou:.4f}")

if __name__ == "__main__":
        main()
