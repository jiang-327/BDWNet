# 在文件开头添加导入
import torch
import os
import os.path as osp

# 参数配置
class Config:
    """配置类，存储所有训练和模型参数"""
    
    def __init__(self):
        # 获取项目根目录
        self.project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        
        # 保存配置
        self.save_dir = 'outputs'
        self.save_freq = 5  # 每隔多少个epoch保存一次模型
        
        # 可视化配置
        self.vis_dir = os.path.join(self.save_dir, 'visualizations')
        
        # 数据集路径配置 - 使用相对于项目根目录的路径
        self.data_dir = osp.join(self.project_root, "data")  # 添加数据根目录
        self.train_image_dir = osp.join(self.data_dir, 'train/images')
        self.train_mask_dir = osp.join(self.data_dir, 'train/masks')
        self.val_image_dir = osp.join(self.data_dir, 'val/images')
        self.val_mask_dir = osp.join(self.data_dir, 'val/masks')
        self.test_image_dir = osp.join(self.data_dir, 'test/images')
        self.test_mask_dir = osp.join(self.data_dir, 'test/masks')
        
        # 确保数据目录存在
        self._ensure_directories_exist()

        # 图像处理配置
        self.image_size = 512  # 提高图像尺寸以改善精度

        # 训练超参数配置 - 调整以适应A6000显卡
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 200    # 增加训练轮数提高模型收敛
        self.batch_size = 2     # 增加批次大小利用A6000的大显存
        self.initial_lr = 3e-4   # 适当调整初始学习率
        self.num_workers = 8     # 增加工作线程数提高数据加载速度
        
        # 优化器配置
        self.weight_decay = 1e-5  # 适当调整正则化强度
        
        # 损失函数权重 - 调整以平衡各类损失
        self.edge_weight = 0.5
        self.threshold_weight = 0.2
        self.dice_weight = 2.0      # 大幅增加Dice损失权重
        self.tversky_weight = 1.0   # 添加对小目标敏感的Tversky损失
        self.focal_weight = 0.7     # 添加对困难样本敏感的Focal损失
        
        # Tversky损失参数 - 对FN给予更高权重
        self.tversky_alpha = 0.3    # FP权重
        self.tversky_beta = 0.7     # FN权重
        
        # 早停配置
        self.early_stopping_patience = 15  # 增加早停耐心
        self.early_stopping_min_delta = 0.001  # 提高提前停止的阈值
        
        # 学习率调度
        self.lr_scheduler = 'onecycle'  # 使用OneCycle策略
        self.max_lr = 1e-3  # 最大学习率
        self.warmup_epochs = 5  # 增加预热轮数
        self.min_lr = 1e-6  # 最小学习率
        
        # 模型配置
        self.pretrained = True  # 使用预训练权重

        # 分类相关配置
        self.num_classes = 1  # 缺陷分割是二分类问题

        # 添加MTAM相关配置
        self.mtam_dilations = [1, 2, 4, 8]  # 保持原有空洞卷积的膨胀率
        self.mtam_reduction = 4  # 通道压缩比例

        # 学习率预热
        self.warmup_factor = 0.001
        self.warmup_iters = 1000  # 增加预热迭代次数
        
        # 优化器配置
        self.backbone_lr_factor = 0.1  # 减少主干网络的学习率因子

        # 如果使用GPU，设置一些CUDA相关配置
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # 加速训练
            torch.backends.cudnn.deterministic = False  # 允许非确定性运行以提高性能
        
        # 添加正则化
        self.dropout_rate = 0.2    # 增加dropout以防止过拟合

        # 可视化配置
        self.vis_interval = 2    # 每2个epoch保存可视化结果
        self.vis_samples = 8     # 每次保存8个样本
        self.save_vis = True     # 是否保存可视化结果

        # 简化日志配置
        self.print_freq = 10     # 每10个batch打印一次
        self.eval_freq = 1      # 每epoch评估一次

        # 添加损失函数权重
        self.loss_weights = {       # 损失函数权重
            'bce': 1.0,
            'dice': 2.0,
            'tversky': 1.0,
            'focal': 0.7,
            'edge': 0.5,
            'threshold': 0.2
        }

    def _ensure_directories_exist(self):
        """确保所有必要的目录存在"""
        directories = [
            self.data_dir,
            self.train_image_dir,
            self.train_mask_dir,
            self.val_image_dir,
            self.val_mask_dir,
            self.test_image_dir,
            self.test_mask_dir,
            self.save_dir,
            self.vis_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"确保目录存在: {directory}")

    def get_lr_scheduler(self, optimizer):
        """获取学习率调度器"""
        if self.lr_scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=self.min_lr
            )
        elif self.lr_scheduler == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            return ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif self.lr_scheduler == 'onecycle':
            from torch.optim.lr_scheduler import OneCycleLR
            return OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                epochs=self.num_epochs,
                steps_per_epoch=100,  # 这个值应该在训练时被替换为实际的steps_per_epoch
                pct_start=0.3,  # 30%的时间用于预热
                div_factor=10.0,  # 初始学习率 = max_lr/div_factor
                final_div_factor=1000.0  # 最终学习率 = max_lr/final_div_factor
            )
        else:
            from torch.optim.lr_scheduler import OneCycleLR
            return OneCycleLR(
                optimizer,
                max_lr=self.initial_lr,
                epochs=self.num_epochs,
                steps_per_epoch=100  # 应该传入实际的len(train_loader)
            )

    def get_model(self):
        """获取模型实例"""
        from models.network import WoodDefectDB
        return WoodDefectDB(pretrained=self.pretrained)

    def get_loss_function(self):
        """获取损失函数实例"""
        from models.loss import WoodDefectLoss
        return WoodDefectLoss(
            edge_weight=self.edge_weight,
            threshold_weight=self.threshold_weight,
            dice_weight=self.dice_weight,
            tversky_weight=self.tversky_weight,
            focal_weight=self.focal_weight,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta
        )

    def get_optimizer(self, model):
        """获取优化器实例"""
        from torch.optim import AdamW  # 使用AdamW优化器
        
        # 区分骨干网络和其他部分使用不同的学习率
        backbone_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'mtpe' in name and 'layer' in name:  # 骨干网络参数
                backbone_params.append(param)
            else:  # 其他参数
                other_params.append(param)
                
        param_groups = [
            {'params': backbone_params, 'lr': self.initial_lr * self.backbone_lr_factor},
            {'params': other_params}
        ]
        
        return AdamW(
            param_groups,
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )

    def get_early_stopping(self):
        """获取早停实例"""
        return EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            verbose=True
        )

    # 分类相关配置
    def get_num_classes(self):
        # Implementation of get_num_classes method
        pass

    # 添加MTAM相关配置
    def get_mtam_dilations(self):
        # Implementation of get_mtam_dilations method
        pass

    def get_mtam_reduction(self):
        # Implementation of get_mtam_reduction method
        pass

    # 学习率预热
    def get_warmup_epochs(self):
        # Implementation of get_warmup_epochs method
        pass

    def get_warmup_factor(self):
        # Implementation of get_warmup_factor method
        pass

    def get_warmup_iters(self):
        # Implementation of get_warmup_iters method
        pass

    def get_min_lr(self):
        # Implementation of get_min_lr method
        pass

    # 优化器配置
    def get_backbone_lr_factor(self):
        # Implementation of get_backbone_lr_factor method
        pass

class EarlyStopping:
    """早停类"""
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
