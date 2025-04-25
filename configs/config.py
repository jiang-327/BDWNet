import torch
import os
import os.path as osp
class Config:
    def __init__(self):
        self.project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        self.save_dir = 'outputs'
        self.save_freq = 5
        self.vis_dir = os.path.join(self.save_dir, 'visualizations')
        self.data_dir = osp.join(self.project_root, "data")
        self.train_image_dir = osp.join(self.data_dir, 'train/images')
        self.train_mask_dir = osp.join(self.data_dir, 'train/masks')
        self.val_image_dir = osp.join(self.data_dir, 'val/images')
        self.val_mask_dir = osp.join(self.data_dir, 'val/masks')
        self.test_image_dir = osp.join(self.data_dir, 'test/images')
        self.test_mask_dir = osp.join(self.data_dir, 'test/masks')
        self._ensure_directories_exist()
        self.image_size = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 200
        self.batch_size = 16
        self.initial_lr = 1e-4
        self.num_workers = 8
        self.weight_decay = 1e-5
        self.edge_weight = 0.5
        self.threshold_weight = 0.2
        self.dice_weight = 2.0
        self.tversky_weight = 1.0
        self.focal_weight = 0.7
        self.tversky_alpha = 0.3
        self.tversky_beta = 0.7
        self.early_stopping_patience = 15
        self.early_stopping_min_delta = 0.001
        self.lr_scheduler = 'onecycle'
        self.max_lr = 1e-3
        self.warmup_epochs = 5
        self.min_lr = 1e-6
        self.pretrained = True
        self.num_classes = 1
        self.mtam_dilations = [1, 2, 4, 8]
        self.mtam_reduction = 4
        self.warmup_factor = 0.001
        self.warmup_iters = 1000
        self.backbone_lr_factor = 0.1
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.dropout_rate = 0.2
        self.vis_interval = 2
        self.vis_samples = 8
        self.save_vis = True
        self.print_freq = 10
        self.eval_freq = 1
        self.loss_weights = {
            'bce': 1.0,
            'dice': 2.0,
            'tversky': 1.0,
            'focal': 0.7,
            'edge': 0.5,
            'threshold': 0.2
        }
    def _ensure_directories_exist(self):
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
            print(f"Ensuring directory exists: {directory}")
    def get_lr_scheduler(self, optimizer):
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
                steps_per_epoch=100,
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=1000.0
            )
        else:
            from torch.optim.lr_scheduler import OneCycleLR
            return OneCycleLR(
                optimizer,
                max_lr=self.initial_lr,
                epochs=self.num_epochs,
                steps_per_epoch=100
            )
    def get_model(self):
        from models.network import WoodDefectBD
        return WoodDefectBD(pretrained=self.pretrained)
    def get_loss_function(self):
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
        from torch.optim import AdamW
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'mtpe' in name and 'layer' in name:
                backbone_params.append(param)
            else:
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
        return EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            verbose=True
        )
class EarlyStopping:
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