#数据集加载和预处理
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations import (
    Compose, RandomBrightnessContrast, RandomGamma, HorizontalFlip,
    VerticalFlip, Resize, Normalize, RandomRotate90, GaussNoise, OneOf,
    ElasticTransform, GridDistortion, OpticalDistortion, MotionBlur, MedianBlur, 
    GaussianBlur, Blur, CoarseDropout, ShiftScaleRotate
)
from albumentations.pytorch import ToTensorV2

class WoodDefectDataset(Dataset):
    """
    木材缺陷数据集类
    负责数据加载、预处理和增强
    """
    def __init__(self, image_dir, mask_dir, transform=None, is_train=True, img_size=512):
        """
        初始化数据集
        Args:
            image_dir: 图像目录
            mask_dir: 掩码目录
            transform: 数据增强转换
            is_train: 是否为训练集
            img_size: 图像尺寸
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.img_size = img_size
        
        # 获取所有图像和对应的掩码
        self.images = []
        self.masks = []
        
        # 检查目录是否存在
        if not os.path.exists(image_dir):
            raise RuntimeError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise RuntimeError(f"Mask directory not found: {mask_dir}")
            
        # 获取所有图像文件
        for img_name in os.listdir(image_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(img_name)[0]
                
                # 尝试多种可能的掩码命名模式
                possible_mask_names = [
                    img_name,  # 相同文件名
                    base_name + '_mask.png',  # 添加_mask后缀
                    base_name + '.png'  # 使用PNG扩展名
                ]
                
                mask_found = False
                for mask_name in possible_mask_names:
                    mask_path = os.path.join(mask_dir, mask_name)
                    if os.path.exists(mask_path):
                        self.images.append(img_name)
                        self.masks.append(mask_name)
                        mask_found = True
                        break
                
                if not mask_found:
                    print(f"Warning: No matching mask found for {img_name}")
        
        if len(self.images) == 0:
            raise RuntimeError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
            
        print(f"Found {len(self.images)} valid image-mask pairs")
        if len(self.images) > 0:
            print(f"First few pairs:")
            for i in range(min(3, len(self.images))):
                print(f"  Image: {self.images[i]} -> Mask: {self.masks[i]}")
        
        # 设置默认的数据增强
        if transform is None:
            if is_train:
                self.transform = get_strong_augmentation(img_size)
            else:
                self.transform = Compose([
                    Resize(img_size, img_size),
                    Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # 加载图像
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"加载图像时出错 {img_path}: {e}")
            # 创建一个空的图像作为替代
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # 加载掩码
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"无法读取掩码: {mask_path}")
        except Exception as e:
            print(f"加载掩码时出错 {mask_path}: {e}")
            # 创建一个空的掩码作为替代
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # 确保mask为float32类型
        mask = mask.astype(np.float32)
        
        # 确保mask的值在[0,1]范围内
        mask = mask / 255.0  # 显式归一化
        mask = np.clip(mask, 0, 1)
        
        # 数据增强
        if self.transform:
            try:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            except Exception as e:
                print(f"应用数据增强时出错: {e}")
                # 如果增强失败，至少转换为张量
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                mask = torch.from_numpy(mask).float().unsqueeze(0)

        # 确保mask处理正确
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        mask = torch.clamp(mask, 0, 1)

        return image, mask

    @staticmethod
    def get_transforms(is_train, img_size=512):
        """获取数据转换和增强"""
        if is_train:
            # 训练时使用数据增强
            transforms = Compose([
                # 先调整尺寸
                Resize(img_size, img_size),
                # 应用各种数据增强
                OneOf([
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    RandomRotate90(p=0.5),
                ], p=0.5),
                
                # 随机亮度对比度调整
                RandomBrightnessContrast(p=0.3),
                
                # 随机添加噪声
                OneOf([
                    GaussNoise(p=0.5),
                    GaussianBlur(p=0.5),
                    MotionBlur(p=0.5),
                ], p=0.3),
                
                # 随机光学变形
                OneOf([
                    ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    GridDistortion(p=0.5),
                    OpticalDistortion(distort_limit=1.0, shift_limit=0.5, p=0.5),
                ], p=0.3),
                
                # 归一化并转换为tensor
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # 验证/测试时只调整尺寸并归一化
            transforms = Compose([
                Resize(img_size, img_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        return transforms

def get_training_augmentation(img_size=512):
    """基本训练增强"""
    return Compose([
        RandomRotate90(p=0.5),
        Resize(img_size, img_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            GridDistortion(p=0.5),
            OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        OneOf([
            GaussNoise(p=0.5),
            RandomBrightnessContrast(p=0.5),
            RandomGamma(p=0.5),
        ], p=0.3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_strong_augmentation(img_size=512):
    """增强版的数据增强，用于解决欠拟合"""
    return Compose([
        RandomRotate90(p=0.6),
        Resize(img_size, img_size),
        HorizontalFlip(p=0.6),
        VerticalFlip(p=0.6),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=60, p=0.6),
        OneOf([
            ElasticTransform(alpha=150, sigma=120 * 0.07, alpha_affine=120 * 0.05, p=0.6),
            GridDistortion(num_steps=7, distort_limit=0.4, p=0.6),
            OpticalDistortion(distort_limit=1.3, shift_limit=0.6, p=0.6),
        ], p=0.4),
        OneOf([
            GaussNoise(var_limit=(10.0, 50.0), p=0.6),
            RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            RandomGamma(gamma_limit=(80, 120), p=0.6),
            Blur(blur_limit=5, p=0.3),
        ], p=0.5),
        CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=2, min_height=8, min_width=8, p=0.3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
