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
    def __init__(self, image_dir, mask_dir, transform=None, is_train=True, img_size=512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.img_size = img_size
        self.images = []
        self.masks = []
        if not os.path.exists(image_dir):
            raise RuntimeError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise RuntimeError(f"Mask directory not found: {mask_dir}")
        for img_name in os.listdir(image_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(img_name)[0]
                possible_mask_names = [
                    img_name,
                    base_name + '_mask.png',
                    base_name + '.png'
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
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Cannot read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot read mask: {mask_path}")
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        mask = mask.astype(np.float32)
        mask = mask / 255.0
        mask = np.clip(mask, 0, 1)
        if self.transform:
            try:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            except Exception as e:
                print(f"Error applying data augmentation: {e}")
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                mask = torch.from_numpy(mask).float().unsqueeze(0)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        mask = torch.clamp(mask, 0, 1)
        return image, mask
    @staticmethod
    def get_transforms(is_train, img_size=512):
        if is_train:
            transforms = Compose([
                Resize(img_size, img_size),
                OneOf([
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    RandomRotate90(p=0.5),
                ], p=0.5),
                RandomBrightnessContrast(p=0.3),
                OneOf([
                    GaussNoise(p=0.5),
                    GaussianBlur(p=0.5),
                    MotionBlur(p=0.5),
                ], p=0.3),
                OneOf([
                    ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    GridDistortion(p=0.5),
                    OpticalDistortion(distort_limit=1.0, shift_limit=0.5, p=0.5),
                ], p=0.3),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            transforms = Compose([
                Resize(img_size, img_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        return transforms
def get_training_augmentation(img_size=512):
    return Compose([
        RandomRotate90(p=0.5),
        Resize(img_size, img_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        RandomBrightnessContrast(p=0.2),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
def get_strong_augmentation(img_size=512):
    return Compose([
        Resize(img_size, img_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        OneOf([
            MotionBlur(blur_limit=7, p=0.5),
            MedianBlur(blur_limit=7, p=0.5),
            GaussianBlur(blur_limit=7, p=0.5),
        ], p=0.3),
        CoarseDropout(max_holes=10, max_height=32, max_width=32, p=0.3),
        ElasticTransform(alpha=100, sigma=30, alpha_affine=10, p=0.3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])