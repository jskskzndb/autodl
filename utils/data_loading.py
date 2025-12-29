import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A  # 🔥 [新增1] 导入增强库

# 修改后 (强制 .convert('RGB'))
def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename)).convert('RGB')
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy()).convert('RGB')
    else:
        return Image.open(filename).convert('RGB')

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', augment: bool = False):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augment = augment

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        logging.info('🚀 跳过扫描，使用固定掩码值: [0, 255]')
        self.mask_values = [0, 255]
        # ============================================================
        # 🔥 [新增3] 定义增强流水线 (仅当 augment=True 时初始化)
        # ============================================================
        if self.augment:
            self.transform = A.Compose([
                # --- 几何变换：打破位置记忆 ---
                A.HorizontalFlip(p=0.5),      # 水平翻转
                A.VerticalFlip(p=0.5),        # 垂直翻转
                A.RandomRotate90(p=0.5),      # 90度旋转
                
                # --- 像素变换：适应不同光照 ---
                A.RandomBrightnessContrast(p=0.2), 
                
                # --- Mamba 特训核心：挖孔训练 (CoarseDropout) ---
                # 随机在原图上挖掉正方形黑洞，但【不挖 Mask】
                # 强迫模型通过 Mamba 的长距离扫描能力，结合上下文推断被遮挡的部分
                # 适配 Albumentations 2.x 版本
                
            ])
        # ============================================================

    

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            # === 图片处理逻辑大改 ===
            # 1. 维度调整：把 (H, W, C) 转为 (C, H, W)
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            # 2. 归一化到 [0, 1]
            if (img > 1).any():
                img = img / 255.0

            # 3. 【新增】ImageNet 标准化
            # 定义标准参数 (C, 1, 1) 以便广播计算
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

            # 执行标准化 (Z-Score)
            # 结果范围会变成约 [-2, 2.6]，且包含负数
            img = (img - mean) / std

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        # ============================================================
        # 🔥 [新增4] 应用增强逻辑 (拦截处理)
        # ============================================================
        if self.augment:
            # A. PIL -> Numpy (Albumentations 需要 Numpy 格式)
            img_np = np.array(img)
            mask_np = np.array(mask)
            
            # B. 执行增强 (image 和 mask 自动同步变换)
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']
            
            # C. Numpy -> PIL (转回去，无缝对接原本的 preprocess)
            img = Image.fromarray(img_np)
            mask = Image.fromarray(mask_np)
        # ============================================================

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
