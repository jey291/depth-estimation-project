# 数据加载器（NYUv2/KITTI）


import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import torch

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import h5py


class NYUDepthDataset(Dataset):
    def __init__(self, data_root, split='train', input_size=(518, 518)):
        self.data_root = data_root
        self.input_size = input_size

        # 尝试加载 .h5 或 .mat 文件
        h5_path = os.path.join(data_root, 'nyu_depth_v2.h5')
        mat_path = os.path.join(data_root, 'nyu_depth_v2_labeled.mat')

        if os.path.exists(h5_path):
            file_path = h5_path
        elif os.path.exists(mat_path):
            file_path = mat_path
        else:
            raise FileNotFoundError(f"未找到数据集文件: {data_root}")

        with h5py.File(file_path, 'r') as f:
            if split == 'train':
                # 根据实际数据集结构调整 key 名称
                self.images = f['images'][:] if 'images' in f else f['rgb'][:]
                self.depths = f['depths'][:] if 'depths' in f else f['depth'][:]
            else:
                self.images = f['images_test'][:] if 'images_test' in f else f['rgb_test'][:]
                self.depths = f['depths_test'][:] if 'depths_test' in f else f['depth_test'][:]

        # 转换为 numpy 并调整维度
        if self.images.shape[1] == 3:  # (N, C, H, W) -> (N, H, W, C)
            self.images = np.transpose(self.images, (0, 2, 3, 1))
        if self.depths.ndim == 4:  # (N, 1, H, W) -> (N, H, W)
            self.depths = self.depths.squeeze(1)

        print(f"加载 {len(self.images)} 张 {split} 集图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)  # (H, W, 3)
        depth = self.depths[idx].astype(np.float32)  # (H, W)

        # BGR 格式（Depth Anything 使用 BGR）
        img = img[:, :, ::-1]  # RGB -> BGR

        # 归一化到 [0, 1]
        img = img / 255.0

        # 调整尺寸
        h, w = img.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        depth_resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 填充到目标尺寸
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        img_padded = np.pad(img_resized, ((pad_h, target_h - new_h - pad_h),
                                          (pad_w, target_w - new_w - pad_w),
                                          (0, 0)), mode='constant')
        depth_padded = np.pad(depth_resized, ((pad_h, target_h - new_h - pad_h),
                                              (pad_w, target_w - new_w - pad_w)),
                              mode='constant')

        # ImageNet 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_padded = (img_padded - mean) / std

        # 转为 Tensor (CHW)
        img_tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).float()
        depth_tensor = torch.from_numpy(depth_padded).float().unsqueeze(0)  # (1, H, W)

        return img_tensor, depth_tensor


def get_train_loader(data_root, batch_size=4, input_size=(518, 518), num_workers=0):
    dataset = NYUDepthDataset(data_root, split='train', input_size=input_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    return loader