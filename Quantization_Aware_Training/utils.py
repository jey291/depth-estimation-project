# 辅助函数（预处理等）

import cv2
import numpy as np
import torch

import torch
import torch.nn.functional as F


def compute_loss(pred, target):
    """
    深度估计损失函数
    pred: (B, H, W) 或 (B, 1, H, W)
    target: (B, 1, H, W)
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # L1 损失（SiLog 损失可选）
    return F.l1_loss(pred, target)


def save_checkpoint(model, optimizer, epoch, path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)