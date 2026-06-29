# 训练配置


import os
import torch
# from nncf import torch

# ========== 数据路径 ==========
# NYUv2 数据集路径（包含 nyu_depth_v2_labeled.mat 或 .h5 文件）
DATA_ROOT = r'C:\Users\11650\PycharmProjects\PythonProject1\Quantization_Aware_Training\data'

# ========== 模型路径 ==========
# 原始 FP32 预训练模型（.pth 文件）
PRETRAINED_MODEL = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth'

# ========== 训练超参数 ==========
BATCH_SIZE = 2          # 根据 GPU 显存调整，无 GPU 可设为 2###################调成4##################
LEARNING_RATE = 1e-5    # QAT 必须使用极小学习率
EPOCHS = 5              # 微调 3-5 个 epoch 即可
INPUT_SIZE = (518, 518) # 模型输入尺寸

# ========== 设备 ==========
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 输出目录 ==========
OUTPUT_DIR = './qat_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)