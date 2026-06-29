# 主训练脚本

import sys
import os

# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# import torch
# import torch.optim as optim
# import nncf  # 重要：必须在 torch 之后导入
# from nncf import NNCFConfig,create_compressed_model, register_default_init_args
# # from nncf.torch import create_compressed_model, register_default_init_args

import torch
import torch.optim as optim
import nncf  # 注意：torch 必须在 nncf 之前导入


from depth_anything_v2.dpt import DepthAnythingV2
from Quantization_Aware_Training.dataset import get_train_loader
from Quantization_Aware_Training.config import *
from Quantization_Aware_Training.utils import compute_loss, save_checkpoint
import openvino as ov
import numpy as np

def main():
    print(f"使用设备: {DEVICE}")

    # 1. 加载你的原始 FP32 模型
    print("加载 FP32 预训练模型...")
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }
    model = DepthAnythingV2(**model_configs['vits'])
    state_dict = torch.load(PRETRAINED_MODEL, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()  # 注意：nncf.quantize() 要求模型处于 eval 模式
    print("✅ 模型加载成功")

    # 2. 准备数据加载器
    print("加载训练数据...")
    train_loader = get_train_loader(DATA_ROOT, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)
    # 注意：nncf.quantize() 需要一个 nncf.Dataset 对象
    # 这里将 train_loader 或一个简单的数据列表包装成 nncf.Dataset
    calibration_dataset = nncf.Dataset(train_loader, lambda data: data[0])
    print(f"✅ 数据加载完成")

    # 3. 进行训练后量化 (PTQ)
    print("应用训练后量化 (PTQ)...")
    # 这个 quantized_model 是已经应用了 INT8 量化的模型
    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED,  # 或 PERFORMANCE
        subset_size=50,  # 从数据集中取多少样本用于校准   ##########(往上调)############
    )
    quantized_model = quantized_model.to(DEVICE)
    quantized_model.train()  # 切换回训练模式，准备进行 QAT 微调
    print("✅ PTQ 完成，模型准备就绪")

    # 4. 开始微调（这步就是 Quantization-Aware Training）
    print(f"\n开始 QAT 微调 ({EPOCHS} epochs, 学习率: {LEARNING_RATE})")
    optimizer = optim.Adam(quantized_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, depths) in enumerate(train_loader):
            images = images.to(DEVICE)
            depths = depths.to(DEVICE)

            optimizer.zero_grad()

            # 前向传播（在微调阶段，模型会模拟量化误差）
            outputs = quantized_model(images)
            if outputs.dim() == 4:
                outputs = outputs.squeeze(1)

            loss = compute_loss(outputs, depths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.6f}")

    # 5. 保存模型
    torch.save(quantized_model.state_dict(), os.path.join(OUTPUT_DIR, 'qat_model_final.pth'))
    print(f"\n✅ QAT 模型已保存: {OUTPUT_DIR}/qat_model_final.pth")

    # 6. 导出为 OpenVINO IR
    print("导出为 OpenVINO IR 格式...")
    quantized_model.eval()
    model_cpu = quantized_model.cpu()
    example_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])

    ov_model = ov.convert_model(model_cpu, example_input=example_input)
    ov.save_model(ov_model, os.path.join(OUTPUT_DIR, 'depth_anything_vits_int8_qat.xml'))
    print(f"✅ OpenVINO IR 已保存: {OUTPUT_DIR}/depth_anything_vits_int8_qat.xml")

if __name__ == "__main__":
    main()