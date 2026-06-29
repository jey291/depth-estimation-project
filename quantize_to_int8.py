import nncf
import openvino as ov
import cv2
import glob
##########是量化脚本，轻量化核心脚本，它使用 NNCF（神经网络压缩框架） 对模型进行 INT8 量化，将模型权重从 32 位浮点数压缩为 8 位整数。
# 加载 OpenVINO 模型
# core = ov.Core()
# ov_model = core.read_model('depth_anything_v2_vits.xml')#FP32 模型，这是量化的起点，也是你对比轻量化效果的基准，convert_to_ov.py 将 PyTorch 模型（.pth）转换为 OpenVINO 格式（.xml + .bin）
#
# # 准备校准数据加载器
# def calibration_data():
#     for img_path in glob.glob('calibration_data/*.jpg'):
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (518, 518))
#         img = img.transpose(2, 0, 1)  # HWC -> CHW
#         img = img.astype('float32') / 255.0
#         yield {0: img}  # 输入节点索引
#
# # 量化
# quantized_model = nncf.quantize(
#     ov_model,
#     calibration_dataset=nncf.Dataset(calibration_data_loader()),
#     preset=nncf.QuantizationPreset.MIXED,
# )
#
# # 保存量化模型
# ov.serialize(quantized_model, 'checkpoints/depth_anything_v2_vits_int8.xml')#INT8 模型（使用 NNCF 对 FP32 模型进行 INT8 量化）
# print("✅ INT8 量化完成！")
import os
import cv2
import numpy as np
import openvino as ov
import nncf
from typing import Iterable, Dict

# ==================== 配置 ====================
INPUT_MODEL_XML = r'C:\Users\11650\PycharmProjects\PythonProject1\depth_anything_v2_vits.xml'
CALIBRATION_DATA_DIR = r'C:\Users\11650\PycharmProjects\PythonProject1\calibration_data'
OUTPUT_INT8_XML = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\checkpoints\depth_anything_v2_vits_int8.xml'
INPUT_SIZE = 518

# ==================== 1. 加载 OpenVINO 模型 ====================
core = ov.Core()
model = core.read_model(INPUT_MODEL_XML)
print(f"✅ 加载 OpenVINO 模型: {INPUT_MODEL_XML}")


# ==================== 2. 准备校准数据加载器 ====================
def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    h, w = img.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    pad_h = (INPUT_SIZE - new_h) // 2
    pad_w = (INPUT_SIZE - new_w) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, INPUT_SIZE - new_h - pad_h,
                                pad_w, INPUT_SIZE - new_w - pad_w,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

    image = padded.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image


def calibration_data_loader():
    image_files = [f for f in os.listdir(CALIBRATION_DATA_DIR)
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise ValueError(f"校准数据目录为空: {CALIBRATION_DATA_DIR}")

    print(f"📸 找到 {len(image_files)} 张校准图片")
    for img_file in image_files:
        img_path = os.path.join(CALIBRATION_DATA_DIR, img_file)
        try:
            input_tensor = preprocess_image(img_path)
            # 使用输入节点名称作为键（常见为 'input' 或 'x'）
            yield {model.input(0).get_any_name(): input_tensor}
        except Exception as e:
            print(f"⚠️ 跳过图片 {img_file}: {e}")


# ==================== 3. 运行 NNCF INT8 量化 ====================
print("🔄 开始 INT8 量化...")
calibration_dataset = nncf.Dataset(calibration_data_loader())  # 注意括号！

quantized_model = nncf.quantize(
    model,
    calibration_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    subset_size=len(os.listdir(CALIBRATION_DATA_DIR)),
)

print("✅ 量化完成！")

# ==================== 4. 保存 INT8 模型 ====================
ov.serialize(quantized_model, OUTPUT_INT8_XML)
print(f"✅ INT8 模型已保存: {OUTPUT_INT8_XML}")

# ==================== 5. 对比模型大小 ====================
fp32_bin = INPUT_MODEL_XML.replace('.xml', '.bin')
int8_bin = OUTPUT_INT8_XML.replace('.xml', '.bin')
fp32_size = os.path.getsize(fp32_bin) / (1024 * 1024)
int8_size = os.path.getsize(int8_bin) / (1024 * 1024)
print(f"\n📊 模型大小对比:")
print(f"   FP32 模型: {fp32_size:.2f} MB")
print(f"   INT8 模型: {int8_size:.2f} MB")
print(f"   压缩比: {fp32_size / int8_size:.2f}x")