import cv2
import numpy as np
import torch
import openvino as ov
import os
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from depth_anything_v2.dpt import DepthAnythingV2
from datetime import datetime

# ==================== 配置（请根据你的路径修改） ====================
VIDEO_PATH = r'Depth-Anything-V2/test_resourse/video/Video Project 2.mp4'
FP32_MODEL_PATH = r'Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'
INT8_MODEL_XML = r'Depth-Anything-V2/checkpoints/depth_anything_v2_vits_int8.xml'
OUTPUT_DIR = r'comparison_video'          # 存放深度图视频
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_SIZE = 518
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 是否保存深度图视频（纯深度图，不拼接原图）
SAVE_DEPTH_VIDEOS = True
DEPTH_VIDEO_FPS = 30

# ==================== 加载模型 ====================
print("加载 PyTorch FP32 模型...")
model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
fp32_model = DepthAnythingV2(**model_configs['vits'])
fp32_model.load_state_dict(torch.load(FP32_MODEL_PATH, map_location='cpu'))
fp32_model = fp32_model.to(DEVICE).eval()

print("加载 OpenVINO INT8 模型...")
core = ov.Core()
int8_model = core.read_model(INT8_MODEL_XML)
int8_model.reshape({int8_model.input(0): [1, 3, 518, 518]})
int8_compiled = core.compile_model(int8_model, 'CPU')
int8_input = int8_compiled.input(0)
int8_output = int8_compiled.output(0)

# ==================== 预处理 / 后处理函数 ====================
def preprocess_frame(frame, target_size=518):
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, target_size - new_h - pad_h,
                                pad_w, target_size - new_w - pad_w,
                                cv2.BORDER_CONSTANT, value=[0,0,0])
    image = padded.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = image.transpose(2,0,1)
    image = np.expand_dims(image, axis=0)
    return image, (h, w), (pad_h, pad_w, new_h, new_w)

def postprocess_depth(depth, orig_shape, pad_info):
    h, w = orig_shape
    pad_h, pad_w, new_h, new_w = pad_info
    cropped = depth[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized

def depth_to_uint8(depth):
    """将深度图归一化到0-255并转换为uint8"""
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return depth_norm

# ==================== 打开视频 ====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频信息: {width}x{height}, {fps} fps, 总帧数: {total_frames}")

# ==================== 准备输出视频（纯深度图） ====================
if SAVE_DEPTH_VIDEOS:
    fp32_video_writer = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, 'depth_fp32.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), DEPTH_VIDEO_FPS, (width, height), False
    )
    int8_video_writer = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, 'depth_int8.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), DEPTH_VIDEO_FPS, (width, height), False
    )

# ==================== 逐帧处理 ====================
fp32_times = []
int8_times = []
all_metrics = []  # 每帧的指标 [mae, rmse, psnr, ssim, corr]

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 预处理
    input_tensor, orig_shape, pad_info = preprocess_frame(frame, INPUT_SIZE)

    # ---------- FP32 推理 ----------
    start = time.perf_counter()
    input_torch = torch.from_numpy(input_tensor).to(DEVICE)
    with torch.no_grad():
        depth_fp32 = fp32_model(input_torch)
    fp32_times.append(time.perf_counter() - start)
    depth_fp32 = depth_fp32.squeeze().cpu().numpy()
    depth_fp32 = postprocess_depth(depth_fp32, orig_shape, pad_info)

    # ---------- INT8 推理 ----------
    start = time.perf_counter()
    result = int8_compiled([input_tensor])[int8_output]
    int8_times.append(time.perf_counter() - start)
    depth_int8 = result.squeeze()
    depth_int8 = postprocess_depth(depth_int8, orig_shape, pad_info)

    # ---------- 计算指标（归一化到[0,1]） ----------
    d1 = (depth_fp32 - depth_fp32.min()) / (depth_fp32.max() - depth_fp32.min() + 1e-8)
    d2 = (depth_int8 - depth_int8.min()) / (depth_int8.max() - depth_int8.min() + 1e-8)

    mae = np.mean(np.abs(d1 - d2))
    rmse = np.sqrt(np.mean((d1 - d2)**2))
    psnr_val = psnr(d1, d2, data_range=1.0)
    ssim_val = ssim(d1, d2, data_range=1.0)
    corr = np.corrcoef(d1.flatten(), d2.flatten())[0,1]
    all_metrics.append([mae, rmse, psnr_val, ssim_val, corr])

    # ---------- 保存深度图视频（灰度） ----------
    if SAVE_DEPTH_VIDEOS:
        fp32_gray = depth_to_uint8(depth_fp32)
        int8_gray = depth_to_uint8(depth_int8)
        fp32_video_writer.write(fp32_gray)
        int8_video_writer.write(int8_gray)

    # 进度显示
    if frame_count % 30 == 0:
        print(f"已处理 {frame_count} 帧")

# 释放资源
cap.release()
if SAVE_DEPTH_VIDEOS:
    fp32_video_writer.release()
    int8_video_writer.release()

# ==================== 统计所有帧的指标 ====================
metrics_arr = np.array(all_metrics)  # shape: (N, 5)
mean_metrics = np.mean(metrics_arr, axis=0)
std_metrics = np.std(metrics_arr, axis=0)
min_metrics = np.min(metrics_arr, axis=0)
max_metrics = np.max(metrics_arr, axis=0)

# 速度统计
avg_fp32_time = np.mean(fp32_times)
avg_int8_time = np.mean(int8_times)
speedup = avg_fp32_time / avg_int8_time

# ==================== 打印报告 ====================
print("\n" + "="*60)
print("📊 模型对比报告")
print("="*60)
print(f"总帧数: {frame_count}")
print(f"FP32 平均推理时间: {avg_fp32_time:.4f}s")
print(f"INT8 平均推理时间: {avg_int8_time:.4f}s")
print(f"速度提升: {speedup:.2f}x")
print("\n--- 深度图相似度指标（平均值 ± 标准差）---")
print(f"MAE  : {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}    (min: {min_metrics[0]:.4f}, max: {max_metrics[0]:.4f})")
print(f"RMSE : {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}    (min: {min_metrics[1]:.4f}, max: {max_metrics[1]:.4f})")
print(f"PSNR : {mean_metrics[2]:.2f} ± {std_metrics[2]:.2f} dB (min: {min_metrics[2]:.2f}, max: {max_metrics[2]:.2f})")
print(f"SSIM : {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}    (min: {min_metrics[3]:.4f}, max: {max_metrics[3]:.4f})")
print(f"Corr : {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}    (min: {min_metrics[4]:.4f}, max: {max_metrics[4]:.4f})")
print("="*60)

# ==================== 追加保存到 performance_report_pytorch.txt ====================
report_file = 'performance_report_pytorch.txt'
with open(report_file, 'a', encoding='utf-8') as f:
    f.write("\n" + "="*60 + "\n")
    f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("模型对比报告 (FP32 vs INT8)\n")
    f.write("="*60 + "\n")
    f.write(f"视频: {VIDEO_PATH}\n")
    f.write(f"总帧数: {frame_count}\n")
    f.write(f"FP32 平均推理时间: {avg_fp32_time:.4f}s\n")
    f.write(f"INT8 平均推理时间: {avg_int8_time:.4f}s\n")
    f.write(f"速度提升: {speedup:.2f}x\n")
    f.write("\n--- 深度图相似度指标（平均值 ± 标准差）---\n")
    f.write(f"MAE  : {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}\n")
    f.write(f"RMSE : {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}\n")
    f.write(f"PSNR : {mean_metrics[2]:.2f} ± {std_metrics[2]:.2f} dB\n")
    f.write(f"SSIM : {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}\n")
    f.write(f"Corr : {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}\n")
    f.write("="*60 + "\n")

print(f"报告已追加至 {report_file}")
print(f"深度图视频保存至 {OUTPUT_DIR} 目录")