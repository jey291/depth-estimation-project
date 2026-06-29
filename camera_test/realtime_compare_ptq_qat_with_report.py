import cv2
import numpy as np
import openvino as ov
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import datetime

# ==================== 配置 ====================
PTQ_MODEL_XML = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\checkpoints\depth_anything_v2_vits_int8.xml'
QAT_MODEL_XML = r'C:\Users\11650\PycharmProjects\PythonProject1\Quantization_Aware_Training\qat_output\depth_anything_vits_int8_qat.xml'
INPUT_SIZE = 518
DEVICE = 'CPU'
REPORT_FILE = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\performance_report_pytorch.txt'

# ==================== 预处理 / 后处理 ====================
def preprocess_frame(frame, target_size=INPUT_SIZE):
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, target_size - new_h - pad_h,
                                pad_w, target_size - new_w - pad_w,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image = padded.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image, (h, w), (pad_h, pad_w, new_h, new_w)

def postprocess_depth(depth, orig_shape, pad_info):
    h, w = orig_shape
    pad_h, pad_w, new_h, new_w = pad_info
    cropped = depth[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized

def depth_to_color(depth):
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

# ==================== 加载模型 ====================
print("加载 PTQ 模型...")
core = ov.Core()
ptq_model = core.read_model(PTQ_MODEL_XML)
ptq_model.reshape({ptq_model.input(0): [1, 3, INPUT_SIZE, INPUT_SIZE]})
ptq_compiled = core.compile_model(ptq_model, DEVICE)
ptq_output = ptq_compiled.output(0)

print("加载 QAT 模型...")
qat_model = core.read_model(QAT_MODEL_XML)
qat_model.reshape({qat_model.input(0): [1, 3, INPUT_SIZE, INPUT_SIZE]})
qat_compiled = core.compile_model(qat_model, DEVICE)
qat_output = qat_compiled.output(0)
print("✅ 模型加载完成")

# ==================== 打开摄像头 ====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()
print("✅ 摄像头已打开，按 'q' 退出")

# ==================== 统计变量 ====================
frame_count = 0
ptq_times = []
qat_times = []
mae_sum = 0.0
rmse_sum = 0.0
psnr_sum = 0.0
ssim_sum = 0.0
corr_sum = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 预处理
    input_tensor, orig_shape, pad_info = preprocess_frame(frame)

    # ---- PTQ 推理 ----
    start = time.perf_counter()
    result_ptq = ptq_compiled([input_tensor])[ptq_output]
    ptq_time = time.perf_counter() - start
    ptq_times.append(ptq_time)
    depth_ptq = postprocess_depth(result_ptq.squeeze(), orig_shape, pad_info)

    # ---- QAT 推理 ----
    start = time.perf_counter()
    result_qat = qat_compiled([input_tensor])[qat_output]
    qat_time = time.perf_counter() - start
    qat_times.append(qat_time)
    depth_qat = postprocess_depth(result_qat.squeeze(), orig_shape, pad_info)

    # ---- 计算深度图相似度指标（PTQ vs QAT） ----
    d1 = (depth_ptq - depth_ptq.min()) / (depth_ptq.max() - depth_ptq.min() + 1e-8)
    d2 = (depth_qat - depth_qat.min()) / (depth_qat.max() - depth_qat.min() + 1e-8)

    mae = np.mean(np.abs(d1 - d2))
    rmse = np.sqrt(np.mean((d1 - d2)**2))
    psnr_val = psnr(d1, d2, data_range=1.0)
    ssim_val = ssim(d1, d2, data_range=1.0)
    corr_val = np.corrcoef(d1.flatten(), d2.flatten())[0, 1]

    mae_sum += mae
    rmse_sum += rmse
    psnr_sum += psnr_val
    ssim_sum += ssim_val
    corr_sum += corr_val

    # ---- 显示 ----
    ptq_color = depth_to_color(depth_ptq)
    qat_color = depth_to_color(depth_qat)

    sep = np.ones((frame.shape[0], 10, 3), dtype=np.uint8) * 255
    display = np.hstack([frame, sep, ptq_color, sep, qat_color])

    cv2.putText(display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"PTQ FPS: {frame_count / sum(ptq_times):.2f}", (frame.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"QAT FPS: {frame_count / sum(qat_times):.2f}", (frame.shape[1]*2 + 20 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('PTQ vs QAT Real-time Depth', display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==================== 释放资源 ====================
cap.release()
cv2.destroyAllWindows()

# ==================== 计算汇总统计 ====================
if frame_count == 0:
    print("未处理任何帧，退出")
    exit()

avg_ptq_time = np.mean(ptq_times)
avg_qat_time = np.mean(qat_times)
ptq_fps = frame_count / sum(ptq_times)
qat_fps = frame_count / sum(qat_times)
speed_ratio = avg_ptq_time / avg_qat_time if avg_qat_time > 0 else 0

avg_mae = mae_sum / frame_count
avg_rmse = rmse_sum / frame_count
avg_psnr = psnr_sum / frame_count
avg_ssim = ssim_sum / frame_count
avg_corr = corr_sum / frame_count

# ==================== 追加写入报告文件 ====================
with open(REPORT_FILE, 'a', encoding='utf-8') as f:
    f.write("\n" + "="*60 + "\n")
    f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("实时测试报告 (PTQ vs QAT on USB Camera)\n")
    f.write("="*60 + "\n")
    f.write(f"处理帧数: {frame_count}\n")
    f.write(f"PTQ 平均推理时间: {avg_ptq_time:.4f} s, 等效 FPS: {ptq_fps:.2f}\n")
    f.write(f"QAT 平均推理时间: {avg_qat_time:.4f} s, 等效 FPS: {qat_fps:.2f}\n")
    f.write(f"速度比 (PTQ / QAT): {speed_ratio:.2f}x\n")
    f.write("\n--- 深度图相似度指标 (PTQ vs QAT, 平均值) ---\n")
    f.write(f"MAE  : {avg_mae:.4f}\n")
    f.write(f"RMSE : {avg_rmse:.4f}\n")
    f.write(f"PSNR : {avg_psnr:.2f} dB\n")
    f.write(f"SSIM : {avg_ssim:.4f}\n")
    f.write(f"Corr : {avg_corr:.4f}\n")
    f.write("="*60 + "\n")

print(f"\n✅ 报告已追加至: {REPORT_FILE}")
print("按任意键关闭窗口...")