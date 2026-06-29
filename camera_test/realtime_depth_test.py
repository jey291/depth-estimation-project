import cv2
import numpy as np
import openvino as ov
import time

# ==================== 配置区域（请修改为你的路径） ====================
# 选择你要测试的模型（二选一，或两个都测）
MODEL_XML = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\checkpoints\depth_anything_v2_vits_int8.xml'# INT8 模型
# MODEL_XML = r'Depth-Anything-V2/checkpoints/depth_anything_v2_vits.xml'# FP32 模型

INPUT_SIZE = 518   # 可以改为 320 或 256 以提速
DEVICE = 'CPU'     # 可改为 'GPU' 如果有

# ==================== 预处理与后处理函数 ====================
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

# ==================== 加载 OpenVINO 模型 ====================
print("加载模型...")
core = ov.Core()
model = core.read_model(MODEL_XML)
# 固定输入形状
model.reshape({model.input(0): [1, 3, INPUT_SIZE, INPUT_SIZE]})
compiled = core.compile_model(model, DEVICE)
output = compiled.output(0)
print("✅ 模型加载成功")

# ==================== 打开摄像头 ====================
cap = cv2.VideoCapture(1)  # 若不行，去掉 cv2.CAP_DSHOW
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()
print("✅ 摄像头已打开，按 'q' 退出")

# 可选：降低摄像头分辨率以提高帧率
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==================== 实时推理循环 ====================
frame_count = 0
total_time = 0.0
fps_display = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 读取帧失败")
        break

    frame_count += 1

    # 预处理
    input_tensor, orig_shape, pad_info = preprocess_frame(frame)

    # 推理计时
    start = time.perf_counter()
    result = compiled([input_tensor])[output]
    inference_time = time.perf_counter() - start
    total_time += inference_time

    # 后处理
    depth = result.squeeze()
    depth = postprocess_depth(depth, orig_shape, pad_info)

    # 深度图可视化
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    # 并排显示原图与深度图
    combined = np.hstack([frame, depth_color])
    cv2.imshow('Real-time Depth (INT8)', combined)

    # 打印帧率信息（每 30 帧）
    if frame_count % 30 == 0:
        avg_time = total_time / frame_count
        fps = 1 / avg_time
        print(f"帧 {frame_count}，平均推理时间 {avg_time*1000:.2f}ms，等效 FPS {fps:.2f}")

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==================== 清理 ====================
cap.release()
cv2.destroyAllWindows()
print("✅ 测试结束")


