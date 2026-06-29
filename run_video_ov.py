# #########用 OpenVINO 的运行时替换 PyTorch 模型加载
# import openvino as ov
# import cv2
# import time
#
# # 加载量化后的模型
# core = ov.Core()
# # 选择用 FP16 还是 INT8
# model_path = 'depth_anything_v2_vits.xml'  # 或 .xml
# compiled_model = core.compile_model(model_path, device_name='CPU')
#
# # 视频处理
# cap = cv2.VideoCapture('test_resourse/video/your_video.mp4')
# frame_count = 0
# total_time = 0
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 预处理
#     input_tensor = preprocess(frame)  # 转换为 CHW, 归一化等
#
#     # 推理（计时）
#     start = time.time()
#     result = compiled_model([input_tensor])[0]  # 推理
#     total_time += time.time() - start
#
#     # 后处理
#     depth = result.squeeze()  # 得到深度图
#     # ... 你的可视化代码 ...
#
#     frame_count += 1
#
# print(f"处理了 {frame_count} 帧，平均推理时间: {total_time / frame_count:.4f}s")
# print(f"等效 FPS: {frame_count / total_time:.2f}")


import cv2
import numpy as np
import openvino as ov
import time
import os
from datetime import datetime

# def preprocess_frame(frame, target_size=518):
#     """
#     对输入帧进行预处理，转换为模型要求的格式
#     - 保持宽高比，缩放到 target_size，边缘填充
#     - 归一化
#     - 转为 CHW 格式
#     """
#     h, w = frame.shape[:2]
#     # 1. 缩放（保持宽高比）
#     scale = target_size / max(h, w)
#     new_h, new_w = int(h * scale), int(w * scale)
#     resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#
#     # 2. 填充到 target_size x target_size（上下左右填充）
#     pad_h = (target_size - new_h) // 2
#     pad_w = (target_size - new_w) // 2
#     padded = cv2.copyMakeBorder(resized, pad_h, target_size - new_h - pad_h,
#                                 pad_w, target_size - new_w - pad_w,
#                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
#
#     # 3. 归一化：转为 float32，除以255，并减均值除方差
#     image = padded.astype(np.float32) / 255.0
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#     image = (image - mean) / std
#
#     # 4. 转为 CHW (HWC -> CHW)
#     image = image.transpose(2, 0, 1)  # (3, 518, 518)
#     # 5. 增加 batch 维度
#     image = np.expand_dims(image, axis=0)  # (1, 3, 518, 518)
#     return image, (h, w), (pad_h, pad_w, new_h, new_w)
#
#
# def postprocess_depth(depth, original_shape, pad_info):
#     """
#     将深度图恢复到原始图像尺寸
#     """
#     h, w = original_shape
#     pad_h, pad_w, new_h, new_w = pad_info
#
#     # 去掉填充部分
#     depth_cropped = depth[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
#     # 缩放到原始尺寸
#     depth_resized = cv2.resize(depth_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
#     return depth_resized
#
#
# def main():
#     # -------------------- 配置 --------------------
#     # 选择使用 FP16 还是 INT8 模型
#     # model_xml = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\checkpoints\depth_anything_v2_vits.xml'        # FP16
#     model_xml = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\checkpoints\depth_anything_v2_vits_int8.xml'  # INT8（如果已量化）
#
#     video_path = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\test_resourse\video\Video Project 2.mp4'
#     output_video_path = r'C:\Users\11650\PycharmProjects\PythonProject1\vis_video_depth_ov\output_ov.mp4'
#
#     # -------------------- 加载 OpenVINO 模型 --------------------
#     core = ov.Core()
#     model = core.read_model(model_xml)
#     # 获取输入层，并固定形状为 [1, 3, 518, 518]
#     input_layer = model.input(0)
#     model.reshape({input_layer: [1, 3, 518, 518]})
#
#     compiled_model = core.compile_model(model, device_name='CPU')  # 用 CPU 推理
#
#     # 重新获取输入/输出层（reshape后可能变化）
#     input_layer = compiled_model.input(0)
#     output_layer = compiled_model.output(0)
#     print("✅ OpenVINO 模型加载成功")
#     print(f"   输入形状: {input_layer.shape}")  # 现在可以安全打印
#     print(f"   输出形状: {output_layer.shape}")
#
#     # -------------------- 打开视频 --------------------
#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"❌ 无法打开视频: {video_path}")
#         return
#
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"视频信息: {width}x{height}, {fps} fps, 总帧数: {total_frames}")
#
#     # 输出视频编码器（原尺寸视频，显示深度图并排）
#     out_width = width * 2 + 50  # 原图 + 边距 + 深度图
#     out_height = height
#     out = cv2.VideoWriter(output_video_path,
#                           cv2.VideoWriter_fourcc(*'avc1'),
#                           fps, (out_width, out_height))
#     if not out.isOpened():
#         print("❌ 无法创建输出视频文件")
#         cap.release()
#         return
#
#     # -------------------- 处理视频 --------------------
#     frame_count = 0
#     total_infer_time = 0.0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 预处理
#         input_tensor, original_shape, pad_info = preprocess_frame(frame, target_size=518)
#
#         # 推理
#         start_time = time.time()
#         result = compiled_model([input_tensor])[output_layer]  # 形状 (1, 518, 518)
#         inference_time = time.time() - start_time
#         total_infer_time += inference_time
#
#         # 后处理
#         depth_map = result.squeeze()  # (518, 518)
#         depth_map = postprocess_depth(depth_map, original_shape, pad_info)
#
#         # 归一化到 0-255 用于显示
#         depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#         depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
#
#         # 拼接图像
#         split_region = np.ones((height, 50, 3), dtype=np.uint8) * 255
#         combined = np.hstack([frame, split_region, depth_color])
#
#         # 写入输出视频
#         out.write(combined)
#
#         frame_count += 1
#         if frame_count % 30 == 0:
#             print(f"已处理 {frame_count} 帧, 平均推理时间: {total_infer_time / frame_count:.4f}s")
#
#     # 释放资源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#
#     avg_time = total_infer_time / frame_count if frame_count > 0 else 0
#     print(f"\n✅ 处理完成！共 {frame_count} 帧，平均推理时间: {avg_time:.4f}s")
#     print(f"   等效 FPS: {1 / avg_time:.2f} (仅计算推理，不含前后处理)")
#     print(f"   输出视频保存至: {output_video_path}")
#
#
# if __name__ == "__main__":
#     main()




def preprocess_frame(frame, target_size=518):
    """
    对输入帧进行预处理，转换为模型要求的格式
    - 保持宽高比，缩放到 target_size，边缘填充
    - 归一化
    - 转为 CHW 格式
    """
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

def postprocess_depth(depth, original_shape, pad_info):
    h, w = original_shape
    pad_h, pad_w, new_h, new_w = pad_info
    depth_cropped = depth[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
    depth_resized = cv2.resize(depth_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth_resized

def main():
    # -------------------- 配置 --------------------
    model_xml = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\checkpoints\depth_anything_v2_vits_int8.xml'  # INT8 模型
    video_path = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\test_resourse\video\Video Project 2.mp4'
    output_video_path = r'C:\Users\11650\PycharmProjects\PythonProject1\vis_video_depth_ov\output_ov.mp4'

    # -------------------- 获取模型文件大小（权重文件 .bin）--------------------
    model_bin = model_xml.replace('.xml', '.bin')
    if os.path.exists(model_bin):
        model_size_bytes = os.path.getsize(model_bin)
        model_size_mb = model_size_bytes / (1024 * 1024)
    else:
        model_size_mb = 0.0
        print("⚠️ 未找到 .bin 文件，无法获取模型大小")

    # -------------------- 加载 OpenVINO 模型 --------------------
    core = ov.Core()
    model = core.read_model(model_xml)
    input_layer = model.input(0)
    model.reshape({input_layer: [1, 3, 518, 518]})
    compiled_model = core.compile_model(model, device_name='CPU')
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    print("✅ OpenVINO 模型加载成功")
    print(f"   输入形状: {input_layer.shape}")
    print(f"   输出形状: {output_layer.shape}")

    # -------------------- 打开视频 --------------------
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {width}x{height}, {fps} fps, 总帧数: {total_frames}")

    out_width = width * 2 + 50
    out_height = height
    out = cv2.VideoWriter(output_video_path,
                          cv2.VideoWriter_fourcc(*'avc1'),
                          fps, (out_width, out_height))
    if not out.isOpened():
        print("❌ 无法创建输出视频文件")
        cap.release()
        return

    # -------------------- 处理视频并收集推理时间 --------------------
    frame_count = 0
    inference_times = []  # 存储每帧推理时间（秒）

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor, original_shape, pad_info = preprocess_frame(frame, target_size=518)

        start_time = time.perf_counter()
        result = compiled_model([input_tensor])[output_layer]
        inference_time = time.perf_counter() - start_time
        inference_times.append(inference_time)

        depth_map = result.squeeze()
        depth_map = postprocess_depth(depth_map, original_shape, pad_info)

        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

        split_region = np.ones((height, 50, 3), dtype=np.uint8) * 255
        combined = np.hstack([frame, split_region, depth_color])
        out.write(combined)

        frame_count += 1
        if frame_count % 30 == 0:
            avg_time = np.mean(inference_times[-30:])
            print(f"已处理 {frame_count} 帧, 最近30帧平均推理时间: {avg_time:.4f}s")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # -------------------- 统计计算 --------------------
    if frame_count == 0:
        print("❌ 没有处理任何帧")
        return

    total_time = sum(inference_times)
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    fps_equivalent = frame_count / total_time if total_time > 0 else 0

    print("\n" + "="*60)
    print("📊 OpenVINO 模型推理性能报告")
    print("="*60)
    print(f"模型文件: {model_xml}")
    print(f"模型大小 (权重): {model_size_mb:.2f} MB")
    print(f"设备: CPU")
    print(f"输入尺寸: 518x518")
    print(f"总帧数: {frame_count}")
    print("\n--- 推理时间统计 ---")
    print(f"总推理时间: {total_time:.4f} 秒")
    print(f"平均每帧推理时间: {avg_time:.4f} 秒 ({avg_time*1000:.2f} ms)")
    print(f"标准差: {std_time:.4f} 秒")
    print(f"最小/最大每帧时间: {min_time:.4f} / {max_time:.4f} 秒")
    print(f"等效 FPS: {fps_equivalent:.2f} 帧/秒")
    print("="*60)

    # -------------------- 保存报告到文件 --------------------
    report_path = 'Depth-Anything-V2/performance_report_pytorch.txt'
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write("OpenVINO 模型性能报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"模型文件: {model_xml}\n")
        f.write(f"模型大小 (权重): {model_size_mb:.2f} MB\n")
        f.write(f"设备: CPU\n")
        f.write(f"输入尺寸: 518x518\n")
        f.write(f"总帧数: {frame_count}\n")
        f.write("\n--- 推理时间统计 ---\n")
        f.write(f"总推理时间: {total_time:.4f} 秒\n")
        f.write(f"平均每帧推理时间: {avg_time:.4f} 秒 ({avg_time * 1000:.2f} ms)\n")
        f.write(f"标准差: {std_time:.4f} 秒\n")
        f.write(f"最小/最大每帧时间: {min_time:.4f} / {max_time:.4f} 秒\n")
        f.write(f"等效 FPS: {fps_equivalent:.2f} 帧/秒\n")

if __name__ == "__main__":
    main()