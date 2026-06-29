import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import time
from datetime import datetime
from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str,default='test_resourse/video/Video Project 2.mp4')#加一个强制路径
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        #'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
       # 'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
       # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    

    # ========== 获取模型文件大小（用于对比） ==========
    model_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    model_size_bytes = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    model_size_mb = model_size_bytes / (1024 * 1024)

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(model_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)
    print("输出目录实际位置:", os.path.abspath(args.outdir))

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # ========== 统计变量 ==========
    all_inference_times = []  # 存储每帧推理时间（秒）
    total_frames = 0

    for k, filename in enumerate(filenames):
        print(f'Progress {k + 1}/{len(filenames)}: {filename}')

        raw_video = cv2.VideoCapture(filename)
        if not raw_video.isOpened():
            print(f"❌ 无法打开视频: {filename}")
            continue

        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        total_video_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频信息: {frame_width}x{frame_height}, {frame_rate} fps, 总帧数: {total_video_frames}")

        if args.pred_only:
            output_width = frame_width
        else:
            output_width = frame_width * 2 + margin_width

        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

        frame_count = 0
        # ========== 每帧推理时间统计（仅模型推理部分） ==========
        inference_times = []

        print("开始处理视频...")
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                print(f"⚠️ 在第 {frame_count} 帧读取失败，退出循环")
                break

            frame_count += 1

            # ========== 计时：仅模型推理 ==========
            start_time = time.perf_counter()
            depth = depth_anything.infer_image(raw_frame, args.input_size)
            inference_time = time.perf_counter() - start_time
            inference_times.append(inference_time)
            all_inference_times.append(inference_time)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            if args.pred_only:
                out.write(depth)
            else:
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth])
                out.write(combined_frame)

            # 每30帧打印进度
            if frame_count % 30 == 0:
                avg_time = np.mean(inference_times[-30:]) if len(inference_times) >= 30 else np.mean(inference_times)
                print(f"已处理 {frame_count} 帧, 最近30帧平均推理时间: {avg_time:.4f}s")

        raw_video.release()
        out.release()

        # ========== 当前视频统计 ==========
        print(f"✅ 当前视频处理完成，共 {frame_count} 帧")
        total_frames += frame_count

    # ============================================================
    # ========== 最终统计报告 ======================================
    # ============================================================
    if len(all_inference_times) == 0:
        print("❌ 没有处理任何帧，无法生成统计报告")
        exit()

    total_time = sum(all_inference_times)
    avg_time = np.mean(all_inference_times)
    std_time = np.std(all_inference_times)
    min_time = np.min(all_inference_times)
    max_time = np.max(all_inference_times)
    fps_equivalent = total_frames / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print("📊 PyTorch 模型推理性能报告")
    print("=" * 60)
    print(f"模型文件: {model_path}")
    print(f"模型大小: {model_size_mb:.2f} MB")
    print(f"设备: {DEVICE.upper()}")
    print(f"输入尺寸: {args.input_size}x{args.input_size}")
    print(f"视频总帧数: {total_frames}")
    print("\n--- 推理时间统计 ---")
    print(f"总推理时间: {total_time:.4f} 秒")
    print(f"平均每帧推理时间: {avg_time:.4f} 秒 ({avg_time * 1000:.2f} ms)")
    print(f"标准差: {std_time:.4f} 秒")
    print(f"最小/最大每帧时间: {min_time:.4f} / {max_time:.4f} 秒")
    print(f"等效 FPS: {fps_equivalent:.2f} 帧/秒")
    print("=" * 60)

    # ========== 保存报告到文件 ==========
    report_path = 'performance_report_pytorch.txt'
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write("PyTorch 模型性能报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"模型文件: {model_path}\n")
        f.write(f"模型大小: {model_size_mb:.2f} MB\n")
        f.write(f"设备: {DEVICE.upper()}\n")
        f.write(f"输入尺寸: {args.input_size}x{args.input_size}\n")
        f.write(f"总帧数: {total_frames}\n")
        f.write("\n--- 推理时间统计 ---\n")
        f.write(f"总推理时间: {total_time:.4f} 秒\n")
        f.write(f"平均每帧推理时间: {avg_time:.4f} 秒 ({avg_time * 1000:.2f} ms)\n")
        f.write(f"标准差: {std_time:.4f} 秒\n")
        f.write(f"最小/最大每帧时间: {min_time:.4f} / {max_time:.4f} 秒\n")
        f.write(f"等效 FPS: {fps_equivalent:.2f} 帧/秒\n")

