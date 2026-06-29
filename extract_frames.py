# 抽帧脚本
import cv2
import os

cap = cv2.VideoCapture('Depth-Anything-V2/test_resourse/video/Video Project 2.mp4')#换成我的路径

os.makedirs('calibration_data', exist_ok=True)
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % 30 == 0:  # 每30帧抽一张
        cv2.imwrite(f'calibration_data/img_{saved_count}.jpg', frame)
        saved_count += 1
        if saved_count >= 50:  # 50~100张足够
            break
    frame_count += 1

cap.release()
print(f"✅ 准备了 {saved_count} 张校准图片")