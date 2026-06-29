import cv2
import sys
import os

# 将项目根目录添加到 Python 路径（确保能导入 navigation 包）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from navigation.depth_estimator import DepthEstimator
from navigation.obstacle_avoidance import ObstacleAvoidance
from navigation.config import MODEL_XML, INPUT_SIZE, DEVICE, SAFETY_DISTANCE, TURN_THRESHOLD
###
def main():
    # 1. 初始化深度估计器和避障决策器
    print("正在加载深度估计模型...")
    depth_estimator = DepthEstimator()
    avoidance = ObstacleAvoidance(
        safety_distance=SAFETY_DISTANCE,
        turn_threshold=TURN_THRESHOLD
    )
    print("模型加载完成。")

    # 2. 打开 USB 摄像头（索引 0，若不行可改为 1）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查连接或索引")
        return

    print("✅ 摄像头已打开，按 'q' 退出")
    print("开始实时测试...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取帧失败")
            break

        # 可调整大小以加快速度
        frame = cv2.resize(frame, (640, 480))

        # 3. 深度估计
        depth = depth_estimator.infer(frame)

        # 4. 避障决策
        action, info = avoidance.get_action(depth)

        # 5. 打印决策到终端
        print(f"动作: {action:8} | 中心深度: {info['center_depth']:.3f}m | "
              f"左: {info['left_depth']:.3f}m | 右: {info['right_depth']:.3f}m")

        # 6. 显示原图 + 深度图 + 决策文字
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

        combined = cv2.hconcat([frame, depth_color])
        cv2.putText(combined, f"Action: {action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, f"Center depth: {info['center_depth']:.2f}m", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Depth & Avoidance Test', combined)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("测试结束。")

if __name__ == '__main__':
    main()