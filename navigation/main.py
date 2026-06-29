# 主循环逻辑
# navigation/main.py
import cv2
import time
import numpy as np
from .config import *
from .drone_interface import DroneInterface
from .depth_estimator import DepthEstimator
from .obstacle_avoidance import ObstacleAvoidance
# from .drone_tello import TelloDrone
# from .drone_pioneer import PioneerDrone  # 未来实现

class NavigationController:
    def __init__(self, drone: DroneInterface):
        self.drone = drone
        self.depth_estimator = DepthEstimator()
        self.avoidance = ObstacleAvoidance(
            safety_distance=SAFETY_DISTANCE,
            turn_threshold=TURN_THRESHOLD
        )
        self.running = False
        self.video_writer = None
        if SAVE_DEBUG_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(DEBUG_VIDEO_PATH, fourcc, 15, (640, 480))

    def run(self):
        """主循环"""
        self.drone.connect()
        self.drone.takeoff()
        self.running = True
        print("开始自主导航避障... 按 'q' 终止")

        try:
            while self.running:
                # 1. 获取图像
                frame = self.drone.get_frame()
                if frame is None:
                    continue
                # 可调整大小以加快处理
                frame_resized = cv2.resize(frame, (640, 480))

                # 2. 深度估计
                depth = self.depth_estimator.infer(frame_resized)

                # 3. 避障决策
                action, info = self.avoidance.get_action(depth)

                # 4. 执行动作
                self._execute_action(action)

                # 5. 显示调试信息
                self._display_debug(frame_resized, depth, action, info)

                # 6. 保存视频
                if self.video_writer is not None:
                    self.video_writer.write(self.debug_frame)

                # 按键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("用户中断")
        finally:
            self.stop()

    def _execute_action(self, action):
        """根据动作名称调用无人机接口"""
        if action == 'forward':
            self.drone.move_forward(FORWARD_SPEED)
        elif action == 'backward':
            self.drone.move_backward(FORWARD_SPEED * 0.5)
        elif action == 'turn_left':
            self.drone.turn_left(TURN_SPEED)
            # 短暂转向后悬停，防止过冲
            time.sleep(0.5)
            self.drone.hover()
        elif action == 'turn_right':
            self.drone.turn_right(TURN_SPEED)
            time.sleep(0.5)
            self.drone.hover()
        elif action == 'hover':
            self.drone.hover()
        else:
            self.drone.hover()

    def _display_debug(self, frame, depth, action, info):
        """显示原图、深度图、决策信息"""
        # 深度图彩色化
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        # 并排显示
        combined = np.hstack([frame, depth_color])
        # 添加文字
        cv2.putText(combined, f"Action: {action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(combined, f"Center Depth: {info.get('center_depth',0):.2f}m", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        self.debug_frame = combined
        cv2.imshow('Navigation', combined)

    def stop(self):
        self.running = False
        self.drone.hover()
        self.drone.land()
        self.drone.disconnect()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        print("导航已停止")