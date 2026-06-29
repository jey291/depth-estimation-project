# navigation/config.py
import os

# ========== 模型路径 ==========
# 选择你要使用的模型（PTQ 或 QAT）
MODEL_XML = r'C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\checkpoints\depth_anything_v2_vits_int8.xml'
# 如果 QAT 训练成功，切换为 QAT 模型
# MODEL_XML = r'C:\Users\11650\PycharmProjects\PythonProject1\Quantization_Aware_Training\qat_output\depth_anything_vits_int8_qat.xml'

INPUT_SIZE = 518          # 模型输入尺寸
DEVICE = 'CPU'            # 推理设备

# ========== 避障参数 ==========
SAFETY_DISTANCE = 0.5     # 安全距离（米），低于此值触发避障
TURN_THRESHOLD = 0.3      # 左右转向判断阈值
FORWARD_SPEED = 0.3       # 前进速度（m/s）
TURN_SPEED = 0.5          # 旋转速度（rad/s）

# ========== 无人机选择 ==========
# 可选 'tello', 'pioneer', 'simulator'
DRONE_TYPE = 'simulator'   # 目前无真机，先用模拟器测试

# ========== 日志与调试 ==========
SAVE_DEBUG_VIDEO = True    # 是否保存调试视频
DEBUG_VIDEO_PATH = './debug_navigation.avi'