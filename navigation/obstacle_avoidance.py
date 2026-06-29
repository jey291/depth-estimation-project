# 避障决策算法（复用）
# navigation/obstacle_avoidance.py
import numpy as np

class ObstacleAvoidance:
    def __init__(self, safety_distance=0.5, turn_threshold=0.3,
                 center_region_ratio=0.3, left_region_ratio=0.4):
        self.safety_distance = safety_distance
        self.turn_threshold = turn_threshold
        self.center_region_ratio = center_region_ratio
        self.left_region_ratio = left_region_ratio

    def get_action(self, depth_map):
        """
        输入深度图 (H, W)，返回动作名称和附加信息
        returns: (action, info_dict)
        action: 'forward', 'turn_left', 'turn_right', 'hover', 'backward'
        info_dict: 包含 center_depth, left_depth, right_depth 等，用于调试
        """
        h, w = depth_map.shape
        info = {}

        # 1. 中心区域深度
        center_h = int(h * self.center_region_ratio)
        center_w = int(w * self.center_region_ratio)
        center_crop = depth_map[h//2 - center_h//2 : h//2 + center_h//2,
                                w//2 - center_w//2 : w//2 + center_w//2]
        center_depth = np.mean(center_crop)
        info['center_depth'] = center_depth

        # 2. 左右区域深度
        left_crop = depth_map[:, :int(w * self.left_region_ratio)]
        right_crop = depth_map[:, int(w * (1 - self.left_region_ratio)):]
        left_depth = np.mean(left_crop)
        right_depth = np.mean(right_crop)
        info['left_depth'] = left_depth
        info['right_depth'] = right_depth

        # 3. 决策
        if center_depth < self.safety_distance:
            # 前方危险，比较左右
            if abs(left_depth - right_depth) < self.turn_threshold:
                return 'backward', info
            elif left_depth > right_depth:
                return 'turn_left', info
            else:
                return 'turn_right', info
        else:
            # 前方安全，但可微调方向
            if abs(left_depth - right_depth) > self.turn_threshold * 1.5:
                if left_depth > right_depth:
                    return 'turn_left', info
                else:
                    return 'turn_right', info
            else:
                return 'forward', info