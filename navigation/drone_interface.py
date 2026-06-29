# navigation/drone_interface.py
from abc import ABC, abstractmethod
import numpy as np

class DroneInterface(ABC):
    """无人机控制抽象基类，定义所有必须实现的方法"""

    @abstractmethod
    def connect(self):
        """连接无人机"""
        pass

    @abstractmethod
    def takeoff(self):
        """起飞"""
        pass

    @abstractmethod
    def land(self):
        """降落"""
        pass

    @abstractmethod
    def move_forward(self, speed: float):
        """以给定速度前进 (m/s)"""
        pass

    @abstractmethod
    def move_backward(self, speed: float):
        """后退"""
        pass

    @abstractmethod
    def turn_left(self, angular_speed: float):
        """左转 (rad/s)"""
        pass

    @abstractmethod
    def turn_right(self, angular_speed: float):
        """右转"""
        pass

    @abstractmethod
    def hover(self):
        """悬停 (保持当前位置)"""
        pass

    @abstractmethod
    def get_battery(self) -> int:
        """获取电量百分比"""
        pass

    @abstractmethod
    def get_pose(self) -> tuple:
        """返回 (x, y, z, yaw) 当前位置和偏航角"""
        pass

    @abstractmethod
    def get_frame(self):
        """获取当前摄像头帧 (numpy array BGR)"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接，释放资源"""
        pass