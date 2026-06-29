#深度模型封装
# navigation/depth_estimator.py
import cv2
import numpy as np
import openvino as ov
from .config import MODEL_XML, INPUT_SIZE, DEVICE

class DepthEstimator:
    def __init__(self, model_xml=MODEL_XML, input_size=INPUT_SIZE, device=DEVICE):
        self.input_size = input_size
        self.device = device
        self.core = ov.Core()
        self.model = self.core.read_model(model_xml)
        self.model.reshape({self.model.input(0): [1, 3, input_size, input_size]})
        self.compiled = self.core.compile_model(self.model, self.device)
        self.output = self.compiled.output(0)

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        padded = cv2.copyMakeBorder(resized, pad_h, self.input_size - new_h - pad_h,
                                    pad_w, self.input_size - new_w - pad_w,
                                    cv2.BORDER_CONSTANT, value=[0,0,0])
        image = padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = image.transpose(2,0,1)
        image = np.expand_dims(image, axis=0)
        return image, (h, w), (pad_h, pad_w, new_h, new_w)

    def postprocess(self, depth, orig_shape, pad_info):
        h, w = orig_shape
        pad_h, pad_w, new_h, new_w = pad_info
        cropped = depth[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    def infer(self, frame):
        """返回深度图 (H, W) numpy array"""
        input_tensor, orig_shape, pad_info = self.preprocess(frame)
        result = self.compiled([input_tensor])[self.output]
        depth = self.postprocess(result.squeeze(), orig_shape, pad_info)
        return depth