import cv2
import numpy as np
import openvino as ov
import requests
import socket
import time


# ============================================================
# USER CONFIG
# ============================================================

ESP32_IP = "192.168.4.1"
CAPTURE_URL = f"http://{ESP32_IP}/capture"
UDP_PORT = 5005

# 修改成你自己的 OpenVINO .xml 模型路径
MODEL_PATH = r"C:\Users\11650\PycharmProjects\PythonProject1\Depth-Anything-V2\checkpoints\depth_anything_v2_vits_int8.xml"

# Depth Anything V2 输入尺寸
MODEL_WIDTH = 518
MODEL_HEIGHT = 518

# HTTP 超时
CONNECT_TIMEOUT = 2.0
READ_TIMEOUT = 3.0

# 连续多少次取图失败后继续保持 STOP
MAX_CAPTURE_FAILURES = 3


# ============================================================
# UDP CONTROL
# Python -> ESP32-S3 -> UART -> Motor ESP32
# ============================================================

class CarController:
    def __init__(self, ip, port):
        self.address = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.current_command = "stop"

    def send(self, command):
        self.current_command = command
        self.sock.sendto(command.encode("utf-8"), self.address)

    def stop_repeatedly(self, count=3, interval=0.05):
        for _ in range(count):
            try:
                self.sock.sendto(b"stop", self.address)
            except OSError:
                pass

            time.sleep(interval)

        self.current_command = "stop"

    def set_speed(self, speed):
        speed = max(0, min(255, int(speed)))
        command = f"speed{speed}"

        self.sock.sendto(
            command.encode("utf-8"),
            self.address
        )

        print(f"Speed -> {speed}")

    def close(self):
        try:
            self.stop_repeatedly()
        finally:
            self.sock.close()


# ============================================================
# OPENVINO DEPTH MODEL
# ============================================================

class DepthModel:
    def __init__(self, model_path):
        print()
        print("Loading OpenVINO model...")

        self.core = ov.Core()

        self.model = self.core.read_model(
            model_path
        )

        self.compiled_model = self.core.compile_model(
            self.model,
            "CPU"
        )

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        print("Model loaded successfully.")
        print("Input shape :", self.input_layer.partial_shape)
        print("Output shape:", self.output_layer.partial_shape)

        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

    def predict(self, frame):
        # BGR -> RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # resize to model input
        image = cv2.resize(
            rgb,
            (MODEL_WIDTH, MODEL_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        # 0~255 -> 0~1
        image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        image = (image - self.mean) / self.std

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # CHW -> NCHW
        image = np.expand_dims(
            image,
            axis=0
        )

        # inference
        result = self.compiled_model(
            [image]
        )

        # 不使用 get_any_name，避免 output 没有名字的问题
        depth = next(
            iter(result.values())
        )

        depth = np.squeeze(
            depth
        )

        return depth


# ============================================================
# FRESH CAPTURE CLIENT
#
# 每次只请求一张图。
# 不使用 /stream。
# 不建立视频帧队列。
# ============================================================

class FreshFrameCamera:
    def __init__(self, url):
        self.url = url

        self.session = requests.Session()

        self.session.headers.update({
            "Cache-Control": "no-cache, no-store",
            "Pragma": "no-cache"
        })

    def capture(self):
        start = time.perf_counter()

        response = self.session.get(
            self.url,
            timeout=(
                CONNECT_TIMEOUT,
                READ_TIMEOUT
            )
        )

        response.raise_for_status()

        capture_ms = (
            time.perf_counter() - start
        ) * 1000.0

        # ESP32 官方 /capture 里带的时间戳
        camera_timestamp = response.headers.get(
            "X-Timestamp"
        )

        jpeg_array = np.frombuffer(
            response.content,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            jpeg_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            raise RuntimeError(
                "JPEG decode failed"
            )

        return frame, capture_ms, camera_timestamp

    def close(self):
        self.session.close()


# ============================================================
# DEPTH VISUALIZATION
# ============================================================

def depth_to_color(depth, output_width, output_height):
    depth_norm = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_uint8 = depth_norm.astype(
        np.uint8
    )

    depth_color = cv2.applyColorMap(
        depth_uint8,
        cv2.COLORMAP_INFERNO
    )

    depth_color = cv2.resize(
        depth_color,
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST
    )

    return depth_color


# ============================================================
# DISPLAY
# ============================================================

def make_display(
    frame,
    depth,
    capture_ms,
    infer_ms,
    loop_ms,
    command
):
    height, width = frame.shape[:2]

    depth_color = depth_to_color(
        depth,
        width,
        height
    )

    display = np.hstack(
        (
            frame,
            depth_color
        )
    )

    if loop_ms > 0:
        system_hz = 1000.0 / loop_ms
    else:
        system_hz = 0.0

    cv2.putText(
        display,
        f"Capture: {capture_ms:.0f} ms",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Inference: {infer_ms:.0f} ms",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Loop: {loop_ms:.0f} ms  {system_hz:.2f} Hz",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"CMD: {command.upper()}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "W/A/S/D Move | SPACE/X Stop | 1/2/3 Speed | Q Quit",
        (10, display.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )

    return display


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("================================================")
    print("ESP32-S3 FRESH CAPTURE + DEPTH + CAR TEST")
    print("================================================")
    print("Capture :", CAPTURE_URL)
    print("UDP     :", f"{ESP32_IP}:{UDP_PORT}")
    print("Model   :", MODEL_PATH)
    print()
    print("Keyboard:")
    print("W = forward")
    print("S = backward")
    print("A = left")
    print("D = right")
    print("SPACE or X = stop")
    print("1 = speed 60")
    print("2 = speed 100")
    print("3 = speed 150")
    print("Q = stop and quit")
    print("================================================")
    print()

    camera = FreshFrameCamera(
        CAPTURE_URL
    )

    car = CarController(
        ESP32_IP,
        UDP_PORT
    )

    try:
        model = DepthModel(
            MODEL_PATH
        )

    except Exception:
        camera.close()
        car.close()
        raise

    # 开始时先停车
    car.send("stop")

    current_command = "stop"

    capture_failures = 0
    frame_index = 0

    last_log_time = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()

            # =================================================
            # STEP 1
            # 只请求一张新图片
            # =================================================

            try:
                frame, capture_ms, camera_timestamp = camera.capture()

                capture_failures = 0

            except Exception as e:
                capture_failures += 1

                current_command = "stop"

                try:
                    car.send("stop")
                except OSError:
                    pass

                print(
                    f"[CAPTURE ERROR {capture_failures}] {e}"
                )

                if capture_failures >= MAX_CAPTURE_FAILURES:
                    print("Camera repeatedly failed. Car remains STOP.")
                    print("Waiting 2 seconds before retry...")
                    time.sleep(2.0)
                    capture_failures = 0

                time.sleep(
                    0.5
                )

                continue

            # =================================================
            # STEP 2
            # 对这一张图做 Depth Anything
            # =================================================

            infer_start = time.perf_counter()

            depth = model.predict(
                frame
            )

            infer_ms = (
                time.perf_counter() - infer_start
            ) * 1000.0

            # =================================================
            # STEP 3
            # 计算一次系统循环耗时
            # =================================================

            loop_ms = (
                time.perf_counter() - loop_start
            ) * 1000.0

            # =================================================
            # STEP 4
            # 显示原图 + 深度图
            # =================================================

            display = make_display(
                frame,
                depth,
                capture_ms,
                infer_ms,
                loop_ms,
                current_command
            )

            cv2.imshow(
                "ROBOT VISION - FRESH FRAME MODE",
                display
            )

            # =================================================
            # STEP 5
            # 当前先人工控制
            #
            # 后面这里只要替换成自动避障决策即可
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("w"):
                current_command = "forward"

            elif key == ord("s"):
                current_command = "backward"

            elif key == ord("a"):
                current_command = "left"

            elif key == ord("d"):
                current_command = "right"

            elif key == 32 or key == ord("x"):
                current_command = "stop"

            elif key == ord("1"):
                car.set_speed(60)

            elif key == ord("2"):
                car.set_speed(100)

            elif key == ord("3"):
                car.set_speed(150)

            elif key == ord("q"):
                current_command = "stop"

                car.stop_repeatedly()

                break

            # =================================================
            # STEP 6
            # 每完成一次 AI 推理，就发送一次控制命令
            #
            # 这同时也是 Motor ESP32 的 watchdog heartbeat
            # =================================================

            car.send(
                current_command
            )

            frame_index += 1

            # =================================================
            # Console Log
            # 每约1秒打印一次
            # =================================================

            now = time.perf_counter()

            if now - last_log_time >= 1.0:
                if loop_ms > 0:
                    hz = 1000.0 / loop_ms
                else:
                    hz = 0.0

                print(
                    f"Frame {frame_index:5d} | "
                    f"Capture {capture_ms:6.1f} ms | "
                    f"Infer {infer_ms:6.1f} ms | "
                    f"Loop {loop_ms:6.1f} ms | "
                    f"{hz:4.2f} Hz | "
                    f"CMD {current_command:<8} | "
                    f"X-Timestamp {camera_timestamp}"
                )

                last_log_time = now

    except KeyboardInterrupt:
        print()
        print("Ctrl+C received.")

    finally:
        print()
        print("Stopping car...")

        try:
            car.close()

        except Exception as e:
            print(
                "Car close error:",
                e
            )

        camera.close()

        cv2.destroyAllWindows()

        print(
            "Program stopped safely."
        )


if __name__ == "__main__":
    main()