"""HTTP MJPEG stream -> rectified sensor_msgs/Image + CameraInfo.

Iki kez calistirilir (left/right). Her instance kendi tarafinin
rectification haritasini kalibrasyondan kurar.
"""

import os
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo, Image

from va_wave_mapping.common import StereoCalibration, image_to_msg


class CameraIngestNode(Node):
    def __init__(self):
        super().__init__("camera_ingest")

        self.declare_parameter("side", "left")
        self.declare_parameter("stream_url", "")
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("publish_rate", 15.0)
        self.declare_parameter("reconnect_period", 2.0)
        self.declare_parameter("calib_path", "/ros2_ws/assets/stereo_calibration.npz")

        self.side = self.get_parameter("side").value
        self.stream_url = self.get_parameter("stream_url").value
        # Docker ortam degiskeni her zaman params.yaml'i ezer
        env_key = "LEFT_STREAM_URL" if self.side == "left" else "RIGHT_STREAM_URL"
        env_url = os.environ.get(env_key, "").strip()
        if env_url:
            self.stream_url = env_url
        self.frame_id = self.get_parameter("frame_id").value
        rate = float(self.get_parameter("publish_rate").value)
        self.reconnect_period = float(self.get_parameter("reconnect_period").value)
        calib_path = self.get_parameter("calib_path").value

        self.calib = StereoCalibration(calib_path)
        w, h = self.calib.image_size
        if self.side == "left":
            mtx, dist, R, P = (
                self.calib.mtxL, self.calib.distL, self.calib.R1, self.calib.P1
            )
        else:
            mtx, dist, R, P = (
                self.calib.mtxR, self.calib.distR, self.calib.R2, self.calib.P2
            )
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            mtx, dist, R, P, (w, h), cv2.CV_32FC1
        )
        self.P = P

        ns = f"/stereo/{self.side}"
        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_image = self.create_publisher(Image, f"{ns}/image_rect", qos)
        self.pub_info = self.create_publisher(CameraInfo, f"{ns}/camera_info", qos)

        self._lock = threading.Lock()
        self._latest = None
        self._cap = None
        self._stop = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self.timer = self.create_timer(1.0 / rate, self.publish_latest)
        self.get_logger().info(
            f"[{self.side}] ingest basladi. URL={self.stream_url} "
            f"boyut={w}x{h} calib_err={self.calib.reprojection_error:.2f}px"
        )

    def _open(self):
        cap = cv2.VideoCapture(self.stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _read_loop(self):
        while not self._stop:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    self.get_logger().warning(
                        f"[{self.side}] baglanilamadi, tekrar denenecek..."
                    )
                    self._sleep(self.reconnect_period)
                    continue
                self.get_logger().info(f"[{self.side}] stream baglandi.")
            ok, frame = self._cap.read()
            if not ok:
                self._cap.release()
                self._cap = None
                self._sleep(self.reconnect_period)
                continue
            with self._lock:
                self._latest = frame

    def _sleep(self, seconds):
        end = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while self.get_clock().now().nanoseconds < end and not self._stop:
            import time
            time.sleep(0.05)

    def _camera_info(self, stamp, w, h):
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width = w
        info.height = h
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        K = self.P[:3, :3]
        info.k = K.flatten().tolist()
        info.r = np.eye(3).flatten().tolist()
        info.p = self.P.flatten().tolist()
        return info

    def publish_latest(self):
        with self._lock:
            frame = None if self._latest is None else self._latest.copy()
        if frame is None:
            return
        h, w = self.calib.image_size[1], self.calib.image_size[0]
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        rect = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)

        stamp = self.get_clock().now().to_msg()
        img_msg = image_to_msg(rect, "bgr8", stamp, self.frame_id)
        self.pub_image.publish(img_msg)
        self.pub_info.publish(self._camera_info(stamp, w, h))

    def destroy_node(self):
        self._stop = True
        if self._cap is not None:
            self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraIngestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
