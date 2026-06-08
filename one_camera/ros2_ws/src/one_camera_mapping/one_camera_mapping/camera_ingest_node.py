"""HTTP MJPEG stream -> sensor_msgs/Image + CameraInfo (Single Camera).

Tek kamera yayinini alir ve yayinlar.
"""

import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo, Image

from one_camera_mapping.common import StereoCalibration, image_to_msg


class CameraIngestNode(Node):
    def __init__(self):
        super().__init__("camera_ingest")

        self.declare_parameter("stream_url", "")
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("publish_rate", 15.0)
        self.declare_parameter("reconnect_period", 2.0)
        self.declare_parameter("calib_path", "/ros2_ws/assets/stereo_calibration.npz")

        self.stream_url = self.get_parameter("stream_url").value
        env_url = os.environ.get("STREAM_URL", "").strip()
        if env_url:
            self.stream_url = env_url
            
        self.frame_id = self.get_parameter("frame_id").value
        rate = float(self.get_parameter("publish_rate").value)
        self.reconnect_period = float(self.get_parameter("reconnect_period").value)
        calib_path = self.get_parameter("calib_path").value

        # Eger kalibrasyon dosyasi varsa, P1'i (Sol kamera projektif matrisi) kullan. 
        # Yoksa varsayilan bir matris atariz.
        try:
            self.calib = StereoCalibration(calib_path)
            self.w, self.h = self.calib.image_size
            self.P = self.calib.P1
            mtx, dist, R = self.calib.mtxL, self.calib.distL, self.calib.R1
            self.map_x, self.map_y = cv2.initUndistortRectifyMap(
                mtx, dist, R, self.P, (self.w, self.h), cv2.CV_32FC1
            )
            self.use_rect = True
            self.get_logger().info(f"Kalibrasyon yuklendi. Boyut: {self.w}x{self.h}")
        except Exception as e:
            self.get_logger().warning(f"Kalibrasyon dosyasi okunamadi ({e}). Varsayilan parametreler kullanilacak.")
            self.w, self.h = 640, 480
            # Varsayilan K matrisi (640x480 icin yaklasik)
            self.P = np.array([[500.0, 0, 320.0, 0],
                               [0, 500.0, 240.0, 0],
                               [0, 0, 1.0, 0]])
            self.use_rect = False

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_image = self.create_publisher(Image, "/camera/image_raw", qos)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/camera_info", qos)

        self._lock = threading.Lock()
        self._latest = None
        self._cap = None
        self._stop = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self.timer = self.create_timer(1.0 / rate, self.publish_latest)
        self.get_logger().info(f"Ingest basladi. URL={self.stream_url}")

    def _open(self):
        cap = cv2.VideoCapture(self.stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _read_loop(self):
        while not self._stop:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    self.get_logger().warning("Kameraya baglanilamadi, tekrar denenecek...")
                    self._sleep(self.reconnect_period)
                    continue
                self.get_logger().info("Kamera akisi baglandi.")
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
            
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h))
            
        if self.use_rect:
            frame = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)

        stamp = self.get_clock().now().to_msg()
        img_msg = image_to_msg(frame, "bgr8", stamp, self.frame_id)
        self.pub_image.publish(img_msg)
        self.pub_info.publish(self._camera_info(stamp, self.w, self.h))

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
