"""Stereo derinlik node'u.

- Sol/sag rectified goruntuleri senkron alir
- SGBM (+ opsiyonel WLS) ile disparity
- Q ile 3D'ye projekte, range filtresi
- PointCloud2 (/stereo/points) yayinlar (camera_optical_frame)
- RANSAC ile yer duzlemi tahmin eder, map->camera_optical_frame TF yayinlar
- Disparity onizleme (/stereo/disparity_color)
"""

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image, PointCloud2
from tf2_ros import StaticTransformBroadcaster

import cv2

from va_wave_mapping.common import (
    StereoCalibration,
    fit_ground_plane,
    image_to_msg,
    make_pointcloud2,
    rotation_aligning_vectors,
    rotation_matrix_to_quaternion,
)

try:
    HAS_WLS = hasattr(cv2, "ximgproc")
except Exception:
    HAS_WLS = False


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__("stereo_depth_node")

        self.declare_parameter("calib_path", "/ros2_ws/assets/stereo_calibration.npz")
        self.declare_parameter("optical_frame", "camera_optical_frame")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("min_disparity", 0)
        self.declare_parameter("num_disparities", 96)
        self.declare_parameter("block_size", 7)
        self.declare_parameter("uniqueness_ratio", 10)
        self.declare_parameter("speckle_window_size", 100)
        self.declare_parameter("speckle_range", 2)
        self.declare_parameter("use_wls", True)
        self.declare_parameter("wls_lambda", 8000.0)
        self.declare_parameter("wls_sigma", 1.5)
        self.declare_parameter("min_range", 0.3)
        self.declare_parameter("max_range", 12.0)
        self.declare_parameter("cloud_decimation", 2)
        self.declare_parameter("process_rate", 6.0)
        self.declare_parameter("ground_enable", True)
        self.declare_parameter("ground_threshold", 0.05)
        self.declare_parameter("ground_min_confidence", 0.2)
        self.declare_parameter("fallback_camera_height", 1.0)
        self.declare_parameter("fallback_tilt_deg", 0.0)

        gp = self.get_parameter
        self.optical_frame = gp("optical_frame").value
        self.map_frame = gp("map_frame").value
        self.min_range = float(gp("min_range").value)
        self.max_range = float(gp("max_range").value)
        self.decim = int(gp("cloud_decimation").value)
        self.ground_enable = bool(gp("ground_enable").value)
        self.ground_threshold = float(gp("ground_threshold").value)
        self.ground_min_conf = float(gp("ground_min_confidence").value)

        self.calib = StereoCalibration(gp("calib_path").value)
        self.Q = self.calib.Q.astype(np.float32)

        num_disp = int(gp("num_disparities").value)
        num_disp = max(16, (num_disp // 16) * 16)
        block = int(gp("block_size").value)
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=int(gp("min_disparity").value),
            numDisparities=num_disp,
            blockSize=block,
            P1=8 * 3 * block ** 2,
            P2=32 * 3 * block ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=int(gp("uniqueness_ratio").value),
            speckleWindowSize=int(gp("speckle_window_size").value),
            speckleRange=int(gp("speckle_range").value),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.use_wls = bool(gp("use_wls").value) and HAS_WLS
        if bool(gp("use_wls").value) and not HAS_WLS:
            self.get_logger().warning(
                "cv2.ximgproc yok; WLS kapatildi (opencv-contrib gerekli)."
            )
        if self.use_wls:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.wls.setLambda(float(gp("wls_lambda").value))
            self.wls.setSigmaColor(float(gp("wls_sigma").value))

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_cloud = self.create_publisher(PointCloud2, "/stereo/points", qos)
        self.pub_disp = self.create_publisher(Image, "/stereo/disparity_color", qos)
        self.pub_depth = self.create_publisher(Image, "/stereo/depth", qos)

        self.tf_static = StaticTransformBroadcaster(self)
        self.ground_locked = False
        self._left_msg = None
        self._right_msg = None
        self._empty_count = 0

        # HTTP stream'ler icin en son sol/sag kare (timestamp sync yerine)
        self.create_subscription(
            Image, "/stereo/left/image_rect", self._on_left, qos
        )
        self.create_subscription(
            Image, "/stereo/right/image_rect", self._on_right, qos
        )

        self.timer = self.create_timer(1.0 / float(gp("process_rate").value), self.process)
        self._publish_fallback_tf()
        self.get_logger().info(
            f"stereo_depth basladi. WLS={'acik' if self.use_wls else 'kapali'} "
            f"baseline={self.calib.baseline*100:.1f}cm range=[{self.min_range},{self.max_range}]m"
        )

    def _on_left(self, msg):
        self._left_msg = msg

    def _on_right(self, msg):
        self._right_msg = msg

    @staticmethod
    def _to_bgr(msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        return arr.reshape(msg.height, msg.width, 3)

    def process(self):
        if self._left_msg is None or self._right_msg is None:
            return
        left_msg, right_msg = self._left_msg, self._right_msg
        left = self._to_bgr(left_msg)
        right = self._to_bgr(right_msg)
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        if self.use_wls:
            disp_l = self.left_matcher.compute(gray_l, gray_r)
            disp_r = self.right_matcher.compute(gray_r, gray_l)
            disparity = self.wls.filter(disp_l, gray_l, disparity_map_right=disp_r)
            disparity = disparity.astype(np.float32) / 16.0
        else:
            disparity = self.left_matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0

        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        z = points_3d[:, :, 2]
        stamp = left_msg.header.stamp

        # Tam cozunurluk gecerlilik (derinlik goruntusu + fusion icin)
        valid_full = (
            (disparity > 0) & np.isfinite(z) & (z > self.min_range) & (z < self.max_range)
        )
        depth_img = np.where(valid_full, z, 0.0).astype(np.float32)
        self.pub_depth.publish(image_to_msg(depth_img, "32FC1", stamp, self.optical_frame))

        # Nokta bulutu icin seyreltilmis maske
        cloud_mask = valid_full
        if self.decim > 1:
            stride = np.zeros_like(valid_full)
            stride[:: self.decim, :: self.decim] = True
            cloud_mask = valid_full & stride

        pts = points_3d[cloud_mask].reshape(-1, 3)
        cols = left[cloud_mask].reshape(-1, 3) if pts.shape[0] else None
        if pts.shape[0] > 0:
            cloud = make_pointcloud2(pts, cols, stamp, self.optical_frame)
            self.pub_cloud.publish(cloud)
            if self.ground_enable and not self.ground_locked:
                self._update_ground(pts)
            self._empty_count = 0
        else:
            self._empty_count += 1
            if self._empty_count % 30 == 1:
                valid_px = int(np.sum(disparity > 0))
                self.get_logger().warning(
                    f"Gecerli 3D nokta yok. disparity>0 piksel={valid_px} "
                    f"(kalibrasyon hatasi yuksek olabilir, calib_err="
                    f"{self.calib.reprojection_error:.1f}px)"
                )

        # Disparity onizleme
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = cv2.applyColorMap(disp_vis.astype(np.uint8), cv2.COLORMAP_JET)
        self.pub_disp.publish(image_to_msg(disp_vis, "bgr8", stamp, self.optical_frame))

    # --- yer duzlemi ---
    def _update_ground(self, pts):
        sample = pts
        if pts.shape[0] > 20000:
            idx = np.random.default_rng(0).choice(pts.shape[0], 20000, replace=False)
            sample = pts[idx]
        normal, d, conf = fit_ground_plane(
            sample, threshold=self.ground_threshold
        )
        if normal is None or conf < self.ground_min_conf:
            return
        # Normal kameraya dogru (yukari) baksin: optik cercevede yukari ~ -Y
        if normal[1] > 0:
            normal = -normal
            d = -d
        height = abs(d)
        R = rotation_aligning_vectors(normal, np.array([0.0, 0.0, 1.0]))
        self._publish_ground_tf(R, height)
        self.ground_locked = True
        self.get_logger().info(
            f"Yer duzlemi kilitlendi. conf={conf:.2f} kamera_yuksekligi={height:.2f}m"
        )

    def _publish_ground_tf(self, R, height):
        q = rotation_matrix_to_quaternion(R)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.optical_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = float(height)
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.tf_static.sendTransform(t)

    def _publish_fallback_tf(self):
        """Yer duzlemi bulunana kadar fallback TF (yukseklik + tilt)."""
        height = float(self.get_parameter("fallback_camera_height").value)
        tilt = np.deg2rad(float(self.get_parameter("fallback_tilt_deg").value))
        # Optik cerceve (Z ileri, Y asagi) -> map (Z yukari)
        # Once optik->ENU benzeri temel rotasyon, sonra tilt
        base = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ], dtype=float)
        ct, st = np.cos(tilt), np.sin(tilt)
        tilt_R = np.array([
            [1, 0, 0],
            [0, ct, -st],
            [0, st, ct],
        ])
        R = base @ tilt_R
        self._publish_ground_tf(R, height)


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
