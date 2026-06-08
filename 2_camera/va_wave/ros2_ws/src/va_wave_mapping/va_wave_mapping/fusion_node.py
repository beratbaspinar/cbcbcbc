"""Fusion node: 2D tespit + derinlik -> 3D tespit.

- /detections_2d (Detection2DArray) ve /stereo/depth (32FC1) alir
- Her kutunun merkez bolgesinden medyan derinlik orneklenir
- Sol kamera intrinsik (P1) ile 3D konum hesaplanir (camera_optical_frame)
- vision_msgs/Detection3DArray (/detections_3d) + MarkerArray (/detections_3d/markers)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    Detection2DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from va_wave_mapping.common import StereoCalibration


class FusionNode(Node):
    def __init__(self):
        super().__init__("fusion_node")

        self.declare_parameter("calib_path", "/ros2_ws/assets/stereo_calibration.npz")
        self.declare_parameter("optical_frame", "camera_optical_frame")
        self.declare_parameter("depth_sample_ratio", 0.5)
        self.declare_parameter("min_range", 0.3)
        self.declare_parameter("max_range", 12.0)
        self.declare_parameter("marker_lifetime", 1.0)

        self.optical_frame = self.get_parameter("optical_frame").value
        self.sample_ratio = float(self.get_parameter("depth_sample_ratio").value)
        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.marker_lifetime = float(self.get_parameter("marker_lifetime").value)

        calib = StereoCalibration(self.get_parameter("calib_path").value)
        P = calib.P1
        self.fx = float(P[0, 0])
        self.fy = float(P[1, 1])
        self.cx = float(P[0, 2])
        self.cy = float(P[1, 2])

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_det3d = self.create_publisher(Detection3DArray, "/detections_3d", 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, "/detections_3d/markers", 10
        )

        self._depth = None
        self.create_subscription(Image, "/stereo/depth", self._on_depth, qos)
        self.create_subscription(
            Detection2DArray, "/detections_2d", self._on_detections, 10
        )
        self.get_logger().info(
            f"fusion_node basladi. fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}"
        )

    def _on_depth(self, msg):
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.width
        )
        self._depth = depth

    def _sample_depth(self, cx, cy, w, h):
        if self._depth is None:
            return None
        dh, dw = self._depth.shape
        rw = max(2, int(w * self.sample_ratio / 2))
        rh = max(2, int(h * self.sample_ratio / 2))
        x0 = max(0, int(cx - rw))
        x1 = min(dw, int(cx + rw))
        y0 = max(0, int(cy - rh))
        y1 = min(dh, int(cy + rh))
        patch = self._depth[y0:y1, x0:x1]
        vals = patch[(patch > self.min_range) & (patch < self.max_range)]
        if vals.size < 5:
            return None
        return float(np.median(vals))

    def _on_detections(self, msg):
        out = Detection3DArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.optical_frame

        markers = MarkerArray()
        # Onceki marker'lari temizle
        clear = Marker()
        clear.header.frame_id = self.optical_frame
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        mid = 0
        for det in msg.detections:
            if not det.results:
                continue
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y
            bw = det.bbox.size_x
            bh = det.bbox.size_y
            z = self._sample_depth(u, v, bw, bh)
            if z is None:
                continue
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy
            name = det.results[0].hypothesis.class_id
            score = det.results[0].hypothesis.score

            # Fiziksel boyut tahmini (kutu acisal boyutundan)
            size_x = max(0.1, bw * z / self.fx)
            size_y = max(0.1, bh * z / self.fy)
            size_z = max(0.2, 0.5 * (size_x + size_y))

            d3 = Detection3D()
            d3.header = out.header
            bbox = BoundingBox3D()
            bbox.center.position.x = x
            bbox.center.position.y = y
            bbox.center.position.z = z
            bbox.center.orientation.w = 1.0
            bbox.size.x = float(size_x)
            bbox.size.y = float(size_y)
            bbox.size.z = float(size_z)
            d3.bbox = bbox
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = float(score)
            hyp.pose.pose.position.x = x
            hyp.pose.pose.position.y = y
            hyp.pose.pose.position.z = z
            hyp.pose.pose.orientation.w = 1.0
            d3.results.append(hyp)
            out.detections.append(d3)

            markers.markers.append(
                self._cube_marker(mid, x, y, z, size_x, size_y, size_z)
            )
            mid += 1
            markers.markers.append(
                self._text_marker(mid, x, y, z, size_z, f"{name} {z:.1f}m")
            )
            mid += 1

        self.pub_det3d.publish(out)
        self.pub_markers.publish(markers)

    def _cube_marker(self, mid, x, y, z, sx, sy, sz):
        m = Marker()
        m.header.frame_id = self.optical_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "obstacles"
        m.id = mid
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = float(sx)
        m.scale.y = float(sy)
        m.scale.z = float(sz)
        m.color.r = 0.95
        m.color.g = 0.4
        m.color.b = 0.1
        m.color.a = 0.6
        m.lifetime = rclpy.duration.Duration(seconds=self.marker_lifetime).to_msg()
        return m

    def _text_marker(self, mid, x, y, z, sz, text):
        m = Marker()
        m.header.frame_id = self.optical_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "labels"
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z + sz / 2 + 0.15)
        m.pose.orientation.w = 1.0
        m.scale.z = 0.25
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.9
        m.text = text
        m.lifetime = rclpy.duration.Duration(seconds=self.marker_lifetime).to_msg()
        return m


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
