"""Monocular Metric Depth Node using Depth Anything V2.

Yayinlananlar:
- /depth/image_raw (sensor_msgs/Image - 32FC1): Metrik derinlik (metre)
- /depth/points (sensor_msgs/PointCloud2): Renkli 3D nokta bulutu
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

import cv2
import torch
from transformers import pipeline
from ultralytics import YOLO

from one_camera_mapping.common import image_to_msg


class MonocularDepthNode(Node):
    def __init__(self):
        super().__init__("monocular_depth")

        self.declare_parameter("model_id", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
        self.declare_parameter("metric_scale", 10.0)  # Eger model relative ise, sahte metrik olcegi
        self.declare_parameter("max_depth", 15.0)

        self.model_id = self.get_parameter("model_id").value
        self.metric_scale = float(self.get_parameter("metric_scale").value)
        self.max_depth = float(self.get_parameter("max_depth").value)

        self.get_logger().info(f"Depth modeli yukleniyor: {self.model_id}...")
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.depth_estimator = pipeline("depth-estimation", model=self.model_id, device=device)
            self.get_logger().info("Model basariyla yuklendi.")
        except Exception as e:
            self.get_logger().error(f"Model yuklenirken hata: {e}")
            self.depth_estimator = None

        self.get_logger().info("Segmentasyon (Human Masking) modeli yukleniyor: yolov8n-seg.pt")
        try:
            self.seg_model = YOLO("yolov8n-seg.pt")
            self.get_logger().info("Segmentasyon modeli basariyla yuklendi.")
        except Exception as e:
            self.get_logger().error(f"Segmentasyon modeli yuklenirken hata: {e}")
            self.seg_model = None

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_depth = self.create_publisher(Image, "/depth/image_raw", qos)
        self.pub_points = self.create_publisher(PointCloud2, "/depth/points", qos)

        self._camera_info = None
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, qos)
        self.create_subscription(Image, "/camera/image_raw", self._on_image, qos)

    def _on_info(self, msg):
        if self._camera_info is None:
            self._camera_info = msg
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def _on_image(self, msg):
        if self.depth_estimator is None or self._camera_info is None:
            return

        # BGR goruntuyu PIL veya numpy RGB yap
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(rgb_frame)

        # Derinlik tahmini
        result = self.depth_estimator(pil_img)
        # Metric model ciktisi dogrudan metre cinsinden (float)
        # HF pipeline'i ham tensörü 'predicted_depth' altinda veya 'depth' PIL image olarak dondurur.
        # Biz ham tensörü kullanacagiz:
        if "predicted_depth" in result:
            depth_array = np.array(result["predicted_depth"]).astype(np.float32)
            # Tensor boyutunda ise sıkıştır
            if depth_array.ndim > 2:
                depth_array = depth_array.squeeze()
        else:
            # Fallback (eger model hala PIL image dondururse ve predicted_depth yoksa)
            depth_array = np.array(result["depth"]).astype(np.float32)
            depth_array = (255.0 - depth_array) / 255.0 * self.metric_scale

        depth_array = np.clip(depth_array, 0.1, self.max_depth)
        
        # Kamera cozunurlugune yeniden boyutlandir (Model ciktisi farkli olabilir)
        if depth_array.shape != (msg.height, msg.width):
            depth_array = cv2.resize(depth_array, (msg.width, msg.height), interpolation=cv2.INTER_LINEAR)

        # Derinlik imajini yayinla
        depth_msg = Image()
        depth_msg.header = msg.header
        depth_msg.height = msg.height
        depth_msg.width = msg.width
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = False
        depth_msg.step = msg.width * 4
        depth_msg.data = depth_array.tobytes()
        self.pub_depth.publish(depth_msg)

        # Insan Maskeleme (Segmentation)
        human_mask = np.zeros((msg.height, msg.width), dtype=bool)
        if hasattr(self, 'seg_model') and self.seg_model is not None:
            seg_results = self.seg_model(rgb_frame, classes=[0], verbose=False)
            for r in seg_results:
                if r.masks is not None:
                    masks = r.masks.data.cpu().numpy()
                    combined = np.any(masks > 0.5, axis=0).astype(np.uint8)
                    combined_resized = cv2.resize(combined, (msg.width, msg.height), interpolation=cv2.INTER_NEAREST)
                    human_mask = human_mask | (combined_resized > 0)

        # Nokta bulutu yayinla
        if self.pub_points.get_subscription_count() > 0:
            self._publish_pointcloud(msg.header, frame, depth_array, human_mask)

    def _publish_pointcloud(self, header, color, depth, human_mask):
        h, w = depth.shape
        y, x = np.mgrid[0:h, 0:w]

        # 3D koordinatlar
        z = depth.flatten()
        valid = (z > 0.1) & (z < self.max_depth) & (~human_mask.flatten())
        
        x_c = ((x.flatten()[valid] - self.cx) * z[valid] / self.fx).astype(np.float32)
        y_c = ((y.flatten()[valid] - self.cy) * z[valid] / self.fy).astype(np.float32)
        z_c = z[valid].astype(np.float32)

        # Renkleri paketle
        color_flat = color.reshape(-1, 3)
        b = color_flat[valid, 0].astype(np.uint32)
        g = color_flat[valid, 1].astype(np.uint32)
        r = color_flat[valid, 2].astype(np.uint32)
        rgba = (255 << 24) | (r << 16) | (g << 8) | b
        rgba = rgba.astype(np.uint32).view(np.float32)

        points = np.zeros(valid.sum(), dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
        ])
        points['x'] = x_c
        points['y'] = y_c
        points['z'] = z_c
        points['rgb'] = rgba

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = False
        msg.is_bigendian = False
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.data = points.tobytes()
        self.pub_points.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MonocularDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
