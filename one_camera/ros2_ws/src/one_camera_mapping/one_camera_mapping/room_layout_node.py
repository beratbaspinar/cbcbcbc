"""Room Layout Node using RoomNet (or equivalent).

Yayinlananlar:
- /room/boundaries (visualization_msgs/MarkerArray): Oda kosesi ve duvar cizgileri
- /room/mask (sensor_msgs/Image): Yer zeminini gosteren 2D maske
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray

import cv2
import torch
from one_camera_mapping.common import image_to_msg


class RoomLayoutNode(Node):
    def __init__(self):
        super().__init__("room_layout")

        self.declare_parameter("model_path", "/ros2_ws/assets/roomnet.pth")
        self.model_path = self.get_parameter("model_path").value

        # Eger model varsa PyTorch ile yuklenir, yoksa simdilik dummy/mock calisir
        self.model_loaded = False
        self.get_logger().info(f"RoomNet modeli yukleniyor: {self.model_path}")
        try:
            # self.model = torch.load(self.model_path)
            # self.model.eval()
            # self.model_loaded = True
            self.get_logger().warning("Model bulunamadi, dummy layout maskesi uretecek.")
        except Exception as e:
            self.get_logger().error(f"Model yuklenirken hata: {e}")

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_mask = self.create_publisher(Image, "/room/mask", qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/room/boundaries", 10)

        self._camera_info = None
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, qos)
        
        # Derinlik haritasini ve raw image'i senkronize okumak gerekebilir.
        # Basitlik acisindan sadece raw image uzerinden calisip, sonradan 3D ye izdusum yapariz.
        self.create_subscription(Image, "/camera/image_raw", self._on_image, qos)

    def _on_info(self, msg):
        self._camera_info = msg

    def _on_image(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        h, w = msg.height, msg.width

        # Dummy model calismasi: (Eger model_loaded True ise modelden cikacak veriyi simule ederiz)
        # Odanin zeminini resmin alt %40'lik kismi gibi farz edelim.
        mask = np.zeros((h, w), dtype=np.uint8)
        ground_horizon = int(h * 0.6)
        mask[ground_horizon:, :] = 255

        # Maske yayinla
        mask_msg = image_to_msg(mask, "mono8", msg.header.stamp, msg.header.frame_id)
        self.pub_mask.publish(mask_msg)

        # Oda cizgileri (MarkerArray)
        if self._camera_info is not None:
            self._publish_room_boundaries(msg.header)

    def _publish_room_boundaries(self, header):
        markers = MarkerArray()
        
        # Onceki markerlari temizle
        clear = Marker()
        clear.header.frame_id = header.frame_id
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        # Cizgiler (Ufuk cizgisi vb. temsil)
        m = Marker()
        m.header.frame_id = header.frame_id
        m.header.stamp = header.stamp
        m.ns = "room_layout"
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.8
        m.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
        
        # Z varsayilan uzaklik (örneğin 5m)
        z = 5.0
        cx = self._camera_info.k[2]
        cy = self._camera_info.k[5]
        fx = self._camera_info.k[0]
        fy = self._camera_info.k[4]

        # Sahte zemin köseleri (2D -> 3D izdusum)
        # Sol ufuk, Sag ufuk, Sag Alt, Sol Alt
        pts_2d = [
            (0, int(self._camera_info.height * 0.6)),
            (self._camera_info.width, int(self._camera_info.height * 0.6)),
            (self._camera_info.width, self._camera_info.height),
            (0, self._camera_info.height),
            (0, int(self._camera_info.height * 0.6))
        ]

        from geometry_msgs.msg import Point
        for u, v in pts_2d:
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            m.points.append(p)

        markers.markers.append(m)
        self.pub_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = RoomLayoutNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
