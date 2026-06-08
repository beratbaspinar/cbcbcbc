"""Mapping node: kusbakisi occupancy grid + birikmis engel bulutu.

- /stereo/points (PointCloud2, optical) -> TF ile map'e tasinir
- map cercevesinde (Z yukari) belirli yukseklik araligi engel sayilir
- nav_msgs/OccupancyGrid (/map): olasiliksal hit + zaman sonumu (decay)
- /obstacle_cloud (PointCloud2, map): guncel engel noktalari
- /detections_3d ile semantik engeller de grid'e islenir
- Harita kaydet/yukle (kalicilik)
"""

import os

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray

from one_camera_mapping.common import (
    make_pointcloud2,
    quaternion_to_rotation_matrix,
    read_points_xyz,
)


class MappingNode(Node):
    def __init__(self):
        super().__init__("mapping_node")

        self.declare_parameter("input_topic", "/depth/points")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("resolution", 0.10)
        self.declare_parameter("width", 200)
        self.declare_parameter("height", 200)
        self.declare_parameter("origin_x", -10.0)
        self.declare_parameter("origin_y", -10.0)
        self.declare_parameter("hit_increment", 25)
        self.declare_parameter("miss_decrement", 2)
        self.declare_parameter("decay_period", 5.0)
        self.declare_parameter("use_obstacle_cloud", True)
        self.declare_parameter("use_detections", True)
        self.declare_parameter("obstacle_min_height", 0.10)
        self.declare_parameter("obstacle_max_height", 2.5)
        self.declare_parameter("map_save_path", "/ros2_ws/assets/occupancy_map.npy")
        self.declare_parameter("publish_rate", 2.0)

        gp = self.get_parameter
        self.map_frame = gp("map_frame").value
        self.res = float(gp("resolution").value)
        self.W = int(gp("width").value)
        self.H = int(gp("height").value)
        self.ox = float(gp("origin_x").value)
        self.oy = float(gp("origin_y").value)
        self.hit = int(gp("hit_increment").value)
        self.miss = int(gp("miss_decrement").value)
        self.obs_min_h = float(gp("obstacle_min_height").value)
        self.obs_max_h = float(gp("obstacle_max_height").value)
        self.save_path = gp("map_save_path").value
        self.optical_frame = "camera_optical_frame"

        self.grid = np.zeros((self.H, self.W), dtype=np.float32)
        self.known = np.zeros((self.H, self.W), dtype=bool)
        self._load_map()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_map = self.create_publisher(OccupancyGrid, "/map", 1)
        self.pub_obstacles = self.create_publisher(PointCloud2, "/obstacle_cloud", qos)

        if gp("use_obstacle_cloud").value:
            self.create_subscription(PointCloud2, gp("input_topic").value, self._on_cloud, qos)
        if gp("use_detections").value:
            self.create_subscription(
                Detection3DArray, "/detections_3d", self._on_detections, 10
            )

        self.create_timer(1.0 / float(gp("publish_rate").value), self.publish_map)
        self.create_timer(float(gp("decay_period").value), self.apply_decay)
        self.create_timer(30.0, self._save_map)
        self.get_logger().info(
            f"mapping_node basladi. grid {self.W}x{self.H} @ {self.res}m "
            f"({self.W*self.res:.0f}x{self.H*self.res:.0f}m)"
        )

    # --- TF yardimcisi ---
    def _lookup(self, source_frame):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, source_frame, rclpy.time.Time()
            )
        except Exception:
            return None, None
        t = tf.transform.translation
        q = tf.transform.rotation
        R = quaternion_to_rotation_matrix([q.x, q.y, q.z, q.w])
        T = np.array([t.x, t.y, t.z])
        return R, T

    def _to_map(self, pts, source_frame):
        R, T = self._lookup(source_frame)
        if R is None:
            return None
        return (R @ pts.T).T + T

    def _world_to_cell(self, xs, ys):
        cx = ((xs - self.ox) / self.res).astype(np.int32)
        cy = ((ys - self.oy) / self.res).astype(np.int32)
        inside = (cx >= 0) & (cx < self.W) & (cy >= 0) & (cy < self.H)
        return cx[inside], cy[inside]

    # --- callbacks ---
    def _on_cloud(self, msg):
        pts = read_points_xyz(msg)
        if pts.shape[0] == 0:
            return
        map_pts = self._to_map(pts, msg.header.frame_id or self.optical_frame)
        if map_pts is None:
            return
        z = map_pts[:, 2]
        # Bosluklari (zemin) bul ve temizle
        free = (z <= self.obs_min_h)
        free_pts = map_pts[free]
        if free_pts.shape[0] > 0:
            fx, fy = self._world_to_cell(free_pts[:, 0], free_pts[:, 1])
            self.grid[fy, fx] = np.clip(self.grid[fy, fx] - 2, 0, 100)
            self.known[fy, fx] = True

        # Engelleri ekle
        obs = (z > self.obs_min_h) & (z < self.obs_max_h)
        obs_pts = map_pts[obs]
        if obs_pts.shape[0] > 0:
            cx, cy = self._world_to_cell(obs_pts[:, 0], obs_pts[:, 1])
            np.add.at(self.grid, (cy, cx), self.hit)
            self.grid[cy, cx] = np.clip(self.grid[cy, cx], 0, 100)
            self.known[cy, cx] = True

        cloud = make_pointcloud2(
            obs_pts.astype(np.float32), None, msg.header.stamp, self.map_frame
        )
        self.pub_obstacles.publish(cloud)

    def _on_detections(self, msg):
        R, T = self._lookup(msg.header.frame_id or self.optical_frame)
        if R is None:
            return
        for det in msg.detections:
            p = det.bbox.center.position
            world = R @ np.array([p.x, p.y, p.z]) + T
            cx, cy = self._world_to_cell(
                np.array([world[0]]), np.array([world[1]])
            )
            if cx.size:
                # Tespitleri daha guclu isle (yaricap 1 hucre)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        x, y = cx[0] + dx, cy[0] + dy
                        if 0 <= x < self.W and 0 <= y < self.H:
                            self.grid[y, x] = min(100, self.grid[y, x] + self.hit * 2)
                            self.known[y, x] = True

    def apply_decay(self):
        self.grid[self.known] = np.clip(self.grid[self.known] - self.miss, 0, 100)

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.res
        msg.info.width = self.W
        msg.info.height = self.H
        msg.info.origin.position.x = self.ox
        msg.info.origin.position.y = self.oy
        msg.info.origin.orientation.w = 1.0

        data = np.full((self.H, self.W), -1, dtype=np.int8)
        data[self.known] = self.grid[self.known].astype(np.int8)
        msg.data = data.flatten().tolist()
        self.pub_map.publish(msg)

    # --- kalicilik ---
    def _save_map(self):
        try:
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            np.savez(
                self.save_path if self.save_path.endswith(".npz") else self.save_path + ".npz",
                grid=self.grid, known=self.known,
                res=self.res, W=self.W, H=self.H, ox=self.ox, oy=self.oy,
            )
        except Exception as exc:
            self.get_logger().warning(f"Harita kaydedilemedi: {exc}")

    def _load_map(self):
        path = self.save_path if self.save_path.endswith(".npz") else self.save_path + ".npz"
        if not os.path.exists(path):
            return
        try:
            d = np.load(path)
            if int(d["W"]) == self.W and int(d["H"]) == self.H:
                self.grid = d["grid"].astype(np.float32)
                self.known = d["known"].astype(bool)
                self.get_logger().info(f"Onceki harita yuklendi: {path}")
        except Exception as exc:
            self.get_logger().warning(f"Harita yuklenemedi: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save_map()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
