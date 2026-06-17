import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from transformers import pipeline
from PIL import Image as PILImage

class WallFinderNode(Node):
    def __init__(self):
        super().__init__('wall_finder_node')
        
        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, '/independent_scene_pc', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/room_structure', 10)
        self.img_pub = self.create_publisher(Image, '/camera_view', 10)
        
        self.bridge = CvBridge()
        
        # Connect to camera stream
        self.stream_url = 'http://host.docker.internal:5000/video_feed'
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera stream {self.stream_url}!")
        else:
            self.get_logger().info("Connected to camera stream.")
            
        # Load Depth Anything V2
        self.get_logger().info("Loading Depth Anything V2...")
        self.depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cpu")
        self.get_logger().info("Depth model loaded! System ready.")
        
        # Camera intrinsics (approximate standard web camera)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        
        # Thread for lag-free camera reading
        self.latest_frame = None
        import threading
        self.cam_thread = threading.Thread(target=self.update_camera)
        self.cam_thread.daemon = True
        self.cam_thread.start()
        
        # Timer to run the pipeline
        self.timer = self.create_timer(0.5, self.timer_callback)

    def update_camera(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame

    def timer_callback(self):
        if self.latest_frame is None:
            return
            
        frame = self.latest_frame.copy()
            
        # 1. Run Depth Estimation
        depth_result = self.depth_estimator(PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        depth_map = np.array(depth_result["depth"]).astype(np.float32)
        
        if depth_map.shape != (frame.shape[0], frame.shape[1]):
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
            
        # Convert disparity to relative depth. 
        # Using 0.5 puts the room roughly between 0.5m and 12 meters distance.
        # This makes it human-scale.
        arbitrary_scale = 0.5
        scene_z = (255.0 / (depth_map + 1.0)) * arbitrary_scale
        
        # 2. Build Point Cloud (Downsampled for performance)
        h, w = scene_z.shape
        v_all, u_all = np.mgrid[0:h, 0:w]
        
        # Take 1 pixel out of every 4 (16x reduction) to be fast
        step = 4
        v = v_all[::step, ::step].flatten()
        u = u_all[::step, ::step].flatten()
        z = scene_z[::step, ::step].flatten()
        colors = frame[v, u]
        
        # Convert to 3D Camera Coordinates
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Publish PointCloud2
        self.publish_pointcloud(x, y, z, colors)
        
        # Publish Image
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera_link'
        self.img_pub.publish(img_msg)
        
        # 3. RANSAC Room Extraction (Open3D)
        points = np.column_stack((x, y, z))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Voxel downsample again for RANSAC speed
        downpcd = pcd.voxel_down_sample(voxel_size=0.1)
        
        planes_found = [] # (Name, center, scale, quaternion, color)
        
        # Y axis in camera frame is DOWN
        y_axis = np.array([0, 1, 0])
        
        # Extract up to 6 planes
        max_planes = 6
        for i in range(max_planes):
            if len(downpcd.points) < 100:
                break
                
            plane_model, inliers = downpcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
            if len(inliers) < 50:
                break
                
            normal = np.array(plane_model[:3])
            normal = normal / np.linalg.norm(normal)
            
            dot_product = np.dot(normal, y_axis)
            abs_dot = abs(dot_product)
            
            plane_cloud = downpcd.select_by_index(inliers)
            center, scale, quat = self.calculate_plane_properties(plane_cloud, normal)
            
            # Determine plane type based on normal vector
            if abs_dot > 0.8:
                # Parallel to Y axis. Is it floor or ceiling?
                if dot_product < 0: # Normal points UP (-Y) -> Floor
                    name = "Floor"
                    color = ColorRGBA(r=0.4, g=0.4, b=0.4, a=0.6) # Gray
                else: # Normal points DOWN (+Y) -> Ceiling
                    name = "Ceiling"
                    color = ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.3) # Light gray
            else:
                # Perpendicular to Y axis -> Wall
                name = f"Wall_{i+1}"
                color = ColorRGBA(r=0.0, g=0.6, b=1.0, a=0.5) # Blue
                
            planes_found.append((name, center, scale, quat, color))
            
            # Remove plane points to find the next one
            downpcd = downpcd.select_by_index(inliers, invert=True)
            
        # 4. Publish MarkerArray
        self.publish_markers(planes_found)

    def calculate_plane_properties(self, pcd, normal):
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)
        z_axis = np.array([0, 0, 1])
        
        rot_matrix, _ = R.align_vectors([normal], [z_axis])
        quat = rot_matrix.as_quat()
        
        inv_rot = rot_matrix.inv()
        local_points = inv_rot.apply(points - center)
        
        min_bounds = np.min(local_points, axis=0)
        max_bounds = np.max(local_points, axis=0)
        
        size_x = max_bounds[0] - min_bounds[0]
        size_y = max_bounds[1] - min_bounds[1]
        size_z = 0.05 # Wall thickness
        
        local_center = (min_bounds + max_bounds) / 2.0
        refined_center = center + rot_matrix.apply(local_center)
        
        return refined_center, [size_x, size_y, size_z], quat
        
    def publish_pointcloud(self, x, y, z, colors):
        structured_array = np.zeros(len(x), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'u4')])
        structured_array['x'] = x.astype(np.float32)
        structured_array['y'] = y.astype(np.float32)
        structured_array['z'] = z.astype(np.float32)
        
        b = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        r = colors[:, 2].astype(np.uint32)
        a = np.full_like(b, 255, dtype=np.uint32)
        rgba = (a << 24) | (r << 16) | (g << 8) | b
        structured_array['rgb'] = rgba
        
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.height = 1
        msg.width = len(x)
        msg.is_dense = False
        msg.is_bigendian = False
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * len(x)
        msg.data = structured_array.tobytes()
        
        self.pc_pub.publish(msg)

    def publish_markers(self, planes_found):
        if not planes_found:
            return
            
        marker_array = MarkerArray()
        for idx, (plane_type, center, scale, quat, color) in enumerate(planes_found):
            marker = Marker()
            marker.header.frame_id = 'camera_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "room_structure"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            
            marker.pose.orientation.x = float(quat[0])
            marker.pose.orientation.y = float(quat[1])
            marker.pose.orientation.z = float(quat[2])
            marker.pose.orientation.w = float(quat[3])
            
            marker.scale.x = float(scale[0])
            marker.scale.y = float(scale[1])
            marker.scale.z = float(scale[2])
            
            marker.color = color
            marker.lifetime.sec = 1
            
            marker_array.markers.append(marker)
            
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = WallFinderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
