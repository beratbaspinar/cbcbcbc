import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class SemanticMapperNode(Node):
    def __init__(self):
        super().__init__('semantic_mapper_node')
        
        # Subscribe to the point cloud topic (from Phase 1 of roadmap)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/scene_point_cloud',
            self.pc_callback,
            10
        )
        
        # Publisher for the MarkerArray (from Phase 2 of roadmap)
        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_room', 10)
        
        self.get_logger().info("Semantic Mapper Node initialized. Waiting for point clouds...")

    def pc_callback(self, msg):
        self.get_logger().info("Received PointCloud2 message. Starting processing pipeline...")
        
        # --- PHASE 1: Geometry with Python (Open3D RANSAC) ---
        
        # 1. Data Preparation (Downsampling)
        # Convert ROS PointCloud2 to numpy array
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
            
        if len(points) < 100:
            self.get_logger().warn("Not enough points to process.")
            return
            
        np_points = np.array(points)
        
        # Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_points)
        
        # Downsample for performance (Voxel Downsampling)
        voxel_size = 0.05 # 5cm voxels
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        self.get_logger().info(f"Downsampled points: {len(downpcd.points)} / {len(pcd.points)}")
        
        planes_found = [] # We will store (type, center, sizes, quaternion, color) here
        
        # 2. Floor Extraction
        # We assume the floor is the largest plane.
        # equation: ax + by + cz + d = 0
        floor_plane_model, floor_inliers = downpcd.segment_plane(distance_threshold=0.05,
                                                                 ransac_n=3,
                                                                 num_iterations=1000)
        if len(floor_inliers) > 0:
            [a, b, c, d] = floor_plane_model
            normal = np.array([a, b, c])
            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)
            
            # Validation: Floor normal should be parallel to Y axis (0, 1, 0)
            # In camera_optical_frame, Y is down, so the floor normal is [0, -1, 0] or [0, 1, 0].
            # So the dot product with [0,1,0] should be close to 1 or -1
            y_axis = np.array([0, 1, 0])
            dot_product = abs(np.dot(normal, y_axis))
            
            if dot_product > 0.8: # Close to 1, meaning it's horizontal
                self.get_logger().info(f"Floor found! Normal: {normal}")
                floor_cloud = downpcd.select_by_index(floor_inliers)
                
                # Extract mathematical properties
                center, scale, quat = self.calculate_plane_properties(floor_cloud, normal)
                
                # Gray semi-transparent color for floor
                color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.5)
                planes_found.append(("Floor", center, scale, quat, color))
                
                # Remove floor points from the cloud so we can find walls
                downpcd = downpcd.select_by_index(floor_inliers, invert=True)
            else:
                self.get_logger().warn(f"Largest plane is not a floor. Normal: {normal}")

        # 3. Wall Extraction
        # Now we look for walls in the remaining points
        wall_count = 0
        while len(downpcd.points) > 100 and wall_count < 4:
            wall_model, wall_inliers = downpcd.segment_plane(distance_threshold=0.05,
                                                             ransac_n=3,
                                                             num_iterations=1000)
            if len(wall_inliers) < 50:
                break # Not enough points to be a wall
                
            [a, b, c, d] = wall_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            
            # Validation: Wall normal should be perpendicular to Y axis (Y ~ 0)
            dot_product = abs(np.dot(normal, y_axis))
            
            if dot_product < 0.3: # Close to 0, meaning it's vertical
                self.get_logger().info(f"Wall {wall_count+1} found! Normal: {normal}")
                wall_cloud = downpcd.select_by_index(wall_inliers)
                
                # Extract mathematical properties
                center, scale, quat = self.calculate_plane_properties(wall_cloud, normal)
                
                # Blue semi-transparent color for walls
                color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.5)
                planes_found.append((f"Wall_{wall_count+1}", center, scale, quat, color))
                wall_count += 1
                
            # Always remove the found plane to continue searching
            downpcd = downpcd.select_by_index(wall_inliers, invert=True)


        # --- PHASE 2: ROS 2 Broadcast and Visualization ---
        
        if not planes_found:
            return
            
        marker_array = MarkerArray()
        
        # Build MarkerArray message
        for idx, (plane_type, center, scale, quat, color) in enumerate(planes_found):
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id # Same frame as the point cloud
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "semantic_room"
            marker.id = idx
            
            # 2. Fill Marker properties
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Set the position (Centroid)
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            
            # 3. Critical Step - Orientation (Quaternion Transformation)
            marker.pose.orientation.x = float(quat[0])
            marker.pose.orientation.y = float(quat[1])
            marker.pose.orientation.z = float(quat[2])
            marker.pose.orientation.w = float(quat[3])
            
            # Scale (Dimensions)
            marker.scale.x = float(scale[0])
            marker.scale.y = float(scale[1])
            marker.scale.z = float(scale[2]) # This will be thin (0.01) because of our local coordinate trick
            
            # Color
            marker.color = color
            
            # Lifetime
            marker.lifetime.sec = 1 # Disappear if not updated
            
            marker_array.markers.append(marker)
            
        # 4. Publishing
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} semantic markers to RViz.")


    # 4. Size and Center Calculation (Bounding Box)
    def calculate_plane_properties(self, pcd, normal):
        """
        Takes a point cloud (inliers of a plane) and its normal vector.
        Calculates the centroid, exact length/width, and orientation quaternion.
        """
        # Convert Open3D points back to NumPy
        points = np.asarray(pcd.points)
        
        # Calculate standard center (mean of points)
        center = np.mean(points, axis=0)
        
        # We need to find the rotation that aligns the plane's normal to the Z-axis [0, 0, 1].
        # Why? Because a standard ROS Cube Marker's "thin" side is its Z scale. 
        # By finding this rotation, we can orient the cube so its Z points along the normal.
        z_axis = np.array([0, 0, 1])
        
        # Find rotation using scipy
        # align_vectors returns a Rotation object that rotates the second vector to align with the first
        # We want to rotate [0,0,1] to become 'normal'
        rot_matrix, _ = R.align_vectors([normal], [z_axis])
        
        # Get quaternion in [x, y, z, w] format for ROS
        quat = rot_matrix.as_quat()
        
        # To find accurate width and height (Length/Width ratios), we temporarily 
        # rotate the points so the plane lies flat on the Z=0 plane.
        # This prevents diagonal planes from having overly large bounding boxes.
        inv_rot = rot_matrix.inv()
        local_points = inv_rot.apply(points - center)
        
        # Now find the min and max in X and Y (which represent the planar dimensions)
        min_bounds = np.min(local_points, axis=0)
        max_bounds = np.max(local_points, axis=0)
        
        size_x = max_bounds[0] - min_bounds[0]
        size_y = max_bounds[1] - min_bounds[1]
        size_z = 0.01 # Plane is paper thin!
        
        # Optional: Refine center based on the exact middle of the bounding box
        local_center = (min_bounds + max_bounds) / 2.0
        refined_center = center + rot_matrix.apply(local_center)
        
        scale = [size_x, size_y, size_z]
        
        return refined_center, scale, quat


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
