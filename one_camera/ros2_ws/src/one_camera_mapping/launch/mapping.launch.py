"""one_camera_mapping tum hatti tek launch ile baslatir.

Ortam degiskenleri ile override:
  STREAM_URL
  ENABLE_DETECTION  (1/0)
  ENABLE_FOXGLOVE   (1/0)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("one_camera_mapping")
    params = os.path.join(pkg, "config", "params.yaml")

    stream_url = os.environ.get("STREAM_URL", "")
    enable_detection = LaunchConfiguration("enable_detection")
    enable_foxglove = LaunchConfiguration("enable_foxglove")

    nodes = [
        DeclareLaunchArgument(
            "enable_detection",
            default_value=os.environ.get("ENABLE_DETECTION", "1"),
        ),
        DeclareLaunchArgument(
            "enable_foxglove",
            default_value=os.environ.get("ENABLE_FOXGLOVE", "1"),
        ),

        Node(
            package="one_camera_mapping", executable="camera_ingest",
            name="camera_ingest_node", output="screen",
            parameters=[params, {"stream_url": stream_url}] if stream_url else [params],
        ),
        Node(
            package="one_camera_mapping", executable="monocular_depth",
            name="monocular_depth_node", output="screen",
            parameters=[params],
        ),
        Node(
            package="one_camera_mapping", executable="room_layout",
            name="room_layout_node", output="screen",
            parameters=[params],
        ),
        Node(
            package="one_camera_mapping", executable="fusion",
            name="fusion_node", output="screen",
            parameters=[params],
        ),
        Node(
            package="one_camera_mapping", executable="mapping",
            name="mapping_node", output="screen",
            parameters=[params],
        ),
        Node(
            package="one_camera_mapping", executable="wave_bridge",
            name="wave_bridge_node", output="screen",
            parameters=[params],
        ),
        Node(
            package="one_camera_mapping", executable="detection",
            name="detection_node", output="screen",
            parameters=[params],
            condition=IfCondition(enable_detection),
        ),
        Node(
            package="foxglove_bridge", executable="foxglove_bridge",
            name="foxglove_bridge", output="screen",
            parameters=[{"port": 8766, "address": "0.0.0.0"}],
            condition=IfCondition(enable_foxglove),
        ),
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="static_tf_pub",
            arguments=["0", "0", "1.0", "-1.5708", "0", "-1.5708", "map", "camera_optical_frame"],
        ),
    ]

    return LaunchDescription(nodes)
