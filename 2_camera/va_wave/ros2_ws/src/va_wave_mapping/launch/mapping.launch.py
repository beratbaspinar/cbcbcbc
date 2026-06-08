"""va_wave_mapping tum hatti tek launch ile baslatir.

Ortam degiskenleri ile override:
  LEFT_STREAM_URL   (varsayilan params.yaml)
  RIGHT_STREAM_URL
  CALIB_PATH
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
    pkg = get_package_share_directory("va_wave_mapping")
    params = os.path.join(pkg, "config", "params.yaml")

    calib = os.environ.get("CALIB_PATH", "/ros2_ws/assets/stereo_calibration.npz")
    left_url = os.environ.get("LEFT_STREAM_URL", "")
    right_url = os.environ.get("RIGHT_STREAM_URL", "")

    enable_detection = LaunchConfiguration("enable_detection")
    enable_foxglove = LaunchConfiguration("enable_foxglove")

    def cam_overrides(url):
        ov = {"calib_path": calib}
        if url:
            ov["stream_url"] = url
        return ov

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
            package="va_wave_mapping", executable="camera_ingest",
            name="camera_ingest_left", output="screen",
            parameters=[params, cam_overrides(left_url)],
        ),
        Node(
            package="va_wave_mapping", executable="camera_ingest",
            name="camera_ingest_right", output="screen",
            parameters=[params, cam_overrides(right_url)],
        ),
        Node(
            package="va_wave_mapping", executable="stereo_depth",
            name="stereo_depth_node", output="screen",
            parameters=[params, {"calib_path": calib}],
        ),
        Node(
            package="va_wave_mapping", executable="fusion",
            name="fusion_node", output="screen",
            parameters=[params, {"calib_path": calib}],
        ),
        Node(
            package="va_wave_mapping", executable="mapping",
            name="mapping_node", output="screen",
            parameters=[params],
        ),
        Node(
            package="va_wave_mapping", executable="wave_bridge",
            name="wave_bridge_node", output="screen",
            parameters=[params],
        ),
        Node(
            package="va_wave_mapping", executable="detection",
            name="detection_node", output="screen",
            parameters=[params, {"calib_path": calib}],
            condition=IfCondition(enable_detection),
        ),
        Node(
            package="foxglove_bridge", executable="foxglove_bridge",
            name="foxglove_bridge", output="screen",
            parameters=[{"port": 8766, "address": "0.0.0.0"}],
            condition=IfCondition(enable_foxglove),
        ),
    ]

    return LaunchDescription(nodes)
