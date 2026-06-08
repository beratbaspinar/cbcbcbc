import os
from glob import glob

from setuptools import find_packages, setup

package_name = "va_wave_mapping"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="va_wave",
    maintainer_email="dev@vawave.local",
    description="Sabit stereo kamera ile ROS2 algilama + haritalama hatti",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_ingest = va_wave_mapping.camera_ingest_node:main",
            "stereo_depth = va_wave_mapping.stereo_depth_node:main",
            "detection = va_wave_mapping.detection_node:main",
            "fusion = va_wave_mapping.fusion_node:main",
            "mapping = va_wave_mapping.mapping_node:main",
            "wave_bridge = va_wave_mapping.wave_bridge_node:main",
        ],
    },
)
