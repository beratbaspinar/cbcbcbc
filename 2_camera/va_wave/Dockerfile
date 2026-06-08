FROM ros:humble-ros-base

# Sistem + ROS bagimliliklari
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ros-humble-vision-msgs \
    ros-humble-foxglove-bridge \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Python bagimliliklari (ximgproc/WLS icin contrib)
RUN pip3 install --no-cache-dir \
    ultralytics \
    opencv-contrib-python-headless \
    openai-clip \
    websockets

WORKDIR /ros2_ws

# ROS2 paketi
COPY ros2_ws/src ./src

# Model + kalibrasyon (params.yaml /ros2_ws/assets bekliyor)
COPY yolov8s-worldv2.pt ./assets/yolov8s-worldv2.pt
COPY calib_images/stereo_calibration.npz ./assets/stereo_calibration.npz

# colcon build
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

# Eski tek-kamera test node'lari (opsiyonel geriye uyum)
COPY vision_node.py test.avi ./

EXPOSE 8000 8765 8766

# Varsayilan: tum mapping hatti
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 launch va_wave_mapping mapping.launch.py"]
