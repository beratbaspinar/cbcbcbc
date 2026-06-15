FROM ros:humble-ros-base

# Gerekli sistem paketleri (X11, RViz2, OpenCV için GLX)
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-rviz2 \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ultralytics (YOLO) ve OpenCV kurulumu
RUN pip3 install numpy opencv-python ultralytics flask transformers pillow

# Ortam ayarları
ENV DISPLAY=host.docker.internal:0.0
ENV QT_X11_NO_MITSHM=1

WORKDIR /app
COPY sam_rviz2_node.py /app/

# Container çalıştığında ROS ortamını kaynakla ve ana düğüm ile RViz'i başlat
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && (rviz2 &) && python3 sam_rviz2_node.py"]
