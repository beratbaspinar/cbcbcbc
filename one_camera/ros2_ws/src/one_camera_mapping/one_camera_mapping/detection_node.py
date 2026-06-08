"""YOLO-World tespit node'u.

- Rectified sol goruntude calisir (/stereo/left/image_rect)
- Acik kelime dagarcigi (custom classes) ile tespit
- vision_msgs/Detection2DArray yayinlar (/detections_2d)
- Anotasyonlu onizleme (/detections_2d/image)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

import cv2

from one_camera_mapping.common import image_to_msg


class DetectionNode(Node):
    def __init__(self):
        super().__init__("detection_node")

        self.declare_parameter("model_path", "/ros2_ws/assets/yolov8s-worldv2.pt")
        self.declare_parameter(
            "classes",
            ["traffic cone", "curbstone", "pole", "concrete barrier",
             "car", "truck", "person", "tripod", "box",
             "table", "chair", "desk", "door", "wall"],
        )
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("process_rate", 4.0)
        self.declare_parameter("input_topic", "/camera/image_raw")

        self.classes = list(self.get_parameter("classes").value)
        self.confidence = float(self.get_parameter("confidence").value)
        input_topic = self.get_parameter("input_topic").value

        from ultralytics import YOLOWorld
        model_path = self.get_parameter("model_path").value
        self.get_logger().info(f"YOLO-World yukleniyor: {model_path}")
        self.model = YOLOWorld(model_path)
        self.model.set_classes(self.classes)

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub_det = self.create_publisher(Detection2DArray, "/detections_2d", 10)
        self.pub_img = self.create_publisher(Image, "/detections_2d/image", qos)

        self._latest = None
        self.create_subscription(Image, input_topic, self._on_image, qos)
        self.timer = self.create_timer(
            1.0 / float(self.get_parameter("process_rate").value), self.process
        )
        self.get_logger().info(
            f"detection_node basladi. {len(self.classes)} sinif, conf>={self.confidence}"
        )

    def _on_image(self, msg):
        self._latest = msg

    def process(self):
        if self._latest is None:
            return
        msg = self._latest
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )

        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        out = Detection2DArray()
        out.header = msg.header
        annotated = frame.copy()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                name = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)

                det = Detection2D()
                det.header = msg.header
                bbox = BoundingBox2D()
                bbox.center.position.x = cx
                bbox.center.position.y = cy
                bbox.center.theta = 0.0
                bbox.size_x = x2 - x1
                bbox.size_y = y2 - y1
                det.bbox = bbox

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = name
                hyp.hypothesis.score = score
                det.results.append(hyp)
                out.detections.append(det)

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(annotated, f"{name} {score:.2f}",
                            (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

        self.pub_det.publish(out)
        self.pub_img.publish(
            image_to_msg(annotated, "bgr8", msg.header.stamp, msg.header.frame_id)
        )


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
