import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from http.server import BaseHTTPRequestHandler, HTTPServer

latest_preview = None


class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='background:#111;color:#fff;text-align:center'>"
                b"<h1>Stereo Depth (ROS)</h1>"
                b"<img src='/video_feed' width='900'></body></html>"
            )
        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header(
                "Content-type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            while True:
                if latest_preview is not None:
                    ok, jpeg = cv2.imencode(".jpg", latest_preview)
                    if ok:
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header(
                                "Content-Length", str(len(jpeg.tobytes()))
                            )
                            self.end_headers()
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b"\r\n")
                        except Exception:
                            break
                time.sleep(0.05)


def start_preview_server(port):
    server = HTTPServer(("0.0.0.0", port), PreviewHandler)
    server.serve_forever()


def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if isinstance(source, str) and source.startswith("http"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class StereoNode(Node):
    def __init__(self):
        super().__init__("stereo_node")

        calib_path = os.environ.get("CALIB_PATH", "stereo_calibration.npz")
        if not os.path.exists(calib_path):
            calib_path = "calib_images/stereo_calibration.npz"
        calib = np.load(calib_path)

        self.image_size = (
            int(calib["image_width"]),
            int(calib["image_height"]),
        )
        self.Q = calib["Q"]

        left_mtx, left_dist = calib["camera_matrix_left"], calib["dist_coeffs_left"]
        right_mtx, right_dist = (
            calib["camera_matrix_right"],
            calib["dist_coeffs_right"],
        )
        R, T = calib["R"], calib["T"]
        R1, R2, P1, P2 = calib["R1"], calib["R2"], calib["P1"], calib["P2"]

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            left_mtx, left_dist, R1, P1, self.image_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            right_mtx, right_dist, R2, P2, self.image_size, cv2.CV_32FC1
        )

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
        )

        windows_ip = os.environ.get("WINDOWS_IP", "192.168.1.191")
        cam1_url = os.environ.get("CAM1_URL") or (
            f"http://{windows_ip}:5000/video_feed"
        )
        cam2_url = os.environ.get("CAM2_URL") or (
            "http://host.docker.internal:5001/video_feed"
        )

        self.cam1_url = cam1_url
        self.cam2_url = cam2_url
        self.cap_left = None
        self.cap_right = None

        self.publisher_ = self.create_publisher(
            Float32MultiArray, "/va_wave/stereo_point", 10
        )
        self.timer = self.create_timer(0.2, self.timer_callback)

        preview_port = int(os.environ.get("STEREO_PREVIEW_PORT", "8001"))
        threading.Thread(
            target=start_preview_server, args=(preview_port,), daemon=True
        ).start()

        self.get_logger().info(f"Kalibrasyon: {calib_path}")
        self.get_logger().info(f"Sol kamera (Windows): {cam1_url}")
        self.get_logger().info(f"Sag kamera (Mac): {cam2_url}")
        self.get_logger().info(
            f"Stereo ROS node basladi. Onizleme: http://localhost:{preview_port}"
        )

    def ensure_cameras(self):
        if self.cap_left is None or not self.cap_left.isOpened():
            self.cap_left = open_capture(self.cam1_url)
        if self.cap_right is None or not self.cap_right.isOpened():
            self.cap_right = open_capture(self.cam2_url)

        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            self.get_logger().warning(
                f"Kamera bekleniyor... CAM1={self.cam1_url} CAM2={self.cam2_url}"
            )
            return False
        return True

    def timer_callback(self):
        global latest_preview

        if not self.ensure_cameras():
            return

        ok_l, frame_l = self.cap_left.read()
        ok_r, frame_r = self.cap_right.read()
        if not ok_l or not ok_r:
            self.get_logger().warning("Stereo frame alinamadi")
            return

        frame_l = cv2.resize(frame_l, self.image_size)
        frame_r = cv2.resize(frame_r, self.image_size)

        rect_l = cv2.remap(frame_l, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, self.map2x, self.map2y, cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)

        h, w = disparity.shape
        cx, cy = w // 2, h // 2

        valid_disp = disparity[disparity > 0]
        if len(valid_disp) > 100:
            threshold = np.percentile(valid_disp, 30)
            mask = disparity > threshold
            valid = points_3d[mask]
            valid = valid[valid[:, 2] > 0]
            if len(valid) > 0:
                centroid = valid.mean(axis=0)
                x, y, z = map(float, centroid)
            else:
                x, y, z = 0.0, 0.0, 0.0
        else:
            point = points_3d[cy, cx]
            x, y, z = map(float, point)

        if z > 0:
            msg = Float32MultiArray()
            msg.data = [x, y, z]
            self.publisher_.publish(msg)
            self.get_logger().info(f"Stereo nokta -> X:{x:.3f} Y:{y:.3f} Z:{z:.3f} m")

        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = cv2.applyColorMap(disp_vis.astype(np.uint8), cv2.COLORMAP_JET)
        preview = np.hstack([rect_l, rect_r, disp_vis])
        cv2.putText(
            preview,
            f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}m",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        latest_preview = preview


def main(args=None):
    rclpy.init(args=args)
    node = StereoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap_left is not None:
            node.cap_left.release()
        if node.cap_right is not None:
            node.cap_right.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
