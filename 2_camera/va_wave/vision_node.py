import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

latest_annotated_frame = None

# Basit bir Web Sunucusu: Görüntüyü tarayiciya aktarmak icin
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head><title>Vision Node</title></head><body style="background:#111;color:white;text-align:center;"><h1>Canli Engel Tespiti</h1><img src="/video_feed" width="800" style="border:3px solid white;border-radius:10px;"></body></html>')
        elif self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                if latest_annotated_frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', latest_annotated_frame)
                    if ret:
                        try:
                            self.wfile.write(b'--frame\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', str(len(jpeg.tobytes())))
                            self.end_headers()
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                        except:
                            break # Tarayici kapatilirsa donguyu bitir
                time.sleep(0.05)

def run_server():
    server = HTTPServer(('0.0.0.0', 8000), VideoStreamHandler)
    server.serve_forever()

from ultralytics import YOLOWorld

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/va_wave/vision_target', 10)
        
        self.timer = self.create_timer(0.1, self.timer_callback)

        source = os.environ.get('VIDEO_SOURCE', 'test.avi')
        if source.isdigit():
            source = int(source)
        self.is_file_source = isinstance(source, str) and os.path.isfile(source)
        self.cap = cv2.VideoCapture(source)

        # Otonom arac / engel tespiti icin YOLO-World modelini (Acik Kelime Dagarcigi) kullaniyoruz!
        self.model = YOLOWorld('yolov8s-worldv2.pt')

        # İstediginiz her engeli buraya Ingilizce kelime olarak yazabilirsiniz. YOLO-World onlari taniyacaktir!
        self.custom_obstacles = ["traffic cone", "curbstone", "pole", "concrete barrier", "car", "truck", "building", "tripod"]
        self.model.set_classes(self.custom_obstacles)

        self.get_logger().info(f"Video kaynagi: {source}")
        self.get_logger().info("YOLO-World Engel Tespiti Basladi... Tarayicinizdan http://localhost:8000 adresine gidin.")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def timer_callback(self):
        global latest_annotated_frame
        ret, frame = self.cap.read()
        
        if not ret:
            if self.is_file_source:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning("Kameradan/Videodan goruntu alinamadi!")
                return

        # YOLO-World sadece belirledigimiz listeyi arayacak
        results = self.model(frame, verbose=False)
        
        # YOLO'nun cizdigi kutulu/isimli goruntuyu al
        annotated_frame = results[0].plot() 
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                # YOLOWorld icin tum siniflar (0,1,2,3..) artik bizim sectigimiz custom_obstacles listesine aittir.
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = float((x1 + x2) / 2)
                center_y = float((y1 + y2) / 2)

                msg = Float32MultiArray()
                msg.data = [center_x, center_y]
                self.publisher_.publish(msg)
                
                # Ekrana objenin adini yazdiralim
                obj_name = self.custom_obstacles[cls_id]
                self.get_logger().info(f"Engel ({obj_name}) Yakalandi -> X: {center_x:.1f}, Y: {center_y:.1f}")
                
                cv2.circle(annotated_frame, (int(center_x), int(center_y)), 10, (0, 0, 255), -1)
                break

        latest_annotated_frame = annotated_frame

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
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