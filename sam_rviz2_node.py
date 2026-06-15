import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import struct
from transformers import pipeline
from PIL import Image as PILImage
from ultralytics import YOLO

# Web Sunucusu için eklentiler
import threading
from flask import Flask, Response, jsonify

app = Flask(__name__)

# Web arayüzüne gönderilecek anlık veriler
global_status = {
    "detected": False,
    "distance": 0.0
}
global_frame = None

WEB_UI_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>⚡ vA-Vision Tespit Arayüzü</title>
    <style>
        body {
            background-color: #0a0a0a;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
        }
        h1 {
            letter-spacing: 2px;
            font-weight: 300;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 20px;
        }
        .video-feed {
            border: 2px solid #333;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.8);
            max-width: 80%;
            height: auto;
        }
        .status-panel {
            margin-top: 20px;
            padding: 20px;
            background: #111;
            border-radius: 12px;
            width: 50%;
            display: flex;
            justify-content: space-around;
            box-shadow: 0 5px 15px rgba(0,0,0,0.5);
        }
        .indicator {
            font-size: 24px;
            font-weight: bold;
        }
        .searching { color: #FF9800; }
        .detected { color: #4CAF50; }
        .distance { color: #2196F3; }
    </style>
</head>
<body>
    <h1>vA-Vision</h1>
    
    <div class="container">
        <img class="video-feed" src="/video_feed" alt="Kamera Bekleniyor...">
        
        <div class="status-panel">
            <div id="detection-status" class="indicator searching">Aranıyor...</div>
            <div id="distance-status" class="indicator distance">Mesafe: 0.00m</div>
        </div>
    </div>

    <script>
        // Her 500ms'de bir durumu sunucudan çek
        setInterval(() => {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const statusEl = document.getElementById('detection-status');
                    const distEl = document.getElementById('distance-status');
                    
                    if (data.detected) {
                        statusEl.innerText = "Kupa Bulundu!";
                        statusEl.className = "indicator detected";
                        distEl.innerText = "Mesafe: " + data.distance + "m";
                    } else {
                        statusEl.innerText = "Aranıyor";
                        statusEl.className = "indicator searching";
                        distEl.innerText = "Mesafe: --";
                    }
                });
        }, 500);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return WEB_UI_HTML

@app.route('/status')
def status():
    return jsonify(global_status)

def generate_web_frame():
    global global_frame
    while True:
        if global_frame is not None:
            # Görüntüyü JPEG formatına çevir
            ret, buffer = cv2.imencode('.jpg', global_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Görüntü yoksa CPU'yu yormamak için çok az bekle
            cv2.waitKey(10)

@app.route('/video_feed')
def video_feed():
    return Response(generate_web_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask Sunucusunu Arka Planda Başlatma Fonksiyonu
def start_flask_app():
    # host='0.0.0.0' sayesinde Windows üzerinden 8080 portuyla erişilebilecek
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)


class SamRviz2Node(Node):
    def __init__(self):
        super().__init__('sam_rviz2_node')
        
        self.pc_pub = self.create_publisher(PointCloud2, '/cup_point_cloud', 10)
        self.img_pub = self.create_publisher(Image, '/detection_image', 10)
        self.bridge = CvBridge()
        
        # Docker içerisinden Windows host'una bağlanmak için stream adresi:
        self.stream_url = 'http://host.docker.internal:5000/video_feed'
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            self.get_logger().error(f"HATA: {self.stream_url} açılamadı! Lütfen Windows'ta camera_stream.py'nin çalıştığından emin ol.")
        else:
            self.get_logger().info(f"Kamera yayınına ({self.stream_url}) başarıyla bağlanıldı.")
            
        self.get_logger().info("YOLO Segmentasyon modeli yükleniyor... Lütfen bekleyin.")
        self.model = YOLO('yolov8n-seg.pt')
        self.get_logger().info("YOLO Segmentasyon modeli yüklendi.")
        
        self.get_logger().info("Depth Anything V2 modeli yükleniyor... (Bu işlem biraz zaman alabilir)")
        # Arka planda devasa derinlik modelini başlatıyoruz
        self.depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cpu")
        self.get_logger().info("Depth Anything V2 yüklendi! Sistem tamamen hazır.")
        
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        
        # Sınıflara göre tahmini GERÇEK HAYAT boyları (Metre cinsinden)
        self.real_heights_m = {
            41: 0.11, # Cup (11 cm)
            73: 0.24  # Book (24 cm)
        }
        
        self.class_names = {
            41: "Cup",
            73: "Book"
        }
        
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        global global_status, global_frame
        
        # Kamera önbelleğinde (buffer) biriken eski ve gecikmeli görüntüleri çöpe at.
        # Sadece TAM O ANKİ canlı kareyi al (Gecikmeyi/Lag'ı önler)
        for _ in range(4):
            self.cap.grab()
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        clean_frame = frame.copy() # Nokta bulutunu boyarken yeşil renk bulaşmasın diye
        results = self.model(frame, conf=0.15, verbose=False)
        detected = False
        
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
                
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            all_u_coords = []
            all_v_coords = []
            all_z_coords = []
            
            # Ekranda aradığımız hedef nesnelerden (Cup, Book) var mı diye kontrol et
            has_target = any(int(c) in self.real_heights_m for c in classes)
            if not has_target:
                continue # Ekranda nesne yoksa devasa 3D derinlik modelini boşuna çalıştırıp bilgisayarı kilitme!
            
            # Tüm sahne için derinlik haritasını SADECE bir kez çıkar (Çok ağır bir işlem olduğu için)
            depth_result = self.depth_estimator(PILImage.fromarray(cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)))
            depth_map = np.array(depth_result["depth"]).astype(np.float32)
            if depth_map.shape != (clean_frame.shape[0], clean_frame.shape[1]):
                depth_map = cv2.resize(depth_map, (clean_frame.shape[1], clean_frame.shape[0]))
            
            # Disparity (yakın yerler daha parlak/büyük) den relative derinliğe dönüştür (Z)
            z_rel_map = 255.0 / (depth_map + 1.0)
            
            for i, cls in enumerate(classes):
                cls_id = int(cls)
                if cls_id in self.real_heights_m:
                    mask = masks[i]
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    y_coords, x_coords = np.where(mask > 0)
                    if len(y_coords) == 0:
                        continue
                        
                    pixel_height = np.max(y_coords) - np.min(y_coords)
                    
                    if pixel_height > 10:
                        detected = True
                        real_h = self.real_heights_m[cls_id]
                        estimated_distance = (self.fy * real_h) / pixel_height
                        
                        # --- GÖRSEL ARAYÜZ (WEB) İÇİN ÇİZİMLER ---
                        # Maskeyi yeşile boya
                        colored_mask = np.zeros_like(frame)
                        colored_mask[mask > 0] = [0, 255, 0]
                        cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0, frame)
                        
                        # Bounding Box çiz
                        x1, y1, x2, y2 = map(int, boxes[i][:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{self.class_names[cls_id]}: {estimated_distance:.2f}m", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        global_status["detected"] = True
                        global_status["distance"] = round(float(estimated_distance), 2)
                        
                        all_u_coords.append(x_coords)
                        all_v_coords.append(y_coords)
                        
                        obj_rel_depths = z_rel_map[y_coords, x_coords]
                        
                        mean_rel_depth = np.mean(obj_rel_depths)
                        scale_factor = estimated_distance / mean_rel_depth
                        
                        z_coords_true = obj_rel_depths * scale_factor
                        all_z_coords.append(z_coords_true.astype(np.float32))
            
            if detected and len(all_u_coords) > 0:
                combined_u = np.concatenate(all_u_coords)
                combined_v = np.concatenate(all_v_coords)
                combined_z = np.concatenate(all_z_coords)
                self.publish_pointcloud(combined_u, combined_v, combined_z, 
                                        clean_frame)
                        
        if not detected:
            global_status["detected"] = False
            global_status["distance"] = 0.0
            
        # İşlenmiş görüntüyü RViz2'ye gönder
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera_link'
            self.img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Görüntü yayınlanamadı: {e}")

        # İşlenmiş (veya işlenmemiş) görüntüyü Web arayüzü için kaydet
        global_frame = frame.copy()


    def publish_pointcloud(self, u_coords, v_coords, z_coords, frame):
        z = z_coords
        x = (u_coords - self.cx) * z / self.fx
        y = (v_coords - self.cy) * z / self.fy
        
        colors = frame[v_coords, u_coords]
        
        structured_array = np.zeros(len(x), dtype=[('x', 'f4'), ('y', 'f4'), 
                                                   ('z', 'f4'), ('rgb', 'u4')])
        structured_array['x'] = x.astype(np.float32)
        structured_array['y'] = y.astype(np.float32)
        structured_array['z'] = z.astype(np.float32)
        
        b = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        r = colors[:, 2].astype(np.uint32)
        a = np.full_like(b, 255, dtype=np.uint32)
        
        # Bit düzeyinde (bitwise) renkleri tek bir sayıya birleştir
        rgba = (a << 24) | (r << 16) | (g << 8) | b
        structured_array['rgb'] = rgba
        
        points_byte_array = structured_array.tobytes()
            
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_link' 
        
        pc2_msg = PointCloud2()
        pc2_msg.header = header
        pc2_msg.height = 1
        pc2_msg.width = len(x)
        pc2_msg.is_dense = False
        pc2_msg.is_bigendian = False
        pc2_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        pc2_msg.point_step = 16
        pc2_msg.row_step = pc2_msg.point_step * len(x)
        pc2_msg.data = bytes(points_byte_array)
        
        self.pc_pub.publish(pc2_msg)


def main(args=None):
    # Flask sunucusunu arka planda başlat
    flask_thread = threading.Thread(target=start_flask_app, daemon=True)
    flask_thread.start()

    # ROS 2 düğümünü başlat
    rclpy.init(args=args)
    node = SamRviz2Node()
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
