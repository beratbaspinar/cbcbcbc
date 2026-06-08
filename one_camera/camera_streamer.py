import cv2
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


# Laptop 1'in kendi yerlesik (built-in) kamerasini aciyoruz
cap = cv2.VideoCapture(0)

# Agda gecikme olmamasi icin cozunurlugu sabitliyoruz (Isterseniz 1280x720 yapabilirsiniz)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

latest_frame = None

class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                if latest_frame is not None:
                    # Goruntuyu JPEG formatina cevirip ag uzerinden gonderiyoruz
                    ret, jpeg = cv2.imencode('.jpg', latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        try:
                            self.wfile.write(b'--frame\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', str(len(jpeg.tobytes())))
                            self.end_headers()
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                        except:
                            break
                time.sleep(0.03)

def run_server():
    server = ThreadingHTTPServer(('0.0.0.0', 5000), VideoStreamHandler)
    server.serve_forever()

# Web sunucusunu ayri bir kanalda baslat
threading.Thread(target=run_server, daemon=True).start()

local_ip = get_local_ip()
print("==================================================")
print("Laptop 1 Yayina Basladi!")
print(f"IP adresiniz: {local_ip}")
print(f"Stream URL:   http://{local_ip}:5000/video_feed")
print("Bu IP'yi diger laptopta laptop2_capture.py icin kullanin.")
print("Bu pencereyi kapatmadiginiz surece yayin devam eder.")
print("==================================================")

while True:
    ret, frame = cap.read()
    if ret:
        latest_frame = frame
