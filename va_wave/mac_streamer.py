"""Mac lokal kamerayi HTTP stream olarak yayinlar (port 5001).

Docker icindeki stereo_node bu yayini host.docker.internal:5001 uzerinden okur.

Kullanim:
    python3 mac_streamer.py
"""

import socket
import threading
import time

import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 5001
latest_frame = None


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/video_feed":
            self.send_response(200)
            self.send_header(
                "Content-type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            while True:
                if latest_frame is not None:
                    ok, jpeg = cv2.imencode(
                        ".jpg", latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
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
                time.sleep(0.03)


def run_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), VideoStreamHandler)
    server.serve_forever()


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

threading.Thread(target=run_server, daemon=True).start()

local_ip = get_local_ip()
print("=" * 50)
print("Mac Kamera Yayini Basladi!")
print(f"IP: {local_ip}")
print(f"URL: http://{local_ip}:{PORT}/video_feed")
print("Docker stereo icin: CAM2_URL=http://host.docker.internal:5001/video_feed")
print("=" * 50)

while True:
    ok, frame = cap.read()
    if ok:
        latest_frame = frame
