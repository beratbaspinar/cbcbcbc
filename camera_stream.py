import cv2
from flask import Flask, Response
import time

app = Flask(__name__)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("HATA: Bilgisayarının kamerası açılamıyor. Başka bir uygulama (Zoom vb.) kamerayı kullanıyor olabilir.")
    exit(1)

print("Kamera başarıyla açıldı. Görüntü Docker'a aktarılmak üzere yayınlanıyor...")
print("Lütfen bu siyah ekranı KAPATMAYIN. Sistemi durdurmak için CTRL+C yapabilirsiniz.")

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Görüntüyü hafifletip JPEG'e çeviriyoruz
            # Çözünürlüğü biraz düşürmek performansı artırır (İsteğe bağlı)
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
@app.route('/video_feed')
def video_feed():
    # Docker veya Tarayıcı bu URL'den videoyu çekecek
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Tüm ağdan erişilebilir olması için host='0.0.0.0'
    app.run(host='0.0.0.0', port=5000, threaded=True)
