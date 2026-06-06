# va_wave — Sabit Stereo Kamera → ROS2 Mapping

İki laptop (Windows + Mac) stereo kamera kurulumu ile ROS2 tabanlı algılama ve haritalama sistemi.

## İçerik

- `va_wave/` — Docker, ROS2 paketi, kalibrasyon araçları, streamer scriptleri
- `wave adaptation/` — Wave radar arayüzü (`wave screen.html`)

## Hızlı başlangıç

Ayrıntılı adımlar: [`va_wave/RUNBOOK.md`](va_wave/RUNBOOK.md)

```bash
# Windows
python laptop1_streamer.py

# Mac
python3 mac_streamer.py
cd va_wave
LEFT_STREAM_URL=http://<WINDOWS_IP>:5000/video_feed \
docker compose -f docker-compose.stereo.yml up --build
```

## Gerekli dosyalar (git'e dahil değil)

Docker build öncesi `va_wave/` içine koy:

- `yolov8s-worldv2.pt` — YOLO-World modeli
- `calib_images/stereo_calibration.npz` — stereo kalibrasyon (veya yeniden üret)

Kalibrasyon:

```bash
python3 laptop2_capture.py <WINDOWS_IP>
python3 stereo_calibrate.py --cols 9 --rows 6
```
