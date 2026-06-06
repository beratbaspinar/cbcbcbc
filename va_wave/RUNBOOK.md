# va_wave - Sabit Stereo Kamera → ROS2 Mapping Runbook

İki laptop (Windows = sol kamera, Mac = sağ kamera) ile sabit stereo kurulumdan
canlı occupancy map + 3D tespit üretir. Çıktı hem Foxglove/RViz2 hem de
`wave screen.html` arayüzünde görünür.

## Mimari (özet)
```
Windows kamera :5000 ─┐
                      ├─> camera_ingest (rectify) ─> stereo_depth ─> /stereo/points, /stereo/depth, map TF
Mac kamera     :5001 ─┘                          │
                         detection (YOLO-World) ──┴─> fusion ─> /detections_3d ─┐
                                                                                ├─> mapping ─> /map, /obstacle_cloud
                                                                                │
                              wave_bridge (ws :8765) <── /map, /detections_3d ──┘
                              foxglove_bridge (:8766)
```

## 0) Kalibrasyon (yalnızca bir kez veya kurulum değişince)
Reprojection error < 1.0px hedefle.

Windows'ta (sol kamera):
```
python laptop1_streamer.py
```
Mac'te (sağ kamera + yakalama):
```
python3 laptop2_capture.py <WINDOWS_IP>      # ör. 192.168.1.191
```
- İki pencerede de satranç tahtasını gezdir. Köşeler bulunup net olunca
  **otomatik** kaydedilir (`A` ile aç/kapa, `S` manuel, `Q` çıkış).
- 12–20 çift topla (farklı açı/mesafe/köşe).

Kalibrasyonu hesapla:
```
python3 stereo_calibrate.py --cols 9 --rows 6 --square-size 0.025
```
- `calib_images/stereo_calibration.npz` üretir. "MUKEMMEL" görürsen hazırsın.

## 1) Canlı yayınları başlat
Windows (sol):
```
python laptop1_streamer.py            # http://<WINDOWS_IP>:5000/video_feed
```
Mac (sağ):
```
python3 mac_streamer.py               # http://localhost:5001/video_feed
```

## 2) Mapping hattını başlat (Docker, Mac)
```
cd "va_wave"
mkdir -p maps
LEFT_STREAM_URL=http://<WINDOWS_IP>:5000/video_feed docker compose -f docker-compose.stereo.yml up --build
```
İlk çalıştırmada imaj derlenir (colcon build dahil). Sonraki başlatmalarda
`--build` olmadan açabilirsin. Kod değiştirdiysen sadece `restart` yeterli
(src bind-mount + `--symlink-install`).

Faydalı override'lar:
- `ENABLE_DETECTION=0` → YOLO'yu kapat (hafif test)
- `ENABLE_FOXGLOVE=0` → foxglove_bridge'i kapat

## 3) Görselleştirme
- **wave arayüzü**: `wave adaptation/wave screen.html` aç. Otomatik
  `ws://localhost:8765/ws/sensing`'e bağlanır; occupancy ızgarası + tespitler.
- **Foxglove Studio**: `Open connection` → `ws://localhost:8766`. Eklenecekler:
  `/map`, `/stereo/points`, `/obstacle_cloud`, `/detections_3d/markers`,
  `/detections_2d/image`, `TF`.
- **RViz2** (ROS2 kurulu masaüstünde):
  ```
  ros2 launch va_wave_mapping mapping.launch.py   # veya
  rviz2 -d ros2_ws/src/va_wave_mapping/rviz/mapping.rviz
  ```

## Topic'ler
| Topic | Tip | Açıklama |
|------|-----|----------|
| `/stereo/left/image_rect` | sensor_msgs/Image | rectified sol |
| `/stereo/right/image_rect`| sensor_msgs/Image | rectified sağ |
| `/stereo/points` | sensor_msgs/PointCloud2 | renkli 3D bulut |
| `/stereo/depth` | sensor_msgs/Image (32FC1) | derinlik (m) |
| `/detections_2d` | vision_msgs/Detection2DArray | YOLO kutular |
| `/detections_3d` | vision_msgs/Detection3DArray | 3D tespitler |
| `/detections_3d/markers` | visualization_msgs/MarkerArray | etiketli küpler |
| `/map` | nav_msgs/OccupancyGrid | kuşbakışı harita |
| `/obstacle_cloud` | sensor_msgs/PointCloud2 | map'te engeller |
| TF `map→camera_optical_frame` | yer düzleminden | otomatik |

## Sorun giderme
- **Kamera bağlanmıyor**: `laptop1_streamer.py` / `mac_streamer.py` çalışıyor mu?
  Aynı ağ + güvenlik duvarı (Windows: 5000 portuna izin). Hotspot kullanılıyorsa
  IP değişir → `LEFT_STREAM_URL`'i güncelle.
- **Harita boş / yer düzlemi yok**: kamera zemini görmeli; bulunamazsa
  `fallback_camera_height` / `fallback_tilt_deg` (params.yaml) kullanılır.
- **Derinlik gürültülü**: kalibrasyon hatası yüksek olabilir → Faz 0'ı tekrarla.
- **WLS yok uyarısı**: `opencv-contrib-python-headless` kurulu olmalı (Dockerfile'da var).
- **Port çakışması**: wave_bridge 8765, foxglove 8766 — ikisi farklı.

## Parametreler
Tümü `ros2_ws/src/va_wave_mapping/config/params.yaml` içinde. Önemliler:
`num_disparities`, `max_range`, `resolution`/`width`/`height`,
`obstacle_min_height`, `hit_increment`/`miss_decrement`/`decay_period`.
