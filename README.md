# vA-Vision Otonom Semantik Görüş Sistemi

Bu sistem, standart bir web kamerasını kullanarak **YOLOv8 Segmentasyonu** ve **Depth Anything V2** modellerini gerçek zamanlı bir şekilde çalıştırıp, otonom araçların dünyayı gördüğü gibi **3 Boyutlu Hacimsel Nokta Bulutları (Volumetric Point Clouds)** üretir.

## Özellikler
- **Semantik Segmentasyon**: Ekranda ne olursa olsun sadece tanımlı objeleri (Cup ve Book) arka plandan ayırır.
- **Monoküler Derinlik + Yapay Zeka Hacmi**: YOLO'dan gelen metrik uzaklık verisini (kaç santimetre uzakta olduğu), Depth Anything V2'nin sağladığı yoğun kavis ve hacim verisiyle birleştirerek nesnelere gerçekçi bir yuvarlaklık ve 3 boyutlu kıvrım kazandırır.
- **Canlı Web Arayüzü**: Tespitleri ve bounding box'ları anlık olarak tarayıcıdan izlemeni sağlar.
- **RViz2 Entegrasyonu**: Kavisli, hacimli 3 Boyutlu nesneleri orijinal renkleriyle dijital ROS 2 uzayında simüle eder.

---

## Çalıştırma Adımları

**Önemli Not:** Sistemi başlatırken terminalinizin (PowerShell veya CMD) `cbcbcbc` klasöründe (proje dizini) olduğundan ve **Docker Desktop** uygulamasının arka planda açık olduğundan emin olun.

### 1. Sistemi Başlatma
Terminalinizi açın, `cbcbcbc` klasörüne gidin ve şu komutu çalıştırın:
```powershell
docker compose up --build
```
*(İlk çalıştırmada devasa Depth Anything V2 modeli internetten indirileceği için "build" ve indirme işlemi birkaç dakika sürebilir. Lütfen terminalde "Depth Anything V2 yüklendi! Sistem tamamen hazır." yazısını görene kadar bekleyin).*

### 2. Canlı 2D Görüntüyü İzleme
Yapay zekanın vizöründen bakmak ve tespitleri canlı takip etmek için herhangi bir tarayıcıdan şu adrese gidin:
**[http://localhost:8080](http://localhost:8080)**

### 3. Asıl Büyü: 3 Boyutlu RViz2 Dünyası
Sistem açıldığında **RViz2** penceresi otomatik olarak karşınıza gelecektir. Hacimleri görmek için:

- **Görüntüyü Eklemek İçin:**
  Sol alttan `Add` butonuna tıklayın -> `Image` seçeneğini ekleyin.
  Sol menüye gelen Image ayarlarını açıp **Topic** kısmına tıklayarak `/detection_image` seçin. 
  *(Eğer görüntü donuksa altındaki Reliability Policy ayarını "Best Effort" yapın).*

- **3 Boyutlu Nokta Bulutunu Eklemek İçin:**
  Sol alttan tekrar `Add` butonuna tıklayın -> `PointCloud2` seçeneğini ekleyin.
  Sol menüye gelen PointCloud2 ayarlarından **Topic** kısmını `/cup_point_cloud` olarak seçin.
  Noktaları daha net görmek için altındaki **Size (m)** değerini `0.03` yapabilirsiniz.

### 4. Test Aşaması
Artık hazırsınız! Kameraya bir kupa ve kitap gösterdiğinizde, her ikisini aynı anda algılayacak ve RViz2 ekranında onlara düz kağıtlar gibi değil; kıvrımları, iç ve dış derinlikleri olan **gerçek hacimli cisimler** olarak bakabileceksiniz! Havada süzülen bu teknolojik sanat eserinin tadını çıkarın!
