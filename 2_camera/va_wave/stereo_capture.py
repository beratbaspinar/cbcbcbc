import cv2
import os

# Kalibrasyon fotograflari icin klasorler olusturalim
os.makedirs("calib_images/cam1", exist_ok=True)
os.makedirs("calib_images/cam2", exist_ok=True)

# 0 ve 1 genelde Windows'a bagli ilk iki kamerayi (Webcam 1 ve Webcam 2) temsil eder
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

if not cap1.isOpened() or not cap2.isOpened():
    print("HATA: Iki kamera ayni anda acilamadi! Lutfen bilgisayara 2 kameranin da bagli oldugundan emin olun.")
    print("Eger sadece 1 kamera bagliysa, program hata verecektir.")
    exit()

print("Kameralar basariyla acildi!")
print("=========================================")
print("KLAVYE KISAYOLLARI:")
print(" 'S' veya 's' -> İki kameradan ayni anda SATRANC TAHTASI fotografi cek (Kaydet)")
print(" 'Q' veya 'q' -> Cikis")
print("=========================================")

img_count = 0

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("HATA: Kameralarin birinden goruntu akisi kesildi.")
        break

    # Goruntuleri ekranda gosterelim
    cv2.imshow("Kamera 1 (Sol)", frame1)
    cv2.imshow("Kamera 2 (Sag)", frame2)

    # 1 ms bekle ve klavye tusunu dinle
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') or key == ord('S'):
        # Fotograflari ayni anda kaydediyoruz
        img_name1 = f"calib_images/cam1/stereo_{img_count:02d}.png"
        img_name2 = f"calib_images/cam2/stereo_{img_count:02d}.png"
        cv2.imwrite(img_name1, frame1)
        cv2.imwrite(img_name2, frame2)
        print(f"[{img_count}] Fotograf Cifti Kaydedildi! (Satranc tahtasini baska bir aciya cevirip tekrar cekin)")
        img_count += 1
    elif key == ord('q') or key == ord('Q'):
        print("Cikis yapiliyor...")
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
