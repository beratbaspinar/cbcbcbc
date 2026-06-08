"""Stereo kalibrasyon goruntu yakalama (iki laptop).

Sol kamera  = Windows HTTP stream (cam1)
Sag kamera  = Mac lokal kamera   (cam2)

Ozellikler:
- Canli satranc tahtasi kose tespiti (iki kamerada da yesil overlay)
- Otomatik yakalama: iki kamerada da kose bulunur + netse + onceki kareden
  yeterince farkliysa otomatik kaydeder
- Manuel 'S' ile de kaydedilir
- Netlik (Laplacian varyansi) ve kapsama kontrolu

Kullanim:
    python3 laptop2_capture.py 192.168.1.191
    python3 laptop2_capture.py 192.168.1.191 --cols 9 --rows 6
    python3 laptop2_capture.py 192.168.1.191 --no-auto
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iki laptop stereo kalibrasyon goruntu yakalama"
    )
    parser.add_argument(
        "remote_ip",
        nargs="?",
        default=os.environ.get("LAPTOP1_IP", ""),
        help="Laptop 1 (Windows) IP adresi, ornek: 192.168.1.191",
    )
    parser.add_argument("--cols", type=int, default=9, help="Ic kose sayisi (yatay)")
    parser.add_argument("--rows", type=int, default=6, help="Ic kose sayisi (dikey)")
    parser.add_argument(
        "--local-camera", type=int, default=0, help="Mac lokal kamera index"
    )
    parser.add_argument(
        "--no-auto", action="store_true", help="Otomatik yakalamayi kapat"
    )
    parser.add_argument(
        "--sharpness", type=float, default=80.0,
        help="Minimum netlik (Laplacian varyansi) esigi",
    )
    parser.add_argument(
        "--output-dir", default="calib_images", help="Cikti klasoru"
    )
    return parser.parse_args()


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_corners(gray, pattern_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    return found, corners


def corner_signature(corners):
    """Kareler arasi farki olcmek icin basit imza (kose merkezi)."""
    return corners.reshape(-1, 2).mean(axis=0)


def main():
    args = parse_args()
    if not args.remote_ip:
        print("HATA: Laptop 1 (Windows) IP adresi gerekli!")
        print("  python3 laptop2_capture.py 192.168.1.191")
        sys.exit(1)

    pattern_size = (args.cols, args.rows)
    cam1_dir = os.path.join(args.output_dir, "cam1")
    cam2_dir = os.path.join(args.output_dir, "cam2")
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(cam2_dir, exist_ok=True)

    stream_url = f"http://{args.remote_ip}:5000/video_feed"
    print(f"Laptop 1'e baglaniliyor... ({stream_url})")
    cap_remote = cv2.VideoCapture(stream_url)
    cap_remote.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap_local = cv2.VideoCapture(args.local_camera)
    cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap_local.isOpened() or not cap_remote.isOpened():
        print("HATA: Kameralarin ikisine birden ulasilamadi!")
        print("- Windows'ta laptop1_streamer.py calisiyor mu?")
        print("- IP dogru mu, iki cihaz ayni agda mi?")
        sys.exit(1)

    print("=" * 55)
    print("STEREO KALIBRASYON YAKALAMA")
    print(f"Tahta: {args.cols}x{args.rows} ic kose")
    print(f"Otomatik yakalama: {'KAPALI' if args.no_auto else 'ACIK'}")
    print("KISAYOLLAR:  S=manuel kaydet  Q=cikis  A=oto ac/kapat")
    print("=" * 55)

    img_count = 0
    last_sig = None
    last_auto_time = 0.0
    auto_enabled = not args.no_auto

    while True:
        ok_r, frame_remote = cap_remote.read()
        ok_l, frame_local = cap_local.read()
        if not ok_r or not ok_l:
            continue

        gray_r = cv2.cvtColor(frame_remote, cv2.COLOR_BGR2GRAY)
        gray_l = cv2.cvtColor(frame_local, cv2.COLOR_BGR2GRAY)

        found_r, corners_r = find_corners(gray_r, pattern_size)
        found_l, corners_l = find_corners(gray_l, pattern_size)

        sh_r = sharpness(gray_r)
        sh_l = sharpness(gray_l)

        disp_r = frame_remote.copy()
        disp_l = frame_local.copy()
        if found_r:
            cv2.drawChessboardCorners(disp_r, pattern_size, corners_r, found_r)
        if found_l:
            cv2.drawChessboardCorners(disp_l, pattern_size, corners_l, found_l)

        both_found = found_r and found_l
        sharp_ok = sh_r >= args.sharpness and sh_l >= args.sharpness

        # Durum yazilari
        def status(img, found, sh, name):
            color = (0, 255, 0) if found else (0, 0, 255)
            cv2.putText(img, f"{name}: {'KOSE VAR' if found else 'KOSE YOK'}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            scolor = (0, 255, 0) if sh >= args.sharpness else (0, 165, 255)
            cv2.putText(img, f"netlik: {sh:.0f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, scolor, 1)

        status(disp_l, found_l, sh_l, "cam2 (Mac/Sag)")
        status(disp_r, found_r, sh_r, "cam1 (Win/Sol)")
        cv2.putText(disp_l, f"Kaydedilen cift: {img_count}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(disp_l, f"OTO: {'ACIK' if auto_enabled else 'KAPALI'}",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("cam1 (Windows / Sol)", disp_r)
        cv2.imshow("cam2 (Mac / Sag)", disp_l)

        should_save = False
        now = time.time()

        # Otomatik yakalama kosullari
        if auto_enabled and both_found and sharp_ok and (now - last_auto_time) > 1.0:
            sig = corner_signature(corners_l)
            moved_enough = last_sig is None or np.linalg.norm(sig - last_sig) > 15.0
            if moved_enough:
                should_save = True
                last_sig = sig
                last_auto_time = now

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("s"), ord("S")):
            should_save = True
            if both_found:
                last_sig = corner_signature(corners_l)
        elif key in (ord("a"), ord("A")):
            auto_enabled = not auto_enabled
        elif key in (ord("q"), ord("Q")):
            break

        if should_save:
            if not both_found:
                print("[ATLA] Iki kamerada da tahta gorunmuyor, kaydedilmedi.")
            else:
                name1 = os.path.join(cam1_dir, f"stereo_{img_count:02d}.png")
                name2 = os.path.join(cam2_dir, f"stereo_{img_count:02d}.png")
                cv2.imwrite(name1, frame_remote)
                cv2.imwrite(name2, frame_local)
                tag = "OTO" if (auto_enabled and key not in (ord("s"), ord("S"))) else "MANUEL"
                print(f"[{img_count}] ({tag}) kaydedildi  netlik L={sh_r:.0f} R={sh_l:.0f}")
                img_count += 1

    cap_local.release()
    cap_remote.release()
    cv2.destroyAllWindows()
    print(f"\nToplam {img_count} cift kaydedildi.")
    if img_count >= 12:
        print("Iyi sayida cift var. Kalibrasyon:")
        print(f"  python3 stereo_calibrate.py --cols {args.cols} --rows {args.rows}")
    else:
        print("En az 12-20 cift onerilir (farkli aci/mesafe/koseler).")


if __name__ == "__main__":
    main()
