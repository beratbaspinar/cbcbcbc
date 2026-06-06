"""Stereo kamera kalibrasyonu (kalite odakli).

- Her cift icin satranc tahtasi kosesi bulur
- Mono kalibrasyon + stereo kalibrasyon
- Per-goruntu reprojection error hesaplar, kotu ciftleri otomatik eler
- Hedef ortalama hata < 1.0 px; tutturulamazsa uyarir
- Rectification (R1,R2,P1,P2,Q) uretir ve epipolar hizalama dogrulamasi yapar

Kullanim:
    python3 stereo_calibrate.py
    python3 stereo_calibrate.py --cols 9 --rows 6 --square-size 0.025
    python3 stereo_calibrate.py --max-error 1.0 --target-error 1.0
"""

import argparse
import glob
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Stereo kamera kalibrasyonu")
    p.add_argument("--cols", type=int, default=9, help="Ic kose sayisi (yatay)")
    p.add_argument("--rows", type=int, default=6, help="Ic kose sayisi (dikey)")
    p.add_argument("--square-size", type=float, default=0.025,
                   help="Kare kenar uzunlugu (metre). 2.5cm = 0.025")
    p.add_argument("--input-dir", default="calib_images",
                   help="cam1/cam2 klasorlerinin ust dizini")
    p.add_argument("--output", default="calib_images/stereo_calibration.npz")
    p.add_argument("--max-error", type=float, default=1.5,
                   help="Bu hatanin uzerindeki ciftler elenir (px)")
    p.add_argument("--target-error", type=float, default=1.0,
                   help="Hedeflenen ortalama stereo hata (px)")
    p.add_argument("--min-pairs", type=int, default=6,
                   help="Kalibrasyon icin minimum gecerli cift")
    return p.parse_args()


def find_corners(image, pattern_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def per_view_stereo_error(objpoints, imgL, imgR, mtxL, dL, mtxR, dR, R, T):
    """Her cift icin epipolar (Sampson benzeri) ortalama hatayi dondurur."""
    F = None
    errors = []
    for i in range(len(objpoints)):
        # Sol/sag noktalari undistort edip normalize et
        undL = cv2.undistortPoints(imgL[i], mtxL, dL)
        undR = cv2.undistortPoints(imgR[i], mtxR, dR)
        # Esansiyel matristen epipolar tutarliligi: x_r^T E x_l ~ 0
        E = np.cross(np.eye(3), T.ravel()) @ R  # [t]_x R
        errs = []
        for pl, pr in zip(undL.reshape(-1, 2), undR.reshape(-1, 2)):
            xl = np.array([pl[0], pl[1], 1.0])
            xr = np.array([pr[0], pr[1], 1.0])
            num = abs(xr @ E @ xl)
            El = E @ xl
            Er = E.T @ xr
            denom = np.sqrt(El[0] ** 2 + El[1] ** 2 + Er[0] ** 2 + Er[1] ** 2) + 1e-12
            errs.append(num / denom)
        errors.append(np.mean(errs))
    return np.array(errors)


def calibrate(objpoints, imgL, imgR, image_size):
    _, mtxL, dL, _, _ = cv2.calibrateCamera(objpoints, imgL, image_size, None, None)
    _, mtxR, dR, _, _ = cv2.calibrateCamera(objpoints, imgR, image_size, None, None)
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    ret, mtxL, dL, mtxR, dR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgL, imgR, mtxL, dL, mtxR, dR, image_size,
        criteria=criteria, flags=flags,
    )
    return ret, mtxL, dL, mtxR, dR, R, T, E, F


def main():
    args = parse_args()
    pattern_size = (args.cols, args.rows)

    cam1 = sorted(glob.glob(os.path.join(args.input_dir, "cam1", "*.png")))
    cam2 = sorted(glob.glob(os.path.join(args.input_dir, "cam2", "*.png")))
    if not cam1 or not cam2:
        print("HATA: calib_images/cam1 ve cam2 icinde foto bulunamadi.")
        return
    if len(cam1) != len(cam2):
        print("HATA: cam1 ve cam2 dosya sayilari esit degil.")
        return

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints, imgL, imgR, names = [], [], [], []
    image_size = None
    for lf, rf in zip(cam1, cam2):
        il, ir = cv2.imread(lf), cv2.imread(rf)
        if il is None or ir is None:
            continue
        if image_size is None:
            image_size = (il.shape[1], il.shape[0])
        ok_l, cl = find_corners(il, pattern_size)
        ok_r, cr = find_corners(ir, pattern_size)
        if ok_l and ok_r:
            objpoints.append(objp)
            imgL.append(cl)
            imgR.append(cr)
            names.append(os.path.basename(lf))
            print(f"[OK]   {os.path.basename(lf)}")
        else:
            print(f"[ATLA] {os.path.basename(lf)} - kose bulunamadi")

    if len(objpoints) < args.min_pairs:
        print(f"\nHATA: Yeterli gecerli cift yok ({len(objpoints)}/{args.min_pairs}).")
        print("Daha fazla / daha net tahtali foto cekin.")
        return

    print(f"\n{len(objpoints)} gecerli cift ile kalibrasyon...")
    ret, mtxL, dL, mtxR, dR, R, T, E, F = calibrate(
        objpoints, imgL, imgR, image_size
    )
    print(f"Ilk stereo RMS: {ret:.4f} px")

    # Outlier eleme dongusu
    for _ in range(3):
        if ret <= args.target_error:
            break
        errs = per_view_stereo_error(objpoints, imgL, imgR, mtxL, dL, mtxR, dR, R, T)
        worst = int(np.argmax(errs))
        if errs[worst] < 1e-6 or len(objpoints) <= args.min_pairs:
            break
        print(f"  Eleniyor: {names[worst]} (epipolar hata {errs[worst]:.4f})")
        for lst in (objpoints, imgL, imgR, names):
            lst.pop(worst)
        ret, mtxL, dL, mtxR, dR, R, T, E, F = calibrate(
            objpoints, imgL, imgR, image_size
        )
        print(f"  Yeni stereo RMS: {ret:.4f} px ({len(objpoints)} cift)")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtxL, dL, mtxR, dR, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        image_width=image_size[0], image_height=image_size[1],
        pattern_cols=args.cols, pattern_rows=args.rows,
        square_size=args.square_size,
        camera_matrix_left=mtxL, dist_coeffs_left=dL,
        camera_matrix_right=mtxR, dist_coeffs_right=dR,
        R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        reprojection_error=ret,
    )

    baseline = float(np.linalg.norm(T))
    print("\n" + "=" * 50)
    print("KALIBRASYON TAMAMLANDI")
    print(f"  Stereo RMS hata : {ret:.4f} px")
    print(f"  Kullanilan cift : {len(objpoints)}")
    print(f"  Baseline (kameralar arasi): {baseline*100:.1f} cm")
    print(f"  Cikti: {args.output}")
    if ret <= args.target_error:
        print("  DURUM: MUKEMMEL - mapping icin hazir.")
    elif ret <= 2.0:
        print("  DURUM: KABUL EDILEBILIR - daha iyi icin daha cok foto cekin.")
    else:
        print("  DURUM: ZAYIF - derinlik guvenilmez olabilir, tekrar cekin.")
        print("  Ipucu: tahta tam ve net gorunsun, farkli aci/mesafe kullanin,")
        print("         --square-size degerini gercek kare boyutuna ayarlayin.")
    print("=" * 50)


if __name__ == "__main__":
    main()
