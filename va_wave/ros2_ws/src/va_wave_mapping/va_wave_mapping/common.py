"""Ortak yardimcilar: goruntu/pointcloud donusumleri, geometri, kalibrasyon yukleme.

cv_bridge ve sensor_msgs_py'a kasitli olarak bagimli DEGILIZ; numpy 2.x ile ROS
Humble (numpy 1.x'e gore derlenmis) arasindaki ABI catismasini onlemek icin
Image ve PointCloud2 paketleme/cozme islemleri elle yapilir.
"""

import struct

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header


# ---------------------------------------------------------------------------
# Image <-> numpy (bgr8 / mono8 / 32FC1)
# ---------------------------------------------------------------------------
def image_to_msg(array, encoding, stamp, frame_id):
    msg = Image()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = array.shape[0]
    msg.width = array.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    if array.ndim == 3:
        msg.step = array.shape[1] * array.shape[2]
    else:
        msg.step = array.shape[1] * array.itemsize
    msg.data = np.ascontiguousarray(array).tobytes()
    return msg


def msg_to_image(msg):
    enc = msg.encoding
    if enc in ("bgr8", "rgb8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        return arr.reshape(msg.height, msg.width, 3)
    if enc in ("mono8", "8UC1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        return arr.reshape(msg.height, msg.width)
    if enc in ("32FC1",):
        arr = np.frombuffer(msg.data, dtype=np.float32)
        return arr.reshape(msg.height, msg.width)
    raise ValueError(f"Desteklenmeyen encoding: {enc}")


# ---------------------------------------------------------------------------
# PointCloud2 (xyz + rgb)
# ---------------------------------------------------------------------------
def make_pointcloud2(points_xyz, colors_bgr, stamp, frame_id):
    """points_xyz: (N,3) float32, colors_bgr: (N,3) uint8 veya None."""
    n = points_xyz.shape[0]
    has_rgb = colors_bgr is not None and len(colors_bgr) == n

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    if has_rgb:
        fields.append(
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)
        )
        point_step = 16
    else:
        point_step = 12

    buf = bytearray(point_step * n)
    xyz = np.ascontiguousarray(points_xyz, dtype=np.float32)
    if has_rgb:
        b = colors_bgr[:, 0].astype(np.uint32)
        g = colors_bgr[:, 1].astype(np.uint32)
        r = colors_bgr[:, 2].astype(np.uint32)
        rgb_int = (r << 16) | (g << 8) | b
        rgb_float = rgb_int.view(np.float32)
        packed = np.zeros((n, 4), dtype=np.float32)
        packed[:, 0:3] = xyz
        packed[:, 3] = rgb_float
        buf = packed.tobytes()
    else:
        buf = xyz.tobytes()

    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * n
    msg.is_dense = True
    msg.data = bytes(buf)
    return msg


def read_points_xyz(msg):
    """PointCloud2 -> (N,3) float32. Sadece x,y,z alanlarini okur."""
    offs = {f.name: f.offset for f in msg.fields}
    if not all(k in offs for k in ("x", "y", "z")):
        return np.empty((0, 3), np.float32)
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    out = np.empty((n, 3), np.float32)
    for i, k in enumerate(("x", "y", "z")):
        o = offs[k]
        out[:, i] = raw[:, o:o + 4].copy().view(np.float32).ravel()
    finite = np.isfinite(out).all(axis=1)
    return out[finite]


# ---------------------------------------------------------------------------
# Geometri: rotasyon matrisi <-> quaternion
# ---------------------------------------------------------------------------
def rotation_matrix_to_quaternion(R):
    """3x3 -> (x,y,z,w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) + 1e-12)


def quaternion_to_rotation_matrix(q):
    """(x,y,z,w) -> 3x3."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_aligning_vectors(a, b):
    """a vektorunu b'ye getiren rotasyon matrisi (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


# ---------------------------------------------------------------------------
# Yer duzlemi tahmini (RANSAC)
# ---------------------------------------------------------------------------
def fit_ground_plane(points, iterations=200, threshold=0.05, min_inliers_frac=0.15):
    """points: (N,3). Donen: (normal(3), d) ax+by+cz+d=0, normal birim.

    Yer duzlemi genelde en buyuk yatay yuzeydir. Kamera optigi cercevesinde
    Y ekseni asagi baktigi icin normal genelde -Y'ye yakin olur.
    """
    n = points.shape[0]
    if n < 50:
        return None, None, 0.0

    best_inliers = 0
    best_plane = None
    rng = np.random.default_rng(42)

    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p1))
        dist = np.abs(points @ normal + d)
        inliers = int(np.sum(dist < threshold))
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal, d)

    if best_plane is None or best_inliers < n * min_inliers_frac:
        return None, None, 0.0

    # Inlier'larla yeniden fit (en kucuk kareler)
    normal, d = best_plane
    dist = np.abs(points @ normal + d)
    inlier_pts = points[dist < threshold]
    centroid = inlier_pts.mean(axis=0)
    u, s, vt = np.linalg.svd(inlier_pts - centroid)
    normal = vt[2]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -float(np.dot(normal, centroid))
    confidence = best_inliers / n
    return normal, d, confidence


# ---------------------------------------------------------------------------
# Kalibrasyon yukleme
# ---------------------------------------------------------------------------
class StereoCalibration:
    def __init__(self, path):
        data = np.load(path)
        self.image_size = (int(data["image_width"]), int(data["image_height"]))
        self.mtxL = data["camera_matrix_left"]
        self.distL = data["dist_coeffs_left"]
        self.mtxR = data["camera_matrix_right"]
        self.distR = data["dist_coeffs_right"]
        self.R = data["R"]
        self.T = data["T"]
        self.R1 = data["R1"]
        self.R2 = data["R2"]
        self.P1 = data["P1"]
        self.P2 = data["P2"]
        self.Q = data["Q"]
        self.reprojection_error = float(data.get("reprojection_error", -1))
        self.baseline = float(np.linalg.norm(self.T))
