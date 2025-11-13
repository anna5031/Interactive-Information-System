# H_calculator.py
import numpy as np
import cv2

try:
    from nudge_test.motor.config_loader import load_config
except ModuleNotFoundError:  # pragma: no cover
    from config_loader import load_config


def _safe_vector(value, default):
    """Return np.array for vector settings, falling back to default on bad input."""
    arr = np.asarray(default, dtype=np.float64)
    if value is None:
        return arr
    try:
        candidate = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError):
        return arr
    if candidate.shape != arr.shape:
        try:
            candidate = candidate.reshape(arr.shape)
        except ValueError:
            return arr
    return candidate


def _parse_ratio(value, default):
    """Parse width/height ratio from numeric, iterable, or 'a/b' string."""
    if value is None:
        return default
    if isinstance(value, str) and "/" in value:
        num, den = value.split("/", 1)
        try:
            num, den = float(num), float(den)
            return default if abs(den) < 1e-9 else num / den
        except ValueError:
            return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            num, den = float(value[0]), float(value[1])
            return default if abs(den) < 1e-9 else num / den
        except (TypeError, ValueError):
            return default
    try:
        ratio = float(value)
        return default if abs(ratio) < 1e-9 else ratio
    except (TypeError, ValueError):
        return default


config = load_config()

################### configurations
# Pan Tilt geo config
pt_h = float(config.get("pan_tilt_height", 1600.0))   # projector height (mm)
pt_pos = np.array([0.0, 0.0, pt_h], dtype=np.float64)   # projector origin

# Projector geo config
beam_offset_vec = _safe_vector(
    config.get("projector_offset_vector"),
    [50.0, 0.0, 80.0],
)  # pan tilt에서 본 빔프로젝터 light source 위치 벡터 (mm)

# Projector config
PROJ_W, PROJ_H = 3840, 2160
H_FOV_DEG = 45.0

# ui image size
IMG_W, IMG_H = 3840, 2160

# nudge plane config
plane_cfg = config.get("nudge_plane_geometry", {})
ceiling_n_w = _safe_vector(
    plane_cfg.get("normal_vector"),
    [0.0, 0.0, 1.0],
)   # 천장 법선(+z)
ceiling_height = float(plane_cfg.get("nudge_plane_displacement", 2500.0))  # 천장 높이 (mm)

# nudge screen config
screen_cfg = config.get("nudge_screen_geometry", {})
screen_width_mm = float(screen_cfg.get("screen_width", 354.0))
screen_ratio = max(1e-9, _parse_ratio(screen_cfg.get("screen_ratio"), 16.0 / 9.0))  # width / height
screen_height_mm = screen_width_mm / screen_ratio
# Keep calculations in millimeters to stay consistent with other coordinates.
SCREEN_WIDTH_MM  = screen_width_mm
SCREEN_HEIGHT_MM = screen_height_mm
ROLL_DEG = float(screen_cfg.get("roll_deg", 0.0))  # 평면 내 추가 회전

def _intrinsics_from_hfov(width, height, h_fov_deg):
    h = np.deg2rad(h_fov_deg)
    fx = (width/2.0)/np.tan(h/2.0); fy = fx
    cx, cy = width/2.0, height/2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
PROJECTOR_INTRINSICS = _intrinsics_from_hfov(PROJ_W, PROJ_H, H_FOV_DEG)

def _normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

def _rodrigues_rotate(vec, axis, angle_rad):
    a = _normalize(axis); v = vec
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return v*c + np.cross(a, v)*s + a*np.dot(a, v)*(1-c)

# 카메라 좌표계 축: right = down × z (오른손), down = y_cam(+)
# 카메라 좌표축(right, down, forward)을 회전 행렬에서 추출
def _make_camera_axes_from_rotation(R):
    forward_w = _normalize(R[:, 0])
    right_w = _normalize(-R[:, 1])
    down_w = _normalize(-R[:, 2])
    R_wc = np.vstack([right_w, down_w, forward_w])
    return R_wc, (right_w, down_w, forward_w)

def _compute_nudge_center_on_ray(n_w, distance, ray_dir_w, ray_origin_w):
    n = _normalize(n_w) # nudge plane normal vector 
    P0 = n * distance  # point on nudge plane: located `distance` away from origin along normal
    d = _normalize(ray_dir_w) # nudge vector normalized
    o = np.asarray(ray_origin_w, dtype=np.float64) # ray origin
    den = np.dot(n, d) # 비율 계산
    if abs(den) < 1e-9:
        print("nudge_vec:", ray_dir_w)
        raise RuntimeError("nudge_vec이 벽과 거의 평행")
    t = np.dot(n, P0 - o) / den # nudge point와 빔 사이 거리
    if t <= 0: raise RuntimeError("광선이 벽을 향하지 않음")
    return o + d * t # 빔 시작 지점 + 거리*방향 벡터

# 월드 y(왼쪽)과 가로(16)를 평행 정렬
def _plane_basis_align_with_world_y(n_w, roll_deg=0.0, ensure_positive=True):
    n = _normalize(n_w)
    world_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u0 = world_y - np.dot(world_y, n)*n
    if np.linalg.norm(u0) < 1e-9:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u0 = helper - np.dot(helper, n)*n
    u = _normalize(u0)
    if ensure_positive and np.dot(u, world_y) < 0: u = -u
    v = _normalize(np.cross(n, u))  # 오른손: u×v = n
    if abs(roll_deg) > 1e-9:
        a = np.deg2rad(roll_deg)
        u, v = _rodrigues_rotate(u, n, a), _rodrigues_rotate(v, n, a)
    return u, v, n

def _project_points_world(K, R_wc, Xw, cam_origin_w):
    Xw = np.asarray(Xw, dtype=np.float64)
    cam_origin = np.asarray(cam_origin_w, dtype=np.float64)
    Xc = (R_wc @ (Xw - cam_origin).T).T
    uvw = (K @ Xc.T).T
    return uvw[:, :2]/uvw[:, 2:3]

def _cal_beam_rotion_matrix(z_rad, y_rad):
    Rz = np.array([
        [ np.cos(z_rad), -np.sin(z_rad), 0.0],
        [ np.sin(z_rad),  np.cos(z_rad), 0.0],
        [      0.0    ,       0.0     , 1.0]
    ], dtype=np.float64)
    Ry = np.array([
        [ np.cos(y_rad), 0.0, np.sin(y_rad)],
        [     0.0     , 1.0,      0.0     ],
        [-np.sin(y_rad), 0.0, np.cos(y_rad)]
    ], dtype=np.float64)
    return Rz @ Ry

def _cal_nudge_vector(R):
    return R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)

def _cal_nudge_corners(u_w, v_w, center_w):
    half_w, half_h = SCREEN_WIDTH_MM/2.0, SCREEN_HEIGHT_MM/2.0
    P_TL = center_w + (-half_w)*u_w + (+half_h)*v_w
    P_TR = center_w + (+half_w)*u_w + (+half_h)*v_w
    P_BR = center_w + (+half_w)*u_w + (-half_h)*v_w
    P_BL = center_w + (-half_w)*u_w + (-half_h)*v_w
    return np.stack([P_TL, P_TR, P_BR, P_BL], axis=0)

def H_caculator(pan, tilt, src_pts=None, return_wall_center=False):
    # 월드 좌표계: x=앞, y=왼, z=위

    # radian 변환
    tilt_rad = np.deg2rad(tilt)
    pan_rad  = np.deg2rad(pan)

    # world 좌표계에서의 회전 각도
    z_rad = pan_rad
    y_rad = -tilt_rad

    # beam position 회전 행렬 R = Rz(pan) Ry(-tilt)
    R = _cal_beam_rotion_matrix(z_rad, y_rad)

    # beam starting point
    beam_pos = pt_pos + R @ beam_offset_vec  # 빔프로젝터 월드 좌표

    # nudge vector in world coordinates
    nudge_vec_w = _cal_nudge_vector(R)

    # world 기준 빔프로젝터 좌표계 축들
    R_wc, _ = _make_camera_axes_from_rotation(R) 

    # 벽면 중심점 계산
    nudge_center_w = _compute_nudge_center_on_ray(
        ceiling_n_w,
        ceiling_height,
        nudge_vec_w,
        beam_pos
    )

    # 가로=월드 y 정렬
    u_w, v_w, _ = _plane_basis_align_with_world_y(ceiling_n_w, roll_deg=ROLL_DEG, ensure_positive=True)
    nudge_corners_w = _cal_nudge_corners(u_w, v_w, nudge_center_w)
    dst_pts = _project_points_world(PROJECTOR_INTRINSICS, R_wc, nudge_corners_w, beam_pos).astype(np.float64)
    if src_pts is None:
        src_pts = np.array([[0,0],[IMG_W-1,0],[IMG_W-1,IMG_H-1],[0,IMG_H-1]], dtype=np.float64)

    H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
    if return_wall_center:
        return H, nudge_center_w
    return H

def test_homography(test_pan, test_tilt):
    INPUT_IMG_PATH = "input_img.jpg"

    img = cv2.imread(INPUT_IMG_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {INPUT_IMG_PATH}")

    H_img, W_img = img.shape[:2]
    # print("Image size:", W_img, "x", H_img)

    src_pts = np.array([[0,0],[W_img-1,0],[W_img-1,H_img-1],[0,H_img-1]], dtype=np.float64)
    H, wall_center_w = H_caculator(test_pan, test_tilt, src_pts=src_pts, return_wall_center=True)

    distorted = cv2.warpPerspective(
        img,
        H,
        (PROJ_W, PROJ_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0)
    )
    cv2.imwrite("distorted_img.jpg", distorted)

    print("wall_center_w:", wall_center_w)
    print("Saved: distorted_img.jpg")

        # Show the distorted image in fullscreen for visual verification.
    window_name = "Distorted Projection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, distorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_pan = 45
    test_tilt = 45
    test_homography(test_pan, test_tilt)
