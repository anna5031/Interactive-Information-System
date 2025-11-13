# prewarp_projector_plane_from_distance.py
import numpy as np
import cv2

# -----------------------------
# 사용자 설정값
# -----------------------------
PROJ_W, PROJ_H   = 3840, 2160      # 프로젝터 해상도(px)
H_FOV_DEG        = 45.0            # 수평 FOV (deg)
INPUT_IMG_PATH   = "input_img.jpg"  # 원본 이미지

screen_vec  = np.array([1.0, 0.0, 1.0], dtype=np.float64)  # 프로젝터 중앙선(광축) 방향
wall_n_vec  = np.array([0.0, 0.0,  1.0], dtype=np.float64)  # 벽 법선 (정규화는 아래에서)
beam_wall_displacement = 0.4  # 원점(프로젝터)에서 벽 평면까지의 수직거리 [m]

# 벽 위 출력 화면 물리 크기 (16:9)
screen_width_m  = 1.77*0.2
screen_height_m = screen_width_m * 9.0 / 16.0

# -----------------------------
# 유틸
# -----------------------------
def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

def intrinsics_from_hfov(width, height, h_fov_deg):
    h_fov = np.deg2rad(h_fov_deg)
    fx = (width / 2.0) / np.tan(h_fov / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,  1]], dtype=np.float64)

def make_camera_rotation(forward_vec):
    z = normalize(forward_vec)                    # 카메라 z축(전방)
    up_world = np.array([0, 1, 0], np.float64)
    if abs(np.dot(z, up_world)) > 0.99:
        up_world = np.array([1, 0, 0], np.float64)
    x = normalize(np.cross(up_world, z))          # 카메라 x축(우)
    y = np.cross(z, x)                            # 카메라 y축(상)
    return np.vstack([x, y, z])                   # world->camera 회전

def plane_basis(n): # wall normal vector
    n = normalize(n)
    helper = np.array([0, 1, 0], np.float64)
    if abs(np.dot(n, helper)) > 0.99:
        helper = np.array([1, 0, 0], np.float64)
    u = normalize(np.cross(helper, n))            # 벽 평면 가로
    v = np.cross(n, u)                            # 벽 평면 세로
    return u, v, n

def project_points(K, R_wc, Xw):
    Xc = (R_wc @ Xw.T).T
    uvw = (K @ Xc.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv

# -----------------------------
# 1) 내부/외부 파라미터
# -----------------------------
K   = intrinsics_from_hfov(PROJ_W, PROJ_H, H_FOV_DEG)
R_wc = make_camera_rotation(screen_vec)

# -----------------------------
# 2) 벽 평면 정의 & screen_vec과 교점 계산
#     - 벽 평면: n·(X - P0) = 0,  P0 = n * beam_wall_displacement
#     - 광선:    X(t) = 0 + t * d,  d = normalize(screen_vec)
#     - 교점 t*: t* = n·P0 / (n·d)
# -----------------------------
n = normalize(wall_n_vec)
P0 = n * beam_wall_displacement
d  = normalize(screen_vec)

den = np.dot(n, d)
if abs(den) < 1e-9:
    raise RuntimeError("screen_vec이 벽 평면과 거의 평행합니다. 다른 screen_vec을 사용하세요.")

t_hit = np.dot(n, P0) / den
if t_hit <= 0:
    raise RuntimeError("광선이 벽 쪽으로 향하지 않습니다. screen_vec 방향을 확인하세요.")

wall_center_point = d * t_hit  # 요구한 방식으로 계산된 중심점

# -----------------------------
# 3) 벽 평면 위 16:9 사각형 모서리 3D 점 생성
# -----------------------------
u, v, _ = plane_basis(n)
half_w, half_h = screen_width_m * 0.5, screen_height_m * 0.5

P_BL = wall_center_point + (-half_w) * u + (+half_h) * v
P_BR = wall_center_point + (+half_w) * u + (+half_h) * v
P_TR = wall_center_point + (+half_w) * u + (-half_h) * v
P_TL = wall_center_point + (-half_w) * u + (-half_h) * v

wall_corners_3d = np.stack([P_TL, P_TR, P_BR, P_BL], axis=0)

# -----------------------------
# 4) 모서리들을 프로젝터 픽셀로 투영
# -----------------------------
dst_pts = project_points(K, R_wc, wall_corners_3d).astype(np.float64)  # (4,2)

# -----------------------------
# 5) 원본 이미지 4귀퉁이와 Homography 계산
# -----------------------------
img = cv2.imread(INPUT_IMG_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Cannot read {INPUT_IMG_PATH}")
H_img, W_img = img.shape[:2]

src_pts = np.array([[0,         0        ],  # TL
                    [W_img - 1, 0        ],  # TR
                    [W_img - 1, H_img - 1],  # BR
                    [0,         H_img - 1]], # BL
                   dtype=np.float64)

H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
np.save("distortion_matrix.npy", H)

# -----------------------------
# 6) 프리왜프 이미지 생성
# -----------------------------
distorted = cv2.warpPerspective(
    img, H, (PROJ_W, PROJ_H),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)
cv2.imwrite("distorted_img.jpg", distorted)

print("Done.")
print("wall_center_point:", wall_center_point)
print("Saved: distortion_matrix.npy, distorted_img.jpg")
