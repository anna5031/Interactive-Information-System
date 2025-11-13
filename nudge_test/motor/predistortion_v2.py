# prewarp_projector_refactored.py
import numpy as np
import cv2
import json

# =============================
# 사용자 설정 (x=앞, y=왼쪽, z=위) -> opencv 좌표계 (x=오른쪽, y=아래, z=앞)
# =============================
PROJ_W, PROJ_H = 3840, 2160        # 프로젝터 해상도(px)
H_FOV_DEG      = 45.0              # 수평 FOV (deg)
INPUT_IMG_PATH = "input_img.jpg"   # 원본 이미지

# 광축(화면이 '앞으로' 길어지는 케이스 예: x=앞, z=위)
screen_vec_user = np.array([1.0, 0.0, 1.0], dtype=np.float64)

# 벽의 법선: '위'를 바라보는 벽이면 z=1
wall_n_vec_user = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# 프로젝터 원점에서 벽까지 수직거리(법선방향) [m]
beam_wall_displacement = 0.4

# 벽 위 16:9 화면 물리 크기 [m]
screen_width_m  = 1.77 * 0.2
screen_height_m = screen_width_m * 9.0 / 16.0

# =============================
# 유틸
# =============================
def normalize(v):
    v = np.asarray(v, dtype=np.float64) # 배열 변환
    n = np.linalg.norm(v) # 벡터 크기
    return v if n < 1e-12 else v / n


def user_to_internal(vu):
    """사용자(x=앞, y=왼쪽, z=위) -> 내부(X=오른쪽, Y=아래, Z=앞)"""
    x_u, y_u, z_u = vu
    X_i = -y_u          # 오른쪽 = -왼쪽
    Y_i = -z_u          # 위     =  위
    Z_i = x_u          # 앞     =  앞
    return np.array([X_i, Y_i, Z_i], dtype=np.float64)

def intrinsics_from_hfov(width, height, h_fov_deg):
    h_fov = np.deg2rad(h_fov_deg)
    fx = (width / 2.0) / np.tan(h_fov / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,  1]], dtype=np.float64)

def make_camera_rotation(forward_vec_int):
    """카메라 z축이 forward_vec을 향하도록 world->camera 회전"""
    z = normalize(forward_vec_int)
    up_world = np.array([0, 1, 0], np.float64)
    if abs(np.dot(z, up_world)) > 0.99:
        up_world = np.array([1, 0, 0], np.float64)
    x = normalize(np.cross(up_world, z))  # 우
    y = np.cross(z, x)                    # 하
    return np.vstack([x, y, z])           # rows: cam axes in world

def plane_basis(n):
    """벽 평면 위 직교기저 (u,v), 오른손계 (u×v=+n)"""
    n = normalize(n)
    helper = np.array([0, 1, 0], np.float64)
    if abs(np.dot(n, helper)) > 0.99:
        helper = np.array([1, 0, 0], np.float64)
    u = normalize(np.cross(helper, n))    # 벽 가로
    v = np.cross(n, u)                    # 벽 세로
    return u, v, n

def project_points(K, R_wc, Xw):
    Xc = (R_wc @ Xw.T).T
    uvw = (K @ Xc.T).T
    return uvw[:, :2] / uvw[:, 2:3]

# =============================
# 1) 좌표 변환 + 카메라 파라미터
# =============================
screen_vec = user_to_internal(screen_vec_user)
wall_n_vec = user_to_internal(wall_n_vec_user)

K   = intrinsics_from_hfov(PROJ_W, PROJ_H, H_FOV_DEG)
R_wc = make_camera_rotation(screen_vec)

# =============================
# 2) 벽 평면 & 광선 교점으로 중심점 계산
#     평면: n·(X - P0)=0,  P0 = n * d
#     광선: X(t)=0 + t*d_ray,  t*=(n·P0)/(n·d_ray)
# =============================
n = normalize(wall_n_vec)
P0 = n * beam_wall_displacement
d_ray = normalize(screen_vec)

den = float(np.dot(n, d_ray))
if abs(den) < 1e-9:
    raise RuntimeError("screen_vec이 벽 평면과 거의 평행합니다.")

t_hit = float(np.dot(n, P0) / den)
if t_hit <= 0:
    raise RuntimeError("광선이 벽을 향하지 않습니다. screen_vec 방향을 확인하세요.")

wall_center_point = d_ray * t_hit

# =============================
# 3) 벽 위 16:9 모서리 생성 (TL, TR, BR, BL)
# =============================
u, v, _ = plane_basis(n)
half_w, half_h = screen_width_m * 0.5, screen_height_m * 0.5

P_TL = wall_center_point + (-half_w) * u + (+half_h) * v
P_TR = wall_center_point + (+half_w) * u + (+half_h) * v
P_BR = wall_center_point + (+half_w) * u + (-half_h) * v
P_BL = wall_center_point + (-half_w) * u + (-half_h) * v

wall_corners_3d = np.stack([P_TL, P_TR, P_BR, P_BL], axis=0)

# =============================
# 4) 모서리를 프로젝터 픽셀로 투영
# =============================
dst_pts = project_points(K, R_wc, wall_corners_3d).astype(np.float64)  # (4,2)

# =============================
# 5) 원본 4귀퉁이 ↔ 투영점 Homography
# =============================
img = cv2.imread(INPUT_IMG_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Cannot read {INPUT_IMG_PATH}")
H_img, W_img = img.shape[:2]

src_pts = np.array([[0,         0        ],   # TL
                    [W_img - 1, 0        ],   # TR
                    [W_img - 1, H_img - 1],   # BR
                    [0,         H_img - 1]],  # BL
                   dtype=np.float64)

H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
np.save("distortion_matrix.npy", H)

# (선택) 프런트엔드 전달용 JSON도 덤프
with open("distortion_matrix.json", "w", encoding="utf-8") as f:
    json.dump(H.tolist(), f)

# =============================
# 6) 사전왜곡 이미지 출력 (검증용)
# =============================
distorted = cv2.warpPerspective(
    img, H, (PROJ_W, PROJ_H),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)
cv2.imwrite("distorted_img.jpg", distorted)

print("Done.")
print("wall_center_point (internal coords):", wall_center_point)
print("Saved: distortion_matrix.npy, distortion_matrix.json, distorted_img.jpg")
