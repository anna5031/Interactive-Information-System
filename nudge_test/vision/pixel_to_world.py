# pixel_to_world.py
import cv2 as cv
import numpy as np

cal = np.load('camera_calib.npz')
K, dist = cal['K'], cal['dist']
ext = np.load('camera_extrinsics.npz')
R, t = ext['R'], ext['t']

Kinv = np.linalg.inv(K)
n = np.array([[0.0],[0.0],[1.0]])   # 바닥 법선 (Z=0)
d = 0.0

def pixel_to_world_on_floor(u, v):
    """픽셀(u,v) -> 바닥(Z=0) 위 실세계 좌표(X,Y,0)"""
    # 1) 픽셀 -> 정규화 카메라 좌표 (왜곡 보정)
    pts = np.array([[[u, v]]], dtype=np.float32)
    x_n, y_n = cv.undistortPoints(pts, K, dist, P=None)[0,0]
    ray_c = np.array([[x_n], [y_n], [1.0]], dtype=np.float64)  # 카메라 좌표계 광선

    # 2) camera→world로 변환 준비
    Rcw, tcw = R, t                         # solvePnP 결과 (world->camera)
    Rwc = Rcw.T                             # (camera->world) 회전
    Cw  = -Rwc @ tcw                        # 카메라 중심 (world 좌표)
    d_w = Rwc @ ray_c                       # 광선 방향 (world 좌표)

    # 3) Z=0 평면과 교점: X = C + λ d,  n=[0,0,1], d=0 → λ = -Cz / dz
    dz = float(d_w[2,0])
    if abs(dz) < 1e-12:                     # 평면과 평행
        return None
    lam = - float(Cw[2,0]) / dz
    Xw = Cw + lam * d_w                     # (3,1)

    return float(Xw[0,0]), float(Xw[1,0]), float(Xw[2,0])  # ≈ 0


if __name__ == "__main__":
    pts = np.load('image_pts.npy')  # [[u,v], [u,v], ...]
    for (u, v) in pts:
        Xw = pixel_to_world_on_floor(u, v)
        print(f"pixel=({u:.1f}, {v:.1f}) -> world={Xw}")