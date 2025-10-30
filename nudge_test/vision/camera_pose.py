# estimate_pose.py
import cv2 as cv
import numpy as np

# 내부파라미터 로드
data = np.load('camera_calib.npz')
K = data['K']
dist = data['dist']

# === 월드 좌표계 정의: 바닥이 Z=0 ===
# 단위(mm)로 추천. 예: 바닥에 테이프로 사각형 4점 붙이고 실측
world_pts = np.array([
    [   0.0,    0.0, 0.0],   # P1
    [   0.0, 1200.0, 0.0],   # P2
    [1800.0,    0.0, 0.0],   # P3
    [1800.0, 1200.0, 0.0],   # P4
], dtype=np.float32)

# === 이미지 픽셀 좌표 (P1~P4의 이미지 위치) ===
# 수동으로 클릭해서 넣거나, 이미지 뷰어로 좌표 읽어서 입력
image_pts = np.array([
    [553,683],
    [317, 561],
    [793, 507],
    [596, 438],
], dtype=np.float32)

# 왜곡 보정 후 좌표 쓰는 것이 정확
image_pts_und = cv.undistortPoints(image_pts.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)

# PnP(Perspective-n-Point)로 외부파라미터 구하기
success, rvec, tvec = cv.solvePnP(world_pts, image_pts, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
if not success:
    raise RuntimeError("solvePnP failed")

R, _ = cv.Rodrigues(rvec)          # 회전벡터 -> 회전행렬
t = tvec.reshape(3,1)               # (3,1)

print("R=\n", R)
print("t=\n", t)

np.savez('camera_extrinsics.npz', R=R, t=t)