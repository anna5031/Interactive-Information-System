import numpy as np
import cv2 as cv
import glob # 파일 경로 패턴을 한 번에 불러오는 모듈, 특정 폴더 안의 여러 이미지를 한꺼번에 읽을 수 있음.
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7,3), np.float32) # 63개의 3D 포인트 생성
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2) # [x, y] 쌍이 63개 생성, z 좌표는 0으로 유지

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

image_dir = os.path.join(os.path.dirname(__file__), "checkboard")
images = glob.glob(os.path.join(image_dir, "*.jpg")) # 디렉토리의 모든 jpg 파일 이름을 리스트로 반환

# 전체화면 모드 설정
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.setWindowProperty('img', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 그레이스케일로 변환

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,7), None) # ret: return value

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,7), corners2, ret)

        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("RMS reprojection error:", ret)
print("K=\n", mtx)
print("dist=", dist.ravel())

# 저장
np.savez('camera_calib.npz', K=mtx, dist=dist)