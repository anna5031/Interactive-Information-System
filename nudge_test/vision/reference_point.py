# click_points.py
import cv2 as cv, numpy as np

img = cv.imread('reference.jpg')              # 좌표를 딸 이미지
pts = []                                  # 클릭한 (u,v) 목록

def cb(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN and len(pts)<4:
        pts.append([x,y])
        cv.circle(img,(x,y),5,(0,255,0),-1)
        cv.putText(img,f"P{len(pts)}",(x+6,y-6),0,0.6,(0,255,0),2)

cv.namedWindow('Click P1->P2->P3->P4')
cv.setMouseCallback('Click P1->P2->P3->P4', cb)

while True:
    cv.imshow('Click P1->P2->P3->P4', img)
    k = cv.waitKey(1) & 0xFF
    if k==27: break                      # ESC로 종료
    if len(pts)==4:                      # 4점 찍으면 자동 저장/종료
        pts = np.array(pts, np.float32)
        np.save('image_pts.npy', pts)    # estimate_pose.py에서 로드해서 사용
        print("image_pts:\n", pts)
        # break

# cv.destroyAllWindows()
