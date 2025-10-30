# click_points.py
import cv2 as cv, numpy as np

img = cv.imread('frame.jpg')  # 좌표를 찍을 이미지
pts = []

def cb(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv.putText(img, f"{len(pts)}", (x+8, y-8), 0, 0.6, (0,255,0), 2)
        print(f"Clicked: ({x}, {y})")


print("이미지에서 클릭 → ESC로 종료")
while True:
    cv.imshow('Click points', img)
    cv.setMouseCallback('Click points', cb)
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
np.save('image_pts.npy', np.array(pts, np.float32))
print("저장 완료: image_pts.npy")
