import ctypes
from pathlib import Path

import cv2 as cv
import numpy as np


IMAGE_PATH = Path(__file__).with_name("sts_hallway.jpg")
img = cv.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f"Unable to load image at {IMAGE_PATH}")

pts = []
scale = 1.0
display_img = img.copy()

try:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    try:
        user32.SetProcessDPIAware()
    except AttributeError:
        pass
    screen_w = max(user32.GetSystemMetrics(0) - 120, 1)
    screen_h = max(user32.GetSystemMetrics(1) - 160, 1)
    scale = min(screen_w / img.shape[1], screen_h / img.shape[0], 1.0)
except (AttributeError, OSError):
    max_w, max_h = 1280, 720
    scale = min(max_w / img.shape[1], max_h / img.shape[0], 1.0)

if scale < 1.0:
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    display_img = cv.resize(img, new_size, interpolation=cv.INTER_AREA)
else:
    scale = 1.0


def cb(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        orig_x = max(0, min(int(round(x / scale)), img.shape[1] - 1))
        orig_y = max(0, min(int(round(y / scale)), img.shape[0] - 1))
        pts.append([orig_x, orig_y])
        cv.circle(display_img, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv.putText(
            display_img,
            str(len(pts)),
            (int(x) + 8, int(y) - 8),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv.imshow("Click points", display_img)
        print(f"Clicked: ({orig_x}, {orig_y})")


print("Click on the image to record points. Press ESC to finish.")
cv.namedWindow("Click points", cv.WINDOW_NORMAL)
cv.setMouseCallback("Click points", cb)

while True:
    cv.imshow("Click points", display_img)
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
np.save("image_pts.npy", np.array(pts, dtype=np.float32))
print("Saved image_pts.npy")
