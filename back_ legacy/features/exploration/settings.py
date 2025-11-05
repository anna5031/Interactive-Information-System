from __future__ import annotations

"""Exploration pipeline default parameters (edit here for quick tuning)."""

# Device selection
DEVICE_PREFERENCE = ("cuda:0", "mps", "cpu")
FORCE_DEVICE = None  # 예: "cuda:0", "cpu"

# Camera settings
CAMERA_SOURCE = 0
CAMERA_FRAME_SIZE = (1280, 720)  # (width, height) 또는 None
CAMERA_FOURCC = "MJPG"  # 예: "MJPG", "YUYV", None
CAMERA_TARGET_FPS = None  # float 또는 None
CAMERA_REFERENCE_IMAGE = "back/features/homography/calibration/sts_hallway.jpg"  # reference image for pose calibration (or None)

# Model thresholds
# 우선 TensorRT 엔진을 시도하고, 실패 시 파이토치 가중치(.pt)를 사용합니다.
MODEL_PATH = "yolo11n-pose.engine"
MODEL_CONFIDENCE_THRESHOLD = 0.3
MODEL_IOU_THRESHOLD = 0.7
MODEL_KEYPOINT_THRESHOLD = 0.3

# Tracking parameters
TRACK_DISTANCE_THRESHOLD = 80.0  # 픽셀
TRACK_MAX_AGE = 30  # 누락 허용 프레임 수
TRACK_STATIONARY_SPEED_THRESHOLD = 40.0  # px/s 이하이면 정지
TRACK_VELOCITY_SMOOTHING = 0.6  # 0~1, 클수록 과거 속도 반영
TRACK_ANGLE_SPEED_THRESHOLD = 60.0  # 방향 추정 최소 속도
TRACK_STATIONARY_DURATION = 3.0  # 초

# Assistance heuristics
ASSIST_STATIONARY_SECONDS = 3.0
ASSIST_COOLDOWN_SECONDS = 2.0

# Image post-processing
CROP_ENABLED = False
CROP_RATIO = 0.5

# Debug overlay
DEBUG_DISPLAY = False
DEBUG_WINDOW_NAME = "Exploration Debug"
