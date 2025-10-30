# motor.py
import numpy as np
from config_loader import load_config
from serial_comm import SimpleSerialMotor
import time

# ── config 로드 ─────────────────────────────────────────────
config = load_config()

# geometry
z_offset = float(config["geometry"]["z_offset"])  # tilt축~프로젝터 거리(mm)
H        = float(config["geometry"]["H"])         # 바닥~tilt축 높이(mm)

# serial
port     = str(config["serial"]["port"])          
baudrate = int(config["serial"]["baudrate"])
timeout  = float(config["serial"]["timeout"])

# motor limits & init
pan_min  = float(config["motor"]["pan"]["min_deg"])
pan_max  = float(config["motor"]["pan"]["max_deg"])
pan_init = float(config["motor"]["pan"]["init_deg"])

tilt_min  = float(config["motor"]["tilt"]["min_deg"])
tilt_max  = float(config["motor"]["tilt"]["max_deg"])
tilt_init = float(config["motor"]["tilt"]["init_deg"])

# ── 통신 링크 ──────────────────────────────────────────────
link = SimpleSerialMotor(port=port, baudrate=baudrate, timeout=timeout)
print("[Serial Check]", "PING?", link.ping())

# ── 유틸 ───────────────────────────────────────────────────
def _clip(v: float, vmin: float, vmax: float) -> float:
    return min(max(v, vmin), vmax)

# ── 각도 계산 (radian → degree) ────────────────────────────
def _calAngle(target):
    x, y, z = map(float, target)
    r_xy = np.hypot(x, y)                 # sqrt(x^2 + y^2)
    dz = z-H
    r = np.hypot(r_xy, dz)            # sqrt(x^2 + y^2 + (H-z)^2)

    theta_p = np.arctan2(y, x)

    ratio = np.clip(z_offset / max(r, 1e-9), -1.0, 1.0)
    theta_1 = np.arccos(ratio)
    theta_2 = np.arctan2(r_xy, dz)
    print("theta_1, theta_2:", np.degrees(theta_1), np.degrees(theta_2))
    theta_t = np.arccos(ratio) - np.arctan2(r_xy, dz)
    
    return float(np.degrees(theta_t)), float(np.degrees(theta_p))

# ── 모터 제어 ──────────────────────────────────────────────
def _setMotor(tilt_deg, pan_deg):
    # int로 반올림 후 전송
    print(f"Set motor to: tilt {tilt_deg:.2f}, pan {pan_deg:.2f}")
    print(link.send(int(round(tilt_deg)), int(round(pan_deg))))

# ── 공개 API ───────────────────────────────────────────────
def move_to(target):
    """
    1) target으로부터 '순수' 각도 계산
    2) 각도에 init 보정(+= init_deg)
    3) min/max 클램프
    4) 모터 전송
    """
    tilt_raw, pan_raw = _calAngle(target)
    print(f"Raw angles: tilt {tilt_raw:.2f}, pan {pan_raw:.2f}")
    tilt_cmd = _clip(tilt_init+tilt_raw, tilt_min, tilt_max)
    pan_cmd  = _clip(pan_raw  + pan_init,  pan_min,  pan_max)

    _setMotor(tilt_cmd, pan_cmd)
    return tilt_cmd, pan_cmd  # 디버깅용 반환

# ── 테스트 ─────────────────────────────────────────────────
if __name__ == "__main__":
    _setMotor(70, 20)
    time.sleep(1)