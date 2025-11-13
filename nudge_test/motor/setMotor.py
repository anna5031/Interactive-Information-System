# setMotor.py
import time
import numpy as np
from config_loader import load_config
from serial_comm import SimpleSerialMotor

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
config = load_config()
# Toggle to disable serial writes when Arduino is not present.
ARDUINO_CONNECTED = False

def _vector_setting(name, default):
    """Read a 3D vector from the config with a safe fallback."""
    value = config.get(name, default)
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray(default, dtype=np.float64)
    if arr.shape != np.asarray(default).shape:
        return np.asarray(default, dtype=np.float64)
    return arr

# Geometry
PT_HEIGHT = float(config.get("pan_tilt_height", 1350.0))
PT_POS = np.array([0.0, 0.0, PT_HEIGHT], dtype=np.float64)
PROJECTOR_OFFSET_LOCAL = _vector_setting("projector_offset_vector", [50.0, 0.0, 150.0])

# Serial settings
port     = str(config["serial"]["port"])
baudrate = int(config["serial"]["baudrate"])
timeout  = float(config["serial"]["timeout"])

# Motor limits & init
pan_min  = float(config["motor"]["pan"]["min_deg"])
pan_max  = float(config["motor"]["pan"]["max_deg"])
pan_init = float(config["motor"]["pan"]["init_deg"])

tilt_min  = float(config["motor"]["tilt"]["min_deg"])
tilt_max  = float(config["motor"]["tilt"]["max_deg"])
tilt_init = float(config["motor"]["tilt"]["init_deg"])

# -------------------------------------------------------------
# Serial link
# -------------------------------------------------------------
link = None
if ARDUINO_CONNECTED:
    link = SimpleSerialMotor(port=port, baudrate=baudrate, timeout=timeout)
    print("[Serial Check]", "PING?", link.ping())
else:
    print("[Serial Check] Arduino link disabled (ARDUINO_CONNECTED=False)")

# -------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------
def _clip(v: float, vmin: float, vmax: float, type) -> float:
    clipped_v = min(max(v, vmin), vmax)
    if clipped_v != v:
        print(type, " Motor Clipped", v, "to", clipped_v)
    return clipped_v

def _rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def _rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]], dtype=np.float64)

def _angles_to_R(pan_deg, tilt_deg):
    """Return world rotation for given pan (about Z) and tilt (about Y)."""
    p = np.deg2rad(pan_deg)
    t = np.deg2rad(tilt_deg)
    return _rot_z(p) @ _rot_y(t)

def _beam_from_angles(pan_deg, tilt_deg):
    """Compute beam origin and direction in world coordinates."""
    R = _angles_to_R(pan_deg, tilt_deg)
    beam_pos = PT_POS + R @ PROJECTOR_OFFSET_LOCAL
    local_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    nudge_vec = R @ local_forward
    norm = np.linalg.norm(nudge_vec)
    if norm < 1e-9:
        raise RuntimeError("Invalid nudge vector norm (too small).")
    return beam_pos, nudge_vec / norm

def _ray_target_distance_sq(pan_deg, tilt_deg, target):
    """Squared shortest distance between the beam ray and target."""
    beam_pos, nudge_vec = _beam_from_angles(pan_deg, tilt_deg)
    target = np.asarray(target, dtype=np.float64)
    v = target - beam_pos
    s = max(0.0, np.dot(v, nudge_vec))  # ray only fires forward
    closest = beam_pos + s * nudge_vec
    diff = closest - target
    return float(np.dot(diff, diff))

def _initial_guess_without_offset(target):
    """Pan/tilt guess if the beam originated at PT_POS (offset ignored)."""
    x, y, z = map(float, target)
    dz = z - PT_POS[2]
    pan = np.degrees(np.arctan2(y, x))
    r_xy = np.hypot(x, y)
    tilt = np.degrees(np.arctan2(-dz, r_xy))
    return pan, tilt

def solve_pan_tilt_for_target(target,
                              pan_bounds=(pan_min, pan_max),
                              tilt_bounds=(tilt_min, tilt_max)):
    """Numerically minimize beam-to-target distance over pan/tilt."""
    target = np.asarray(target, dtype=np.float64).reshape(3,)
    pan, tilt = _initial_guess_without_offset(target)
    pan = float(np.clip(pan, *pan_bounds))
    tilt = float(np.clip(tilt, *tilt_bounds))
    best_loss = _ray_target_distance_sq(pan, tilt, target)

    for step in (4.0, 1.0, 0.25):
        improved = True
        while improved:
            improved = False
            for d_pan in (-step, 0.0, step):
                for d_tilt in (-step, 0.0, step):
                    if d_pan == 0.0 and d_tilt == 0.0:
                        continue
                    cand_pan = float(np.clip(pan + d_pan, *pan_bounds))
                    cand_tilt = float(np.clip(tilt + d_tilt, *tilt_bounds))
                    loss = _ray_target_distance_sq(cand_pan, cand_tilt, target)
                    if loss + 1e-6 < best_loss:
                        best_loss = loss
                        pan, tilt = cand_pan, cand_tilt
                        improved = True
    return pan, tilt

def _calAngle(target):
    start = time.perf_counter()
    result = solve_pan_tilt_for_target(target)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"_calAngle computation took {elapsed_ms:.2f} ms")
    return result

# -------------------------------------------------------------
# Motor command helpers
# -------------------------------------------------------------
def _setAngle(tilt_deg, pan_deg):
    if ARDUINO_CONNECTED and link is not None:
        print(link.send(int(round(tilt_deg)), int(round(pan_deg))))
        print(f"Set motor to: tilt {tilt_deg:.2f}, pan {pan_deg:.2f}")
    else:
        print("[Serial Disabled] Command not sent.")
        print(f"Simulated command: tilt {tilt_deg:.2f}, pan {pan_deg:.2f}")

# -------------------------------------------------------------
# Public API
# -------------------------------------------------------------
def setMotor(target):
    """
    1) solve pan/tilt that align the beam with target
    2) add init offsets
    3) clip to motor bounds
    4) command motors
    """
    pan_raw, tilt_raw = _calAngle(target)
    
    tilt_cmd = _clip(tilt_init + tilt_raw, tilt_min, tilt_max, "Tilt")
    pan_cmd  = _clip(pan_init  + pan_raw,  pan_min,  pan_max, "Pan")
    pan_raw = pan_cmd - pan_init
    tilt_raw = tilt_cmd - tilt_init

    print(f"Angles in degree: pan {pan_raw:.2f}, tilt {tilt_raw:.2f}")
    # print(f"Command angles: pan {pan_cmd:.2f}, tilt {tilt_cmd:.2f}")
    _setAngle(tilt_cmd, pan_cmd)
    return pan_raw, tilt_raw

# -------------------------------------------------------------
# Manual test
# -------------------------------------------------------------
if __name__ == "__main__":
    setMotor((1000, 0, 2500.0))
