from __future__ import annotations

"""Default configuration values for the real motor controller."""

# Serial communication defaults (can be overridden by environment variables)
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0  # seconds

# Motion limits (degrees)
PAN_MIN_DEG = 0.0
PAN_MAX_DEG = 180.0
PAN_INIT_DEG = 0.0

TILT_MIN_DEG = 45.0
TILT_MAX_DEG = 170.0
TILT_INIT_DEG = 110.0

# Geometric parameters (millimetres)
TILT_AXIS_HEIGHT_MM = 1200.0  # Height of the tilt axis from the floor
PROJECTOR_OFFSET_MM = 150.0  # Distance from tilt axis to projector lens along beam
PROJECTION_AHEAD_MM = 350.0  # Default distance in front of the foot to aim

# Smoothing / retries
COMMAND_RETRY_DELAY_S = 0.1
PING_ON_STARTUP = True
