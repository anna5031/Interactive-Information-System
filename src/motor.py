from math import pi
from numpy import arccos, arctan2
from config_loader import load_config
# 예: motor.py 안에서
from serial_comm import SimpleSerialMotor
import time

link = SimpleSerialMotor(port="COM5", baudrate=115200)
print("PING?", link.ping())


# 설정 파일(config.yaml 등)에서 기하 파라미터 로드
config = load_config()
z_offset = config["geometry"]["z_offset"] # 프로젝터/광축의 z-오프셋 [mm]
H = config["geometry"]["H"] # 기준 높이(예: 천장 높이 등) [mm]

def _calAngle(target):
    x, y, z = target
    theta_t = arccos(z_offset,((x**2 + y**2 + (z-H)**2)**0.5))-arctan2((x**2 + y**2)**0.5, H-z)
    theta_p = arctan2(y, x)
    return (theta_t, theta_p)

def _setMotor(tilt_deg, pan_deg):
    link.send(tilt_deg, pan_deg)

def move_to(target):
    _setMotor(_calAngle(target))

def main():
    _setMotor(110, 0)
    time.sleep(1)

if __name__ == "__main__":
    main()