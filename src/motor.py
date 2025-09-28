from numpy import arccos, arctan2

z_offset = 1000 # mm
H = 2000 # mm

def _calAngle(target):
    x, y, z = target
    theta_t = arccos(z_offset,((x**2 + y**2 + (z-H)**2)**0.5))-arctan2((x**2 + y**2)**0.5, H-z)
    theta_p = arctan2(y, x)
    return (theta_t, theta_p)

def move_to(target):
    theta_t, theta_p = _calAngle(target)
    # 모터 제어 코드
    pass # TODO