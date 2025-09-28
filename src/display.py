z_offset = 1000 # mm
H = 2000 # mm
standard = 1328 # mm


def _calSizeRatio(target):
    x, y, z = target
    distance = (x**2 + y**2 + (z-H)**2 - z_offset**2)**0.5
    return standard / distance
