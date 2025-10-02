# display.py
from config_loader import load_config
import pygame, sys

config = load_config()
z_offset = config["geometry"]["z_offset"]
H = config["geometry"]["H"]
scale_factor = config["geometry"]["scale_factor"]


def _calSizeRatio(distance):
    return scale_factor / distance *2

def _calDistance(target):
    x, y, z = target
    distance = (x**2 + y**2 + (z-H)**2 - z_offset**2)**0.5
    return distance

def _keystone(img, ratio):
    # 이미지 키스톤 보정 코드
    pass # TODO

def draw_circle(target=None, distance=None, screen=None):
    """
    target 또는 distance 중 하나를 입력받아 원을 그린다.
    원 안에 꽉 찬 화살표를 표시한다.
    """
    if screen is None:
        raise RuntimeError("draw_circle 호출 전 screen을 먼저 만들어야 합니다.")

    if target is None and distance is None:
        raise ValueError("target이나 distance 중 하나는 반드시 입력해야 합니다.")

    if target is not None:
        distance = _calDistance(target)

    # 반지름 계산
    base_radius_px = 200
    ratio = max(0.0, _calSizeRatio(distance))
    radius = int(base_radius_px * ratio)

    w, h = screen.get_size()
    cx, cy = w // 2, h // 2

    # 색상
    bg_color = (0, 0, 0)
    circle_color = (255, 255, 255)
    arrow_color = (255, 0, 255)  # 핑크색

    # 배경/원
    screen.fill(bg_color)
    pygame.draw.circle(screen, circle_color, (cx, cy), radius)

        # 화살표 크기 비율
    arrow_length = int(radius * 1.2)       # 전체 화살표 길이
    body_thickness = int(radius * 0.35)    # 몸통 두께
    head_length = int(radius * 0.5)        # 머리 길이 (몸통보다 확실히 길게)
    head_width  = int(radius * 0.8)        # 머리 폭 (몸통보다 넓게)

    # 몸통 (사각형)
    body_length = arrow_length - head_length
    rect_x = cx - arrow_length // 2
    rect_y = cy - body_thickness // 2
    pygame.draw.rect(screen, arrow_color, (rect_x, rect_y, body_length, body_thickness))

    # 머리 (삼각형)
    tip_x = rect_x + body_length
    tip_y = cy
    pygame.draw.polygon(
        screen,
        arrow_color,
        [
            (tip_x, tip_y - head_width // 2),
            (tip_x, tip_y + head_width // 2),
            (tip_x + head_length, tip_y),
        ],
    )

    pygame.display.flip()
    return screen



if __name__ == "__main__":
    draw_circle(distance = 2000)
