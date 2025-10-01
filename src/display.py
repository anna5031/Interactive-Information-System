# display.py
from config_loader import load_config
import pygame, sys

config = load_config()
z_offset = config["geometry"]["z_offset"]
H = config["geometry"]["H"]
scale_factor = config["geometry"]["scale_factor"]


def _calSizeRatio(distance):
    return scale_factor / distance

def _calDistance(target):
    x, y, z = target
    distance = (x**2 + y**2 + (z-H)**2 - z_offset**2)**0.5
    return distance
def _keystone(img, ratio):
    # 이미지 키스톤 보정 코드
    pass # TODO

def draw_circle_from_distance(distance, base_radius_px=200,
                              circle_color=(255,255,255), bg_color=(0,0,0)):
    """
    distance 값 → scale ratio 계산 → 원 크기 반영 → 전체화면 출력
    ESC 키를 누르면 종료
    """
    # ratio & 반지름 계산
    ratio = _calSizeRatio(distance)
    radius = int(base_radius_px * ratio)

    # pygame 초기화 및 전체화면 창 생성
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    center = (w//2, h//2)

    # 화면 그리기
    screen.fill(bg_color)
    pygame.draw.circle(screen, circle_color, center, radius)
    pygame.display.flip()

    # ESC로 종료
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

if __name__ == "__main__":
    draw_circle_from_distance(2000)
