# test_motor_aim.py
# target 좌표 하나만 넣고 pygame으로 원 그리는 단순 테스트

import pygame, sys
from display import draw_circle
from nudge_test.motor.setMotor import move_to

# ▶ 여기 target 좌표 하나만 수정하세요
TARGET = (10000, 0, 2540)

def main():
    print("=== Single Target Circle Test ===")
    print("ESC 또는 Q 누르면 종료합니다.")

    pygame.init()
    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)  # 창 모드 (필요시 FULLSCREEN으로 변경 가능)
    pygame.display.set_caption("Single Circle Test")

    # 한 번만 그림
    move_to(TARGET)
    screen = draw_circle(target=TARGET, screen=screen)
    

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
