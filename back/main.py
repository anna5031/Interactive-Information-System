import asyncio
import logging

from devices import DeviceManager, load_device_preferences
from runtime.application import Application
from session.runner import SessionRunnerFactory
from websocket.server import WebSocketServer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

SHOW_EXPLORATION_OVERLAY = True  # 주석 해제 시 YOLO 결과 창 표시
USE_DUMMY_ARDUINO = True  # 주석 해제 시 아두이노 연결 확인 건너뜀
USE_DUMMY_NUDGE_PASS = True  # 주석 해제 시 타겟 접근 조건을 더미로 통과

SHOW_EXPLORATION_OVERLAY = bool(globals().get("SHOW_EXPLORATION_OVERLAY", False))
USE_DUMMY_ARDUINO = bool(globals().get("USE_DUMMY_ARDUINO", False))
USE_DUMMY_NUDGE_PASS = bool(globals().get("USE_DUMMY_NUDGE_PASS", False))


async def main_async() -> None:
    preferences = load_device_preferences()
    device_manager = DeviceManager(
        preferences=preferences,
        use_dummy_arduino=USE_DUMMY_ARDUINO,
    )
    session_factory = SessionRunnerFactory(
        show_exploration_overlay=SHOW_EXPLORATION_OVERLAY,
        use_dummy_nudge_pass=USE_DUMMY_NUDGE_PASS,
    )
    application = Application(
        device_manager=device_manager,
        session_factory=session_factory,
    )

    server = WebSocketServer(application=application)

    logger.info("장치 초기화 시작")
    try:
        application.startup()
    except Exception:
        logger.exception("장치 초기화 실패")
        application.shutdown()
        return
    logger.info("장치 초기화 완료")

    try:
        await server.serve_forever()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("서버 실행 중 오류 발생")
    finally:
        await server.stop()
        application.shutdown()
        logger.info("애플리케이션 종료 완료")


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("사용자 입력으로 종료")


if __name__ == "__main__":
    main()
