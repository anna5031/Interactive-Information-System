from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import suppress

from app import Application, DebugConfig, RuntimeOverrides
from websocket.server import WebSocketServer
from websocket.log_handler import BroadcastLogHandler


# 디버그/테스트 토글 (라인을 주석 처리해 on/off 조절)
# LOG_EXPLORATION = True       # 탐색 파이프라인 로그 출력
# LOG_MOTOR = True             # 모터 제어 로그 출력
# LOG_HOMOGRAPHY = True        # 호모그래피 계산 로그 출력
LOG_COMMANDS = True            # 명령 송수신 로그 출력
# SHOW_EXPLORATION_OVERLAY = True  # 디버그용 탐색 오버레이 창 표시
# USE_DUMMY_EXPLORATION = True     # 실제 탐색 대신 스텁 사용
# USE_DUMMY_MOTOR = True           # 실제 모터 대신 스텁 사용
# USE_DUMMY_HOMOGRAPHY = True      # 실제 호모그래피 대신 스텁 사용
# SKIP_TO_QA_AUTO = True           # 탐색 후 QA 자동 진입 건너뛰기

# 위 라인을 주석 처리했을 때 기본값 False로 설정
LOG_EXPLORATION = bool(globals().get("LOG_EXPLORATION", False))
LOG_MOTOR = bool(globals().get("LOG_MOTOR", False))
LOG_HOMOGRAPHY = bool(globals().get("LOG_HOMOGRAPHY", False))
LOG_COMMANDS = bool(globals().get("LOG_COMMANDS", False))
SHOW_EXPLORATION_OVERLAY = bool(globals().get("SHOW_EXPLORATION_OVERLAY", False))
USE_DUMMY_EXPLORATION = bool(globals().get("USE_DUMMY_EXPLORATION", False))
USE_DUMMY_MOTOR = bool(globals().get("USE_DUMMY_MOTOR", False))
USE_DUMMY_HOMOGRAPHY = bool(globals().get("USE_DUMMY_HOMOGRAPHY", False))
SKIP_TO_QA_AUTO_OVERRIDE = globals().get("SKIP_TO_QA_AUTO", None)
if isinstance(SKIP_TO_QA_AUTO_OVERRIDE, bool):
    SKIP_TO_QA_AUTO = SKIP_TO_QA_AUTO_OVERRIDE
else:
    SKIP_TO_QA_AUTO = None


async def main_async() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    debug_config = DebugConfig(
        log_exploration=LOG_EXPLORATION,
        log_motor=LOG_MOTOR,
        log_homography=LOG_HOMOGRAPHY,
        log_commands=LOG_COMMANDS,
        show_exploration_overlay=SHOW_EXPLORATION_OVERLAY,
    )

    overrides = RuntimeOverrides(
        force_dummy_exploration=USE_DUMMY_EXPLORATION,
        force_dummy_motor=USE_DUMMY_MOTOR,
        force_dummy_homography=USE_DUMMY_HOMOGRAPHY,
        skip_to_qa_auto=SKIP_TO_QA_AUTO,
    )

    application = Application(debug=debug_config, overrides=overrides)
    log_handler = BroadcastLogHandler(application.frame_broadcaster)
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(log_handler)

    server = WebSocketServer(application)

    loop = asyncio.get_running_loop()

    async def shutdown() -> None:
        logging.info("Shutdown initiated.")
        await server.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown()))

    server_task = asyncio.create_task(server.start(), name="websocket-server")

    try:
        await server_task
    finally:
        logging.getLogger().removeHandler(log_handler)
        if not server_task.done():
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
