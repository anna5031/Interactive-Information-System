from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import suppress

from app import Application, DebugConfig
from websocket.server import WebSocketServer


# 디버그 로그 토글 (라인을 주석 처리해 on/off 조절)
# LOG_EXPLORATION = True  # 주석 처리하면 로그 비활성화
# LOG_MOTOR = True  # 주석 처리하면 로그 비활성화
# LOG_HOMOGRAPHY = True  # 주석 처리하면 로그 비활성화
LOG_COMMANDS = True  # 주석 처리하면 로그 비활성화

# 위 라인을 주석 처리했을 때 기본값 False로 설정
LOG_EXPLORATION = bool(globals().get("LOG_EXPLORATION", False))
LOG_MOTOR = bool(globals().get("LOG_MOTOR", False))
LOG_HOMOGRAPHY = bool(globals().get("LOG_HOMOGRAPHY", False))
LOG_COMMANDS = bool(globals().get("LOG_COMMANDS", False))


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
    )

    application = Application(debug=debug_config)
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
        if not server_task.done():
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
