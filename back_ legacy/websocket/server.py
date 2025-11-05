from __future__ import annotations

import asyncio
import logging

import websockets
from websockets.legacy.server import WebSocketServerProtocol

from app.application import Application
from websocket.session import ClientSession

logger = logging.getLogger(__name__)


class WebSocketServer:
    """웹소켓 서버 수명주기 관리."""

    def __init__(self, application: Application) -> None:
        self.application = application
        self._server = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        config = self.application.config
        logger.info(
            "Starting WebSocket server on ws://%s:%s",
            config.websocket.host,
            config.websocket.port,
        )
        await self.application.start_background_services()
        self._server = await websockets.serve(
            self._handle_client,
            config.websocket.host,
            config.websocket.port,
        )
        try:
            await self._stop_event.wait()
        finally:
            await self.application.stop_background_services()

    async def stop(self) -> None:
        server = self._server
        if server is not None:
            logger.info("Stopping WebSocket server.")
            # Close active client sockets first so their background tasks unwind cleanly.
            websockets_to_close = list(getattr(server, "websockets", []))
            await asyncio.gather(
                *[ws.close(code=1001, reason="Server shutdown") for ws in websockets_to_close],
                return_exceptions=True,
            )
            server.close()
            await server.wait_closed()
        self._stop_event.set()

    async def _handle_client(
        self, websocket: WebSocketServerProtocol, path: str | None = None
    ) -> None:  # noqa: ARG002
        deps = self.application.create_session()
        session = ClientSession(websocket, deps)
        await session.run()
