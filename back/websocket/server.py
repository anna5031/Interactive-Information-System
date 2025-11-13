from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from websockets.asyncio.server import Server, ServerConnection, serve

from websocket.connection import ClientConnection

if TYPE_CHECKING:
    from runtime.application import Application

logger = logging.getLogger(__name__)


class WebSocketServer:
    def __init__(self, application: "Application", host: str = "0.0.0.0", port: int = 8765) -> None:
        self.application = application
        self.host = host
        self.port = port
        self._server: Optional[Server] = None

    async def serve_forever(self) -> None:
        logger.info("WebSocket 서버 시작: ws://%s:%s", self.host, self.port)
        self._server = await serve(self._handle_client, self.host, self.port)
        try:
            await self._server.serve_forever()
        finally:
            await self._shutdown_server()
            logger.info("WebSocket 서버 종료")

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def _shutdown_server(self) -> None:
        await self.stop()

    async def _handle_client(self, connection: ServerConnection) -> None:
        client = ClientConnection(connection=connection)
        session_runner = self.application.create_session_runner(client)

        logger.info("클라이언트 연결 시작: %s", client.identifier)
        try:
            await session_runner.run()
        finally:
            await client.close()
            logger.info("클라이언트 연결 종료: %s", client.identifier)
