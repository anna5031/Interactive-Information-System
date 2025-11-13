from __future__ import annotations

import logging
from typing import Optional

from websockets.asyncio.server import ServerConnection
from websockets.protocol import State

logger = logging.getLogger(__name__)


class ClientConnection:
    def __init__(self, connection: ServerConnection, identifier: Optional[str] = None):
        self._connection = connection
        self.identifier = identifier or self._format_identifier(connection)

    @property
    def is_open(self) -> bool:
        return self._connection.state is State.OPEN

    async def receive(self) -> Optional[str]:
        try:
            message = await self._connection.recv()
        except Exception:
            logger.info("클라이언트 %s 수신 중 연결 종료", self.identifier)
            return None
        logger.info("클라이언트 %s 수신: %s", self.identifier, message)
        return message

    async def send(self, payload: str) -> None:
        try:
            await self._connection.send(payload)
        except Exception:
            logger.info("클라이언트 %s 전송 실패", self.identifier)

    async def close(self) -> None:
        try:
            await self._connection.close()
        finally:
            await self._connection.wait_closed()

    @staticmethod
    def _format_identifier(connection: ServerConnection) -> str:
        peer = connection.remote_address
        if peer is None:
            return "unknown"
        host, port = peer[:2]
        return f"{host}:{port}"
