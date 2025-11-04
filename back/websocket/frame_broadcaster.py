from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Optional, Set, Tuple

import cv2
import numpy as np
import websockets
from websockets.server import WebSocketServer, WebSocketServerProtocol

logger = logging.getLogger(__name__)


class FrameBroadcaster:
    """Broadcast JPEG-encoded exploration frames to remote viewers via WebSocket."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9100, *, jpeg_quality: int = 70) -> None:
        self._host = host
        self._port = port
        self._jpeg_quality = int(max(10, min(100, jpeg_quality)))
        self._clients: Set[WebSocketServerProtocol] = set()
        self._clients_lock = asyncio.Lock()
        self._server: Optional[WebSocketServer] = None
        self._server_task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def address(self) -> tuple[str, int]:
        return (self._host, self._port)

    async def start(self) -> None:
        if self._server is not None:
            return

        async def _handler(websocket: WebSocketServerProtocol) -> None:
            async with self._clients_lock:
                self._clients.add(websocket)
            logger.info("Frame viewer connected: %s", websocket.remote_address)
            try:
                await websocket.wait_closed()
            finally:
                async with self._clients_lock:
                    self._clients.discard(websocket)
                logger.info("Frame viewer disconnected: %s", websocket.remote_address)

        self._server = await websockets.serve(_handler, self._host, self._port)
        self._loop = asyncio.get_running_loop()
        logger.info("Frame broadcaster listening on ws://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._server is None:
            return

        server = self._server
        self._server = None
        server.close()
        await server.wait_closed()

        async with self._clients_lock:
            clients = tuple(self._clients)
            self._clients.clear()

        await asyncio.gather(
            *[self._safe_close(client) for client in clients],
            return_exceptions=True,
        )
        self._loop = None
        logger.info("Frame broadcaster stopped.")

    async def publish(self, frame_bgr: np.ndarray) -> None:
        if self._server is None:
            return
        clients = await self._snapshot_clients()
        if not clients:
            return

        loop = asyncio.get_running_loop()
        success, encoded = await loop.run_in_executor(
            None,
            cv2.imencode,
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not success:
            logger.warning("Failed to encode frame for broadcast.")
            return

        payload = base64.b64encode(encoded).decode("ascii")
        message = {
            "type": "exploration_frame",
            "image_jpeg": payload,
        }
        await self._broadcast_json(message, clients=clients)

    async def publish_log(self, *, level: str, logger_name: str, message: str, created: float) -> None:
        if self._server is None:
            return
        iso_time = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
        payload = {
            "type": "log_entry",
            "level": level,
            "logger": logger_name,
            "message": message,
            "timestamp": iso_time,
        }
        await self._broadcast_json(payload)

    @staticmethod
    async def _send_safe(client: WebSocketServerProtocol, message: str) -> None:
        try:
            await client.send(message)
        except Exception:
            logger.debug("Dropping frame viewer due to send failure.", exc_info=True)
            try:
                await client.close()
            except Exception:
                pass

    @staticmethod
    async def _safe_close(client: WebSocketServerProtocol) -> None:
        with contextlib.suppress(Exception):
            await client.close()

    async def _snapshot_clients(self) -> Tuple[WebSocketServerProtocol, ...]:
        async with self._clients_lock:
            if not self._clients:
                return ()
            return tuple(self._clients)

    async def _broadcast_json(
        self,
        payload: dict[str, Any],
        *,
        clients: Optional[Tuple[WebSocketServerProtocol, ...]] = None,
    ) -> None:
        if self._server is None:
            return
        if clients is None:
            clients = await self._snapshot_clients()
        if not clients:
            return
        message = json.dumps(payload, ensure_ascii=False)
        await asyncio.gather(
            *[self._send_safe(client, message) for client in clients],
            return_exceptions=True,
        )

    def submit(self, coro: Awaitable[None]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def _runner() -> None:
            task = asyncio.create_task(coro)
            task.add_done_callback(self._log_task_error)

        loop.call_soon_threadsafe(_runner)

    @staticmethod
    def _log_task_error(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("Broadcast task failed: %s", exc, exc_info=exc)
