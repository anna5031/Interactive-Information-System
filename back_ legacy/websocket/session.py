from __future__ import annotations

import asyncio
import json
import logging
import inspect
from typing import Any

from websockets import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol

from app.application import SessionDependencies
from app.events import HomographyEvent
from app.state import CommandManager
from websocket.schemas import CommandMessage, HomographyMessage, SyncMessage

logger = logging.getLogger(__name__)


class ClientSession:
    """단일 WebSocket 클라이언트 세션 관리."""

    def __init__(
        self,
        websocket: WebSocketServerProtocol,
        deps: SessionDependencies,
    ) -> None:
        self.websocket = websocket
        self.deps = deps
        self.debug = deps.debug
        self.command_manager = CommandManager(
            websocket, resend_interval=self.deps.config.command_resend_interval
        )
        self._tasks: list[asyncio.Task[Any]] = []

    async def run(self) -> None:
        logger.info("Client connected: %s", self.websocket.remote_address)
        await self._send_sync()

        self._tasks = [
            asyncio.create_task(self._homography_loop(), name="homography-loop"),
            asyncio.create_task(self._command_loop(), name="command-loop"),
            asyncio.create_task(self._incoming_loop(), name="incoming-loop"),
        ]

        try:
            done, pending = await asyncio.wait(
                self._tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                exception = task.exception()
                if exception:
                    raise exception
        except ConnectionClosed:
            logger.info("Connection closed by client.")
        finally:
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            await self.command_manager.shutdown()
            await self._run_cleanup()
            logger.info("Client session finished.")

    async def _send_sync(self) -> None:
        message = SyncMessage(
            state={"phase": "idle", "sequence": 0},
        ).as_dict()
        await self._send_json(message)

    async def _homography_loop(self) -> None:
        async for vision_event in self.deps.exploration.stream():
            motor_candidate = self.deps.motor.update(vision_event)
            if inspect.isawaitable(motor_candidate):
                motor_state = await motor_candidate
            else:
                motor_state = motor_candidate
            homography_event = self.deps.homography.build(motor_state)

            if self.debug.log_exploration:
                logger.info(
                    "Vision result | has_target=%s pos=%s gaze=%s conf=%.2f",
                    vision_event.has_target,
                    _safe_tuple(vision_event.target_position),
                    _safe_tuple(vision_event.gaze_vector),
                    vision_event.confidence,
                )
            if self.debug.log_motor:
                logger.info(
                    "Motor state | pan=%.2f tilt=%.2f has_target=%s",
                    motor_state.pan,
                    motor_state.tilt,
                    motor_state.has_target,
                )
            if self.debug.log_homography:
                logger.info(
                    "Homography matrix | %s",
                    _format_matrix(homography_event.matrix),
                )

            if hasattr(self.deps.flow, "process_vision"):
                self.deps.flow.process_vision(vision_event)
            if hasattr(self.deps.flow, "process_motor"):
                self.deps.flow.process_motor(motor_state)

            await self._send_homography(homography_event)

    async def _command_loop(self) -> None:
        async for command_event in self.deps.flow.command_stream():
            if self.debug.log_commands:
                logger.info(
                    "Issue command | action=%s requires_completion=%s context=%s",
                    command_event.action,
                    command_event.requires_completion,
                    command_event.context,
                )
            tracker = await self.command_manager.issue_command(
                action=command_event.action,
                context=command_event.context,
                requires_completion=command_event.requires_completion,
            )
            if command_event.requires_completion:
                await tracker.wait_completed()

    async def _incoming_loop(self) -> None:
        try:
            async for raw in self.websocket:
                payload = json.loads(raw)
                msg_type = payload.get("type")
                if msg_type == "ack":
                    if self.debug.log_commands:
                        logger.info(
                            "ACK received | action=%s status=%s commandId=%s",
                            payload.get("action"),
                            payload.get("status"),
                            payload.get("commandId"),
                        )
                    self.command_manager.handle_ack(payload)
                elif msg_type == "ping":
                    await self._send_json({"type": "pong", "timestamp": payload.get("timestamp")})
                else:
                    logger.debug("Unhandled incoming message: %s", payload)
        except ConnectionClosed:
            logger.debug("Incoming loop connection closed.")
            raise

    async def _send_homography(self, event: HomographyEvent) -> None:
        message = HomographyMessage(
            matrix=event.matrix,
            timestamp=event.timestamp,
        ).as_dict()
        await self._send_json(message)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        await self.websocket.send(data)

    async def _run_cleanup(self) -> None:
        callbacks = getattr(self.deps, "cleanup", ())
        for callback in callbacks:
            if callback is None:
                continue
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.warning("Session cleanup callback failed.", exc_info=True)


def _safe_tuple(value: Any) -> Any:
    if value is None:
        return None
    return tuple(round(v, 3) for v in value)


def _format_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [[round(cell, 5) for cell in row] for row in matrix]
