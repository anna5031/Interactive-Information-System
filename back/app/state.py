from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from websockets import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol

Message = Dict[str, Any]
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CommandTracker:
    websocket: WebSocketServerProtocol
    message: Message
    resend_interval: float
    completed_event: asyncio.Event = field(default_factory=asyncio.Event)
    received: bool = False
    _task: Optional[asyncio.Task[None]] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(), name=f"command-resend-{self.command_id}"
            )

    async def _run(self) -> None:
        payload = json.dumps(self.message, ensure_ascii=False)
        while not self.completed_event.is_set():
            try:
                await self.websocket.send(payload)
                logger.debug(
                    "Sent command %s (action=%s)",
                    self.command_id,
                    self.message["action"],
                )
            except ConnectionClosed:
                logger.info(
                    "Stopping command %s because connection closed.", self.command_id
                )
                break

            try:
                await asyncio.wait_for(
                    self.completed_event.wait(), timeout=self.resend_interval
                )
            except asyncio.TimeoutError:
                continue

    @property
    def command_id(self) -> str:
        return str(self.message["commandId"])

    def mark_received(self) -> None:
        if not self.received:
            logger.info("Command %s acknowledged (received).", self.command_id)
        self.received = True

    def mark_completed(self) -> None:
        if not self.completed_event.is_set():
            logger.info("Command %s acknowledged (completed).", self.command_id)
            self.completed_event.set()

    async def wait_completed(self) -> None:
        await self.completed_event.wait()


class CommandManager:
    def __init__(self, websocket: WebSocketServerProtocol, resend_interval: float):
        self.websocket = websocket
        self.resend_interval = resend_interval
        self._sequence = 0
        self._trackers: Dict[str, CommandTracker] = {}

    def _next_command_id(self) -> str:
        self._sequence += 1
        return f"cmd-{self._sequence}"

    async def issue_command(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        requires_completion: bool = False,
    ) -> CommandTracker:
        command_id = self._next_command_id()
        message: Message = {
            "type": "command",
            "commandId": command_id,
            "sequence": self._sequence,
            "action": action,
            "context": context or {},
        }
        if requires_completion:
            message["requiresCompletion"] = True

        tracker = CommandTracker(
            websocket=self.websocket,
            message=message,
            resend_interval=self.resend_interval,
        )
        self._trackers[command_id] = tracker
        tracker.start()
        return tracker

    def handle_ack(self, payload: Message) -> None:
        command_id = payload.get("commandId")
        status = payload.get("status")
        tracker = self._trackers.get(command_id)
        if not tracker:
            logger.debug("Ack for unknown command %s ignored.", command_id)
            return
        if status == "received":
            tracker.mark_received()
        elif status == "completed":
            tracker.mark_completed()

    async def shutdown(self) -> None:
        for tracker in self._trackers.values():
            tracker.mark_completed()
        await asyncio.gather(
            *[
                tracker._task
                for tracker in self._trackers.values()
                if tracker._task is not None
            ],
            return_exceptions=True,
        )
        self._trackers.clear()
