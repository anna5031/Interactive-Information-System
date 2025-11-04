from __future__ import annotations

import logging

from .frame_broadcaster import FrameBroadcaster


class BroadcastLogHandler(logging.Handler):
    """Logging handler that forwards records to the frame broadcaster."""

    def __init__(self, broadcaster: FrameBroadcaster, level: int = logging.INFO) -> None:
        super().__init__(level)
        self._broadcaster = broadcaster

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        broadcaster = self._broadcaster
        if broadcaster is None:
            return

        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - mirrors logging.Handler implementation
            self.handleError(record)
            return

        coro = broadcaster.publish_log(
            level=record.levelname,
            logger_name=record.name,
            message=message,
            created=record.created,
        )
        broadcaster.submit(coro)
