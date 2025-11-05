from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterable, List, Tuple

from app.events import CommandEvent


@dataclass(slots=True)
class QACommandSpec:
    delay: float
    action: str
    context: Dict[str, object]
    requires_completion: bool = False


class QAStub:
    """QA 플로우에서 사용할 더미 커맨드 시퀀스를 반복 전송."""

    def __init__(self, loop: bool = True, loop_delay: float = 5.0) -> None:
        self._loop = loop
        self._loop_delay = loop_delay
        self._script: List[QACommandSpec] = [
            QACommandSpec(2.0, "start_landing", {"message": "시스템 준비 중입니다."}, True),
            QACommandSpec(2.5, "start_nudge", {}),
            QACommandSpec(
                4.0,
                "start_qa",
                {"initialPrompt": "안녕하세요! 무엇을 도와드릴까요?"},
                True,
            ),
            QACommandSpec(3.0, "start_listening", {"message": "듣는 중..."}),
            QACommandSpec(4.0, "start_thinking", {"message": "생각 중..."}),
            QACommandSpec(
                3.0, "start_speaking", {"message": "사용자 질문에 대한 더미 답변입니다."}
            ),
            QACommandSpec(2.5, "stop_speaking", {}),
            QACommandSpec(1.5, "stop_all", {"reason": "세션 종료"}, True),
        ]

    async def sequence(self) -> AsyncIterator[CommandEvent]:
        while True:
            for spec in self._script:
                await asyncio.sleep(spec.delay)
                yield CommandEvent(
                    action=spec.action,
                    context=spec.context,
                    requires_completion=spec.requires_completion,
                    message={},
                )

            if not self._loop:
                break

            await asyncio.sleep(self._loop_delay)
