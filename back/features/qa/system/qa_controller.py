from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator, Iterable, Optional, Union

from app.events import CommandEvent, MotorStateEvent, VisionResultEvent
from ..managers import STTManager, VoiceInterfaceManager


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QAIntroSpec:
    action: str
    context: dict
    requires_completion: bool = False


class QAController:
    """단일 QA 라운드를 처리하는 컨트롤러."""

    def __init__(
        self,
        voice_manager: VoiceInterfaceManager,
        stt_manager: STTManager,
        default_response: str = "더미 답변입니다.",
    ) -> None:
        self.voice_manager = voice_manager
        self.stt_manager = stt_manager
        self._default_response = default_response

    async def run_once(
        self, intro_message: Optional[str] = None
    ) -> AsyncIterator[CommandEvent]:
        if intro_message:
            intro_text = intro_message.strip()
            if intro_text:
                logger.info("QA 안내 음성: %s", intro_text)
                async for event in self._speak(intro_text):
                    yield event

        yield CommandEvent(
            action="start_listening",
            context={"message": "듣는 중..."},
            requires_completion=False,
            message={},
        )

        audio_bytes = await asyncio.to_thread(self.voice_manager.listen_and_record)

        yield CommandEvent(
            action="stop_listening",
            context={},
            requires_completion=False,
            message={},
        )

        if not audio_bytes:
            await self._handle_no_audio()
            yield self._thinking_event("음성이 감지되지 않았습니다.")
            async for event in self._speak("음성이 감지되지 않았습니다. 다시 시도해주세요."):
                yield event
            yield CommandEvent(
                action="stop_all",
                context={"reason": "음성 미검출"},
                requires_completion=True,
                message={},
            )
            return

        transcription = await asyncio.to_thread(
            self.stt_manager.transcribe,
            audio_bytes,
        )

        if not transcription:
            yield self._thinking_event("음성을 이해하지 못했습니다.")
            async for event in self._speak(
                "죄송해요. 음성을 이해하지 못했습니다. 다시 말씀해 주세요."
            ):
                yield event
            yield CommandEvent(
                action="stop_all",
                context={"reason": "STT 실패"},
                requires_completion=True,
                message={},
            )
            return

        logger.info("STT 결과: %s", transcription)

        yield self._thinking_event("답변을 준비 중입니다...")

        response_text = self._build_response(transcription)
        async for event in self._speak(response_text):
            yield event

        yield CommandEvent(
            action="stop_all",
            context={"reason": "QA 응답 완료"},
            requires_completion=True,
            message={},
        )

    async def _handle_no_audio(self) -> None:
        logger.info("음성 입력이 감지되지 않았습니다.")

    def _thinking_event(self, message: str) -> CommandEvent:
        return CommandEvent(
            action="start_thinking",
            context={"message": message},
            requires_completion=False,
            message={},
        )

    async def _speak(self, message: str) -> AsyncIterator[CommandEvent]:
        yield CommandEvent(
            action="start_speaking",
            context={"message": message},
            requires_completion=False,
            message={},
        )
        try:
            await asyncio.to_thread(self.voice_manager.speak, message)
        except Exception as exc:
            logger.error("TTS 실행 실패: %s", exc)
        yield CommandEvent(
            action="stop_speaking",
            context={},
            requires_completion=False,
            message={},
        )

    def _build_response(self, transcription: str) -> str:
        _ = transcription
        return self._default_response


class SessionFlowCoordinator:
    """탐색 → 유도 → QA 단계를 순차적으로 조율한다."""

    class Phase(Enum):
        SCANNING = auto()
        LANDING = auto()
        QA_ACTIVE = auto()
        AWAIT_RESET = auto()

    _RUN_QA = object()

    def __init__(
        self,
        qa_controller: QAController,
        landing_script: Iterable[QAIntroSpec],
        qa_entry: Optional[QAIntroSpec] = None,
        detection_hold_seconds: float = 5.0,
        alignment_hold_seconds: float = 2.0,
        alignment_tolerance_deg: float = 3.0,
    ) -> None:
        self.qa_controller = qa_controller
        self.landing_script = list(landing_script)
        self.qa_entry = qa_entry
        self.detection_hold_seconds = detection_hold_seconds
        self.alignment_hold_seconds = alignment_hold_seconds
        self.alignment_tolerance_deg = alignment_tolerance_deg

        self._command_queue: asyncio.Queue[Union[CommandEvent, object]] = asyncio.Queue()
        self._phase = self.Phase.SCANNING
        self._landing_enqueued = False
        self._qa_trigger_enqueued = False
        self._first_detected_at: Optional[float] = None
        self._alignment_started_at: Optional[float] = None
        self._pending_intro_prompt: Optional[str] = None

    def process_vision(self, vision_event: VisionResultEvent) -> None:
        if self._phase == self.Phase.AWAIT_RESET:
            if not vision_event.has_target:
                self._reset_to_scanning()
            return

        if self._phase == self.Phase.QA_ACTIVE:
            return

        if not vision_event.has_target:
            self._first_detected_at = None
            self._alignment_started_at = None
            if self._phase == self.Phase.LANDING:
                logger.info("타겟을 잃어 Landing 단계를 취소합니다.")
                self._reset_to_scanning()
            return

        if self._phase != self.Phase.SCANNING:
            return

        if self._first_detected_at is None:
            self._first_detected_at = vision_event.timestamp
            logger.debug("타겟 감지 시작: %.2f", vision_event.timestamp)
            return

        elapsed = vision_event.timestamp - self._first_detected_at
        if elapsed >= self.detection_hold_seconds:
            logger.info("타겟 고정 %.2fs 경과. Landing 단계 진입.", elapsed)
            self._begin_landing()

    def process_motor(self, motor_event: MotorStateEvent) -> None:
        if self._phase != self.Phase.LANDING:
            self._alignment_started_at = None
            return

        if not motor_event.has_target or not self._is_aligned(motor_event):
            self._alignment_started_at = None
            return

        if self._alignment_started_at is None:
            self._alignment_started_at = motor_event.timestamp
            logger.debug("정렬 유지 시작: %.2f", motor_event.timestamp)
            return

        hold_time = motor_event.timestamp - self._alignment_started_at
        if hold_time >= self.alignment_hold_seconds:
            self._trigger_qa_phase()

    async def command_stream(self) -> AsyncIterator[CommandEvent]:
        while True:
            item = await self._command_queue.get()

            if item is self._RUN_QA:
                self._phase = self.Phase.QA_ACTIVE
                logger.info("QA 컨트롤러 실행 시작.")
                try:
                    async for command_event in self.qa_controller.run_once(self._pending_intro_prompt):
                        yield command_event
                finally:
                    logger.info("QA 컨트롤러 종료. 탐색 재개 대기.")
                    self._phase = self.Phase.AWAIT_RESET
                    self._landing_enqueued = False
                    self._qa_trigger_enqueued = False
                    self._pending_intro_prompt = None
                    self._first_detected_at = None
                    self._alignment_started_at = None
                continue

            yield item

    def _begin_landing(self) -> None:
        if self._landing_enqueued:
            return

        self._phase = self.Phase.LANDING
        self._landing_enqueued = True
        self._qa_trigger_enqueued = False
        self._pending_intro_prompt = None
        self._first_detected_at = None
        self._alignment_started_at = None

        if not self.landing_script and self.qa_entry is None:
            self._pending_intro_prompt = None
            self._command_queue.put_nowait(self._RUN_QA)
            self._qa_trigger_enqueued = True
            logger.info("Landing 단계 스크립트가 없어 QA를 즉시 실행합니다.")
            return

        for spec in self.landing_script:
            event = self._spec_to_event(spec)
            self._command_queue.put_nowait(event)
            logger.debug("Landing 명령 큐잉: %s", spec.action)

    def _trigger_qa_phase(self) -> None:
        if self._qa_trigger_enqueued:
            return

        intro_prompt = None
        if self.qa_entry is not None:
            raw_prompt = self.qa_entry.context.get("initialPrompt")
            if raw_prompt is None or isinstance(raw_prompt, str):
                intro_prompt = raw_prompt
            else:
                intro_prompt = str(raw_prompt)
            event = self._spec_to_event(self.qa_entry)
            self._command_queue.put_nowait(event)
            logger.debug("QA 진입 명령 큐잉: %s", self.qa_entry.action)
        else:
            intro_prompt = None

        self._pending_intro_prompt = intro_prompt
        self._command_queue.put_nowait(self._RUN_QA)
        self._qa_trigger_enqueued = True
        self._alignment_started_at = None
        logger.info("모터 정렬 완료. QA 단계 대기열에 추가.")

    def _reset_to_scanning(self) -> None:
        self._phase = self.Phase.SCANNING
        self._landing_enqueued = False
        self._qa_trigger_enqueued = False
        self._pending_intro_prompt = None
        self._first_detected_at = None
        self._alignment_started_at = None

    def _is_aligned(self, motor_event: MotorStateEvent) -> bool:
        return (
            abs(motor_event.pan) <= self.alignment_tolerance_deg
            and abs(motor_event.tilt) <= self.alignment_tolerance_deg
        )

    @staticmethod
    def _spec_to_event(spec: QAIntroSpec) -> CommandEvent:
        return CommandEvent(
            action=spec.action,
            context=spec.context,
            requires_completion=spec.requires_completion,
            message={},
        )
