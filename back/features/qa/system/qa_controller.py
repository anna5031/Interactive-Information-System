from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator, Iterable, Optional, Union, Callable

from app.events import CommandEvent, MotorStateEvent, VisionResultEvent
from rag_service import QAServiceResult, RAGQAService
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
        qa_service: Optional[RAGQAService] = None,
        stream_processing_log: bool = False,
    ) -> None:
        self.voice_manager = voice_manager
        self.stt_manager = stt_manager
        self._default_response = default_response
        self.qa_service = qa_service or RAGQAService()
        self._stream_processing_log = stream_processing_log
        self._last_result: Optional[QAServiceResult] = None

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

        response_text, rag_result = await self._generate_response(transcription)

        if rag_result is not None:
            logger.debug(
                "RAG sources=%d navigation=%s",
                len(rag_result.sources),
                rag_result.needs_navigation,
            )

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

    async def _generate_response(self, transcription: str) -> tuple[str, Optional[QAServiceResult]]:
        try:
            result = await self.qa_service.query(
                transcription,
                emit_processing_log=self._stream_processing_log,
            )
        except Exception as exc:
            logger.exception("RAG 응답 생성 실패: %s", exc)
            self._last_result = None
            return self._default_response, None

        self._last_result = result

        if not result.is_safe:
            logger.warning("RAG 응답이 필터에 의해 제한되었습니다: %s", result.summary())
        else:
            logger.info("RAG 응답 생성 완료: %s", result.summary())

        response_text = result.answer.strip() if result.answer else ""
        if not response_text:
            response_text = self._default_response

        return response_text, result

    @property
    def last_result(self) -> Optional[QAServiceResult]:
        return self._last_result


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
        initial_script: Optional[Iterable[QAIntroSpec]] = None,
        qa_entry: Optional[QAIntroSpec] = None,
        detection_hold_seconds: float = 0.0,
        alignment_hold_seconds: float = 2.0,
        alignment_tolerance_deg: float = 3.0,
        qa_start_delay_seconds: float = 5.0,
        on_enter_qa: Optional[Callable[[], None]] = None,
        on_exit_qa: Optional[Callable[[], None]] = None,
        distance_threshold_mm: float = 800.0,
        approach_timeout_seconds: float = 10.0,
        approach_delta_min_mm: float = 100.0,
        skip_to_qa_auto: bool = False,
        qa_auto_delay_seconds: float = 5.0,
        lost_target_grace_seconds: float = 2.0,
    ) -> None:
        self.qa_controller = qa_controller
        self.landing_script = list(landing_script)
        self.initial_script = list(initial_script or [])
        self.qa_entry = qa_entry
        self.detection_hold_seconds = detection_hold_seconds
        self.alignment_hold_seconds = alignment_hold_seconds
        self.alignment_tolerance_deg = alignment_tolerance_deg
        self.qa_start_delay_seconds = qa_start_delay_seconds
        self._on_enter_qa = on_enter_qa
        self._on_exit_qa = on_exit_qa
        self.distance_threshold_mm = distance_threshold_mm
        self.approach_timeout_seconds = approach_timeout_seconds
        self.approach_delta_min_mm = approach_delta_min_mm
        self.skip_to_qa_auto = skip_to_qa_auto
        self.qa_auto_delay_seconds = qa_auto_delay_seconds
        self.lost_target_grace_seconds = lost_target_grace_seconds

        self._command_queue: asyncio.Queue[Union[CommandEvent, object]] = asyncio.Queue()
        self._phase = self.Phase.SCANNING
        self._landing_enqueued = False
        self._qa_trigger_enqueued = False
        self._first_detected_at: Optional[float] = None
        self._alignment_started_at: Optional[float] = None
        self._pending_intro_prompt: Optional[str] = None
        self._initial_script_sent = False
        self._qa_timer_handle: Optional[asyncio.TimerHandle] = None
        self._landing_started_at: Optional[float] = None
        self._initial_distance_mm: Optional[float] = None
        self._last_distance_mm: Optional[float] = None
        self._last_distance_timestamp: Optional[float] = None
        self._has_distance_progress: bool = False
        self._qa_auto_armed: bool = False
        self._last_seen_timestamp: Optional[float] = None

        self._enqueue_initial_script()

    def process_vision(self, vision_event: VisionResultEvent) -> None:
        if self._phase == self.Phase.AWAIT_RESET:
            if not vision_event.has_target:
                self._reset_to_scanning()
            return

        if self._phase == self.Phase.QA_ACTIVE:
            return

        if not vision_event.has_target:
            if (
                self._phase == self.Phase.LANDING
                and self._last_seen_timestamp is not None
                and vision_event.timestamp - self._last_seen_timestamp <= self.lost_target_grace_seconds
            ):
                logger.debug(
                    "타겟 일시 미검출 %.2fs 이내. Landing을 유지합니다.",
                    vision_event.timestamp - self._last_seen_timestamp,
                )
                return
            self._first_detected_at = None
            self._alignment_started_at = None
            self._last_seen_timestamp = None
            if self._phase == self.Phase.LANDING:
                logger.info("타겟을 잃어 Landing 단계를 취소합니다.")
                self._reset_to_scanning()
            return

        self._last_seen_timestamp = vision_event.timestamp

        if self._phase != self.Phase.SCANNING:
            return

        if self._first_detected_at is None:
            self._first_detected_at = vision_event.timestamp
            logger.debug("타겟 감지 시작: %.2f", vision_event.timestamp)
            return

        elapsed = vision_event.timestamp - self._first_detected_at
        if elapsed >= self.detection_hold_seconds:
            logger.info("타겟 고정 %.2fs 경과. Landing 단계 진입.", elapsed)
            self._begin_landing(vision_event.timestamp)

    def process_motor(self, motor_event: MotorStateEvent) -> None:
        if self._phase != self.Phase.LANDING:
            return

        distance = motor_event.distance_to_projector
        if distance is None:
            return

        logger.info(
            "타겟-프로젝터 거리 %.1fmm (phase=%s)",
            distance,
            self._phase.name,
        )

        if self._initial_distance_mm is None:
            self._initial_distance_mm = distance
            self._last_distance_mm = distance
            self._last_distance_timestamp = motor_event.timestamp
            return

        self._last_distance_timestamp = motor_event.timestamp
        self._last_distance_mm = distance

        progress = self._initial_distance_mm - distance
        if progress >= self.approach_delta_min_mm:
            if not self._has_distance_progress:
                logger.debug("타겟이 빔 방향으로 %.1fmm 접근했습니다.", progress)
            self._has_distance_progress = True

        if (
            not self.skip_to_qa_auto
            and not self._has_distance_progress
            and self._landing_started_at is not None
            and motor_event.timestamp - self._landing_started_at >= self.approach_timeout_seconds
        ):
            logger.info(
                "타겟이 %.1fs 동안 빔 방향으로 이동하지 않아 세션을 초기화합니다.",
                motor_event.timestamp - self._landing_started_at,
            )
            self._reset_to_scanning()
            return

        if distance <= self.distance_threshold_mm:
            logger.info(
                "타겟이 프로젝터에서 %.1fmm 거리로 접근했습니다. QA 전환을 준비합니다.",
                distance,
            )
            self._trigger_qa_phase()

    async def command_stream(self) -> AsyncIterator[CommandEvent]:
        while True:
            item = await self._command_queue.get()

            if item is self._RUN_QA:
                self._phase = self.Phase.QA_ACTIVE
                logger.info("QA 컨트롤러 실행 시작.")
                self._notify_enter_qa()
                try:
                    async for command_event in self.qa_controller.run_once(self._pending_intro_prompt):
                        yield command_event
                finally:
                    logger.info("QA 컨트롤러 종료. 탐색 재개 대기.")
                    self._notify_exit_qa()
                    self._phase = self.Phase.AWAIT_RESET
                    self._landing_enqueued = False
                    self._qa_trigger_enqueued = False
                    self._pending_intro_prompt = None
                    self._first_detected_at = None
                    self._alignment_started_at = None
                    self._landing_started_at = None
                    self._initial_distance_mm = None
                    self._last_distance_mm = None
                    self._last_distance_timestamp = None
                    self._has_distance_progress = False
                    self._qa_auto_armed = False
                    self._last_seen_timestamp = None
                    self._cancel_qa_timer()
                continue

            yield item

    def _begin_landing(self, start_timestamp: float) -> None:
        if self._landing_enqueued:
            return

        self._phase = self.Phase.LANDING
        self._landing_enqueued = True
        self._qa_trigger_enqueued = False
        self._pending_intro_prompt = None
        self._first_detected_at = None
        self._alignment_started_at = None
        self._landing_started_at = start_timestamp
        self._initial_distance_mm = None
        self._last_distance_mm = None
        self._last_distance_timestamp = None
        self._has_distance_progress = False
        self._qa_auto_armed = False
        self._last_seen_timestamp = start_timestamp
        self._cancel_qa_timer()

        if not self.landing_script and self.qa_entry is None:
            self._pending_intro_prompt = None
            self._command_queue.put_nowait(self._RUN_QA)
            self._qa_trigger_enqueued = True
            logger.info("Landing 단계 스크립트가 없어 QA를 즉시 실행합니다.")
            if self.skip_to_qa_auto:
                self._schedule_qa_timer(cancel_existing=False)
            return

        for spec in self.landing_script:
            event = self._spec_to_event(spec)
            self._command_queue.put_nowait(event)
            logger.debug("Landing 명령 큐잉: %s", spec.action)

        if self.skip_to_qa_auto:
            self._schedule_qa_timer(cancel_existing=True)

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
        self._cancel_qa_timer()

    def _reset_to_scanning(self) -> None:
        self._phase = self.Phase.SCANNING
        self._landing_enqueued = False
        self._qa_trigger_enqueued = False
        self._pending_intro_prompt = None
        self._first_detected_at = None
        self._alignment_started_at = None
        self._landing_started_at = None
        self._initial_distance_mm = None
        self._last_distance_mm = None
        self._last_distance_timestamp = None
        self._has_distance_progress = False
        self._qa_auto_armed = False
        self._last_seen_timestamp = None
        self._cancel_qa_timer()

    def _enqueue_initial_script(self) -> None:
        if self._initial_script_sent or not self.initial_script:
            return

        for spec in self.initial_script:
            event = self._spec_to_event(spec)
            self._command_queue.put_nowait(event)
            logger.debug("초기 명령 큐잉: %s", spec.action)

        self._initial_script_sent = True

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

    def _schedule_qa_timer(self, cancel_existing: bool = True) -> None:
        if not self.skip_to_qa_auto:
            return
        if cancel_existing:
            self._cancel_qa_timer()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def _delayed_trigger() -> None:
            self._qa_timer_handle = None
            if self._qa_trigger_enqueued:
                return
            self._trigger_qa_phase()

        self._qa_timer_handle = loop.call_later(self.qa_auto_delay_seconds, _delayed_trigger)
        self._qa_auto_armed = True

    def _cancel_qa_timer(self) -> None:
        handle = self._qa_timer_handle
        if handle is not None:
            handle.cancel()
            self._qa_timer_handle = None
        self._qa_auto_armed = False

    def _notify_enter_qa(self) -> None:
        if self._on_enter_qa is not None:
            try:
                self._on_enter_qa()
            except Exception as exc:  # pragma: no cover
                logger.warning("on_enter_qa callback failed: %s", exc)

    def _notify_exit_qa(self) -> None:
        if self._on_exit_qa is not None:
            try:
                self._on_exit_qa()
            except Exception as exc:  # pragma: no cover
                logger.warning("on_exit_qa callback failed: %s", exc)
