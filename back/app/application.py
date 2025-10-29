from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any, AsyncIterator

from app.config import AppConfig, load_config
from app.events import CommandEvent
from features.exploration.stub import ExplorationStub, ExplorationStubConfig
from features.homography.stub import HomographyStub
from features.motor.stub import MotorStub, MotorStubConfig
from features.qa import (
    QAController,
    QAIntroSpec,
    QAStub,
    SessionFlowCoordinator,
    SpeechToTextManager,
    VoiceInterfaceManager,
)
from rag_test_fin import RAGQAService


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DebugConfig:
    log_exploration: bool = False
    log_motor: bool = False
    log_homography: bool = False
    log_commands: bool = False


@dataclass(slots=True)
class SessionDependencies:
    config: AppConfig
    exploration: ExplorationStub
    motor: MotorStub
    homography: HomographyStub
    flow: Any
    debug: DebugConfig


class QAStubFlowAdapter:
    """QAStub을 SessionFlowCoordinator 인터페이스에 맞춰 래핑."""

    def __init__(self, stub: QAStub) -> None:
        self.stub = stub

    def process_vision(self, vision_event: Any) -> None:  # noqa: ARG002
        # 더미 구현에서는 비동기 탐색 신호를 사용하지 않는다.
        return

    def process_motor(self, motor_state: Any) -> None:  # noqa: ARG002
        return

    async def command_stream(self) -> AsyncIterator[CommandEvent]:
        async for event in self.stub.sequence():
            yield event


class Application:
    """세션 생성과 공용 리소스를 관리하는 애플리케이션 컨테이너."""

    def __init__(
        self, config: Optional[AppConfig] = None, debug: Optional[DebugConfig] = None
    ) -> None:
        self.config = config or load_config()
        self.debug = debug or DebugConfig()
        self._qa_service: Optional[RAGQAService] = None

    def create_session(self) -> SessionDependencies:
        exploration = ExplorationStub(
            ExplorationStubConfig(interval=self.config.vision_interval)
        )
        motor = MotorStub(MotorStubConfig())
        homography = HomographyStub()
        flow = self._create_qa_pipeline()
        return SessionDependencies(
            config=self.config,
            exploration=exploration,
            motor=motor,
            homography=homography,
            flow=flow,
            debug=self.debug,
        )

    def _create_qa_pipeline(self) -> Any:
        try:
            voice_manager = VoiceInterfaceManager()
            stt_manager = SpeechToTextManager()

            if self._qa_service is None:
                self._qa_service = RAGQAService()

            qa_controller = QAController(
                voice_manager=voice_manager,
                stt_manager=stt_manager,
                qa_service=self._qa_service,
                stream_processing_log=self.debug.log_commands,
            )

            landing_script = [
                QAIntroSpec(
                    action="start_landing",
                    context={"message": "시스템 준비 중입니다."},
                    requires_completion=True,
                ),
                QAIntroSpec(action="start_nudge", context={}),
            ]
            qa_entry = QAIntroSpec(
                action="start_qa",
                context={"initialPrompt": "안녕하세요! 무엇을 도와드릴까요?"},
                requires_completion=True,
            )

            flow = SessionFlowCoordinator(
                qa_controller=qa_controller,
                landing_script=landing_script,
                qa_entry=qa_entry,
                detection_hold_seconds=self.config.detection_hold_seconds,
                alignment_hold_seconds=2.0,
                alignment_tolerance_deg=2.5,
            )
            logger.info("QAController 초기화 완료")
            return flow
        except Exception as exc:
            logger.warning("QA 파이프라인 초기화 실패. 스텁으로 대체합니다: %s", exc)
            return QAStubFlowAdapter(QAStub())
