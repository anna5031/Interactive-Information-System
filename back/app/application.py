from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Tuple

from app.config import AppConfig, load_config
from app.events import CommandEvent
from features.exploration import (
    ExplorationPipeline,
    ExplorationStub,
    ExplorationStubConfig,
    load_exploration_config,
)
from features.homography import (
    HomographyCalculator,
    HomographyStub,
    PixelToWorldMapper,
    load_calibration_bundle,
)
from features.motor import (
    MotorStub,
    MotorStubConfig,
    RealMotorController,
)
from features.qa import (
    QAController,
    QAIntroSpec,
    QAStub,
    SessionFlowCoordinator,
    SpeechToTextManager,
    VoiceInterfaceManager,
)
from rag_service import RAGQAService


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DebugConfig:
    log_exploration: bool = False
    log_motor: bool = False
    log_homography: bool = False
    log_commands: bool = False
    show_exploration_overlay: bool = False


@dataclass(slots=True)
class RuntimeOverrides:
    force_dummy_exploration: bool = False
    force_dummy_motor: bool = False
    force_dummy_homography: bool = False
    skip_to_qa_auto: Optional[bool] = None


@dataclass(slots=True)
class SessionDependencies:
    config: AppConfig
    exploration: Any
    motor: Any
    homography: Any
    flow: Any
    debug: DebugConfig
    mapper: Optional[PixelToWorldMapper]
    cleanup: Tuple[Callable[[], Awaitable[None] | None], ...]


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
        self,
        config: Optional[AppConfig] = None,
        debug: Optional[DebugConfig] = None,
        overrides: Optional[RuntimeOverrides] = None,
    ) -> None:
        self.config = config or load_config()
        self.debug = debug or DebugConfig()
        self._qa_service: Optional[RAGQAService] = None
        self.overrides = overrides or RuntimeOverrides()

    def create_session(self) -> SessionDependencies:
        cleanup_callbacks: list[Callable[[], Awaitable[None] | None]] = []

        exploration = self._create_exploration()

        mapper: Optional[PixelToWorldMapper] = None
        calibration_bundle = None
        needs_calibration = (
            (self.config.motor.backend != "stub" and not self.overrides.force_dummy_motor)
            or (self.config.homography.backend != "stub" and not self.overrides.force_dummy_homography)
        )
        if needs_calibration:
            try:
                calibration_bundle = load_calibration_bundle(
                    self.config.homography.files.camera_calibration_file,
                    self.config.homography.files.camera_extrinsics_file,
                )
                mapper = PixelToWorldMapper(calibration_bundle)
                logger.info("Calibration bundle loaded successfully.")
            except Exception as exc:
                logger.warning("Calibration assets unavailable: %s", exc)

        motor, motor_cleanup = self._create_motor(mapper)
        if motor_cleanup is not None:
            cleanup_callbacks.append(motor_cleanup)

        homography = self._create_homography(calibration_bundle=calibration_bundle)

        flow = self._create_qa_pipeline(exploration)

        return SessionDependencies(
            config=self.config,
            exploration=exploration,
            motor=motor,
            homography=homography,
            flow=flow,
            debug=self.debug,
            mapper=mapper,
            cleanup=tuple(cleanup_callbacks),
        )

    def _create_exploration(self) -> Any:
        if self.overrides.force_dummy_exploration:
            logger.info("Exploration pipeline forced to stub via override.")
            return ExplorationStub(
                ExplorationStubConfig(interval=self.config.vision_interval)
            )
        try:
            exploration_config = load_exploration_config()
            exploration_config.debug_display = self.debug.show_exploration_overlay
            pipeline = ExplorationPipeline(
                exploration_config, log_details=self.debug.log_exploration
            )
            logger.info("Exploration pipeline initialised (device=%s)", pipeline.device)
            return pipeline
        except Exception as exc:
            logger.warning("Exploration pipeline init failed. Falling back to stub: %s", exc)
            return ExplorationStub(
                ExplorationStubConfig(interval=self.config.vision_interval)
            )

    def _create_qa_pipeline(self, exploration: Any) -> Any:
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

            initial_script = [
                QAIntroSpec(
                    action="start_landing",
                    context={"message": "시스템 준비 중입니다."},
                    requires_completion=True,
                )
            ]
            landing_script = [
                QAIntroSpec(action="start_nudge", context={}),
            ]
            qa_entry = QAIntroSpec(
                action="start_qa",
                context={"initialPrompt": "안녕하세요! 무엇을 도와드릴까요?"},
                requires_completion=True,
            )

            engagement_cfg = self.config.engagement
            skip_to_qa_auto = (
                engagement_cfg.skip_to_qa_auto
                if self.overrides.skip_to_qa_auto is None
                else self.overrides.skip_to_qa_auto
            )

            flow = SessionFlowCoordinator(
                qa_controller=qa_controller,
                landing_script=landing_script,
                initial_script=initial_script,
                qa_entry=qa_entry,
                detection_hold_seconds=self.config.detection_hold_seconds,
                alignment_hold_seconds=2.0,
                alignment_tolerance_deg=2.5,
                qa_start_delay_seconds=engagement_cfg.qa_auto_delay_seconds,
                on_enter_qa=getattr(exploration, "suspend", None),
                on_exit_qa=getattr(exploration, "resume", None),
                distance_threshold_mm=engagement_cfg.distance_threshold_mm,
                approach_timeout_seconds=engagement_cfg.approach_timeout_seconds,
                approach_delta_min_mm=engagement_cfg.approach_delta_min_mm,
                skip_to_qa_auto=skip_to_qa_auto,
                qa_auto_delay_seconds=engagement_cfg.qa_auto_delay_seconds,
                lost_target_grace_seconds=engagement_cfg.lost_target_grace_seconds,
            )
            logger.info("QAController 초기화 완료")
            return flow
        except Exception as exc:
            logger.warning("QA 파이프라인 초기화 실패. 스텁으로 대체합니다: %s", exc)
            return QAStubFlowAdapter(QAStub())


    def _create_motor(
        self,
        mapper: Optional[PixelToWorldMapper],
    ) -> Tuple[Any, Optional[Callable[[], Awaitable[None] | None]]]:
        backend = getattr(self.config.motor, "backend", "stub").lower()
        if self.overrides.force_dummy_motor:
            logger.info("Motor controller forced to stub via override.")
            return (
                MotorStub(
                    MotorStubConfig(),
                    motor_config=self.config.motor,
                    homography_config=self.config.homography,
                    mapper=mapper,
                ),
                None,
            )
        if backend == "stub":
            return (
                MotorStub(
                    MotorStubConfig(),
                    motor_config=self.config.motor,
                    homography_config=self.config.homography,
                    mapper=mapper,
                ),
                None,
            )
        if backend == "serial":
            if mapper is None:
                raise RuntimeError(
                    "Motor backend 'serial' requires calibration assets, but none were loaded."
                )
            controller = RealMotorController(
                motor_config=self.config.motor,
                homography_config=self.config.homography,
                mapper=mapper,
            )
            return controller, controller.shutdown
        raise ValueError(f"Unsupported motor backend: {backend}")

    def _create_homography(
        self,
        calibration_bundle: Optional[Any],
    ) -> Any:
        backend = getattr(self.config.homography, "backend", "stub").lower()
        if self.overrides.force_dummy_homography:
            logger.info("Homography calculator forced to stub via override.")
            return HomographyStub(debug_logs=self.debug.log_homography)
        if backend == "stub":
            return HomographyStub(debug_logs=self.debug.log_homography)
        if calibration_bundle is None:
            logger.warning("Homography backend requested but calibration unavailable. Using stub.")
            return HomographyStub(debug_logs=self.debug.log_homography)
        try:
            return HomographyCalculator(
                calibration=calibration_bundle,
                config=self.config.homography,
            )
        except Exception as exc:
            logger.warning("Homography calculator init failed. Falling back to stub: %s", exc)
            return HomographyStub(debug_logs=self.debug.log_homography)
