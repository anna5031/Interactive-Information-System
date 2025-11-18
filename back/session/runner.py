import asyncio
import logging
from dataclasses import dataclass

from features.exploration.config import AssistanceConfig
from features.exploration.pipeline import ExplorationPipeline
from features.qa import QAPipeline
from websocket.connection import ClientConnection

logger = logging.getLogger(__name__)


@dataclass
class SessionRunner:
    connection: ClientConnection
    exploration: ExplorationPipeline
    qa: QAPipeline

    async def run(self) -> None:
        try:
            while self.connection.is_open:
                logger.info("세션 루프 시작: %s", self.connection.identifier)
                await self._run_cycle()
                logger.info("세션 루프 종료: %s", self.connection.identifier)
                await asyncio.sleep(1)
        finally:
            logger.info("세션 종료 정리: %s", self.connection.identifier)

    async def _run_cycle(self) -> None:
        await self.exploration.run()
        decision = getattr(self.exploration, "_last_assistance_decision", None)
        if decision and decision.transition_to_qa:
            logger.info("더미 패스 이벤트 감지 → QA 파이프라인 실행")
            await self.qa.run()
        else:
            logger.info("QA 단계 진입 조건 미충족: skip")


class SessionRunnerFactory:
    def __init__(
        self,
        show_exploration_overlay: bool = False,
        use_dummy_nudge_pass: bool = False,
    ) -> None:
        self.show_exploration_overlay = show_exploration_overlay
        self.use_dummy_nudge_pass = use_dummy_nudge_pass

    def shutdown(self) -> None:
        logger.info("세션 러너 팩토리 종료 처리")

    def create(self, connection: ClientConnection) -> SessionRunner:
        assistance_cfg = AssistanceConfig(
            dummy_nudge_pass_enabled=self.use_dummy_nudge_pass,
        )
        exploration = ExplorationPipeline(
            show_overlay=self.show_exploration_overlay,
            assistance_config=assistance_cfg,
        )
        qa = QAPipeline(connection=connection)
        return SessionRunner(
            connection=connection,
            exploration=exploration,
            qa=qa,
        )
