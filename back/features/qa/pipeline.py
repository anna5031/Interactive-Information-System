from __future__ import annotations

import logging
from typing import Optional

from .services.voice_io import VoiceIOService, create_voice_service

logger = logging.getLogger(__name__)


class QAPipeline:
    def __init__(
        self,
        *,
        voice_service: Optional[VoiceIOService] = None,
        greeting: str = "안녕하세요. 무엇을 도와드릴까요?",
    ) -> None:
        self.voice_service = voice_service or create_voice_service()
        self.greeting = greeting

    async def run(self) -> None:
        logger.info("QA 파이프라인 시작")
        try:
            if self.greeting:
                self.voice_service.speak(self.greeting)

            audio = self.voice_service.record_audio()
            if not audio:
                logger.warning("녹음 실패로 QA 파이프라인을 종료합니다.")
                return
            transcription = self.voice_service.transcribe(audio) or ""
            logger.info("사용자 발화: %s", transcription)

            response = self._generate_response(transcription)
            self.voice_service.speak(response)
        finally:
            logger.info("QA 파이프라인 종료")

    def _generate_response(self, transcription: str) -> str:
        # TODO: integrate RAG/LLM; currently simple echo
        transcription = transcription.strip()
        if not transcription:
            return "말씀을 이해하지 못했습니다. 다시 말씀해주세요."
        return f"당신이 말씀하신 '{transcription}'에 대해 준비 중입니다."
