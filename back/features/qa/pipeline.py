from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

from rag_system import SessionResult, StreamingRAGService, SessionConfig
from websocket.connection import ClientConnection

from .services.rag_access import get_rag_service
from .services.voice_io import VoiceIOService, create_voice_service

logger = logging.getLogger(__name__)


class QAPipeline:
    def __init__(
        self,
        *,
        voice_service: Optional[VoiceIOService] = None,
        rag_service: Optional[StreamingRAGService] = None,
        connection: Optional[ClientConnection] = None,
        greeting: str = "안녕하세요. 무엇을 도와드릴까요?",
        closing_message: str = "필요하시면 다시 불러주세요.",
        followup_idle_timeout: Optional[float] = None,
    ) -> None:
        self.voice_service = voice_service or create_voice_service()
        self._rag_service = rag_service
        self.connection = connection
        self.greeting = greeting
        default_timeout = SessionConfig().idle_timeout_seconds
        self.followup_idle_timeout = (
            float(followup_idle_timeout)
            if followup_idle_timeout is not None
            else float(default_timeout)
        )
        self.closing_message = closing_message
        self._closing_message_played = False

    async def run(self) -> None:
        logger.info("QA 파이프라인 시작")
        try:
            if self.greeting:
                self.voice_service.speak(self.greeting)

            is_first_turn = True
            while True:
                audio = self._record_user_audio(is_first_turn=is_first_turn)
                if not audio:
                    if is_first_turn:
                        logger.warning("녹음 실패로 QA 파이프라인을 종료합니다.")
                    else:
                        logger.info("추가 질문이 감지되지 않아 세션을 종료합니다.")
                        self._speak_closing_message()
                    break

                transcription = self.voice_service.transcribe(audio) or ""
                logger.info("사용자 발화: %s", transcription)

                final_answer, nav_summary, rag_result = await self._generate_response(transcription)
                if final_answer:
                    self.voice_service.speak(final_answer)
                #todo: nav 쪽 변경하기. 빔프로젝터로 안내하는걸로
                should_speak_nav = bool(
                    nav_summary
                    and not (rag_result and rag_result.navigation.get("success"))
                )
                if should_speak_nav:
                    self.voice_service.speak(nav_summary)
                await self._emit_event(transcription, final_answer, nav_summary, rag_result)
                if rag_result:
                    if rag_result.needs_retry:
                        logger.info("질문을 다시 받아야 하므로 대기 상태로 돌아갑니다.")
                        continue
                    if rag_result.session_should_end:
                        logger.info("RAG 파이프라인이 세션 종료를 요청했습니다.")
                        self._speak_closing_message()
                        break
                is_first_turn = False
        finally:
            if self._rag_service:
                self._rag_service.reset()
            logger.info("QA 파이프라인 종료")

    async def _generate_response(self, transcription: str) -> Tuple[str, Optional[str], Optional[SessionResult]]:
        user_text = (transcription or "").strip()
        if not user_text:
            fallback = "말씀을 이해하지 못했습니다. 다시 말씀해주세요."
            return fallback, None, None

        rag_service = await self._ensure_rag_service()
        if not rag_service:
            fallback = "RAG 시스템 초기화에 실패했습니다. 잠시 후 다시 시도해주세요."
            return fallback, None, None

        result = await self._query_rag(rag_service, user_text)
        if not result:
            fallback = "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다. 다시 말씀해 주세요."
            return fallback, None, None

        answer_text = result.answer.strip() or "죄송합니다. 적절한 답변을 찾지 못했습니다."
        nav_summary: Optional[str] = None
        if result.navigation.get("success"):
            nav_summary = result.navigation.get("message") or ""
        elif result.navigation_request.get("destination"):
            dest = result.navigation_request.get("destination")
            nav_summary = f"{dest} 경로 안내를 준비 중입니다."

        return answer_text, nav_summary, result

    async def _ensure_rag_service(self) -> Optional[StreamingRAGService]:
        if self._rag_service is None:
            try:
                self._rag_service = await get_rag_service()
            except Exception:
                logger.exception("RAG 서비스 초기화 실패")
                return None
        return self._rag_service

    async def _query_rag(self, rag_service: StreamingRAGService, question: str) -> Optional[SessionResult]:
        try:
            return await rag_service.answer(question)
        except Exception:
            logger.exception("RAG 질의 중 오류 발생")
            return None

    async def _emit_event(
        self,
        question: str,
        answer: str,
        nav_summary: Optional[str],
        result: Optional[SessionResult],
    ) -> None:
        if not self.connection or not self.connection.is_open:
            return

        payload = {
            "type": "qa_result",
            "question": question,
            "answer": answer,
            "navigation": result.navigation if result else {},
            "navigation_request": result.navigation_request if result else {},
            "navigation_summary": nav_summary or "",
            "documents": self._format_documents(result),
            "processing_log": result.processing_log if result else [],
        }
        try:
            await self.connection.send(json.dumps(payload, ensure_ascii=False))
        except Exception:
            logger.exception("QA 결과 전송 실패")

    def _record_user_audio(self, *, is_first_turn: bool) -> Optional[bytes]:
        timeout = None if is_first_turn else self.followup_idle_timeout
        effective_timeout = timeout if (timeout and timeout > 0) else None
        audio = self.voice_service.record_audio(max_idle_seconds=effective_timeout)
        if not audio and effective_timeout:
            logger.info("음성 입력 대기 시간 %.1fs 초과", effective_timeout)
        return audio

    def _speak_closing_message(self) -> None:
        if self.closing_message and not self._closing_message_played:
            self.voice_service.speak(
                self.closing_message,
                warmup=True,
            )
            self._closing_message_played = True

    @staticmethod
    def _format_documents(result: Optional[SessionResult]) -> list[dict]:
        if not result or not result.documents:
            return []
        items = []
        scores = list(result.scores or [])
        for index, doc in enumerate(result.documents):
            score = scores[index] if index < len(scores) else 0.0
            items.append(
                {
                    "source": doc.metadata.get("doc_id") or doc.metadata.get("source") or f"doc_{index+1}",
                    "score": round(float(score), 3),
                    "excerpt": doc.page_content[:400],
                }
            )
        return items
