from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional

from .info_system import UniversityMEInfoSystem


@dataclass(slots=True)
class QAServiceResult:
    """QAController와 통합하기 위한 정규화된 RAG 응답."""

    question: str
    answer: str
    intent: str
    confidence: float
    is_safe: bool
    blocked_by: List[str]
    sources: List[str]
    needs_navigation: bool
    navigation_info: Dict[str, Any]
    map_path: Optional[str]
    processing_time: float
    processing_log: List[str]
    raw: Dict[str, Any]

    def summary(self) -> str:
        """로그 용도로 사용할 간략 요약."""
        status = "safe" if self.is_safe else f"blocked({','.join(self.blocked_by) or '?'})"
        return f"intent={self.intent or 'unknown'}[{status}] answer_len={len(self.answer)} time={self.processing_time:.2f}s"


class RAGQAService:
    """QAController에서 직접 사용할 수 있는 RAG 서비스."""

    def __init__(
        self,
        info_system: Optional[UniversityMEInfoSystem] = None,
        *,
        default_emit_processing_log: bool = False,
    ) -> None:
        self.info_system = info_system or UniversityMEInfoSystem()
        self.default_emit_processing_log = default_emit_processing_log
        self._warmup_lock: asyncio.Lock = asyncio.Lock()
        self._warmed_up: bool = False

    async def warm_up(self, include_navigation: bool = False) -> None:
        """백그라운드 초기화를 명시적으로 수행."""
        await self.info_system.warm_up(include_navigation=include_navigation)
        self._warmed_up = True

    async def ensure_ready(self) -> None:
        """동일한 워커를 여러 곳에서 호출해도 1회만 초기화되도록 보장."""
        if self._warmed_up:
            return

        async with self._warmup_lock:
            if self._warmed_up:
                return
            await self.info_system.ensure_rag_ready()
            await asyncio.to_thread(self.info_system.rag_system.db_manager.ensure_embeddings_ready)
            self._warmed_up = True

    async def query(self, question: str, *, emit_processing_log: Optional[bool] = None) -> QAServiceResult:
        """사용자 질문에 대한 RAG 응답을 생성."""
        if emit_processing_log is None:
            emit_processing_log = self.default_emit_processing_log

        await self.ensure_ready()

        start = perf_counter()
        result = await self.info_system.process_query(question)
        elapsed = perf_counter() - start

        processing_log = list(result.get("processing_log") or [])
        if emit_processing_log:
            self._emit_processing_log(question, processing_log)

        filter_results = result.get("filter_results") or {}
        blocked_by = filter_results.get("blocked_by") or []
        is_safe = bool(filter_results.get("is_safe", True))

        navigation_info = dict(result.get("navigation_info") or {})
        map_path = navigation_info.get("map_path")

        sources = list(result.get("retrieved_documents") or [])

        return QAServiceResult(
            question=question,
            answer=result.get("final_output", ""),
            intent=result.get("detected_intent") or "",
            confidence=float(result.get("intent_confidence") or 0.0),
            is_safe=is_safe,
            blocked_by=list(blocked_by),
            sources=sources,
            needs_navigation=bool(result.get("needs_navigation")),
            navigation_info=navigation_info,
            map_path=map_path,
            processing_time=elapsed,
            processing_log=processing_log,
            raw=result,
        )

    def query_sync(self, question: str, *, emit_processing_log: Optional[bool] = None) -> QAServiceResult:
        """CLI나 테스트용으로 동기 실행을 지원."""
        return asyncio.run(self.query(question, emit_processing_log=emit_processing_log))

    @staticmethod
    def _emit_processing_log(question: str, log_entries: List[str]) -> None:
        header = f"[RAG] 질문: {question}"
        print(header)
        if not log_entries:
            print("[RAG]  (processing log 비어 있음)")
            return
        for entry in log_entries:
            print(f"[RAG]  - {entry}")


__all__ = ["QAServiceResult", "RAGQAService"]
