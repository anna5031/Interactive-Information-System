from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.documents import Document

from .config import INDEX_CONFIG, SOURCE_DOCS_DIR, INDEX_DIR
from .graph import RagWorkflowBuilder
from .state import RagState
from .text_ingest import TextCorpusBuilder
from .vector_index import LocalVectorIndex


def build_index_from_source() -> None:
    corpus = TextCorpusBuilder(
        SOURCE_DOCS_DIR,
        chunk_size=INDEX_CONFIG.chunk_size,
        chunk_overlap=INDEX_CONFIG.chunk_overlap,
    )
    chunks = corpus.load()
    if not chunks:
        raise RuntimeError(f"{SOURCE_DOCS_DIR} 디렉터리에 텍스트 파일이 존재하지 않습니다.")
    LocalVectorIndex(INDEX_DIR).build(chunks)


@dataclass(slots=True)
class SessionConfig:
    idle_timeout_seconds: int = 7
    relative_threshold: float = INDEX_CONFIG.relative_threshold


@dataclass
class SessionResult:
    question: str
    answer: str
    documents: List[Document]
    navigation: Dict
    processing_log: List[str]


class StreamingRAGService:
    def __init__(self, config: SessionConfig | None = None) -> None:
        self.config = config or SessionConfig()
        self.vector_index = LocalVectorIndex(INDEX_DIR)
        self._workflow = RagWorkflowBuilder(vector_index=self.vector_index).build()
        self._conversation_history: List[Dict[str, str]] = []
        self._lock = asyncio.Lock()

    def build_index(self) -> None:
        build_index_from_source()

    async def answer(self, question: str) -> SessionResult:
        if not question.strip():
            raise ValueError("질문이 비어 있습니다.")
        async with self._lock:
            self.vector_index.load()
            state = await self._workflow.ainvoke(
                RagState(
                    question=question,
                    sanitized_question=question,
                    conversation_history=list(self._conversation_history),
                    guardrail_reason="",
                    needs_retry=False,
                    retrieved_documents=[],
                    retrieval_scores=[],
                    answer_text="",
                    needs_navigation=False,
                    navigation_payload={},
                    processing_log=[],
                    abort_message="",
                )
            )
            answer = state.get("answer_text", "")
            navigation = state.get("navigation_payload", {})
            documents = state.get("retrieved_documents", [])
            log = state.get("processing_log", [])
            self._conversation_history = state.get("conversation_history", [])
            return SessionResult(question=question, answer=answer, documents=documents, navigation=navigation, processing_log=log)

    def reset(self) -> None:
        self._conversation_history = []

    async def run_interactive(self) -> None:
        print("🧠 새 QA 세션을 시작합니다. 종료하려면 exit/quit 입력.")
        timeout = self.config.idle_timeout_seconds
        while True:
            try:
                question = await self._prompt(timeout=timeout)
            except TimeoutError:
                print("⏱️ 입력이 없어 세션을 초기화합니다.")
                self.reset()
                continue
            except (KeyboardInterrupt, EOFError):
                print("\n세션을 종료합니다.")
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            result = await self.answer(question)
            print("\n답변:", result.answer)
            if result.navigation.get("success"):
                print("경로 안내:", result.navigation["message"])

    async def _prompt(self, timeout: int | None = None) -> str:
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, lambda: input("질문> ").strip())
        if timeout and timeout > 0:
            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError from exc
        return await future
