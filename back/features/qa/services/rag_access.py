from __future__ import annotations

from asyncio import Lock
from typing import Optional

from rag_system import StreamingRAGService

_rag_service: Optional[StreamingRAGService] = None
_rag_lock: Optional[Lock] = None


async def get_rag_service() -> StreamingRAGService:
    """RAG 서비스를 싱글턴으로 초기화."""
    global _rag_service, _rag_lock
    if _rag_service is None:
        if _rag_lock is None:
            _rag_lock = Lock()
        async with _rag_lock:
            if _rag_service is None:
                _rag_service = StreamingRAGService()
    return _rag_service
