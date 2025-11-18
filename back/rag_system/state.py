from __future__ import annotations

from typing import Annotated, Dict, List, Optional, TypedDict, TypeVar

from langchain_core.documents import Document


T = TypeVar("T")


def _override(existing: Optional[T], new: Optional[T]) -> T:
    if new is not None:
        return new
    if existing is not None:
        return existing
    raise ValueError("State value missing for override channel.")


def _extend_list(existing: Optional[List[T]], new: Optional[List[T]]) -> List[T]:
    return [*(existing or []), *(new or [])]


class RagState(TypedDict):
    question: Annotated[str, _override]
    sanitized_question: Annotated[str, _override]
    conversation_history: Annotated[List[Dict[str, str]], _extend_list]
    guardrail_reason: Annotated[str, _override]
    needs_retry: Annotated[bool, _override]
    retrieved_documents: Annotated[List[Document], _override]
    retrieval_scores: Annotated[List[float], _override]
    indoor_map: Annotated[Dict, _override]
    answer_text: Annotated[str, _override]
    needs_navigation: Annotated[bool, _override]
    navigation_payload: Annotated[Dict, _override]
    navigation_request: Annotated[Dict, _override]
    processing_log: Annotated[List[str], _extend_list]
    abort_message: Annotated[str, _override]
    session_should_end: Annotated[bool, _override]
