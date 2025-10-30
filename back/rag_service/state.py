from typing import Dict, List, Optional, TypedDict, Annotated, TypeVar

T = TypeVar("T")


def keep_initial_query(existing: Optional[str], new: Optional[str]) -> str:
    """
    LangGraph는 병렬 노드가 동일한 키를 동시에 갱신하려고 하면
    merge 단계에서 충돌을 일으킨다. user_query는 최초 입력 이후
    수정되지 않아야 하므로, 최초 값만 유지하도록 리듀서를 정의한다.
    """
    if isinstance(existing, str) and existing.strip():
        return existing
    return new or ""


def overwrite_value(existing: Optional[T], new: Optional[T]) -> T:
    """
    병렬 노드가 동일 키를 갱신하더라도 최신 값을 유지한다.
    LangGraph는 기본적으로 스칼라 타입을 병합하지 않으므로
    새 값을 우선하는 커스텀 리듀서를 둔다.
    """
    if new is not None:
        return new
    if existing is not None:
        return existing
    raise ValueError("State reducer received None for both existing and new values.")


class UniversityMEState(TypedDict):
    user_query: Annotated[str, keep_initial_query]
    detected_intent: Annotated[str, overwrite_value]
    intent_confidence: Annotated[float, overwrite_value]
    retrieved_documents: Annotated[List[str], overwrite_value]
    backup_documents: Annotated[List[str], overwrite_value]
    llm_response: Annotated[str, overwrite_value]
    structured_output: Annotated[Dict, overwrite_value]
    final_output: Annotated[str, overwrite_value]
    filter_checks: Annotated[Dict, overwrite_value]
    filter_results: Annotated[Dict, overwrite_value]
    llm_evaluation: Annotated[Dict, overwrite_value]
    needs_navigation: Annotated[bool, overwrite_value]
    navigation_info: Annotated[Dict, overwrite_value]
    processing_log: Annotated[List[str], overwrite_value]
    question_index: Annotated[int, overwrite_value]
    total_questions: Annotated[int, overwrite_value]
    preprocessing_notes: Annotated[List[str], overwrite_value]
    sanitized_query: Annotated[str, overwrite_value]
    effective_query: Annotated[str, overwrite_value]
    rewritten_query: Annotated[str, overwrite_value]
    rewrite_attempted: Annotated[bool, overwrite_value]
    llm_prompt_query: Annotated[str, overwrite_value]
    retry_query: Annotated[str, overwrite_value]
    llm_retry_used: Annotated[bool, overwrite_value]
    intent_is_malicious: Annotated[bool, overwrite_value]
    abort_message: Annotated[str, overwrite_value]
    retry_mode: Annotated[str, overwrite_value]
    needs_retry: Annotated[bool, overwrite_value]
    retry_reason: Annotated[str, overwrite_value]
