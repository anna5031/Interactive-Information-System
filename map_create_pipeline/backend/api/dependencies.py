from __future__ import annotations

from typing import Optional

from fastapi import Request

from services.user_storage import (
    DEFAULT_USER_ID,
    USER_ID_HEADER,
    USER_ID_QUERY_PARAM,
    sanitize_user_id,
)


def resolve_active_user_id(request: Request, fallback: Optional[str] = None) -> str:
    """요청 헤더/쿼리에서 사용자 ID를 추출해 정규화한다."""
    user_token = request.query_params.get(USER_ID_QUERY_PARAM)
    if not user_token:
        user_token = request.headers.get(USER_ID_HEADER)
    if not user_token:
        user_token = fallback or DEFAULT_USER_ID
    return sanitize_user_id(user_token)
