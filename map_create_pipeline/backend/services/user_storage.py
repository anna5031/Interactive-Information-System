from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_USER_ID = "anonymous"
USER_ID_QUERY_PARAM = "userId"
USER_ID_HEADER = "X-User-ID"
_USER_ID_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")


def sanitize_user_id(raw_user_id: Optional[str], default: str = DEFAULT_USER_ID) -> str:
    """입력 문자열을 파일 시스템에서 안전한 사용자 ID로 정규화한다."""
    if raw_user_id is None:
        return default
    if not isinstance(raw_user_id, str):
        raw_user_id = str(raw_user_id)
    trimmed = raw_user_id.strip()
    if not trimmed:
        return default
    sanitized = _USER_ID_SANITIZE_PATTERN.sub("_", trimmed)
    return sanitized or default


@dataclass(frozen=True)
class UserResolverResult:
    """정규화된 사용자 ID와 해당 디렉터리를 함께 전달하는 컨테이너."""

    user_id: str
    root: Path


class UserScopedStorage:
    """data/<user_id> 구조를 추상화해 경로 생성/조회 로직을 한 곳에서 관리한다."""

    def __init__(self, base_root: Path, *, default_user_id: str = DEFAULT_USER_ID):
        self.base_root = base_root
        self.default_user_id = default_user_id
        self.base_root.mkdir(parents=True, exist_ok=True)

    def resolve(self, raw_user_id: Optional[str], *, create: bool = True) -> UserResolverResult:
        """정규화된 사용자 ID와 해당 루트 경로를 돌려준다."""
        user_id = sanitize_user_id(raw_user_id, default=self.default_user_id)
        user_root = self.base_root / user_id
        if create:
            user_root.mkdir(parents=True, exist_ok=True)
        return UserResolverResult(user_id=user_id, root=user_root)
