"""
테스트 공용 헬퍼: src, config 경로 추가 및 .env 로드
"""

import os
import sys
from pathlib import Path


def ensure_paths():
    repo_root = Path(__file__).resolve().parent.parent

    # 필요 시 SDL 오디오 드라이버를 더미로 설정할 때 사용
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    root_path = str(repo_root)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    config_path = str(repo_root / "config")
    if config_path not in sys.path:
        sys.path.insert(0, config_path)

    # .env 로드 (선택적)
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=repo_root / ".env")
    except Exception:
        pass
