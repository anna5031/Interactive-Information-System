from __future__ import annotations

import sys

try:
    from .app import main, run_cli, run_cli_async
except ImportError:  # pragma: no cover - 직접 실행 대응
    from app import main, run_cli, run_cli_async  # type: ignore

__all__ = ["main", "run_cli", "run_cli_async"]


if __name__ == "__main__":
    sys.exit(main())
