from __future__ import annotations

import sys
from pathlib import Path

_BACK_DIR = Path(__file__).resolve().parent
back_dir_str = str(_BACK_DIR)
if back_dir_str not in sys.path:
    sys.path.insert(0, back_dir_str)
