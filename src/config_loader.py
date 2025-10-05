# config_loader.py
from pathlib import Path
import yaml

def load_config(filename: str = "config.yaml"):
    # 프로젝트 루트/현재 파일 기준 안전한 경로 해석
    here = Path(__file__).resolve().parent
    candidates = [
        here / filename,                  # src/config.yaml (예)
        here.parent / filename,           # 프로젝트 루트/config.yaml
        Path.cwd() / filename,            # 현재 작업 디렉토리/config.yaml
    ]
    cfg_path = next((p for p in candidates if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"Config not found. Tried: {', '.join(map(str, candidates))}")

    # 1) 우선 UTF-8로 시도, 2) 안 되면 cp949 등으로 폴백
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except UnicodeDecodeError:
        # 한국어 윈도우에서 작성된 파일이라면 cp949/ansi일 수 있음
        with cfg_path.open("r", encoding="cp949", errors="strict") as f:
            return yaml.safe_load(f)
