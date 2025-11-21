from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DOCS_DIR = DATA_DIR / "source_documents"
INDOOR_MAP_PATH = DATA_DIR / "indoor_map.json"
INDEX_DIR = BASE_DIR / "storage"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
RERANKER_DIR = BASE_DIR / "onnx_reranker"
RERANKER_DIR.mkdir(parents=True, exist_ok=True)

# 환경 변수 로딩 (.env는 back/.env 경로에 존재)
ENV_PATH = BASE_DIR.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# 토크나이저 병렬 경고 및 불필요한 트레이싱 비활성화
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


@dataclass(slots=True)
class IndexConfig:
    chunk_size: int = 600
    chunk_overlap: int = 120
    information_threshold: float = 0.7
    information_min_k: int = 3
    recommendation_threshold: float = 0.6
    recommendation_min_k: int = 5
    recommendation_vector_k: int = 100
    recommendation_bm25_k: int = 100
    recommendation_rrf_k: int = 60


INDEX_CONFIG = IndexConfig()


@dataclass(slots=True)
class RerankerConfig:
    model_name: str = "Dongjin-kr/ko-reranker"
    model_dir: Path = RERANKER_DIR
    max_length: int = 512
    batch_size: int = 16
    max_candidates: int = 50
    normalize_scores: bool = False


RERANKER_CONFIG = RerankerConfig()
