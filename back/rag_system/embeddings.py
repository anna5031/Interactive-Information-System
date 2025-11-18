from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import onnxruntime as ort
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer

from .config import BASE_DIR

ONNX_MODEL_DIR = BASE_DIR / "onnx_model"
DEFAULT_BATCH = 16
DEFAULT_MAX_LENGTH = 512


class ONNXEmbeddingModel(Embeddings):
    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir or ONNX_MODEL_DIR)
        self.model_path = self.model_dir / "model.onnx"
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX 모델을 찾을 수 없습니다: {self.model_path}. 먼저 convert_to_onnx.py를 실행하세요."
            )
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"], sess_options=session_options)
        self.output_name = self.session.get_outputs()[0].name
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0].tolist()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for batch in self._batch(texts):
            tokenized = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=DEFAULT_MAX_LENGTH,
                return_tensors="np",
            )
            ort_inputs = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
            outputs = self.session.run([self.output_name], ort_inputs)[0]
            pooled = self._mean_pool(outputs, tokenized["attention_mask"])
            normalized = self._l2_normalize(pooled)
            vectors.append(normalized.astype(np.float32))
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(vectors)

    def _batch(self, texts: Sequence[str]) -> Iterator[Sequence[str]]:
        for start in range(0, len(texts), DEFAULT_BATCH):
            yield texts[start : start + DEFAULT_BATCH]

    @staticmethod
    def _mean_pool(embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = np.expand_dims(attention_mask.astype(np.float32), axis=-1)
        summed = np.sum(embeddings * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return vectors / norms


@lru_cache(maxsize=1)
def get_embedding_model() -> ONNXEmbeddingModel:
    return ONNXEmbeddingModel()
