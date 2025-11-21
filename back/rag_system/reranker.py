from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .config import RERANKER_CONFIG, RerankerConfig
from .vector_index import ScoredDocument


class KoReranker:
    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RERANKER_CONFIG
        self.model_dir = Path(self.config.model_dir)
        self.model_path = self.model_dir / "model.onnx"
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Ko-Reranker ONNX 모델이 없습니다: {self.model_path}. "
                "먼저 `python back/rag_system/convert_to_onnx.py` 를 실행해 주세요."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"], sess_options=session_options
        )

    def rerank(self, query: str, hits: Sequence[ScoredDocument]) -> List[ScoredDocument]:
        if not hits:
            return []
        limited = list(hits[: self.config.max_candidates])
        texts = [hit.document.page_content for hit in limited]
        scores = self._score(query, texts)
        if not scores:
            return list(limited)
        if self.config.normalize_scores:
            scores = self._softmax(scores)
        scored = [
            ScoredDocument(document=hit.document, score=float(score))
            for hit, score in zip(limited, scores)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    def _score(self, query: str, docs: Sequence[str]) -> List[float]:
        outputs: List[float] = []
        for batch_docs in self._chunk(docs, self.config.batch_size):
            batch_queries = [query] * len(batch_docs)
            tokenized = self.tokenizer(
                batch_queries,
                list(batch_docs),
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="np",
            )
            inputs = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
            if "token_type_ids" in tokenized:
                inputs["token_type_ids"] = tokenized["token_type_ids"]
            logits = self.session.run(None, inputs)[0]
            arr = np.array(logits)
            if arr.ndim == 1:
                batch_scores = arr.tolist()
            elif arr.shape[1] == 1:
                batch_scores = arr[:, 0].tolist()
            else:
                batch_scores = arr[:, 0].tolist()
            outputs.extend(batch_scores)
        return outputs

    @staticmethod
    def _softmax(scores: Sequence[float]) -> List[float]:
        arr = np.array(scores, dtype=np.float32)
        max_score = float(np.max(arr))
        exps = np.exp(arr - max_score)
        denom = float(np.sum(exps)) or 1.0
        return (exps / denom).tolist()

    @staticmethod
    def _chunk(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
        if size <= 0:
            size = len(items)
        for idx in range(0, len(items), size):
            yield items[idx : idx + size]
