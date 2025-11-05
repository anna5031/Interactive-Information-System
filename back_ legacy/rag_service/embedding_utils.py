from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import onnxruntime as ort
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer

COLLECTION_NAME = "mech_eng_information"
ONNX_MODEL_DIR = Path(__file__).parent / "onnx_model"
VECTORSTORE_DIR = Path(__file__).parent / "chroma_db"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 512


class ONNXMeanPoolingEmbeddings(Embeddings):
    """ONNXRuntime 기반 임베딩 래퍼."""

    def __init__(
        self,
        model_dir: Path,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        normalize: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize

        self._model_path = self.model_dir / "model.onnx"
        if not self.model_dir.exists() or not self._model_path.exists():
            raise FileNotFoundError(
                "ONNX 모델을 찾을 수 없습니다. convert_to_onnx.py을 실행하여 모델을 생성하세요."
            )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        self._output_name = self._session.get_outputs()[0].name
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

    # ------------------------------------------------------------------
    # LangChain Embeddings API
    # ------------------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0].tolist()

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for batch in self._batch_iter(texts):
            encoded = self._tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            ort_inputs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
            outputs = self._session.run([self._output_name], ort_inputs)[0]
            pooled = self._mean_pool(outputs, encoded["attention_mask"])
            if self.normalize:
                pooled = self._l2_normalize(pooled)
            chunks.append(pooled.astype(np.float32))

        if not chunks:
            return np.empty((0, 0), dtype=np.float32)

        return np.vstack(chunks)

    def _batch_iter(self, texts: Sequence[str]) -> Iterator[Sequence[str]]:
        for start in range(0, len(texts), self.batch_size):
            yield texts[start : start + self.batch_size]

    @staticmethod
    def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask_expanded = np.expand_dims(attention_mask.astype(np.float32), axis=-1)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return embeddings / norms


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return ONNXMeanPoolingEmbeddings(ONNX_MODEL_DIR)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=get_embeddings(),
    )


def build_retriever(
    documents: Sequence[Document],
    *,
    bm25_k: int = 6,
    vector_k: int = 6,
    use_bm25: bool = True,
) -> object:
    """문서 리스트를 받아 앙상블 또는 단일 벡터 리트리버를 반환."""
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": vector_k})

    if use_bm25 and documents:
        bm25_retriever = BM25Retriever.from_documents(list(documents))
        bm25_retriever.k = bm25_k
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],
        )
        return ensemble

    return vector_retriever
