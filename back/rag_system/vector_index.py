from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from langchain_core.documents import Document

from .embeddings import get_embedding_model
from .text_ingest import TextChunk


@dataclass(slots=True)
class IndexedChunk:
    content: str
    metadata: dict


@dataclass(slots=True)
class ScoredDocument:
    document: Document
    score: float


class LocalVectorIndex:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_path = self.storage_dir / "embeddings.npy"
        self.metadata_path = self.storage_dir / "metadata.json"
        self._embeddings: np.ndarray | None = None
        self._documents: List[Document] = []

    def build(self, chunks: Sequence[TextChunk]) -> None:
        embedder = get_embedding_model()
        texts = [chunk.content for chunk in chunks]
        vectors = embedder.embed_documents(texts)
        self._embeddings = np.array(vectors, dtype=np.float32)
        self._documents = [chunk.to_document() for chunk in chunks]
        np.save(self.embedding_path, self._embeddings)
        payload = [doc.metadata | {"content": doc.page_content} for doc in self._documents]
        self.metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.embedding_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                "벡터 인덱스가 없습니다. `python -m back.rag_system.cli --build-index` 로 먼저 생성하세요."
            )
        self._embeddings = np.load(self.embedding_path)
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._documents = [Document(page_content=item.pop("content", ""), metadata=item) for item in payload]

    @property
    def documents(self) -> List[Document]:
        if not self._documents:
            self.load()
        return self._documents

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self.load()
        if self._embeddings is None:
            raise RuntimeError("임베딩 데이터를 로드하지 못했습니다.")
        return self._embeddings

    def search(self, query: str, top_k: int = 10) -> List[ScoredDocument]:
        if not self.documents:
            return []
        embedder = get_embedding_model()
        query_vec = np.array(embedder.embed_query(query), dtype=np.float32)
        scores = self.embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [ScoredDocument(self.documents[idx], float(scores[idx])) for idx in top_indices]
