from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .config import INDEX_CONFIG
from .vector_index import LocalVectorIndex, ScoredDocument


@dataclass(slots=True)
class RetrievalResult:
    documents: List[Document]
    scores: List[float]


class EnsembleRetriever:
    def __init__(self, vector_index: LocalVectorIndex) -> None:
        self.vector_index = vector_index
        self._bm25: BM25Retriever | None = None

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever.from_documents(self.vector_index.documents)
            self._bm25.k = INDEX_CONFIG.bm25_k
        return self._bm25

    def retrieve(self, query: str) -> RetrievalResult:
        vector_hits = self.vector_index.search(query, top_k=INDEX_CONFIG.vector_k)
        bm25_hits = self._bm25_hits(query)
        combined = self._combine(vector_hits, bm25_hits)
        selected = self._apply_relative_threshold(combined)
        docs = [item.document for item in selected]
        scores = [item.score for item in selected]
        return RetrievalResult(docs, scores)

    def _bm25_hits(self, query: str) -> List[ScoredDocument]:
        docs = self.bm25.invoke(query)
        hits: List[ScoredDocument] = []
        for idx, doc in enumerate(docs):
            score = doc.metadata.get("score")
            numeric = float(score) if isinstance(score, (int, float)) else max(1.0 - idx * 0.05, 0.1)
            hits.append(ScoredDocument(document=doc, score=numeric))
        return hits

    def _combine(
        self,
        vector_hits: List[ScoredDocument],
        bm25_hits: List[ScoredDocument],
    ) -> List[ScoredDocument]:
        all_docs: dict[str, dict[str, float | Document]] = {}
        for weight, hits in ((0.6, vector_hits), (0.4, bm25_hits)):
            if not hits:
                continue
            scores = [hit.score for hit in hits]
            if not scores:
                continue
            min_score = min(scores)
            max_score = max(scores)
            denom = max(max_score - min_score, 1e-9)
            for hit in hits:
                doc_id = f"{hit.document.metadata.get('doc_id', hit.document.page_content[:10])}-{hit.document.metadata.get('chunk_id',0)}"
                scaled = (hit.score - min_score) / denom
                payload = all_docs.setdefault(doc_id, {"document": hit.document, "score": 0.0})
                payload["score"] = float(payload["score"]) + weight * float(scaled)
        scored = [ScoredDocument(document=data["document"], score=float(data["score"])) for data in all_docs.values()]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    def _apply_relative_threshold(self, hits: List[ScoredDocument]) -> List[ScoredDocument]:
        if not hits:
            return []
        max_score = hits[0].score
        if max_score <= 0:
            return hits[: INDEX_CONFIG.min_k]
        threshold = max_score * INDEX_CONFIG.relative_threshold
        filtered: List[ScoredDocument] = [hit for hit in hits if hit.score >= threshold]
        if len(filtered) < INDEX_CONFIG.min_k:
            filtered = hits[: INDEX_CONFIG.min_k]
        return filtered
