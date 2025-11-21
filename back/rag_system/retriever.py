from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .config import INDEX_CONFIG
from .reranker import KoReranker
from .vector_index import LocalVectorIndex, ScoredDocument


@dataclass(slots=True)
class RetrievalResult:
    documents: List[Document]
    scores: List[float]
    max_score: float


class InformationRetriever:
    def __init__(self, vector_index: LocalVectorIndex) -> None:
        self.vector_index = vector_index

    def retrieve(self, query: str) -> RetrievalResult:
        hits = self.vector_index.search(query, top_k=None)
        if not hits:
            return RetrievalResult([], [], 0.0)
        top_score = hits[0].score
        threshold = top_score * INDEX_CONFIG.information_threshold if top_score > 0 else top_score
        selected = [hit for hit in hits if hit.score >= threshold]
        min_k = max(INDEX_CONFIG.information_min_k, 1)
        if len(selected) < min_k:
            selected = hits[:min_k]
        documents = [item.document for item in selected]
        scores = [item.score for item in selected]
        return RetrievalResult(documents, scores, float(top_score))


class RecommendationRetriever:
    def __init__(self, vector_index: LocalVectorIndex, reranker: KoReranker | None = None) -> None:
        self.vector_index = vector_index
        self._bm25: BM25Retriever | None = None
        self._rrf_k = INDEX_CONFIG.recommendation_rrf_k
        self.reranker = reranker or KoReranker()

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            retriever = BM25Retriever.from_documents(self.vector_index.documents)
            retriever.k = INDEX_CONFIG.recommendation_bm25_k
            self._bm25 = retriever
        return self._bm25

    def retrieve(self, query: str) -> RetrievalResult:
        vector_hits = self.vector_index.search(query, top_k=INDEX_CONFIG.recommendation_vector_k)
        bm25_hits = self._bm25_hits(query)
        fused = self._rrf_fuse(vector_hits, bm25_hits)
        if not fused:
            return RetrievalResult([], [], 0.0)
        reranked = self._rerank(query, fused)
        selected = self._apply_threshold(reranked)
        documents = [item.document for item in selected]
        scores = [item.score for item in selected]
        max_score = reranked[0].score if reranked else 0.0
        return RetrievalResult(documents, scores, float(max_score))

    def _bm25_hits(self, query: str) -> List[ScoredDocument]:
        docs = self.bm25.invoke(query)
        hits: List[ScoredDocument] = []
        for idx, doc in enumerate(docs, start=1):
            score = doc.metadata.get("score")
            numeric = float(score) if isinstance(score, (int, float)) else max(1.0 - (idx - 1) * 0.05, 0.1)
            hits.append(ScoredDocument(document=doc, score=numeric))
        return hits

    def _rrf_fuse(
        self,
        vector_hits: List[ScoredDocument],
        bm25_hits: List[ScoredDocument],
    ) -> List[ScoredDocument]:
        rankings: Dict[str, Dict[str, float | Document]] = {}
        for hits in (vector_hits, bm25_hits):
            for rank, hit in enumerate(hits, start=1):
                doc_id = f"{hit.document.metadata.get('doc_id', hit.document.page_content[:10])}-{hit.document.metadata.get('chunk_id', 0)}"
                payload = rankings.setdefault(doc_id, {"document": hit.document, "score": 0.0})
                payload["score"] = float(payload["score"]) + 1.0 / (self._rrf_k + rank)
        scored = [ScoredDocument(document=data["document"], score=float(data["score"])) for data in rankings.values()]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    def _apply_threshold(self, hits: List[ScoredDocument]) -> List[ScoredDocument]:
        if not hits:
            return []
        max_score = hits[0].score
        threshold = max_score * INDEX_CONFIG.recommendation_threshold if max_score > 0 else max_score
        filtered = [hit for hit in hits if hit.score >= threshold]
        min_k = max(INDEX_CONFIG.recommendation_min_k, 1)
        if len(filtered) < min_k:
            filtered = hits[:min_k]
        return filtered

    def _rerank(self, query: str, hits: List[ScoredDocument]) -> List[ScoredDocument]:
        try:
            reranked = self.reranker.rerank(query, hits)
            return reranked or hits
        except Exception:
            return hits
