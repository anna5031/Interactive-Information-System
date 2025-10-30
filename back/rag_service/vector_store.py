from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from langchain_core.documents import Document

from .data_loader import DataLoader, KnowledgeEntry
from .database_manager import DatabaseManager
from .embedding_utils import build_retriever


class MechanicalEngineeringRAG:
    """기계과 RAG 파이프라인의 검색 계층을 담당."""

    def __init__(
        self,
        *,
        data_path: Optional[Path] = None,
        db_manager: Optional[DatabaseManager] = None,
        auto_initialize: bool = True,
    ):
        self.db_manager = db_manager or DatabaseManager()
        self._documents: List[Document] = []
        self._retriever = None
        self._data_path = Path(data_path) if data_path else None
        self._auto_initialize = auto_initialize

        if auto_initialize:
            if self._data_path:
                self.initialize_from_json(self._data_path, rebuild=False)
            else:
                self.refresh_documents()

    # ------------------------------------------------------------------
    # 초기화 및 새로고침
    # ------------------------------------------------------------------
    def initialize_from_json(self, data_path: Path, *, rebuild: bool = False) -> None:
        loader = DataLoader(data_path)
        bundle = loader.load()
        try:
            self.db_manager.load_bundle(bundle, rebuild=rebuild, skip_if_exists=not rebuild)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"임베딩 모델을 찾을 수 없습니다. convert_to_onnx.py를 실행해 ONNX 모델을 생성하세요: {exc}"
            ) from exc
        self.refresh_documents()

    def refresh_documents(self) -> None:
        self._documents = self.db_manager.get_all_documents()
        self._retriever = build_retriever(self._documents)

    # ------------------------------------------------------------------
    # 검색 유틸리티
    # ------------------------------------------------------------------
    def search(self, query: str, *, top_k: int = 6) -> List[Document]:
        if not self._retriever:
            self.refresh_documents()
        retrieved_docs = self._retriever.invoke(query)
        return list(retrieved_docs)[:top_k]

    def search_professors(self, query: str, k: int = 4) -> List[Document]:
        return self._filter_by_type(self.search(query, top_k=k * 2), "professor", limit=k)

    def search_classes(self, query: str, k: int = 4) -> List[Document]:
        return self._filter_by_type(
            self.search(query, top_k=k * 2),
            target_types={"lecture", "class"},
            limit=k,
        )

    def search_seminar_rooms(self, query: str, k: int = 3) -> List[Document]:
        return self._filter_by_type(self.search(query, top_k=k * 2), "seminar_room", limit=k)

    def search_all_seminar_rooms(self, query: str) -> List[Document]:
        """세미나실 추천 의도에서 사용할 전체 세미나실 후보를 가져온다."""
        seminar_total = self.db_manager.count_by_type("seminar_room")
        if seminar_total <= 0:
            return []

        primary_candidates = self.search(query, top_k=seminar_total)
        filtered_primary = self._filter_by_type(primary_candidates, "seminar_room", limit=seminar_total)
        if len(filtered_primary) >= seminar_total:
            return filtered_primary

        needed = seminar_total - len(filtered_primary)
        seen_ids: Set[str] = {
            str(doc.metadata.get("entry_id"))
            for doc in filtered_primary
            if doc.metadata.get("entry_id") is not None
        }

        fallback_docs = self.db_manager.list_documents_by_type("seminar_room")
        additional: List[Document] = []
        for doc in fallback_docs:
            entry_id = doc.metadata.get("entry_id")
            if entry_id is not None and str(entry_id) in seen_ids:
                continue
            additional.append(doc)
            if len(additional) >= needed:
                break

        return filtered_primary + additional

    def _filter_by_type(
        self,
        documents: List[Document],
        target_types: Any,
        *,
        limit: int,
    ) -> List[Document]:
        if isinstance(target_types, str):
            target_types = {target_types}
        filtered = [
            doc
            for doc in documents
            if doc.metadata.get("entry_type") in target_types
            or doc.metadata.get("type") in target_types
        ]
        return filtered[:limit]

    # ------------------------------------------------------------------
    # CRUD helper
    # ------------------------------------------------------------------
    def add_entry(self, payload: Dict[str, Any]) -> KnowledgeEntry:
        entry = self.db_manager.create_entry(payload)
        self.refresh_documents()
        return entry

    def update_entry(self, entry_id: str, payload: Dict[str, Any]) -> KnowledgeEntry:
        entry = self.db_manager.update_entry(entry_id, payload)
        self.refresh_documents()
        return entry

    def delete_entry(self, entry_id: str) -> bool:
        success = self.db_manager.delete_entry(entry_id)
        if success:
            self.refresh_documents()
        return success

    def list_entries(self, entry_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.db_manager.list_entries(entry_type)
