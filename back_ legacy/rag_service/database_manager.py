from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import logging

import chromadb
from chromadb.api.client import ClientAPI
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document

from .data_loader import DataBundle, KnowledgeEntry, build_entry_from_raw
from .embedding_utils import get_embeddings

_RAW_KEY = "_raw_entry"
_ENTRY_TYPE_KEY = "entry_type"

logger = logging.getLogger(__name__)


class DatabaseManager:
    """ChromaDB를 사용하여 정보 CRUD와 초기 적재를 담당."""

    def __init__(
        self,
        *,
        persist_directory: Optional[Path] = None,
        collection_name: str = "mech_eng_information",
    ):
        self.persist_directory = Path(persist_directory or (Path(__file__).parent / "chroma_db"))
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client: ClientAPI = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection: Collection = self._client.get_or_create_collection(name=collection_name)
        self._embeddings_model = None

    # ------------------------------------------------------------------
    # 초기화 & 적재
    # ------------------------------------------------------------------
    def load_bundle(
        self,
        bundle: DataBundle,
        *,
        rebuild: bool = False,
        skip_if_exists: bool = True,
    ) -> bool:
        """JSON 데이터를 기반으로 컬렉션을 초기화하거나 업데이트."""
        if rebuild:
            self._collection.delete(where={})
        elif skip_if_exists:
            try:
                existing = self.count()
            except Exception:
                existing = 0
            if existing > 0:
                logger.info("Chroma 컬렉션이 이미 %d개 항목으로 채워져 있어 초기 적재를 건너뜁니다.", existing)
                return False

        self.upsert_entries(bundle.all_entries())
        return True

    def upsert_entries(self, entries: Iterable[KnowledgeEntry]) -> None:
        entries = list(entries)
        if not entries:
            return

        documents = [entry.content for entry in entries]
        metadatas = [self._build_metadata(entry) for entry in entries]
        ids = [entry.id for entry in entries]
        embeddings = self._embed(documents)

        duplicates = [doc_id for doc_id, count in Counter(ids).items() if count > 1]
        if duplicates:
            duplicate_str = ", ".join(sorted(duplicates))
            raise ValueError(f"중복된 ID가 감지되었습니다: {duplicate_str}")

        self._collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def list_entries(self, entry_type: Optional[str] = None) -> List[Dict[str, Any]]:
        where_clause = {_ENTRY_TYPE_KEY: entry_type} if entry_type else {}
        result = self._collection.get(where=where_clause or None, include=["metadatas", "documents"])
        return self._format_collection_result(result)

    def count_by_type(self, entry_type: str) -> int:
        if not entry_type:
            return 0
        result = self._collection.get(where={_ENTRY_TYPE_KEY: entry_type}, include=[])
        return len(result.get("ids", []))

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        result = self._collection.get(ids=[entry_id], include=["metadatas", "documents"])
        formatted = self._format_collection_result(result)
        return formatted[0] if formatted else None

    def create_entry(self, payload: Dict[str, Any]) -> KnowledgeEntry:
        entry = build_entry_from_raw(payload, fallback_index=self._next_index())
        self.upsert_entries([entry])
        return entry

    def update_entry(self, entry_id: str, payload: Dict[str, Any]) -> KnowledgeEntry:
        stored = self.get_entry(entry_id)
        if not stored:
            raise ValueError(f"ID가 '{entry_id}'인 항목을 찾을 수 없습니다.")

        raw_entry = json.loads(stored["metadata"].get(_RAW_KEY, "{}"))
        raw_entry.update(payload)

        entry = build_entry_from_raw(raw_entry, fallback_index=self._next_index())
        entry.id = entry_id  # ID는 유지

        self.upsert_entries([entry])
        return entry

    def delete_entry(self, entry_id: str) -> bool:
        before = self._collection.count()
        self._collection.delete(ids=[entry_id])
        after = self._collection.count()
        return after < before

    # ------------------------------------------------------------------
    # 검색 & 변환
    # ------------------------------------------------------------------
    def get_all_documents(self) -> List[Document]:
        result = self._collection.get(include=["metadatas", "documents"])
        docs: List[Document] = []
        ids = result.get("ids", [])
        for index, (doc, metadata) in enumerate(
            zip(result.get("documents", []), result.get("metadatas", []))
        ):
            doc_metadata = dict(metadata or {})
            if index < len(ids):
                doc_metadata["entry_id"] = ids[index]
            docs.append(Document(page_content=doc or "", metadata=doc_metadata))
        return docs

    def list_documents_by_type(self, entry_type: str) -> List[Document]:
        if not entry_type:
            return []
        all_docs = self.get_all_documents()
        return [doc for doc in all_docs if doc.metadata.get(_ENTRY_TYPE_KEY) == entry_type]

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # 내부 도우미
    # ------------------------------------------------------------------
    def _format_collection_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []

        for idx, doc_id in enumerate(result.get("ids", [])):
            metadata = (result.get("metadatas") or [])[idx] or {}
            document = (result.get("documents") or [None])[idx]

            formatted.append(
                {
                    "id": doc_id,
                    "metadata": metadata,
                    "document": document,
                }
            )

        return formatted

    def _build_metadata(self, entry: KnowledgeEntry) -> Dict[str, Any]:
        metadata = dict(entry.metadata)
        metadata[_ENTRY_TYPE_KEY] = entry.entry_type
        metadata[_RAW_KEY] = json.dumps(entry.raw, ensure_ascii=False)
        return metadata

    def ensure_embeddings_ready(self) -> None:
        if self._embeddings_model is None:
            self._embeddings_model = get_embeddings()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._embeddings_model is None:
            self._embeddings_model = get_embeddings()
        return self._embeddings_model.embed_documents(texts)

    def _next_index(self) -> int:
        try:
            return self.count() + 1
        except Exception:
            return 0
