from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document


@dataclass(slots=True)
class TextChunk:
    doc_id: str
    chunk_id: int
    content: str
    source_path: str
    extra_info: dict

    def to_document(self) -> Document:
        metadata = {"doc_id": self.doc_id, "chunk_id": self.chunk_id, **self.extra_info, "source": self.source_path}
        return Document(page_content=self.content, metadata=metadata)


class TextCorpusBuilder:
    def __init__(self, source_dir: Path, *, chunk_size: int = 600, chunk_overlap: int = 120) -> None:
        self.source_dir = Path(source_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def discover_files(self) -> List[Path]:
        return sorted([p for p in self.source_dir.glob("**/*.txt") if p.is_file()])

    def load(self) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        for path in self.discover_files():
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            doc_id = path.relative_to(self.source_dir).as_posix()
            for idx, chunk in enumerate(self._split_text(text)):
                chunks.append(
                    TextChunk(
                        doc_id=doc_id,
                        chunk_id=idx,
                        content=chunk,
                        source_path=str(path),
                        extra_info={"relative_path": doc_id},
                    )
                )
        return chunks

    def export_metadata(self, chunks: Iterable[TextChunk], output_path: Path) -> None:
        payload = [
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "source_path": chunk.source_path,
                "extra": chunk.extra_info,
            }
            for chunk in chunks
        ]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _split_text(self, text: str) -> Iterable[str]:
        start = 0
        length = len(text)
        size = max(self.chunk_size, 1)
        overlap = max(min(self.chunk_overlap, size - 1), 0)
        while start < length:
            end = min(length, start + size)
            yield text[start:end]
            if end == length:
                break
            start = end - overlap
