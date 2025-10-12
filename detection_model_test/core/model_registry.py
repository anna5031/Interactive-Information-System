"""Discovery utilities for available model checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from config import (
    BASE_DIR,
    RFDETR_EXTENSIONS,
    RFDETR_MODEL_DIR,
    YOLO_EXTENSIONS,
    YOLO_MODEL_DIR,
)

BACKEND_YOLO = "yolo"
BACKEND_RFDETR = "rf-detr"


@dataclass(frozen=True)
class ModelInfo:
    """Metadata describing an inference-ready model checkpoint."""

    id: str
    name: str
    path: Path
    backend: str
    group: str
    family: str | None = None

    @property
    def cache_key(self) -> str:
        """Key suitable for caching resources keyed by this model."""
        return self.id


def _iter_files(root: Path, extensions: Sequence[str]) -> Iterable[Path]:
    for extension in extensions:
        yield from root.rglob(f"*{extension}")


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _extract_family(relative_path: Path) -> str | None:
    parts = relative_path.parts
    if len(parts) > 1:
        return parts[0]
    return None


def _scan_models(root: Path, extensions: Sequence[str], backend: str, group: str) -> List[ModelInfo]:
    if not root.exists():
        return []

    models: list[ModelInfo] = []
    for path in _iter_files(root, extensions):
        if not path.is_file():
            continue
        relative_path = _safe_relative(path, root)
        family = _extract_family(relative_path)
        identifier = f"{backend}:{path.relative_to(BASE_DIR).as_posix()}"
        models.append(
            ModelInfo(
                id=identifier,
                name=relative_path.as_posix(),
                path=path.resolve(),
                backend=backend,
                group=group,
                family=family,
            )
        )
    models.sort(key=lambda item: item.name.lower())
    return models


def list_available_models() -> list[ModelInfo]:
    """Return all discoverable models in a deterministic order."""
    models = []
    models.extend(_scan_models(YOLO_MODEL_DIR, YOLO_EXTENSIONS, BACKEND_YOLO, "YOLO"))
    models.extend(_scan_models(RFDETR_MODEL_DIR, RFDETR_EXTENSIONS, BACKEND_RFDETR, "RF-DETR"))
    return models


def get_model(model_id: str, models: Sequence[ModelInfo] | None = None) -> ModelInfo | None:
    """Retrieve model metadata by ID."""
    records = models if models is not None else list_available_models()
    for model in records:
        if model.id == model_id:
            return model
    return None
