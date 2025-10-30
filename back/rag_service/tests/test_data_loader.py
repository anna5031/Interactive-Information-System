import pytest

from ..data_loader import KnowledgeEntry, build_entry_from_raw
from ..database_manager import DatabaseManager


def test_build_entry_id_uses_slug_for_ascii():
    entry = build_entry_from_raw({"type": "professor", "name": "John Doe"}, fallback_index=5)
    assert entry.id == "professor-john-doe"


def test_build_entry_id_adds_index_for_non_ascii():
    entry = build_entry_from_raw({"type": "lecture", "name": "기계공학"}, fallback_index=2)
    assert entry.id.startswith("lecture-")
    assert entry.id != "lecture"


def test_build_entry_id_unique_for_duplicate_non_ascii_names():
    entry1 = build_entry_from_raw({"type": "professor", "name": "김교수"}, fallback_index=1)
    entry2 = build_entry_from_raw({"type": "professor", "name": "김교수"}, fallback_index=2)
    assert entry1.id != entry2.id


class DummyCollection:
    def upsert(self, **_):
        raise AssertionError("upsert should not be called when duplicates are present")


def test_upsert_entries_rejects_duplicate_ids():
    manager = DatabaseManager.__new__(DatabaseManager)
    manager._collection = DummyCollection()
    manager._embeddings_model = None
    manager._embed = lambda texts: [[0.0] for _ in texts]

    entry = KnowledgeEntry(
        id="professor",
        entry_type="professor",
        metadata={},
        content="Professor Kim",
        raw={"type": "professor", "name": "김교수"},
    )

    with pytest.raises(ValueError) as exc:
        manager.upsert_entries([entry, entry])

    assert "중복된 ID" in str(exc.value)
