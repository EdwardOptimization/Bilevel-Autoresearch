"""Unit tests for the memory store."""
import json
import tempfile
from pathlib import Path

from src.evolution.lesson_schema import Lesson
from src.evolution.memory_store import MemoryStore


def _make_lesson(lesson_id="l1", confidence=0.85, stage="hypothesis_generation"):
    return Lesson(
        id=lesson_id,
        created_from_run="run-test-001",
        lesson_type="failure_pattern",
        scope="stage",
        stage=stage,
        topic_tags=["test", "pipeline"],
        summary=f"Lesson {lesson_id} summary",
        evidence=[],
        confidence=confidence,
        reuse_rule="Always do X",
        anti_pattern="Never do Y",
    )


class TestMemoryStore:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore(Path(self.tmpdir))

    def test_save_and_load_lessons(self):
        lessons = [_make_lesson("l1"), _make_lesson("l2")]
        self.store.save_lessons(lessons)
        loaded = self.store.load_all_lessons()
        assert len(loaded) == 2
        ids = {l.id for l in loaded}
        assert "l1" in ids
        assert "l2" in ids

    def test_save_appends_to_existing(self):
        self.store.save_lessons([_make_lesson("l1")])
        self.store.save_lessons([_make_lesson("l2")])
        loaded = self.store.load_all_lessons()
        assert len(loaded) == 2

    def test_empty_store_returns_empty_list(self):
        loaded = self.store.load_all_lessons()
        assert loaded == []

    def test_lesson_fields_preserved(self):
        lesson = _make_lesson("l_test", confidence=0.92, stage="draft_writeup")
        self.store.save_lessons([lesson])
        loaded = self.store.load_all_lessons()
        assert len(loaded) == 1
        l = loaded[0]
        assert l.id == "l_test"
        assert l.confidence == 0.92
        assert l.stage == "draft_writeup"
        assert l.summary == "Lesson l_test summary"

    def test_jsonl_format_one_per_line(self):
        lessons = [_make_lesson("la"), _make_lesson("lb")]
        self.store.save_lessons(lessons)
        lessons_file = Path(self.tmpdir) / "lessons.jsonl"
        lines = [l for l in lessons_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "id" in obj
            assert "confidence" in obj

    def test_corrupted_line_skipped(self):
        lessons_file = Path(self.tmpdir) / "lessons.jsonl"
        # Write one valid line and one corrupted line
        valid = _make_lesson("valid")
        lessons_file.write_text(
            json.dumps(valid.to_dict()) + "\n{corrupted_json\n"
        )
        # Should not raise, just skip the bad line
        loaded = self.store.load_all_lessons()
        assert len(loaded) == 1
        assert loaded[0].id == "valid"
