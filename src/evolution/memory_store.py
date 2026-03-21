"""JSONL-based persistent memory store for lessons."""
import json
from pathlib import Path

from .lesson_schema import Lesson


class MemoryStore:
    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.lessons_file = self.memory_dir / "lessons.jsonl"

    def save_lessons(self, lessons: list[Lesson]) -> None:
        """Append lessons to the JSONL store."""
        with open(self.lessons_file, "a", encoding="utf-8") as f:
            for lesson in lessons:
                f.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")

    def load_all_lessons(self) -> list[Lesson]:
        """Load all lessons from JSONL store."""
        if not self.lessons_file.exists():
            return []
        lessons = []
        with open(self.lessons_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        lessons.append(Lesson.from_dict(d))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return lessons

    def count(self) -> int:
        return len(self.load_all_lessons())
