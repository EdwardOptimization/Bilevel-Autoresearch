"""Retrieve relevant lessons from memory by topic/stage/type."""
import json
from pathlib import Path

from .lesson_schema import Lesson


class LessonRetriever:
    def retrieve(
        self,
        lessons: list[Lesson],
        topic_tags: list[str] = None,
        stage: str = None,
        lesson_types: list[str] = None,
        min_confidence: float = 0.0,
        max_results: int = 5,
    ) -> list[Lesson]:
        """
        Filter and rank lessons by relevance.
        Scoring:
          +1 per matching topic_tag
          +2 if stage matches exactly
          lessons below min_confidence are excluded
        """
        topic_tags_set = set(t.lower() for t in (topic_tags or []))

        scored = []
        for lesson in lessons:
            if lesson.confidence < min_confidence:
                continue
            if lesson_types and lesson.lesson_type not in lesson_types:
                continue

            score = 0
            lesson_tags_set = set(t.lower() for t in lesson.topic_tags)
            score += len(topic_tags_set & lesson_tags_set)

            if stage and lesson.stage == stage:
                score += 2

            scored.append((score, lesson))

        # Sort by score descending, then by confidence descending as tiebreaker
        scored.sort(key=lambda x: (x[0], x[1].confidence), reverse=True)
        return [lesson for _, lesson in scored[:max_results]]

    def save_debug(self, lessons: list[Lesson], query: dict, output_path: Path) -> None:
        """Save retrieval debug info for inspection."""
        debug = {
            "query": query,
            "retrieved_count": len(lessons),
            "lessons": [l.to_dict() for l in lessons],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8"
        )
