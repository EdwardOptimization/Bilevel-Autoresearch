"""Unit tests for lesson retrieval."""
from src.evolution.lesson_schema import Lesson
from src.evolution.retrieval import LessonRetriever


def _make_lesson(
    lesson_id="l1",
    stage="hypothesis_generation",
    lesson_type="failure_pattern",
    confidence=0.9,
    topic_tags=None,
    summary="Test lesson",
    reuse_rule="Apply when testing",
    scope="stage",
):
    return Lesson(
        id=lesson_id,
        created_from_run="run-test-abc",
        lesson_type=lesson_type,
        scope=scope,
        stage=stage,
        topic_tags=topic_tags or ["test", "hypothesis"],
        summary=summary,
        evidence=[],
        confidence=confidence,
        reuse_rule=reuse_rule,
        anti_pattern="",
    )


class TestLessonRetriever:
    def setup_method(self):
        self.retriever = LessonRetriever()
        self.lessons = [
            _make_lesson("l1", stage="hypothesis_generation", topic_tags=["feedback", "llm", "pipeline"], confidence=0.9),
            _make_lesson("l2", stage="experiment_plan_or_code", topic_tags=["experiment", "metrics"], confidence=0.85),
            _make_lesson("l3", stage="draft_writeup", topic_tags=["writing", "truncation"], confidence=0.75),
            _make_lesson("l4", stage="hypothesis_generation", topic_tags=["bias", "hypothesis"], confidence=0.6),
            _make_lesson("l5", stage="literature_scan", topic_tags=["papers", "citations"], confidence=0.95),
        ]

    def test_filter_by_stage(self):
        # Stage parameter boosts score (+2) but doesn't filter — it influences ranking
        result = self.retriever.retrieve(
            self.lessons, stage="hypothesis_generation", min_confidence=0.0
        )
        # hypothesis_generation lessons should rank first due to +2 stage score boost
        assert result[0].stage == "hypothesis_generation"
        assert result[1].stage == "hypothesis_generation"

    def test_filter_by_min_confidence(self):
        result = self.retriever.retrieve(self.lessons, min_confidence=0.88, max_results=100)
        assert all(lesson.confidence >= 0.88 for lesson in result)

    def test_tag_matching_boosts_relevance(self):
        result = self.retriever.retrieve(
            self.lessons,
            topic_tags=["feedback", "llm"],
            min_confidence=0.0,
            max_results=5,
        )
        # l1 has "feedback" and "llm" tags — should be ranked first
        assert result[0].id == "l1"

    def test_max_results_respected(self):
        result = self.retriever.retrieve(self.lessons, min_confidence=0.0, max_results=2)
        assert len(result) <= 2

    def test_empty_lessons_returns_empty(self):
        result = self.retriever.retrieve([], topic_tags=["test"])
        assert result == []

    def test_lesson_type_filter(self):
        lessons = [
            _make_lesson("la", lesson_type="failure_pattern", confidence=0.9),
            _make_lesson("lb", lesson_type="successful_pattern", confidence=0.9),
        ]
        result = self.retriever.retrieve(lessons, lesson_types=["failure_pattern"], min_confidence=0.0)
        assert all(lesson.lesson_type == "failure_pattern" for lesson in result)
        assert len(result) == 1

    def test_no_filters_returns_all(self):
        result = self.retriever.retrieve(self.lessons, min_confidence=0.0, max_results=100)
        assert len(result) == len(self.lessons)

    def test_stage_match_adds_score(self):
        # l1 is in hypothesis_generation — should rank above same-tag-count lesson in other stage
        result = self.retriever.retrieve(
            self.lessons,
            stage="hypothesis_generation",
            min_confidence=0.0,
            max_results=5,
        )
        # hypothesis_generation lessons should come first
        assert result[0].stage == "hypothesis_generation"

    def test_confidence_as_tiebreaker(self):
        # Two lessons with same stage, no topic tag match — confidence breaks tie
        lessons = [
            _make_lesson("low", confidence=0.7),
            _make_lesson("high", confidence=0.95),
        ]
        result = self.retriever.retrieve(lessons, min_confidence=0.0, max_results=5)
        assert result[0].id == "high"

    def test_below_min_confidence_excluded(self):
        result = self.retriever.retrieve(self.lessons, min_confidence=0.9, max_results=100)
        assert all(lesson.confidence >= 0.9 for lesson in result)
        # l4 (0.6), l3 (0.75), l2 (0.85) should be excluded
        ids = [lesson.id for lesson in result]
        assert "l4" not in ids
        assert "l3" not in ids
