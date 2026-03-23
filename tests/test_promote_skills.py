"""Tests for InnerLoopState skill management and high-confidence lesson handling.

There is no _promote_skills method in state.py.  These tests cover the
skill surface that does exist: update_skill, inner_skills dict, add_lesson
with high-confidence entries, and lesson_quality_stats.
"""
import pytest

from core.state import InnerLesson, InnerLoopState


def _make_state(article="article text", article_id="art1"):
    return InnerLoopState(original_article=article, article_id=article_id)


def _make_lesson(confidence=0.9, stage="A", lesson_type="success_pattern"):
    return InnerLesson(
        lesson_type=lesson_type,
        stage=stage,
        summary="test summary",
        reuse_rule="do X when Y",
        confidence=confidence,
        run_number=1,
    )


# ---------------------------------------------------------------------------
# update_skill / inner_skills
# ---------------------------------------------------------------------------

class TestUpdateSkill:
    def test_update_skill_stores_text(self):
        state = _make_state()
        state.update_skill("A", "always check structure first")
        assert state.inner_skills["A"] == "always check structure first"

    def test_update_skill_overwrites_existing(self):
        state = _make_state()
        state.update_skill("B", "initial guidance")
        state.update_skill("B", "updated guidance")
        assert state.inner_skills["B"] == "updated guidance"

    def test_update_multiple_stages(self):
        state = _make_state()
        state.update_skill("A", "skill A")
        state.update_skill("C", "skill C")
        assert state.inner_skills == {"A": "skill A", "C": "skill C"}

    def test_inner_skills_empty_on_init(self):
        state = _make_state()
        assert state.inner_skills == {}

    def test_reset_clears_skills(self):
        state = _make_state()
        state.update_skill("A", "some skill")
        state.reset()
        assert state.inner_skills == {}

    def test_skills_not_shared_across_instances(self):
        s1 = _make_state(article_id="a1")
        s2 = _make_state(article_id="a2")
        s1.update_skill("D", "skill for a1")
        assert "D" not in s2.inner_skills


# ---------------------------------------------------------------------------
# add_lesson with high-confidence entries
# ---------------------------------------------------------------------------

class TestAddLessonHighConfidence:
    def test_add_high_confidence_lesson(self):
        state = _make_state()
        lesson = _make_lesson(confidence=0.95)
        state.add_lesson(lesson)
        assert len(state.inner_lessons) == 1
        assert state.inner_lessons[0].confidence == 0.95

    def test_add_low_confidence_lesson(self):
        state = _make_state()
        lesson = _make_lesson(confidence=0.3)
        state.add_lesson(lesson)
        assert len(state.inner_lessons) == 1

    def test_multiple_lessons_appended_in_order(self):
        state = _make_state()
        for conf in [0.5, 0.9, 0.7]:
            state.add_lesson(_make_lesson(confidence=conf))
        assert len(state.inner_lessons) == 3
        assert [l.confidence for l in state.inner_lessons] == [0.5, 0.9, 0.7]

    def test_reset_clears_lessons(self):
        state = _make_state()
        state.add_lesson(_make_lesson(confidence=0.9))
        state.reset()
        assert state.inner_lessons == []


# ---------------------------------------------------------------------------
# lesson_quality_stats
# ---------------------------------------------------------------------------

class TestLessonQualityStats:
    def test_empty_returns_zeros(self):
        state = _make_state()
        stats = state.lesson_quality_stats()
        assert stats == {"total": 0, "high_confidence": 0, "fraction": 0.0}

    def test_all_high_confidence(self):
        state = _make_state()
        for _ in range(4):
            state.add_lesson(_make_lesson(confidence=0.9))
        stats = state.lesson_quality_stats()
        assert stats["total"] == 4
        assert stats["high_confidence"] == 4
        assert stats["fraction"] == 1.0

    def test_none_high_confidence(self):
        state = _make_state()
        for _ in range(3):
            state.add_lesson(_make_lesson(confidence=0.5))
        stats = state.lesson_quality_stats()
        assert stats["total"] == 3
        assert stats["high_confidence"] == 0
        assert stats["fraction"] == 0.0

    def test_mixed_confidence(self):
        state = _make_state()
        # 2 high (>=0.85), 2 low
        for conf in [0.9, 0.85, 0.7, 0.4]:
            state.add_lesson(_make_lesson(confidence=conf))
        stats = state.lesson_quality_stats()
        assert stats["total"] == 4
        assert stats["high_confidence"] == 2
        assert stats["fraction"] == pytest.approx(0.5)

    def test_boundary_confidence_085_counts_as_high(self):
        state = _make_state()
        state.add_lesson(_make_lesson(confidence=0.85))
        stats = state.lesson_quality_stats()
        assert stats["high_confidence"] == 1

    def test_confidence_below_085_not_high(self):
        state = _make_state()
        state.add_lesson(_make_lesson(confidence=0.849))
        stats = state.lesson_quality_stats()
        assert stats["high_confidence"] == 0
