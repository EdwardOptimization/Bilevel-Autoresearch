"""Unit tests for state classes."""
from core.state import InnerLesson, InnerLoopState, RunResult, StageScore


class TestRunResult:
    def test_stage_map(self):
        scores = [
            StageScore(stage="A", score=8, feedback="good"),
            StageScore(stage="B", score=7, feedback="ok"),
        ]
        result = RunResult(run_number=1, scores=scores, overall=7, article_version="test")
        assert result.stage_map == {"A": 8, "B": 7}


class TestInnerLoopState:
    def _make_state(self):
        return InnerLoopState(original_article="test article", article_id="test1")

    def test_initial_state(self):
        state = self._make_state()
        assert state.article_id == "test1"
        assert state.run_trace == []
        assert state.inner_lessons == []

    def test_record_run(self):
        state = self._make_state()
        result = RunResult(
            run_number=1,
            scores=[StageScore(stage="A", score=8, feedback="")],
            overall=8,
            article_version="v2",
        )
        state.record_run(result)
        assert len(state.run_trace) == 1
        assert state.article_working_copy == "v2"

    def test_peak_score(self):
        state = self._make_state()
        for i, score in enumerate([5, 8, 6], 1):
            state.record_run(RunResult(
                run_number=i, scores=[], overall=score, article_version="v",
            ))
        assert state.peak_score() == 8

    def test_peak_score_empty(self):
        state = self._make_state()
        assert state.peak_score() == 0

    def test_runs_to_threshold(self):
        state = self._make_state()
        for i, score in enumerate([5, 6, 8, 9], 1):
            state.record_run(RunResult(
                run_number=i, scores=[], overall=score, article_version="v",
            ))
        assert state.runs_to_threshold(8) == 3

    def test_runs_to_threshold_never(self):
        state = self._make_state()
        state.record_run(RunResult(
            run_number=1, scores=[], overall=5, article_version="v",
        ))
        assert state.runs_to_threshold(8) is None

    def test_add_lesson(self):
        state = self._make_state()
        lesson = InnerLesson(
            lesson_type="failure_pattern",
            stage="edit_planning",
            summary="test lesson",
            reuse_rule="do X when Y",
            confidence=0.9,
            run_number=1,
        )
        state.add_lesson(lesson)
        assert len(state.inner_lessons) == 1
