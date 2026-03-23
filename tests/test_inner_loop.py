"""Unit tests for InnerLoopController."""
from core.inner_loop import InnerLoopController
from core.state import InnerLoopState, OuterLoopState, RunResult, StageScore


class _MockRunner:
    """A minimal runner that returns a fixed score without any LLM calls."""

    def __init__(self, score: int = 9):
        self._score = score
        self.call_count = 0
        self.prompt_overrides: dict = {}
        self.outer_cycle: int = 0
        self.model: str = "mock"

    def run_once(self, inner: InnerLoopState) -> RunResult:
        self.call_count += 1
        run_number = len(inner.run_trace) + 1
        stages = [
            StageScore(stage=s, score=self._score, feedback="ok")
            for s in ["A", "B", "C", "D", "E"]
        ]
        result = RunResult(
            run_number=run_number,
            scores=stages,
            overall=self._score,
            article_version=inner.article_working_copy,
        )
        inner.record_run(result)
        return result


def _make_outer_state(article_id: str = "article1", text: str = "hello") -> OuterLoopState:
    outer = OuterLoopState(base_dir=__import__("pathlib").Path("."), original_articles={article_id: text})
    outer.begin_cycle()
    return outer


class TestInnerLoopControllerInstantiation:
    def test_can_instantiate_with_mock_runner(self):
        runner = _MockRunner()
        ctrl = InnerLoopController(runner=runner)
        assert ctrl.runner is runner
        assert ctrl.max_iterations == 20
        assert ctrl.convergence_threshold == 8

    def test_custom_parameters_stored(self):
        runner = _MockRunner()
        ctrl = InnerLoopController(
            runner=runner,
            max_iterations=5,
            convergence_threshold=7,
            convergence_consecutive=2,
        )
        assert ctrl.max_iterations == 5
        assert ctrl.convergence_threshold == 7
        assert ctrl.convergence_consecutive == 2


class TestRunCycleCallCount:
    def test_converges_early_stops_calling_runner(self):
        """Runner returns score=9 every time; convergence_consecutive=3 so it stops at run 3."""
        runner = _MockRunner(score=9)
        ctrl = InnerLoopController(
            runner=runner,
            max_iterations=20,
            convergence_threshold=8,
            convergence_consecutive=3,
        )
        outer = _make_outer_state()
        inner = ctrl.run_cycle("article1", outer)

        assert runner.call_count == 3
        assert inner.peak_score() == 9
        assert inner.is_converged(threshold=8, consecutive=3)

    def test_budget_exhausted_when_never_converges(self):
        """Runner returns score=5 every time; never converges, should hit max_iterations."""
        runner = _MockRunner(score=5)
        ctrl = InnerLoopController(
            runner=runner,
            max_iterations=4,
            convergence_threshold=8,
            convergence_consecutive=3,
        )
        outer = _make_outer_state()
        inner = ctrl.run_cycle("article1", outer)

        assert runner.call_count == 4
        assert not inner.is_converged(threshold=8, consecutive=3)

    def test_run_once_called_expected_times_with_consecutive_2(self):
        """With convergence_consecutive=2, runner should stop after 2 high-score runs."""
        runner = _MockRunner(score=10)
        ctrl = InnerLoopController(
            runner=runner,
            max_iterations=20,
            convergence_threshold=8,
            convergence_consecutive=2,
        )
        outer = _make_outer_state()
        ctrl.run_cycle("article1", outer)

        assert runner.call_count == 2
