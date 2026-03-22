"""Unit tests for InnerRunner — stage management and inject_stage."""
import pytest

from core.pipeline.base import BaseStage
from core.runner import InnerRunner


class _DummyStage(BaseStage):
    name = "dummy_stage"

    def run(self, context: dict) -> dict:
        return {"content": "dummy output"}


class TestInjectStage:
    def test_inject_after_existing_stage(self):
        runner = InnerRunner()
        original_len = len(runner.stages)
        runner.inject_stage(_DummyStage(), "improvement_hypotheses")
        assert len(runner.stages) == original_len + 1
        names = [s.name for s in runner.stages]
        idx = names.index("dummy_stage")
        assert names[idx - 1] == "improvement_hypotheses"

    def test_inject_after_first_stage(self):
        runner = InnerRunner()
        runner.inject_stage(_DummyStage(), "article_analysis")
        names = [s.name for s in runner.stages]
        assert names[0] == "article_analysis"
        assert names[1] == "dummy_stage"

    def test_inject_after_last_stage(self):
        runner = InnerRunner()
        runner.inject_stage(_DummyStage(), "revised_output")
        names = [s.name for s in runner.stages]
        assert names[-1] == "dummy_stage"

    def test_inject_nonexistent_raises(self):
        runner = InnerRunner()
        with pytest.raises(ValueError, match="not found"):
            runner.inject_stage(_DummyStage(), "nonexistent_stage")

    def test_inject_preserves_original_order(self):
        runner = InnerRunner()
        original = [s.name for s in runner.stages]
        runner.inject_stage(_DummyStage(), "edit_planning")
        names = [s.name for s in runner.stages]
        # Remove the dummy and check original order preserved
        names.remove("dummy_stage")
        assert names == original


class TestRunnerInit:
    def test_default_stages(self):
        runner = InnerRunner()
        names = [s.name for s in runner.stages]
        assert names == [
            "article_analysis",
            "improvement_hypotheses",
            "edit_planning",
            "impact_assessment",
            "revised_output",
        ]

    def test_prompt_overrides_empty_by_default(self):
        runner = InnerRunner()
        assert runner.prompt_overrides == {}
