"""Unit tests for BaseStage and per-stage max_retries override."""
from pathlib import Path

from src.pipeline.base import BaseStage
from src.pipeline.draft_writeup import DraftWriteupStage
from src.pipeline.experiment_plan_or_code import ExperimentPlanStage
from src.pipeline.experiment_result_summary import ExperimentResultSummaryStage
from src.pipeline.hypothesis_generation import HypothesisGenerationStage
from src.pipeline.literature_scan import LiteratureScanStage


class TestBaseStageMaxRetries:
    """Verify per-stage max_retries overrides work as documented."""

    def test_base_stage_max_retries_is_none(self):
        # None means "use global config value"
        assert BaseStage.max_retries is None

    def test_experiment_plan_stage_overrides_max_retries(self):
        # This stage explicitly sets 2 retries due to two-phase generation complexity
        assert ExperimentPlanStage.max_retries == 2

    def test_other_stages_inherit_none(self):
        # Stages that don't override should return None (use global)
        assert HypothesisGenerationStage.max_retries is None
        assert LiteratureScanStage.max_retries is None
        assert ExperimentResultSummaryStage.max_retries is None
        assert DraftWriteupStage.max_retries is None

    def test_stage_name_attributes(self):
        assert LiteratureScanStage.name == "literature_scan"
        assert HypothesisGenerationStage.name == "hypothesis_generation"
        assert ExperimentPlanStage.name == "experiment_plan_or_code"
        assert ExperimentResultSummaryStage.name == "experiment_result_summary"
        assert DraftWriteupStage.name == "draft_writeup"

    def test_max_retries_instance_same_as_class(self):
        stage = ExperimentPlanStage()
        assert stage.max_retries == 2

    def test_none_max_retries_means_use_global(self):
        """Verify the documented pattern: None = defer to global config."""
        hyp_stage = HypothesisGenerationStage()
        # None signals RunManager to use self.qg_max_retries (from config)
        assert hyp_stage.max_retries is None

    def test_stage_model_default_empty(self):
        """Default model is empty string — caller falls back to provider default."""
        stage = LiteratureScanStage()
        assert stage.model == ""

    def test_stage_model_can_be_set(self):
        stage = LiteratureScanStage(model="custom-model")
        assert stage.model == "custom-model"


class TestSaveArtifact:
    """Verify BaseStage._save_artifact writes files correctly."""

    def test_save_artifact_creates_file(self, tmp_path):
        stage = LiteratureScanStage()
        (tmp_path / "stages").mkdir()
        rel_path = stage._save_artifact(tmp_path, "test.md", "Hello artifact")
        written = (tmp_path / rel_path).read_text()
        assert written == "Hello artifact"

    def test_save_artifact_returns_relative_path(self, tmp_path):
        stage = HypothesisGenerationStage()
        (tmp_path / "stages").mkdir()
        rel = stage._save_artifact(tmp_path, "hypotheses.md", "H1: test")
        # Should be relative to run_dir, not absolute
        assert not Path(rel).is_absolute()
        assert rel.startswith("stages/")

    def test_save_artifact_creates_parent_dirs(self, tmp_path):
        stage = DraftWriteupStage()
        # Don't pre-create stages dir — should be auto-created
        rel = stage._save_artifact(tmp_path, "sections/abstract.md", "abstract content")
        assert (tmp_path / rel).exists()
