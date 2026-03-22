"""Unit tests for pipeline base and stage attributes."""
from src.pipeline.article_analysis import ArticleAnalysisStage
from src.pipeline.base import BaseStage
from src.pipeline.edit_planning import EditPlanningStage
from src.pipeline.impact_assessment import ImpactAssessmentStage
from src.pipeline.improvement_hypotheses import ImprovementHypothesesStage
from src.pipeline.revised_output import RevisedOutputStage


class TestBaseStage:
    def test_max_retries_default_none(self):
        assert BaseStage.max_retries is None

    def test_model_default_empty(self):
        stage = ArticleAnalysisStage()
        assert stage.model == ""

    def test_model_can_be_set(self):
        stage = ArticleAnalysisStage(model="custom-model")
        assert stage.model == "custom-model"


class TestStageNames:
    def test_all_stage_names_unique(self):
        stages = [
            ArticleAnalysisStage,
            ImprovementHypothesesStage,
            EditPlanningStage,
            ImpactAssessmentStage,
            RevisedOutputStage,
        ]
        names = [s.name for s in stages]
        assert len(names) == len(set(names))

    def test_expected_names(self):
        assert ArticleAnalysisStage.name == "article_analysis"
        assert ImprovementHypothesesStage.name == "improvement_hypotheses"
        assert EditPlanningStage.name == "edit_planning"
        assert ImpactAssessmentStage.name == "impact_assessment"
        assert RevisedOutputStage.name == "revised_output"


class TestSaveArtifact:
    def test_save_artifact_creates_file(self, tmp_path):
        stage = ArticleAnalysisStage()
        (tmp_path / "stages").mkdir()
        rel_path = stage._save_artifact(tmp_path, "test.md", "Hello artifact")
        written = (tmp_path / rel_path).read_text()
        assert written == "Hello artifact"

    def test_save_artifact_creates_parent_dirs(self, tmp_path):
        stage = RevisedOutputStage()
        rel = stage._save_artifact(tmp_path, "sections/output.md", "content")
        assert (tmp_path / rel).exists()
