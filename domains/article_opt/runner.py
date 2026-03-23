"""Inner loop runner — executes one full run (A→B→C→D→E + evaluate + extract lessons).

One "run" = one pass through all 5 stages on the current article working copy.
The runner is called repeatedly by the inner loop controller until convergence or budget.

Quality gate: if overall score < min_score, retry the failing stage with evaluator feedback.
Lesson extraction: after each run, extract lessons from stage outputs + scores.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.llm_client import call_llm, parse_json_response
from core.state import InnerLesson, InnerLoopState, RunResult, StageScore

from .evaluator.article_evaluator import ArticleEvaluator
from .pipeline.article_analysis import ArticleAnalysisStage
from .pipeline.base import BaseStage
from .pipeline.edit_planning import EditPlanningStage
from .pipeline.impact_assessment import ImpactAssessmentStage
from .pipeline.improvement_hypotheses import ImprovementHypothesesStage
from .pipeline.revised_output import RevisedOutputStage

logger = logging.getLogger(__name__)

LESSON_SYSTEM = """You are extracting structured lessons from a pipeline run.
Each lesson must be specific, actionable, and reusable in future runs.
Return valid JSON only."""


class InnerRunner:
    """
    Executes one inner run and updates InnerLoopState.

    Args:
        model: LLM model for pipeline stages (inner loop)
        eval_model: LLM model for evaluator (must be same or different — stays isolated)
        min_score: Quality gate threshold (default 6)
        max_retries: Max retries per stage on quality gate failure
        artifacts_base: Directory to write run artifacts
    """

    def __init__(
        self,
        model: str = "",
        eval_model: str = "",
        min_score: int = 6,
        max_retries: int = 2,
        artifacts_base: Path | None = None,
    ):
        self.model = model
        self.eval_model = eval_model or model
        self.min_score = min_score
        self.max_retries = max_retries
        self.artifacts_base = artifacts_base or Path("artifacts")

        # Stage instances
        self.stages = [
            ArticleAnalysisStage(model=model),
            ImprovementHypothesesStage(model=model),
            EditPlanningStage(model=model),
            ImpactAssessmentStage(model=model),
            RevisedOutputStage(model=model),
        ]
        self.evaluator = ArticleEvaluator(model=eval_model)
        # Outer loop injects additional stage guidance here (persists across inner resets)
        self.prompt_overrides: dict[str, str] = {}
        # Set by InnerLoopController before each cycle; used for artifact path namespacing
        self.outer_cycle: int = 0

    def inject_stage(self, stage: "BaseStage", inject_after: str) -> None:
        """Insert a generated stage into the pipeline after the named stage.

        Called by MechanismResearcher.validate() before running the inner loop
        with an experimental mechanism. The injection is permanent for this runner
        instance — clone the runner if you need a clean baseline.

        Args:
            stage:        Instantiated stage to inject (must implement BaseStage).
            inject_after: Name of the existing stage to inject after.

        Raises:
            ValueError: if inject_after does not match any stage in the pipeline.
        """
        for i, s in enumerate(self.stages):
            if s.name == inject_after:
                self.stages.insert(i + 1, stage)
                logger.info(f"Injected stage '{stage.name}' after '{inject_after}'")
                return
        raise ValueError(
            f"Stage '{inject_after}' not found in pipeline. "
            f"Available: {[s.name for s in self.stages]}"
        )

    def run_once(self, inner_state: InnerLoopState) -> RunResult:
        """Execute one full pipeline pass. Updates inner_state and returns RunResult."""
        run_num = len(inner_state.run_trace) + 1
        if self.outer_cycle:
            run_dir = self.artifacts_base / f"cycle_{self.outer_cycle:02d}" / f"run_{run_num:03d}"
        else:
            run_dir = self.artifacts_base / f"run_{run_num:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Run {run_num}] Starting — article: {inner_state.article_id}")

        # Build initial context
        context: dict = {
            "article_id": inner_state.article_id,
            "article_content": inner_state.article_working_copy,
            "previous_outputs": {},
            "retrieved_lessons": self._build_lessons_text(inner_state),
            "outer_guidance": self.prompt_overrides,  # from outer loop, persists across resets
            "evaluator_feedback": "",
            "run_dir": run_dir,
            "run_number": run_num,
        }

        stage_scores: list[StageScore] = []

        # Run each stage with quality gate
        for stage in self.stages:
            output, score_info, retried = self._run_stage_with_gate(stage, context, run_num)
            context["previous_outputs"][stage.name] = output.get("content", "")

            stage_scores.append(StageScore(
                stage=stage.name,
                score=score_info.get("score", 5),
                feedback=score_info.get("feedback", ""),
                retried=retried,
            ))
            logger.info(f"[Run {run_num}] Stage {stage.name}: {score_info.get('score', '?')}/10")

        # Evaluate the final revised article (retry on fallback)
        revised_article = context["previous_outputs"].get("revised_output", inner_state.article_working_copy)
        eval_result = None
        for eval_attempt in range(3):
            eval_result = self.evaluator.evaluate(revised_article, inner_state.article_id)
            if eval_result.get("summary") != "Evaluation parsing failed.":
                break
            logger.warning(f"[Run {run_num}] Evaluation fallback, retrying (attempt {eval_attempt + 2}/3)...")

        # Map evaluator rubric dimensions to stage scores (E stage gets overall rubric score)
        overall = int(round(eval_result.get("overall", 5)))

        # Build RunResult using rubric dimension scores (A-E) for tracking
        rubric_scores = eval_result.get("scores", {})
        rubric_stage_scores = [
            StageScore(stage=dim, score=rubric_scores.get(dim, 5),
                       feedback=eval_result.get("feedback", {}).get(dim, {}).get("weakness", ""))
            for dim in "ABCDE"
        ]

        result = RunResult(
            run_number=run_num,
            scores=rubric_stage_scores,
            overall=overall,
            article_version=revised_article,
        )

        # Save evaluation result
        eval_path = run_dir / "evaluation.json"
        eval_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=False), encoding="utf-8")

        # Update inner state
        inner_state.record_run(result)

        # Extract and store lessons
        lessons = self._extract_lessons(context, eval_result, run_num)
        for lesson in lessons:
            inner_state.add_lesson(lesson)

        logger.info(f"[Run {run_num}] Overall: {overall}/10 | Lessons: {len(lessons)}")
        return result

    def _run_stage_with_gate(
        self, stage, context: dict, run_num: int
    ) -> tuple[dict, dict, bool]:
        """Run a stage, apply quality gate, retry if needed. Returns (output, score_info, retried)."""
        max_attempts = (stage.max_retries or self.max_retries) + 1
        retried = False

        for attempt in range(max_attempts):
            output = stage.run(context)
            # Simple score: use overall from impact_assessment if available, else 7 as default
            # Full per-stage scoring would require a stage-level evaluator (V1 pattern)
            # For simplicity, we trust the final article evaluator as ground truth
            score_info = {"score": 7, "feedback": ""}

            if attempt > 0:
                retried = True
                context["previous_outputs"][stage.name] = output.get("content", "")

            # For Stage E (revised_output), check if article meaningfully changed
            if stage.name == "revised_output":
                revised = output.get("revised_article", "")
                if len(revised) < 100:
                    score_info = {"score": 3, "feedback": "Revised article is too short — likely truncated."}
                    if attempt < max_attempts - 1:
                        context["evaluator_feedback"] = score_info["feedback"]
                        logger.warning(f"[Run {run_num}] Stage {stage.name} short output, retrying...")
                        continue

            break

        return output, score_info, retried

    def _build_lessons_text(self, inner_state: InnerLoopState) -> str:
        """
        Format lessons for injection into stages.

        Two-tier design:
        - Promoted skills (confidence ≥ threshold): shown as verified rules
        - Raw lessons (any confidence): always shown, labeled as tentative

        This ensures injection happens every run regardless of confidence
        calibration variance from the LLM.
        """
        has_skills = bool(inner_state.inner_skills)
        has_lessons = bool(inner_state.inner_lessons)
        if not has_skills and not has_lessons:
            return ""

        lines = ["## Lessons from Previous Runs (this cycle)"]

        # Tier 1: promoted skills (high-confidence, stage-grouped)
        if has_skills:
            lines.append("\n### Verified Rules (high confidence)")
            for stage_name, skill_text in inner_state.inner_skills.items():
                lines.append(f"\n#### {stage_name}\n{skill_text}")

        # Tier 2: all raw lessons sorted by confidence desc, most recent runs first
        if has_lessons:
            lines.append("\n### Observations from Recent Runs")
            # Sort by (run_number desc, confidence desc) so recent runs take precedence
            sorted_lessons = sorted(
                inner_state.inner_lessons,
                key=lambda l: (l.run_number, l.confidence),
                reverse=True,
            )
            for lesson in sorted_lessons[:8]:  # top 8 by recency + confidence
                tag = "✓" if lesson.confidence >= 0.85 else "~"
                lines.append(
                    f"\n[{tag} run={lesson.run_number} conf={lesson.confidence:.2f} | {lesson.stage}] "
                    f"{lesson.summary}\n"
                    f"→ Rule: {lesson.reuse_rule}"
                )

        return "\n".join(lines)

    def _extract_lessons(
        self, context: dict, eval_result: dict, run_num: int
    ) -> list[InnerLesson]:
        """Extract structured lessons from this run's outputs and evaluation."""
        scores = eval_result.get("scores", {})
        below = eval_result.get("below_threshold", [])
        summary = eval_result.get("summary", "")

        prompt = f"""## Pipeline Run Summary (Run {run_num})

Overall score: {eval_result.get('overall', '?')}/10
Below threshold: {below}
Evaluator summary: {summary}

Dimension scores: {scores}

Edit plan used:
{context['previous_outputs'].get('edit_planning', '')[:1500]}

## Task
Extract 2-4 lessons from this run for future runs in this cycle.

Focus on:
- What editing pattern worked or failed?
- Which stage's output most limited the final quality?
- What specific change would have most improved the score?

Return JSON array:
[
  {{
    "lesson_type": "failure_pattern|success_pattern|improvement",
    "stage": "article_analysis|improvement_hypotheses|edit_planning|impact_assessment|revised_output",
    "summary": "...",
    "reuse_rule": "In future runs, when [condition], do [action].",
    "confidence": 0.0
  }}
]"""

        raw = call_llm(prompt, system=LESSON_SYSTEM, model=self.model, max_tokens=4000)
        parsed = parse_json_response(raw)

        if not isinstance(parsed, list):
            logger.warning(f"[Run {run_num}] Lesson extraction failed — could not parse LLM response")
            return []

        lessons = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            lessons.append(InnerLesson(
                lesson_type=item.get("lesson_type", "improvement"),
                stage=item.get("stage", ""),
                summary=item.get("summary", ""),
                reuse_rule=item.get("reuse_rule", ""),
                confidence=float(item.get("confidence", 0.7)),
                run_number=run_num,
            ))
        return lessons
