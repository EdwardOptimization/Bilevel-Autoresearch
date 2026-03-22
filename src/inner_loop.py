"""Inner loop controller — runs InnerRunner repeatedly until convergence or budget.

Outer loop calls run_cycle() once per outer iteration.
After each run, checks convergence. Updates inner skills from high-confidence lessons.
"""
from __future__ import annotations

import logging

from .runner import InnerRunner
from .state import InnerLoopState, OuterLoopState

logger = logging.getLogger(__name__)


class InnerLoopController:
    """
    Manages one full inner cycle (up to max_iterations runs on one article).

    Args:
        runner: configured InnerRunner instance
        max_iterations: hard budget per cycle (default 20)
        convergence_threshold: score to declare convergence (default 8)
        convergence_consecutive: runs above threshold before declaring converged (default 3)
        skill_confidence_min: lessons above this confidence become skills (default 0.85)
    """

    def __init__(
        self,
        runner: InnerRunner,
        max_iterations: int = 20,
        convergence_threshold: int = 8,
        convergence_consecutive: int = 3,
        skill_confidence_min: float = 0.85,
    ):
        self.runner = runner
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.convergence_consecutive = convergence_consecutive
        self.skill_confidence_min = skill_confidence_min

    def run_cycle(
        self,
        article_id: str,
        outer_state: OuterLoopState,
    ) -> InnerLoopState:
        """
        Run one full inner cycle for the given article.
        Builds fresh InnerLoopState (article reset), runs until convergence or budget.
        Injects outer_state.prompt_overrides into the runner context.

        Returns the completed InnerLoopState (before reset — outer loop extracts from it).
        """
        inner = outer_state.build_inner_state(article_id)
        self.runner.prompt_overrides = outer_state.prompt_overrides
        self.runner.outer_cycle = outer_state.current_cycle

        logger.info(
            f"[Outer cycle {outer_state.current_cycle}] "
            f"Starting inner cycle — article: {article_id}"
        )

        for i in range(self.max_iterations):
            result = self.runner.run_once(inner)

            logger.info(
                f"  Run {result.run_number}: overall={result.overall}/10 "
                f"| scores={result.stage_map}"
            )

            # Promote high-confidence lessons to skills after each run
            self._promote_skills(inner)

            # Check convergence
            if inner.is_converged(self.convergence_threshold, self.convergence_consecutive):
                logger.info(
                    f"  Converged at run {result.run_number} "
                    f"(≥{self.convergence_threshold}/10 for {self.convergence_consecutive} consecutive runs)"
                )
                break

            if result.overall >= self.convergence_threshold:
                logger.info(f"  Run {result.run_number} reached threshold — watching for stability...")
        else:
            logger.info(
                f"  Budget exhausted ({self.max_iterations} runs). "
                f"Peak score: {inner.peak_score()}/10"
            )

        return inner

    def _promote_skills(self, inner: InnerLoopState) -> None:
        """Promote high-confidence lessons to stage skills for injection."""
        by_stage: dict[str, list] = {}
        for lesson in inner.inner_lessons:
            if lesson.confidence >= self.skill_confidence_min:
                by_stage.setdefault(lesson.stage, []).append(lesson)

        for stage_name, lessons in by_stage.items():
            lines = [f"## Guidance for {stage_name} (from {len(lessons)} lessons)"]
            for l in sorted(lessons, key=lambda x: -x.confidence):
                lines.append(f"\n### Rule (confidence={l.confidence:.2f})")
                lines.append(f"**Summary**: {l.summary}")
                lines.append(f"**Apply when**: {l.reuse_rule}")
            inner.update_skill(stage_name, "\n".join(lines))
