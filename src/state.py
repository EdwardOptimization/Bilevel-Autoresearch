"""
State management for the dual-layer article optimizer.

Two distinct state objects:
  InnerLoopState  — reset at each outer iteration boundary
  OuterLoopState  — persists across all outer iterations

See docs/quality_definitions.md §"State Boundary" for the full design rationale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StageScore:
    stage: str          # "A" | "B" | "C" | "D" | "E"
    score: int          # 0–10
    feedback: str       # evaluator's textual explanation
    retried: bool = False


@dataclass
class RunResult:
    run_number: int
    scores: list[StageScore]
    overall: int
    article_version: str    # full text of the article at end of this run
    strategy_used: str = "" # outer-injected strategy name, if any

    @property
    def stage_map(self) -> dict[str, int]:
        return {s.stage: s.score for s in self.scores}

    def to_dict(self) -> dict:
        return {
            "run_number": self.run_number,
            "scores": {s.stage: s.score for s in self.scores},
            "overall": self.overall,
            "strategy_used": self.strategy_used,
            "feedback": {s.stage: s.feedback for s in self.scores},
            "retried_stages": [s.stage for s in self.scores if s.retried],
        }


@dataclass
class InnerLesson:
    lesson_type: str    # "failure_pattern" | "success_pattern" | "improvement"
    stage: str
    summary: str
    reuse_rule: str
    confidence: float   # 0.0–1.0
    run_number: int


@dataclass
class OuterLesson:
    outer_cycle: int
    lesson_type: str    # "strategy_effectiveness" | "config_change" | "failure_pattern"
    strategy_used: str
    summary: str
    reuse_rule: str
    confidence: float
    inner_convergence_before: int | None = None  # runs to reach 8/10 before change
    inner_convergence_after: int | None = None   # runs to reach 8/10 after change
    stage_affected: str = ""

    def to_dict(self) -> dict:
        return {
            "outer_cycle": self.outer_cycle,
            "lesson_type": self.lesson_type,
            "strategy_used": self.strategy_used,
            "inner_convergence_before": self.inner_convergence_before,
            "inner_convergence_after": self.inner_convergence_after,
            "stage_affected": self.stage_affected,
            "summary": self.summary,
            "reuse_rule": self.reuse_rule,
            "confidence": self.confidence,
        }


# ── Inner loop state ──────────────────────────────────────────────────────────

class InnerLoopState:
    """
    All state that belongs to one inner cycle.
    Cleared completely at each outer iteration boundary.

    Forbidden cross-boundary flows (see quality_definitions.md):
      - article_working_copy must not carry over to the next cycle
      - inner_lessons and inner_skills must not be injected into the next cycle
      - run_trace is extracted by the outer loop BEFORE reset, then discarded
    """

    def __init__(self, original_article: str, article_id: str):
        self.article_id = article_id
        self._original_article = original_article  # immutable reference
        self.article_working_copy: str = original_article
        self.inner_lessons: list[InnerLesson] = []
        self.inner_skills: dict[str, str] = {}   # stage → skill markdown
        self.run_trace: list[RunResult] = []
        self.retry_log: list[dict] = []           # {run, stage, attempt, reason}

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def record_run(self, result: RunResult) -> None:
        self.run_trace.append(result)
        self.article_working_copy = result.article_version

    def add_lesson(self, lesson: InnerLesson) -> None:
        self.inner_lessons.append(lesson)

    def update_skill(self, stage: str, skill_text: str) -> None:
        self.inner_skills[stage] = skill_text

    def log_retry(self, run: int, stage: str, attempt: int, reason: str) -> None:
        self.retry_log.append({"run": run, "stage": stage, "attempt": attempt, "reason": reason})

    # ── Convergence check ─────────────────────────────────────────────────────

    def is_converged(self, threshold: int = 8, consecutive: int = 3) -> bool:
        """True if the last `consecutive` runs all scored >= threshold overall."""
        if len(self.run_trace) < consecutive:
            return False
        return all(r.overall >= threshold for r in self.run_trace[-consecutive:])

    def runs_to_threshold(self, threshold: int = 8) -> int | None:
        """Return the run number when threshold was first reached, or None."""
        for r in self.run_trace:
            if r.overall >= threshold:
                return r.run_number
        return None

    def peak_score(self) -> int:
        if not self.run_trace:
            return 0
        return max(r.overall for r in self.run_trace)

    # ── Convergence trace (extracted by outer loop before reset) ──────────────

    def convergence_trace(self) -> list[dict]:
        return [{"run": r.run_number, "overall": r.overall, **r.stage_map} for r in self.run_trace]

    def stage_failure_pattern(self) -> dict[str, dict]:
        """Per-stage stats: mean score, std, retry count."""
        from statistics import mean, stdev
        pattern = {}
        for stage_id in ["A", "B", "C", "D", "E"]:
            scores = [r.stage_map.get(stage_id, 0) for r in self.run_trace if stage_id in r.stage_map]
            retries = sum(1 for ev in self.retry_log if ev["stage"] == stage_id)
            if scores:
                pattern[stage_id] = {
                    "mean": round(mean(scores), 2),
                    "std": round(stdev(scores), 2) if len(scores) > 1 else 0.0,
                    "retries": retries,
                    "min": min(scores),
                    "max": max(scores),
                }
        return pattern

    def evaluator_dimension_pattern(self) -> dict[str, float]:
        """Mean score per rubric dimension across all runs."""
        from statistics import mean
        dim_scores: dict[str, list[int]] = {d: [] for d in ["A", "B", "C", "D", "E"]}
        for run in self.run_trace:
            for stage in run.scores:
                if stage.stage in dim_scores:
                    dim_scores[stage.stage].append(stage.score)
        return {d: round(mean(v), 2) if v else 0.0 for d, v in dim_scores.items()}

    def lesson_quality_stats(self) -> dict:
        if not self.inner_lessons:
            return {"total": 0, "high_confidence": 0, "fraction": 0.0}
        high = sum(1 for l in self.inner_lessons if l.confidence >= 0.85)
        return {
            "total": len(self.inner_lessons),
            "high_confidence": high,
            "fraction": round(high / len(self.inner_lessons), 2),
        }

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Restore to outer-iteration-start state.
        Call AFTER outer loop has extracted what it needs via convergence_trace(),
        stage_failure_pattern(), etc.
        """
        self.article_working_copy = self._original_article
        self.inner_lessons = []
        self.inner_skills = {}
        self.run_trace = []
        self.retry_log = []


# ── Outer loop state ──────────────────────────────────────────────────────────

class OuterLoopState:
    """
    State that persists across all outer iterations.
    Never reset. This is what the outer loop is actually optimizing.
    """

    def __init__(self, base_dir: Path, original_articles: dict[str, str]):
        self.base_dir = base_dir
        self.original_articles = original_articles  # article_id → original text
        self.outer_lessons: list[OuterLesson] = []
        self.outer_skills: dict[str, str] = {}   # strategy_name → guidance markdown
        self.iteration_summaries: list[dict] = []
        self.strategy_history: list[dict] = []
        self.current_cycle: int = 0
        self._examples_dir = base_dir / "examples"
        self._examples_dir.mkdir(exist_ok=True)
        # Prompt overrides: stage_name → additional guidance appended to stage prompts
        # Updated by outer loop; persists across all inner cycles
        self.prompt_overrides: dict[str, str] = {}

    # ── Extraction from completed inner cycle ─────────────────────────────────

    def extract_from_inner(self, inner: InnerLoopState, strategy_used: str = "") -> dict:
        """
        Pull process-level signals from a completed inner cycle.
        Returns a summary dict that is used to generate outer lessons.

        This is the ONLY sanctioned path for inner → outer information flow.
        Content (article text) is archived but never returned in this dict.
        """
        runs_to_8 = inner.runs_to_threshold(8)
        summary = {
            "outer_cycle": self.current_cycle,
            "article_id": inner.article_id,
            "total_inner_runs": len(inner.run_trace),
            "peak_score": inner.peak_score(),
            "runs_to_threshold_8": runs_to_8,
            "converged": inner.is_converged(),
            "convergence_trace": inner.convergence_trace(),
            "stage_failure_pattern": inner.stage_failure_pattern(),
            "evaluator_dimension_pattern": inner.evaluator_dimension_pattern(),
            "lesson_quality": inner.lesson_quality_stats(),
            "retry_log": inner.retry_log,
            "strategy_used": strategy_used,
        }
        self.iteration_summaries.append(summary)
        self._archive_inner(inner)
        return summary

    def _archive_inner(self, inner: InnerLoopState) -> None:
        """Save best article and full trace to examples/ (archival only)."""
        cycle = self.current_cycle
        if inner.run_trace:
            best_run = max(inner.run_trace, key=lambda r: r.overall)
            best_path = self._examples_dir / f"outer_cycle_{cycle:02d}_best_{inner.article_id}.md"
            best_path.write_text(best_run.article_version, encoding="utf-8")

        trace_path = self._examples_dir / f"outer_cycle_{cycle:02d}_trace_{inner.article_id}.json"
        trace_data = {
            "outer_cycle": cycle,
            "article_id": inner.article_id,
            "run_trace": [r.to_dict() for r in inner.run_trace],
            "retry_log": inner.retry_log,
            "lesson_summary": inner.lesson_quality_stats(),
        }
        trace_path.write_text(json.dumps(trace_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Outer lesson management ───────────────────────────────────────────────

    def add_outer_lesson(self, lesson: OuterLesson) -> None:
        self.outer_lessons.append(lesson)
        self._persist_outer_lessons()

    def add_strategy_result(self, strategy: str, inner_before: int | None, inner_after: int | None, notes: str) -> None:
        self.strategy_history.append({
            "outer_cycle": self.current_cycle,
            "strategy": strategy,
            "convergence_before": inner_before,
            "convergence_after": inner_after,
            "delta": (inner_after - inner_before) if (inner_before and inner_after) else None,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _persist_outer_lessons(self) -> None:
        path = self.base_dir / "memory" / "outer_lessons.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for lesson in self.outer_lessons:
                f.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")

    # ── Convergence check ─────────────────────────────────────────────────────

    def is_outer_converged(self, target_inner_runs: int = 10) -> bool:
        """
        Outer loop converged when inner loop reaches ≥8/10 in ≤target_inner_runs
        across the last 2 consecutive outer iterations.
        """
        recent = self.iteration_summaries[-2:]
        if len(recent) < 2:
            return False
        convergences = [s.get("runs_to_threshold_8") for s in recent]
        return all(c is not None and c <= target_inner_runs for c in convergences)

    # ── Cycle management ──────────────────────────────────────────────────────

    def begin_cycle(self) -> None:
        self.current_cycle += 1

    def build_inner_state(self, article_id: str) -> InnerLoopState:
        """Create a fresh inner state for this article, using the original text."""
        original = self.original_articles[article_id]
        return InnerLoopState(original_article=original, article_id=article_id)

    # ── Context for outer LLM call ────────────────────────────────────────────

    def build_outer_context(self, inner_summary: dict, reference_doc: str = "") -> str:
        """
        Build the context string injected into the outer loop's LLM call.
        Contains process signals (NOT article content).
        """
        lines = [
            f"# Outer Loop Context — Cycle {self.current_cycle}",
            "",
            "## Inner Cycle Summary",
            f"- Article: {inner_summary['article_id']}",
            f"- Total runs: {inner_summary['total_inner_runs']}",
            f"- Peak score: {inner_summary['peak_score']}/10",
            f"- Runs to reach 8/10: {inner_summary['runs_to_threshold_8'] or 'never'}",
            f"- Converged: {inner_summary['converged']}",
            "",
            "## Convergence Trace",
            "run | A | B | C | D | E | overall",
            "----|---|---|---|---|---|--------",
        ]
        for row in inner_summary["convergence_trace"]:
            lines.append(
                f"{row['run']:3d} | {row.get('A','?')} | {row.get('B','?')} | "
                f"{row.get('C','?')} | {row.get('D','?')} | {row.get('E','?')} | {row['overall']}"
            )

        lines += [
            "",
            "## Stage Failure Patterns",
        ]
        for stage, stats in inner_summary["stage_failure_pattern"].items():
            lines.append(
                f"- Stage {stage}: mean={stats['mean']}, std={stats['std']}, "
                f"retries={stats['retries']}, range=[{stats['min']},{stats['max']}]"
            )

        lines += [
            "",
            "## Lesson Quality",
            f"- Total lessons: {inner_summary['lesson_quality']['total']}",
            f"- High-confidence (≥0.85): {inner_summary['lesson_quality']['high_confidence']}",
            f"- Fraction: {inner_summary['lesson_quality']['fraction']}",
            "",
            "## Prior Outer Lessons",
        ]
        for lesson in self.outer_lessons[-5:]:  # last 5 to avoid context overflow
            lines.append(f"- [{lesson.lesson_type}] {lesson.summary}")

        if reference_doc:
            lines += ["", "## Strategy Reference", reference_doc]

        return "\n".join(lines)
