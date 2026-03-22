"""Outer loop controller — meta-analysis via DeepSeek, updates pipeline config.

Each outer iteration:
  1. Run inner cycle (via InnerLoopController) on each article
  2. Extract process signals from inner state
  3. Call DeepSeek to analyze trace → root cause + strategy + config changes
  4. Apply config changes to outer_state.prompt_overrides
  5. Save outer lessons
  6. Check outer convergence
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .llm_client import LLMClient, parse_json_response

from .inner_loop import InnerLoopController
from .runner import InnerRunner
from .state import OuterLesson, OuterLoopState

logger = logging.getLogger(__name__)

OUTER_SYSTEM = """You are the outer optimization loop for a dual-layer article revision system.
Analyze the inner cycle trace and produce actionable pipeline improvements.
Output valid JSON only — no markdown, no explanation outside the JSON."""


class OuterAnalyzer:
    """
    Calls DeepSeek to analyze an inner cycle trace and produce:
      - root cause of slow convergence
      - strategy selection from reference frameworks
      - specific config changes (prompt overrides, token budgets)
      - outer-loop lessons
    """

    def __init__(self, model: str = "deepseek-chat", api_key: str = ""):
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")

    def analyze(
        self,
        inner_summary: dict,
        outer_context: str,
        reference_doc: str,
        outer_cycle: int,
    ) -> dict:
        """Run DeepSeek meta-analysis. Returns structured dict with config changes."""
        # Use LLMClient (instance-scoped) — never mutates the module-level globals
        # that the inner loop's MiniMax stages depend on.
        client = LLMClient("deepseek", self.api_key, self.model)

        prompt = f"""
{outer_context}

## Available Optimization Strategies
{reference_doc[:3000]}

## Your Task

Analyze the inner cycle trace above and produce pipeline improvements.

Output JSON with this exact structure:
{{
  "root_cause": {{
    "primary_bottleneck_dimension": "A|B|C|D|E",
    "diagnosis": "...",
    "evidence": ["...", "..."]
  }},
  "strategy_selected": {{
    "name": "reflexion|self_refine|opro|dspy|textgrad|voyager|other",
    "rationale": "...",
    "decision_rule_applied": "..."
  }},
  "prompt_overrides": [
    {{
      "stage": "article_analysis|improvement_hypotheses|edit_planning|impact_assessment|revised_output",
      "addendum": "Additional instruction to append to this stage's prompt. Write ready-to-inject text."
    }}
  ],
  "outer_lessons": [
    {{
      "lesson_type": "failure_pattern|strategy_effectiveness|config_change",
      "stage_affected": "...",
      "summary": "...",
      "reuse_rule": "In future outer cycles, when [condition], do [action].",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- "prompt_overrides" must contain only stages that genuinely need changes
- "addendum" text must be final, ready-to-inject instructions (not descriptions of instructions)
- Include at least 2 outer_lessons
- If inner loop converged quickly (≤8 runs), focus on why it worked — positive lessons count
"""

        raw = client.call(prompt, system=OUTER_SYSTEM, max_tokens=3000)
        result = parse_json_response(raw)

        if not isinstance(result, dict) or "raw_content" in result:
            logger.warning(f"Outer analyzer returned unparseable response: {raw[:200]}")
            return {"prompt_overrides": [], "outer_lessons": [], "strategy_selected": {"name": "unknown"}}

        return result


class OuterLoopController:
    """
    Manages the full dual-layer experiment.

    Args:
        outer_state: OuterLoopState (persists across cycles)
        inner_controller: InnerLoopController (runs inner cycles)
        analyzer: OuterAnalyzer (DeepSeek meta-analysis)
        article_ids: list of article IDs to optimize (in order)
        max_outer_iterations: outer budget (default 5)
        reference_doc_path: path to reference_frameworks.md
    """

    def __init__(
        self,
        outer_state: OuterLoopState,
        inner_controller: InnerLoopController,
        analyzer: OuterAnalyzer,
        article_ids: list[str],
        max_outer_iterations: int = 5,
        reference_doc_path: Path | None = None,
    ):
        self.outer_state = outer_state
        self.inner_controller = inner_controller
        self.analyzer = analyzer
        self.article_ids = article_ids
        self.max_outer_iterations = max_outer_iterations
        self.reference_doc = ""
        if reference_doc_path and reference_doc_path.exists():
            self.reference_doc = reference_doc_path.read_text(encoding="utf-8")

    def run(self) -> dict:
        """
        Run the full dual-layer experiment.
        Returns a summary dict with per-cycle results and final outer state.
        """
        results = []

        for outer_iter in range(self.max_outer_iterations):
            self.outer_state.begin_cycle()
            cycle = self.outer_state.current_cycle
            logger.info(f"\n{'='*60}")
            logger.info(f"OUTER CYCLE {cycle}/{self.max_outer_iterations}")
            logger.info(f"{'='*60}")

            cycle_results = []

            for article_id in self.article_ids:
                logger.info(f"\n[Outer {cycle}] Article: {article_id}")

                # Run full inner cycle
                inner = self.inner_controller.run_cycle(article_id, self.outer_state)

                # Extract process signals (inner → outer)
                inner_summary = self.outer_state.extract_from_inner(
                    inner, strategy_used=self._last_strategy()
                )

                # Meta-analysis via DeepSeek
                outer_context = self.outer_state.build_outer_context(
                    inner_summary, self.reference_doc[:2000]
                )
                analysis = self.analyzer.analyze(
                    inner_summary, outer_context, self.reference_doc, cycle
                )

                # Apply config changes (prompt overrides)
                self._apply_config_changes(analysis, cycle)

                # Save outer lessons
                self._save_outer_lessons(analysis, inner_summary, cycle)

                # Log strategy
                strategy = analysis.get("strategy_selected", {}).get("name", "unknown")
                logger.info(
                    f"[Outer {cycle}] {article_id}: "
                    f"peak={inner_summary['peak_score']}/10, "
                    f"runs_to_8={inner_summary['runs_to_threshold_8'] or 'never'}, "
                    f"strategy={strategy}"
                )

                cycle_results.append({
                    "article_id": article_id,
                    "peak_score": inner_summary["peak_score"],
                    "runs_to_threshold_8": inner_summary["runs_to_threshold_8"],
                    "converged": inner_summary["converged"],
                    "strategy": strategy,
                    "prompt_overrides_added": len(analysis.get("prompt_overrides", [])),
                })

                # Reset inner state for next article (same outer cycle)
                inner.reset()

            results.append({"cycle": cycle, "articles": cycle_results})
            self._save_cycle_summary(cycle, cycle_results)

            # Check outer convergence (after ≥2 cycles)
            if self.outer_state.is_outer_converged():
                logger.info(f"\nOUTER LOOP CONVERGED at cycle {cycle}!")
                break

        return {
            "total_outer_cycles": self.outer_state.current_cycle,
            "cycle_results": results,
            "final_prompt_overrides": self.outer_state.prompt_overrides,
            "total_outer_lessons": len(self.outer_state.outer_lessons),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _last_strategy(self) -> str:
        if self.outer_state.strategy_history:
            return self.outer_state.strategy_history[-1].get("strategy", "")
        return ""

    def _apply_config_changes(self, analysis: dict, cycle: int) -> None:
        """Apply prompt_overrides from DeepSeek analysis to outer_state."""
        overrides = analysis.get("prompt_overrides", [])
        for override in overrides:
            stage = override.get("stage", "")
            addendum = override.get("addendum", "").strip()
            if not stage or not addendum:
                continue

            # Append to existing override (accumulate guidance across cycles)
            existing = self.outer_state.prompt_overrides.get(stage, "")
            separator = f"\n\n## Outer Cycle {cycle} Guidance\n"
            self.outer_state.prompt_overrides[stage] = existing + separator + addendum

            logger.info(f"[Outer {cycle}] Updated prompt override for stage: {stage}")

        # Record strategy
        strategy = analysis.get("strategy_selected", {})
        if strategy.get("name"):
            self.outer_state.add_strategy_result(
                strategy["name"],
                inner_before=None,
                inner_after=None,
                notes=strategy.get("rationale", ""),
            )

    def _save_outer_lessons(self, analysis: dict, inner_summary: dict, cycle: int) -> None:
        """Convert analysis output to OuterLesson objects and save."""
        for raw_lesson in analysis.get("outer_lessons", []):
            if not isinstance(raw_lesson, dict):
                continue
            lesson = OuterLesson(
                outer_cycle=cycle,
                lesson_type=raw_lesson.get("lesson_type", "config_change"),
                strategy_used=analysis.get("strategy_selected", {}).get("name", ""),
                summary=raw_lesson.get("summary", ""),
                reuse_rule=raw_lesson.get("reuse_rule", ""),
                confidence=float(raw_lesson.get("confidence", 0.7)),
                stage_affected=raw_lesson.get("stage_affected", ""),
            )
            self.outer_state.add_outer_lesson(lesson)

    def _save_cycle_summary(self, cycle: int, results: list[dict]) -> None:
        """Write per-cycle JSON summary to examples/."""
        path = self.outer_state._examples_dir / f"outer_cycle_{cycle:02d}_summary.json"
        summary = {
            "cycle": cycle,
            "articles": results,
            "prompt_overrides_active": list(self.outer_state.prompt_overrides.keys()),
            "total_outer_lessons": len(self.outer_state.outer_lessons),
        }
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
