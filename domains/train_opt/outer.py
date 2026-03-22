"""Outer loop for training optimization (Level 1.5).

Analyzes inner loop trace and modifies the SearchConfig to change
HOW the inner loop searches — not WHAT hyperparameters to try.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.llm_client import LLMClient, parse_json_response

from .runner import TrainRunner

logger = logging.getLogger(__name__)

OUTER_SYSTEM = """You are a meta-optimizer analyzing an ML training experiment loop.
Your job is NOT to propose hyperparameter changes — that is the inner loop's job.
Your job is to optimize HOW the inner loop searches: which parameters to focus on,
which to freeze, what strategy to use, what guidance to give.
Return valid JSON only."""

OUTER_PROMPT = """\
## Inner Loop Trace (last {n_inner} iterations)
{trace_summary}

## Current Search Configuration
Strategy: {strategy}
Active parameters: {active_params}
Frozen parameters: {frozen_params}
Current guidance: {guidance}

## Full Trace Statistics
Total iterations: {total_iters}
Improvements found: {n_keeps}
Crashes: {n_crashes}
Best val_bpb: {best_bpb:.6f} (iteration {best_iter})

## Parameter Change History
{change_history}

## Your Task
Analyze the inner loop's search behavior and optimize its configuration.

Consider:
1. Which parameters have led to improvements? Focus the search there.
2. Which parameters have been tried repeatedly with no gain? Freeze them.
3. Is the search too broad (exploring everything) or too narrow (stuck)?
4. Should the strategy change (explore → exploit, or vice versa)?
5. What specific guidance would help the inner loop make better proposals?

Return JSON:
{{
  "diagnosis": "What is the inner loop doing wrong or right?",
  "strategy": "explore|exploit|focused",
  "freeze_params": ["PARAM1", "PARAM2"],
  "unfreeze_params": ["PARAM3"],
  "guidance": "Specific instruction to inject into the inner loop's proposal prompt. Be concrete.",
  "reasoning": "Why these changes to the search configuration"
}}

Rules:
- Do NOT propose specific hyperparameter values — that is the inner loop's job
- Focus on the PROCESS: what to search, how aggressively, in what order
- If the inner loop has been improving steadily, don't change much
- If the inner loop is stuck, make a significant strategy shift
"""


class TrainOuterLoop:
    """Outer loop that modifies the inner loop's search configuration."""

    def __init__(
        self,
        runner: TrainRunner,
        llm_client: LLMClient,
        max_outer_cycles: int = 3,
        artifacts_dir: Path | None = None,
    ):
        self.runner = runner
        self.client = llm_client
        self.max_outer_cycles = max_outer_cycles
        self.artifacts_dir = artifacts_dir or (runner.artifacts_dir / "outer")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.outer_trace: list[dict] = []

    def run(self) -> dict:
        """Run the full bilevel experiment."""
        # Baseline
        self.runner.run_baseline()
        inner_budget = self.runner.search_config.inner_budget

        for outer_cycle in range(self.max_outer_cycles):
            cycle_dir = self.artifacts_dir / f"cycle_{outer_cycle:02d}"
            cycle_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"\n{'='*60}")
            logger.info(f"OUTER CYCLE {outer_cycle + 1}/{self.max_outer_cycles}")
            logger.info(f"Strategy: {self.runner.search_config.strategy}")
            logger.info(f"Active: {self.runner.search_config.active_params}")
            logger.info(f"Frozen: {self.runner.search_config.frozen_params}")
            logger.info(f"{'='*60}")

            # Run inner loop for inner_budget iterations
            start_iter = len(self.runner.trace.results)
            for i in range(inner_budget):
                iteration = start_iter + i
                result = self.runner.run_iteration(iteration)
                logger.info(
                    f"[Cycle {outer_cycle+1} | Iter {i+1}/{inner_budget}] "
                    f"bpb={result.val_bpb:.6f} [{result.status}] {result.description[:60]}"
                )

            # Outer loop analysis
            logger.info(f"\n[Outer {outer_cycle+1}] Analyzing inner loop trace...")
            analysis = self._analyze(outer_cycle)
            (cycle_dir / "analysis.json").write_text(
                json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # Apply outer loop changes to search config
            self._apply_analysis(analysis, outer_cycle)
            self.outer_trace.append({
                "cycle": outer_cycle,
                "analysis": analysis,
                "best_bpb": self.runner.trace.best_bpb,
                "config_after": {
                    "strategy": self.runner.search_config.strategy,
                    "frozen": self.runner.search_config.frozen_params,
                    "guidance": self.runner.search_config.guidance[:200],
                },
            })

            # Save outer trace
            (self.artifacts_dir / "outer_trace.json").write_text(
                json.dumps(self.outer_trace, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        return self._build_report()

    def _analyze(self, cycle: int) -> dict:
        """Ask outer LLM to analyze inner trace and propose search config changes."""
        trace = self.runner.trace
        config = self.runner.search_config

        # Build change history
        change_counts: dict[str, dict] = {}
        for r in trace.results:
            for param in r.changes:
                if param not in change_counts:
                    change_counts[param] = {"tried": 0, "kept": 0}
                change_counts[param]["tried"] += 1
                if r.status == "keep":
                    change_counts[param]["kept"] += 1

        change_history = "\n".join(
            f"  {p}: tried {s['tried']}x, kept {s['kept']}x"
            for p, s in sorted(change_counts.items(), key=lambda x: -x[1]["tried"])
        ) or "(no changes yet)"

        prompt = OUTER_PROMPT.format(
            n_inner=config.inner_budget,
            trace_summary=trace.summary(),
            strategy=config.strategy,
            active_params=", ".join(config.active_params),
            frozen_params=", ".join(config.frozen_params) or "(none)",
            guidance=config.guidance or "(none)",
            total_iters=len(trace.results),
            n_keeps=sum(1 for r in trace.results if r.status == "keep"),
            n_crashes=sum(1 for r in trace.results if r.status == "crash"),
            best_bpb=trace.best_bpb,
            best_iter=trace.best_iteration,
            change_history=change_history,
        )

        raw = self.client.call(prompt, system=OUTER_SYSTEM, max_tokens=1500)
        result = parse_json_response(raw)

        if not isinstance(result, dict) or "raw_content" in result:
            logger.warning(f"[Outer {cycle}] Failed to parse analysis: {raw[:200]}")
            return {"diagnosis": "parse failure", "strategy": config.strategy,
                    "freeze_params": [], "unfreeze_params": [], "guidance": config.guidance,
                    "reasoning": ""}

        return result

    def _apply_analysis(self, analysis: dict, cycle: int) -> None:
        """Apply outer loop analysis to inner loop's search config."""
        config = self.runner.search_config

        # Update strategy
        new_strategy = analysis.get("strategy", config.strategy)
        if new_strategy in ("explore", "exploit", "focused"):
            old = config.strategy
            config.strategy = new_strategy
            if old != new_strategy:
                logger.info(f"[Outer] Strategy: {old} → {new_strategy}")

        # Freeze params (Improvement 17: limit to max 5 new freezes per cycle,
        # and ensure at least 4 params remain active)
        MAX_FREEZE_PER_CYCLE = 5
        MIN_ACTIVE_PARAMS = 4
        freeze_count = 0
        for p in analysis.get("freeze_params", []):
            if freeze_count >= MAX_FREEZE_PER_CYCLE:
                logger.info(f"[Outer] Freeze limit reached ({MAX_FREEZE_PER_CYCLE}), skipping freeze of {p}")
                break
            # Ensure minimum active params remain
            current_active = len(config.active_params)
            if current_active <= MIN_ACTIVE_PARAMS:
                logger.info(f"[Outer] Only {current_active} active params remain, skipping freeze of {p}")
                break
            if p in config.editable_params and p not in config.frozen_params:
                config.frozen_params.append(p)
                freeze_count += 1
                logger.info(f"[Outer] Frozen: {p}")

        # Unfreeze params
        for p in analysis.get("unfreeze_params", []):
            if p in config.frozen_params:
                config.frozen_params.remove(p)
                logger.info(f"[Outer] Unfrozen: {p}")

        # Update guidance
        new_guidance = analysis.get("guidance", "")
        if new_guidance:
            config.guidance = f"## Outer Loop Guidance (Cycle {cycle + 1})\n{new_guidance}"
            logger.info(f"[Outer] New guidance: {new_guidance[:100]}...")

    def _build_report(self) -> dict:
        """Build final experiment report."""
        trace = self.runner.trace
        return {
            "total_iterations": len(trace.results),
            "best_val_bpb": trace.best_bpb,
            "best_iteration": trace.best_iteration,
            "baseline_bpb": trace.results[0].val_bpb if trace.results else None,
            "improvement": (trace.results[0].val_bpb - trace.best_bpb) if trace.results else 0,
            "keeps": sum(1 for r in trace.results if r.status == "keep"),
            "discards": sum(1 for r in trace.results if r.status == "discard"),
            "crashes": sum(1 for r in trace.results if r.status == "crash"),
            "outer_cycles": len(self.outer_trace),
            "trace": [
                {"iter": r.iteration, "bpb": r.val_bpb, "status": r.status, "desc": r.description}
                for r in trace.results
            ],
            "outer_trace": self.outer_trace,
        }
