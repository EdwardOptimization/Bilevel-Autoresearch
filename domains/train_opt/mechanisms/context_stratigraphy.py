"""Context stratigraphy -- tracks which parameter contexts surrounded each successful change.

In archaeology, context analysis examines not just the artifact itself, but
everything found alongside it -- the "cultural context." A bronze spearhead
means something very different when found in a warrior's grave vs a merchant's
storehouse. The same artifact has different significance depending on what
surrounded it.

Applied to hyperparameter search: a change to MATRIX_LR=0.03 that worked when
DEPTH=10 and WEIGHT_DECAY=0.1 may fail completely when DEPTH=14 and
WEIGHT_DECAY=0.15. The current system records WHAT changed and WHETHER it
helped, but not the full context -- what ALL the other parameters were when
the change succeeded.

This mechanism tracks the full "cultural context" of each successful change:
the complete parameter snapshot that surrounded it. It then identifies
context-dependent patterns:
- "Increasing MATRIX_LR worked when DEPTH<=10 but failed when DEPTH>=12"
- "WEIGHT_DECAY=0.12 was optimal in the context of ADAM_BETAS=(0.9, 0.95)"
- "WARMUP_RATIO changes only helped when combined with high MATRIX_LR"

This allows the LLM to understand that parameter values are not universally
good or bad -- their effectiveness depends on the surrounding context.
"""
from __future__ import annotations

import ast
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContextStratigraphy:
    """Tracks the full parameter context surrounding each successful change.

    For every parameter change, records both the change itself AND the complete
    snapshot of all other parameters. This enables context-dependent analysis:
    which parameter CONTEXTS made a change succeed or fail?

    The key insight: momentum tracking says "increasing MATRIX_LR helped 3/5
    times." Context stratigraphy says "increasing MATRIX_LR helped when DEPTH<=10
    and WEIGHT_DECAY<0.13, but failed when DEPTH>=12."
    """

    def __init__(self, min_observations: int = 3):
        self._min_observations = min_observations
        # For each changed param, track the context and outcome:
        # param -> list of {direction, context_snapshot, delta_bpb, iteration}
        self._context_records: dict[str, list[dict]] = defaultdict(list)

    def record(self, changes: dict, old_config: dict, val_bpb: float,
               best_bpb_before: float, status: str) -> None:
        """Record a change with its full parameter context."""
        if status == "crash":
            return

        delta_bpb = val_bpb - best_bpb_before  # negative = improvement

        # The "context" is all the OTHER parameter values that were in place
        # when this change was made
        for param, new_val in changes.items():
            old_val = old_config.get(param)
            if old_val is None:
                continue

            direction = self._detect_direction(old_val, new_val)

            # Context = all params EXCEPT the one being changed
            context = {
                k: v for k, v in old_config.items() if k != param
            }

            self._context_records[param].append({
                "direction": direction,
                "old_val": str(old_val),
                "new_val": str(new_val),
                "context": context,
                "delta_bpb": delta_bpb,
                "improved": delta_bpb < 0,
            })

    def _detect_direction(self, old_val, new_val) -> str:
        """Detect change direction."""
        try:
            old_num = float(ast.literal_eval(str(old_val)))
            new_num = float(ast.literal_eval(str(new_val)))
            if new_num > old_num:
                return "increase"
            elif new_num < old_num:
                return "decrease"
            return "same"
        except (ValueError, TypeError, SyntaxError, NameError):
            return "changed"

    def find_context_patterns(self) -> dict[str, list[str]]:
        """Identify context-dependent patterns for each parameter.

        For each parameter that has been changed multiple times with mixed
        results (some improvements, some regressions), tries to identify
        which context variables correlate with success vs failure.

        Returns dict of param -> list of pattern description strings.
        """
        patterns: dict[str, list[str]] = {}

        for param, records in self._context_records.items():
            if len(records) < self._min_observations:
                continue

            # Split into successes (improved) and failures (regressed)
            successes = [r for r in records if r["improved"]]
            failures = [r for r in records if not r["improved"]]

            if not successes or not failures:
                continue  # Need both to find discriminating context

            param_patterns = []

            # For each context variable, check if its value differs
            # systematically between successes and failures
            context_params = set()
            for r in records:
                context_params.update(r["context"].keys())

            for ctx_param in context_params:
                success_vals = []
                failure_vals = []
                for r in successes:
                    val = r["context"].get(ctx_param)
                    if val is not None:
                        try:
                            success_vals.append(
                                float(ast.literal_eval(str(val)))
                            )
                        except (ValueError, TypeError, SyntaxError, NameError):
                            pass
                for r in failures:
                    val = r["context"].get(ctx_param)
                    if val is not None:
                        try:
                            failure_vals.append(
                                float(ast.literal_eval(str(val)))
                            )
                        except (ValueError, TypeError, SyntaxError, NameError):
                            pass

                if not success_vals or not failure_vals:
                    continue

                # Compare means -- is there a systematic difference?
                avg_success = sum(success_vals) / len(success_vals)
                avg_failure = sum(failure_vals) / len(failure_vals)

                # Only report if there's a meaningful difference
                spread = max(
                    max(success_vals + failure_vals)
                    - min(success_vals + failure_vals),
                    1e-9,
                )
                if abs(avg_success - avg_failure) > spread * 0.2:
                    direction = (
                        "higher" if avg_success > avg_failure else "lower"
                    )
                    param_patterns.append(
                        f"Changes to {param} worked better when "
                        f"{ctx_param} was {direction} "
                        f"(success avg={avg_success:.4g}, "
                        f"failure avg={avg_failure:.4g})"
                    )

            if param_patterns:
                patterns[param] = param_patterns[:3]  # limit per param

        return patterns

    def get_context_text(self) -> str:
        """Generate context analysis text for the proposal prompt."""
        total_records = sum(
            len(recs) for recs in self._context_records.values()
        )
        if total_records < self._min_observations:
            return ""

        patterns = self.find_context_patterns()
        if not patterns:
            return ""

        lines = [
            "## Context Analysis "
            "(when did parameter changes succeed vs fail?)"
        ]
        lines.append(
            "These patterns show which parameter CONTEXTS made changes "
            "succeed or fail. A change that worked in one context may "
            "fail in another.\n"
        )

        for param, param_patterns in sorted(patterns.items()):
            lines.append(f"  **{param}**:")
            for pattern in param_patterns:
                lines.append(f"    - {pattern}")

        lines.append(
            "\nUse these patterns to choose changes that fit the "
            "CURRENT context of your config."
        )

        return "\n".join(lines)
