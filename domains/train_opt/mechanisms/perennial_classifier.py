"""Perennial classifier — distinguishes settled vs volatile parameters.

In permaculture, perennials are plants that establish once and produce for years
(fruit trees, asparagus, rhubarb). Annuals must be replanted each season (tomatoes,
lettuce, beans). A well-designed permaculture garden has a backbone of perennials
that provide stable structure, with annuals planted in the gaps.

In hyperparameter search, some parameters settle early to near-optimal values and
rarely benefit from further tuning (perennials). Others remain volatile throughout
the search, with small changes continuing to yield improvements (annuals).

Currently the system treats all parameters equally — the LLM can change any active
parameter at any time. But if DEPTH converged to 10 in iteration 3 and every
subsequent attempt to change it caused regressions, continuing to change DEPTH is
wasting iterations. The perennial classifier detects this pattern and advises the
LLM: "DEPTH is a perennial — it is settled. Focus your attention on annual params
like MATRIX_LR and WEIGHT_DECAY that still have room to improve."

This is softer than outer-loop freezing: perennial params remain technically active
(unfrozen) but the LLM is told they are settled. This allows the LLM to still
change them if there is a compelling reason, unlike frozen params which are off-limits.
"""
from __future__ import annotations

import ast
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerennialClassifier:
    """Classifies parameters as perennial (settled) vs annual (volatile).

    A parameter is classified as perennial when:
    1. It has been changed N+ times, AND
    2. Recent changes to it have NOT produced improvements, AND
    3. Its value has converged (recent values cluster tightly)

    A parameter is classified as annual when:
    1. Recent changes to it HAVE produced improvements, OR
    2. Its value has not converged (recent values are spread out)

    The classification is advisory — injected into the prompt as guidance,
    not enforced as a hard constraint.
    """

    def __init__(self, min_trials: int = 3, convergence_threshold: float = 0.05):
        self._min_trials = min_trials
        # What fraction of range constitutes "converged" — 5% means the recent
        # values span less than 5% of the parameter's observed range
        self._convergence_threshold = convergence_threshold

        # param -> list of (new_value_numeric, was_improvement, iteration)
        self._history: dict[str, list[tuple[float, bool, int]]] = defaultdict(list)

    def record(self, changes: dict, val_bpb: float, best_bpb_before: float,
               status: str, iteration: int) -> None:
        """Record a parameter change and its outcome."""
        if status == "crash":
            return

        was_improvement = val_bpb < best_bpb_before

        for param, new_val in changes.items():
            try:
                num = float(ast.literal_eval(str(new_val)))
                self._history[param].append((num, was_improvement, iteration))
            except (ValueError, TypeError, SyntaxError, NameError):
                pass  # Skip non-numeric params

    def classify(self, active_params: list[str]) -> dict[str, str]:
        """Classify each active parameter as perennial or annual.

        Returns dict of param -> "perennial" | "annual" | "unknown".
        """
        classifications = {}

        for param in active_params:
            records = self._history.get(param, [])

            if len(records) < self._min_trials:
                classifications[param] = "unknown"
                continue

            # Check recent improvement rate (last N trials)
            recent = records[-self._min_trials:]
            recent_improvements = sum(1 for _, was_imp, _ in recent if was_imp)

            # Check value convergence
            all_values = [v for v, _, _ in records]
            recent_values = [v for v, _, _ in recent]

            value_range = max(all_values) - min(all_values) if len(all_values) >= 2 else 0
            recent_range = max(recent_values) - min(recent_values) if len(recent_values) >= 2 else 0

            if value_range > 0:
                convergence_ratio = recent_range / value_range
            else:
                convergence_ratio = 0.0  # No variance at all = converged

            # Classification logic
            if recent_improvements == 0 and convergence_ratio < self._convergence_threshold:
                # No recent improvements AND value has converged = perennial
                classifications[param] = "perennial"
            elif recent_improvements >= 2:
                # Multiple recent improvements = annual (still volatile)
                classifications[param] = "annual"
            elif convergence_ratio >= self._convergence_threshold * 3:
                # Values still spread out = annual (still exploring)
                classifications[param] = "annual"
            else:
                classifications[param] = "unknown"

        return classifications

    def get_perennial_text(self, active_params: list[str]) -> str:
        """Generate perennial/annual classification for the proposal prompt."""
        if not self._history:
            return ""

        classifications = self.classify(active_params)

        perennials = [p for p, c in classifications.items() if c == "perennial"]
        annuals = [p for p, c in classifications.items() if c == "annual"]

        if not perennials and not annuals:
            return ""

        lines = ["## Parameter Maturity (perennial vs annual classification)"]
        lines.append(
            "Perennial parameters have settled to near-optimal values — "
            "further changes rarely help. Annual parameters are still volatile "
            "and benefit from continued tuning."
        )

        if perennials:
            lines.append(
                f"\n  PERENNIAL (settled — avoid changing unless you have strong reason): "
                f"{', '.join(sorted(perennials))}"
            )
        if annuals:
            lines.append(
                f"  ANNUAL (still volatile — good candidates for tuning): "
                f"{', '.join(sorted(annuals))}"
            )

        lines.append(
            "\nPrioritize annual parameters for your proposals. "
            "Changing perennials is not forbidden but is usually wasteful."
        )

        return "\n".join(lines)
