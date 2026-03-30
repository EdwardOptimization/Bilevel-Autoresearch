"""Knockout screen — systematically tests parameter importance by resetting to defaults.

In molecular genetics, a gene knockout experiment removes (knocks out) a single
gene to observe its phenotypic effect. If removing a gene causes a severe defect,
that gene is essential. If removing it has no effect, the gene is dispensable.

This mechanism applies the same logic to hyperparameters: periodically "knock out"
a parameter by resetting it to its default value (from the original train.py) and
observing the effect on val_bpb. Parameters whose knockout causes large regressions
are essential and should be protected. Parameters whose knockout has no effect are
dispensable and may be frozen to reduce search dimensionality.

The knockout screen runs passively — it suggests knockout experiments to the LLM
as one of the multi-candidates, rather than requiring dedicated iterations.
"""
from __future__ import annotations

import ast
import logging

logger = logging.getLogger(__name__)


class KnockoutScreen:
    """Systematically tests parameter importance by resetting to defaults.

    Tracks which parameters have been "knocked out" (reset to default) and
    what happened. Builds an importance ranking that can be used to:
    1. Protect essential parameters from over-aggressive outer loop freezing
    2. Identify dispensable parameters that can be safely frozen
    3. Suggest knockout experiments when the search needs diversification
    """

    def __init__(self):
        # param -> default value (from original train.py)
        self._defaults: dict[str, str] = {}
        # param -> list of (delta_bpb_when_knocked_out,)
        # delta > 0 means knockout made things worse (param is important)
        # delta < 0 means knockout made things better (param was hurting)
        self._knockout_results: dict[str, list[float]] = {}
        # Track which params have been knocked out (to avoid repeats)
        self._knocked_out: set[str] = set()
        # Track current config for comparison
        self._current_config: dict[str, str] = {}

    def set_defaults(self, default_config: dict) -> None:
        """Set the default (baseline) parameter values from original train.py."""
        self._defaults = {k: str(v) for k, v in default_config.items()}

    def set_current_config(self, config: dict) -> None:
        """Update the current best config for comparison."""
        self._current_config = {k: str(v) for k, v in config.items()}

    def record_knockout(self, param: str, delta_bpb: float) -> None:
        """Record the result of a knockout experiment.

        Args:
            param: The parameter that was reset to default.
            delta_bpb: val_bpb(knockout) - val_bpb(best). Positive = regression.
        """
        if param not in self._knockout_results:
            self._knockout_results[param] = []
        self._knockout_results[param].append(delta_bpb)
        self._knocked_out.add(param)
        importance = "ESSENTIAL" if delta_bpb > 0.001 else (
            "BENEFICIAL_REMOVAL" if delta_bpb < -0.001 else "DISPENSABLE"
        )
        logger.info(
            f"[Knockout] {param} reset to default: delta={delta_bpb:+.6f} -> {importance}"
        )

    def suggest_knockout(self, active_params: list[str]) -> dict | None:
        """Suggest a knockout experiment candidate.

        Returns a proposal dict (changes + hypothesis) that resets the
        least-tested parameter to its default value, or None if all have
        been tested or no default is available.
        """
        # Find params that differ from default and haven't been knocked out yet
        candidates = []
        for param in active_params:
            if param in self._knocked_out:
                continue
            current = self._current_config.get(param)
            default = self._defaults.get(param)
            if current is None or default is None:
                continue
            if str(current).strip() == str(default).strip():
                continue  # Already at default, no point in knockout
            candidates.append(param)

        if not candidates:
            return None

        # Pick the first untested param (deterministic, no randomness needed)
        param = candidates[0]
        default_val = self._defaults[param]

        # Try to preserve the value type
        try:
            ast.literal_eval(default_val)
            change_val = default_val
        except (ValueError, SyntaxError):
            change_val = default_val

        return {
            "changes": {param: change_val},
            "hypothesis": (
                f"KNOCKOUT TEST: Reset {param} to its default value ({default_val}) "
                f"to measure this parameter's importance. If val_bpb barely changes, "
                f"{param} can be safely frozen. If val_bpb gets much worse, {param} is essential."
            ),
            "expected_direction": "uncertain",
            "risk": "medium",
        }

    def get_importance_text(self) -> str:
        """Generate parameter importance rankings from knockout results."""
        if not self._knockout_results:
            return ""

        lines = ["## Parameter Importance (from knockout experiments)"]
        lines.append(
            "Knockout = reset parameter to its default and measure the effect. "
            "Essential parameters should be kept active. Dispensable ones can be frozen."
        )

        # Sort by importance (largest positive delta = most important)
        importance = []
        for param, deltas in self._knockout_results.items():
            avg_delta = sum(deltas) / len(deltas)
            importance.append((param, avg_delta, len(deltas)))

        importance.sort(key=lambda x: -x[1])

        for param, avg_delta, n_trials in importance:
            if avg_delta > 0.002:
                label = "ESSENTIAL"
            elif avg_delta > 0.0005:
                label = "IMPORTANT"
            elif avg_delta > -0.0005:
                label = "DISPENSABLE"
            else:
                label = "HARMFUL (default was better!)"

            lines.append(
                f"  - {param}: {label} (knockout delta={avg_delta:+.6f}, {n_trials} trial(s))"
            )

        # List untested params
        if self._defaults:
            untested = [p for p in self._defaults if p not in self._knocked_out]
            if untested:
                lines.append(f"\n  Untested: {', '.join(sorted(untested))}")

        return "\n".join(lines)

    def is_knockout_proposal(self, changes: dict) -> str | None:
        """Check if a proposal is actually a knockout (param reset to default).

        Returns the param name if this is a knockout, None otherwise.
        """
        if len(changes) != 1:
            return None
        param = list(changes.keys())[0]
        new_val = str(changes[param]).strip()
        default_val = str(self._defaults.get(param, "")).strip()
        if not default_val:
            return None
        # Check if new value matches default (handle numeric comparison)
        try:
            new_num = float(ast.literal_eval(new_val))
            default_num = float(ast.literal_eval(default_val))
            if abs(new_num - default_num) < 1e-9:
                return param
        except (ValueError, TypeError, SyntaxError, NameError):
            if new_val == default_val:
                return param
        return None
