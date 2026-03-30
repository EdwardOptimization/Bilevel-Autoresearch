"""Plateau detector — detects when the search is stuck with diminishing returns."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PlateauDetector:
    """Detects when the search is stuck on the same parameter set with diminishing returns.

    Tracks recent proposals and identifies when:
    1. The last N iterations all changed the same parameters
    2. The improvements are getting smaller (diminishing returns)

    When a plateau is detected, generates a diversification directive that forces
    the LLM to explore different parameters.
    """

    def __init__(self, window: int = 4, min_improvement_threshold: float = 0.0005):
        self._window = window
        self._min_improvement_threshold = min_improvement_threshold
        # Recent history: (params_changed, delta_bpb, status)
        self._recent: list[tuple[set[str], float, str]] = []
        self._diversification_active = False
        self._recently_plateaued_params: set[str] = set()

    def record(self, changes: dict, val_bpb: float, best_bpb_before: float, status: str) -> None:
        """Record a proposal outcome."""
        if status == "crash":
            return  # Crash results have val_bpb=0 which would corrupt delta
        params = set(changes.keys())
        delta = val_bpb - best_bpb_before  # negative = improvement
        self._recent.append((params, delta, status))
        # Keep window bounded
        if len(self._recent) > self._window * 2:
            self._recent = self._recent[-self._window * 2:]

    def check_plateau(self) -> tuple[bool, str]:
        """Check if we're on a plateau. Returns (is_plateau, directive_text).

        A plateau is detected when the last `window` non-crash iterations:
        1. All touched overlapping parameter sets, AND
        2. None of them improved by more than min_improvement_threshold
        """
        if len(self._recent) < self._window:
            return False, ""

        recent_valid = [(p, d, s) for p, d, s in self._recent if s != "crash"]
        if len(recent_valid) < self._window:
            return False, ""

        last_n = recent_valid[-self._window:]

        # Check if all recent iterations touched overlapping params
        all_params = [p for p, _, _ in last_n]
        common_params = all_params[0]
        for p in all_params[1:]:
            common_params = common_params & p
            if not common_params:
                break

        # Also check union — if all are subsets of a small set
        union_params = set()
        for p in all_params:
            union_params |= p

        # Plateau condition 1: same small param set being tweaked
        params_are_repetitive = len(union_params) <= 3 and len(common_params) >= 1

        # Plateau condition 2: diminishing or no improvements
        improvements = [-d for _, d, _ in last_n if d < 0]
        no_significant_improvement = (
            len(improvements) == 0 or
            max(improvements) < self._min_improvement_threshold
        )

        if params_are_repetitive and no_significant_improvement:
            self._diversification_active = True
            self._recently_plateaued_params = union_params
            directive = (
                f"\n## DIVERSIFICATION REQUIRED (plateau detected)\n"
                f"The last {self._window} iterations all tweaked the same parameters "
                f"({', '.join(sorted(union_params))}) with diminishing returns.\n"
                f"You MUST propose changes to DIFFERENT parameters this time.\n"
                f"DO NOT change: {', '.join(sorted(union_params))}.\n"
                f"Instead, try architectural params (DEPTH, ASPECT_RATIO, HEAD_DIM), "
                f"batch size, or schedule params (WARMUP_RATIO, WARMDOWN_RATIO).\n"
                f"Bold moves are encouraged — try something fundamentally different."
            )
            return True, directive

        self._diversification_active = False
        return False, ""
