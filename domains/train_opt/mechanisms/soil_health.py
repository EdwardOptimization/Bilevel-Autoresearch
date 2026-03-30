"""Soil health monitor — tracks the aggregate health of the search process.

In permaculture, soil health is the foundation of everything. A garden with
depleted soil will fail no matter how good the seeds are. Soil health is measured
by multiple indicators: microbial diversity, organic matter, pH, nutrient balance.
No single number captures it — you need a dashboard.

In this system, the search process has "soil health" that degrades over time:
- Diversity decays as the LLM converges on the same parameter set
- Improvement velocity slows as low-hanging fruit is picked
- The exploration/exploitation balance may drift too far in one direction
- The "microbiome" (variety of hypotheses being tested) may collapse

Currently there is no holistic health metric. PlateauDetector catches one symptom
(same params, no improvement) but misses others (e.g., all proposals are tiny LR
tweaks even when they touch different LR params — still monoculture).

The soil health monitor computes a multi-dimensional health score and injects it
into the proposal prompt as a dashboard. When health is low, it recommends
specific remediation actions.
"""
from __future__ import annotations

import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)


class SoilHealthMonitor:
    """Tracks the holistic health of the hyperparameter search process.

    Computes three health indicators (each 0.0 to 1.0, higher = healthier):

    1. Diversity Index — Shannon entropy of which parameters are being changed,
       normalized by log(num_active_params). Low diversity = monoculture.

    2. Improvement Velocity — exponentially-weighted moving average of improvement
       rate (fraction of recent iterations that improved). Low velocity = depleted soil.

    3. Exploration Breadth — fraction of active parameters that have been changed
       at least once in the last N iterations. Low breadth = only planting one crop.

    The aggregate Soil Health Score is the geometric mean of all three indicators.
    When any indicator drops below a threshold, a remediation directive is generated.
    """

    def __init__(self, lookback_window: int = 8):
        self._lookback = lookback_window
        # Recent history: list of (set_of_params_changed, was_improvement, iteration)
        self._recent: list[tuple[set[str], bool, int]] = []
        # All-time param change counts for overall diversity
        self._all_time_param_counts: Counter = Counter()
        self._total_iterations = 0

    def record(self, changes: dict, val_bpb: float, best_bpb_before: float,
               status: str, iteration: int) -> None:
        """Record an iteration outcome for health tracking."""
        if status == "crash":
            was_improvement = False
        else:
            was_improvement = val_bpb < best_bpb_before

        params = set(changes.keys())
        self._recent.append((params, was_improvement, iteration))
        self._all_time_param_counts.update(params)
        self._total_iterations += 1

        # Keep bounded
        if len(self._recent) > self._lookback * 3:
            self._recent = self._recent[-self._lookback * 3:]

    def compute_health(self, active_params: list[str]) -> dict:
        """Compute the three soil health indicators.

        Returns dict with keys: diversity, velocity, breadth, overall, details.
        """
        if self._total_iterations < 3:
            return {
                "diversity": 1.0, "velocity": 1.0, "breadth": 1.0,
                "overall": 1.0, "details": "Too few iterations to assess health.",
            }

        window = self._recent[-self._lookback:]
        n_active = max(len(active_params), 1)

        # --- Diversity Index ---
        # Shannon entropy of param usage in the lookback window
        param_counts_window: Counter = Counter()
        for params, _, _ in window:
            param_counts_window.update(params)

        total_changes = sum(param_counts_window.values())
        if total_changes > 0 and n_active > 1:
            entropy = 0.0
            for count in param_counts_window.values():
                p = count / total_changes
                if p > 0:
                    entropy -= p * math.log(p)
            max_entropy = math.log(n_active)
            diversity = entropy / max_entropy if max_entropy > 0 else 1.0
            diversity = min(diversity, 1.0)
        else:
            diversity = 0.0

        # --- Improvement Velocity ---
        # Exponentially-weighted fraction of improvements in lookback window
        if window:
            weights = []
            improvements = []
            for i, (_, was_imp, _) in enumerate(window):
                w = 0.8 ** (len(window) - 1 - i)  # more recent = more weight
                weights.append(w)
                improvements.append(1.0 if was_imp else 0.0)
            total_w = sum(weights)
            if total_w > 0:
                velocity = sum(w * imp for w, imp in zip(weights, improvements)) / total_w
            else:
                velocity = 0.0
        else:
            velocity = 0.0

        # --- Exploration Breadth ---
        # Fraction of active params that have been touched in the lookback window
        touched_in_window = set()
        for params, _, _ in window:
            touched_in_window.update(params)

        active_set = set(active_params)
        touched_active = touched_in_window & active_set
        breadth = len(touched_active) / n_active if n_active > 0 else 0.0

        # --- Aggregate Score (geometric mean) ---
        # Add small epsilon to avoid zero killing the product
        eps = 0.01
        overall = (
            (diversity + eps) * (velocity + eps) * (breadth + eps)
        ) ** (1.0 / 3.0)
        overall = min(overall, 1.0)

        # --- Detail string ---
        untouched = active_set - touched_in_window
        details = (
            f"Diversity={diversity:.2f} (entropy of param usage), "
            f"Velocity={velocity:.2f} (recent improvement rate), "
            f"Breadth={breadth:.2f} ({len(touched_active)}/{n_active} active params touched)"
        )
        if untouched:
            details += f". Untouched params: {', '.join(sorted(untouched))}"

        return {
            "diversity": diversity,
            "velocity": velocity,
            "breadth": breadth,
            "overall": overall,
            "details": details,
            "untouched_params": sorted(untouched) if untouched else [],
        }

    def get_health_text(self, active_params: list[str]) -> str:
        """Generate a soil health dashboard for the proposal prompt.

        When health is poor, includes specific remediation directives.
        """
        if self._total_iterations < 4:
            return ""

        health = self.compute_health(active_params)

        lines = ["## Search Health Dashboard (soil health monitor)"]

        # Traffic light indicator
        overall = health["overall"]
        if overall >= 0.4:
            status_icon = "HEALTHY"
        elif overall >= 0.2:
            status_icon = "STRESSED"
        else:
            status_icon = "DEPLETED"

        lines.append(f"Overall search health: {status_icon} ({overall:.2f})")
        lines.append(f"  {health['details']}")

        # Remediation directives for poor indicators
        remediations = []

        if health["diversity"] < 0.3:
            remediations.append(
                "LOW DIVERSITY: You are practicing monoculture — changing the same "
                "parameters repeatedly. Try a DIFFERENT parameter this iteration."
            )

        if health["velocity"] < 0.15:
            remediations.append(
                "LOW VELOCITY: Recent iterations have not produced improvements. "
                "The current search direction may be exhausted. Consider a larger "
                "step size or a completely different parameter."
            )

        if health["breadth"] < 0.3:
            untouched = health.get("untouched_params", [])
            if untouched:
                remediations.append(
                    f"LOW BREADTH: Most active parameters have not been explored recently. "
                    f"Consider trying: {', '.join(untouched[:5])}"
                )

        if remediations:
            lines.append("\n### Remediation Needed")
            for r in remediations:
                lines.append(f"  - {r}")

        return "\n".join(lines)
