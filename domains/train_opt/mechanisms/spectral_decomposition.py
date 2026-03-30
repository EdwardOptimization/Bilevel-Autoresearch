"""Spectral decomposition — attribute bpb improvement to individual parameters.

In astronomy, spectroscopy decomposes starlight into its constituent wavelengths
to determine the star's composition. A single "brightness" measurement (total
flux) is uninformative about WHAT is producing the light. Spectral decomposition
reveals whether the brightness comes from hydrogen, helium, iron, etc.

In this system, when multiple parameters are changed simultaneously, the val_bpb
delta is a single "brightness" measurement. We don't know which parameter
contributed what. Was it the LR change that helped, or the weight decay change?
Without spectral decomposition, the search credits (or blames) all changed
parameters equally, which corrupts the momentum and step-size signals.

This mechanism uses historical single-parameter experiments as "reference spectra"
to decompose multi-parameter results into per-parameter contributions. When
param A was changed alone and produced delta_A, and param B alone produced
delta_B, and (A+B) together produced delta_AB, we can estimate:
  contribution_A ~= delta_A / (delta_A + delta_B) * delta_AB
  contribution_B ~= delta_B / (delta_A + delta_B) * delta_AB

This gives a more accurate picture of which parameters are actually driving
improvements, which directly improves the quality of momentum signals.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SpectralDecomposition:
    """Decomposes multi-parameter bpb deltas into per-parameter attributions.

    Uses historical single-parameter experiments as calibration data to estimate
    how much each parameter contributed to a joint result.
    """

    def __init__(self, min_solo_samples: int = 1):
        self._min_solo_samples = min_solo_samples
        # param -> list of delta_bpb from solo experiments
        self._solo_effects: dict[str, list[float]] = defaultdict(list)
        # Recent decompositions for reporting
        self._decompositions: list[dict] = []

    def record_solo(self, param: str, delta_bpb: float) -> None:
        """Record a single-parameter experiment result.

        Args:
            param: the parameter that was changed alone.
            delta_bpb: val_bpb - best_bpb_before (negative = improvement).
        """
        self._solo_effects[param].append(delta_bpb)

    def record_from_changes(self, changes: dict, val_bpb: float,
                            best_bpb_before: float, status: str) -> None:
        """Record an experiment, routing to solo or multi as appropriate.

        Args:
            changes: param -> value mapping of what was changed.
            val_bpb: the measured val_bpb.
            best_bpb_before: best bpb before this iteration.
            status: "keep", "discard", or "crash".
        """
        if status == "crash":
            return

        delta = val_bpb - best_bpb_before

        if len(changes) == 1:
            param = list(changes.keys())[0]
            self.record_solo(param, delta)

    def decompose(self, changes: dict, total_delta: float) -> dict[str, float]:
        """Decompose a multi-parameter delta into per-parameter attributions.

        Uses the weighted attribution method:
          attribution_i = (avg_solo_effect_i / sum_of_all_solo_effects) * total_delta

        Falls back to equal attribution if no solo data is available.

        Args:
            changes: param -> value mapping of what was changed.
            total_delta: the total bpb delta (negative = improvement).

        Returns:
            dict mapping param -> attributed delta (negative = helped).
        """
        params = list(changes.keys())

        if len(params) <= 1:
            if params:
                return {params[0]: total_delta}
            return {}

        # Gather solo effect estimates for each param
        solo_avgs = {}
        for param in params:
            if param in self._solo_effects and len(self._solo_effects[param]) >= self._min_solo_samples:
                avg = sum(self._solo_effects[param]) / len(self._solo_effects[param])
                solo_avgs[param] = avg

        if not solo_avgs:
            # No calibration data — equal attribution
            equal_share = total_delta / len(params)
            return {p: equal_share for p in params}

        # For params without solo data, assume neutral effect (0)
        for p in params:
            if p not in solo_avgs:
                solo_avgs[p] = 0.0

        # Compute attributions using absolute solo effects as weights
        abs_total = sum(abs(v) for v in solo_avgs.values())
        if abs_total < 1e-12:
            equal_share = total_delta / len(params)
            return {p: equal_share for p in params}

        attributions = {}
        for p in params:
            weight = abs(solo_avgs[p]) / abs_total
            attributions[p] = weight * total_delta

        # Store for reporting
        self._decompositions.append({
            "params": params,
            "total_delta": total_delta,
            "attributions": attributions.copy(),
        })
        # Keep bounded
        if len(self._decompositions) > 20:
            self._decompositions = self._decompositions[-20:]

        return attributions

    def get_attribution_text(self) -> str:
        """Generate a spectral decomposition report for the proposal prompt.

        Shows which parameters have been the strongest contributors to
        improvements (brightest spectral lines) and which are neutral
        (continuum emission — present but not driving change).
        """
        if not self._solo_effects:
            return ""

        lines = ["## Spectral Analysis (per-parameter contribution estimates)"]
        lines.append(
            "Based on single-parameter experiments, here is how much each parameter "
            "typically contributes to val_bpb changes. Use this to focus on parameters "
            "that are the strongest levers, and avoid wasting iterations on parameters "
            "that have minimal individual effect."
        )

        # Rank parameters by their average solo effect magnitude
        param_impacts = []
        for param, deltas in sorted(self._solo_effects.items()):
            avg_delta = sum(deltas) / len(deltas)
            avg_abs = sum(abs(d) for d in deltas) / len(deltas)
            n = len(deltas)
            param_impacts.append((param, avg_delta, avg_abs, n))

        # Sort by absolute impact (biggest lever first)
        param_impacts.sort(key=lambda x: -x[2])

        for param, avg_delta, avg_abs, n in param_impacts:
            if avg_delta < -0.0005:
                role = "STRONG LEVER (tends to improve bpb)"
            elif avg_delta > 0.0005:
                role = "RISKY (tends to hurt bpb)"
            else:
                role = "NEUTRAL (minimal individual effect)"

            lines.append(
                f"  - **{param}**: {role} "
                f"(avg delta={avg_delta:+.6f}, magnitude={avg_abs:.6f}, "
                f"{n} solo trial(s))"
            )

        # Highlight decomposition insights if we have multi-param data
        if self._decompositions:
            recent = self._decompositions[-3:]
            lines.append("\n### Recent Multi-Parameter Decompositions")
            for dec in recent:
                attr_parts = []
                for p, a in sorted(dec["attributions"].items(), key=lambda x: x[1]):
                    attr_parts.append(f"{p}={a:+.6f}")
                lines.append(
                    f"  - Total delta={dec['total_delta']:+.6f} -> "
                    f"{', '.join(attr_parts)}"
                )

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)
