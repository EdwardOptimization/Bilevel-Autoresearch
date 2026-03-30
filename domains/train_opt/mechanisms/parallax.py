"""Parallax estimation — landscape smoothness from repeated near-observations.

In astronomy, parallax measures the distance to a star by observing it from two
different vantage points (opposite sides of Earth's orbit). The apparent shift
between the two observations, combined with the known baseline distance, gives
the true distance via triangulation.

In this system, "parallax" measures how much the val_bpb landscape changes in
response to small parameter perturbations. If two configs that are very close
in parameter space produce very different val_bpb values, the landscape is
"rough" (high parallax) in that region — small changes have outsized effects.
If nearby configs produce similar bpb, the landscape is "smooth" (low parallax)
— the search can take larger steps safely.

This mechanism estimates landscape roughness per parameter by tracking how much
bpb varies per unit of parameter change. Smooth regions allow aggressive steps;
rough regions require cautious, small steps. This is more principled than the
step-size calibrator's approach of just tracking what worked — it gives a
physics-based reason for WHY certain step sizes work.

Key insight: the step-size calibrator knows WHAT step sizes worked but not WHY.
Parallax tells the search whether the landscape is smooth (take big steps) or
rough (take small steps) in each direction, giving a causal model for step-size
selection.
"""
from __future__ import annotations

import ast
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ParallaxEstimator:
    """Estimates landscape roughness per parameter from nearby observations.

    Computes a "gradient roughness" metric: how much does bpb change per unit
    of parameter change? High roughness means the landscape is jagged and
    requires small steps. Low roughness means it's smooth and allows big steps.
    """

    def __init__(self, min_observations: int = 3):
        self._min_observations = min_observations
        # param -> list of (param_value, val_bpb) pairs from solo experiments
        self._observations: dict[str, list[tuple[float, float]]] = defaultdict(list)
        # param -> estimated roughness (bpb change per 1% param change)
        self._roughness: dict[str, float] = {}

    def record(self, changes: dict, old_config: dict, val_bpb: float,
               status: str) -> None:
        """Record an observation for landscape roughness estimation.

        Only records single-parameter changes (solo experiments) because
        multi-parameter changes confound the per-parameter signal.

        Args:
            changes: param -> new_value mapping.
            old_config: config before changes.
            val_bpb: the measured val_bpb.
            status: "keep", "discard", or "crash".
        """
        if status == "crash":
            return

        if len(changes) != 1:
            return  # Only solo experiments give clean parallax signals

        param = list(changes.keys())[0]
        new_val = changes[param]

        try:
            new_num = float(ast.literal_eval(str(new_val)))
        except (ValueError, TypeError, SyntaxError, NameError):
            return

        self._observations[param].append((new_num, val_bpb))

        # Also record the old value's bpb (approximated by best_bpb)
        old_val = old_config.get(param)
        if old_val is not None:
            try:
                float(ast.literal_eval(str(old_val)))  # validate parseable
            except (ValueError, TypeError, SyntaxError, NameError):
                pass

        self._update_roughness(param)

    def _update_roughness(self, param: str) -> None:
        """Update roughness estimate for a parameter.

        Roughness = average |d(bpb)| / |d(param)| across consecutive
        observations, normalized by the parameter's scale.
        """
        obs = self._observations[param]
        if len(obs) < self._min_observations:
            return

        # Sort by parameter value
        sorted_obs = sorted(obs, key=lambda x: x[0])

        # Compute pairwise gradients between consecutive observations
        gradients = []
        for i in range(len(sorted_obs) - 1):
            p1, b1 = sorted_obs[i]
            p2, b2 = sorted_obs[i + 1]
            dp = abs(p2 - p1)
            db = abs(b2 - b1)

            if dp < 1e-12:
                continue

            # Normalize by parameter scale to get relative gradient
            param_scale = max(abs(p1), abs(p2), 1e-9)
            relative_dp = dp / param_scale  # fraction of parameter magnitude
            gradient = db / relative_dp if relative_dp > 1e-9 else 0.0
            gradients.append(gradient)

        if gradients:
            # Use median gradient as robust roughness estimate
            sorted_grads = sorted(gradients)
            self._roughness[param] = sorted_grads[len(sorted_grads) // 2]

    def get_roughness_text(self) -> str:
        """Generate a landscape roughness report for the proposal prompt.

        Tells the LLM which parameters have smooth vs rough landscapes,
        guiding step-size selection with physical intuition.
        """
        if not self._roughness:
            return ""

        lines = ["## Landscape Roughness (parallax estimation)"]
        lines.append(
            "Roughness measures how rapidly val_bpb changes per unit of parameter "
            "change. High roughness = jagged landscape (use small careful steps). "
            "Low roughness = smooth landscape (can use larger steps safely)."
        )

        # Sort by roughness (roughest first — these need the most caution)
        sorted_params = sorted(self._roughness.items(), key=lambda x: -x[1])

        for param, roughness in sorted_params:
            n_obs = len(self._observations[param])

            if roughness > 0.01:
                advice = "ROUGH landscape — use SMALL steps (<5% change)"
            elif roughness > 0.002:
                advice = "MODERATE landscape — use medium steps (5-20% change)"
            else:
                advice = "SMOOTH landscape — can use LARGE steps (>20% change)"

            lines.append(
                f"  - **{param}**: roughness={roughness:.6f} bpb/unit. "
                f"{advice} ({n_obs} observations)"
            )

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)

    def get_recommended_step(self, param: str) -> float | None:
        """Get a recommended relative step size for a parameter.

        Returns a fraction (e.g., 0.05 for 5% change) based on roughness,
        or None if insufficient data.

        The logic: if roughness is R (bpb per unit relative param change),
        and we want to target a delta of ~0.002 bpb (a meaningful improvement),
        then step ~= 0.002 / R.
        """
        if param not in self._roughness:
            return None

        roughness = self._roughness[param]
        if roughness < 1e-9:
            return 0.20  # Smooth landscape, can step big

        target_delta = 0.002  # target meaningful improvement
        step = target_delta / roughness
        # Clamp to reasonable range
        step = max(0.01, min(step, 0.50))
        return step
