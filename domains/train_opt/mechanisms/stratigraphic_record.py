"""Stratigraphic record -- layer-by-layer reconstruction of how the current config was built.

In archaeology, stratigraphy studies the layered deposits (strata) at a dig site.
Each stratum represents a distinct period of activity. By analyzing the sequence of
layers, archaeologists reconstruct the site's history -- which changes came first,
which built on others, and how the site evolved over time.

This mechanism applies the same logic to the hyperparameter search trace. Each "kept"
iteration forms a stratum -- a layer of accepted changes that built on the previous
layer. The mechanism reconstructs the full provenance chain from baseline to current
config, showing:

1. The exact sequence of accepted changes that produced the current config
2. How long each "stratum" survived before being superseded
3. Which changes were load-bearing foundations vs superficial tweaks
4. Whether the current config is built on a deep, stable foundation or a
   shallow, recent chain of changes

This gives the LLM critical context: it can see that "DEPTH was set to 10 in
stratum 2 and has been stable through 15 subsequent layers" vs "WEIGHT_DECAY was
just changed 2 iterations ago and hasn't been validated by further improvements."
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class StratigraphicRecord:
    """Reconstructs the layer-by-layer provenance of the current config.

    Each accepted change forms a "stratum" -- a layer in the geological record.
    The record tracks:
    - The sequence of strata (kept iterations) that built the current config
    - Which parameters were changed in each stratum
    - The val_bpb improvement at each stratum
    - How many iterations each stratum "survived" (remained the active config)

    This information is injected into the proposal prompt so the LLM can see
    the full evolutionary history of the current config.
    """

    def __init__(self):
        # Each stratum: {iteration, changes, val_bpb, improvement_delta,
        #                deposited_at, superseded_at}
        # deposited_at = the iteration when this stratum was laid down
        # superseded_at = the iteration when the next stratum was laid on top
        #                 (or None if current)
        self._strata: list[dict] = []
        # Current iteration counter (for calculating stratum age)
        self._current_iteration: int = 0

    def record(self, iteration: int, changes: dict, val_bpb: float,
               best_bpb_before: float, status: str) -> None:
        """Record an iteration result. Only 'keep' results form new strata."""
        self._current_iteration = iteration

        if status != "keep":
            return

        is_improvement = val_bpb < best_bpb_before
        improvement_delta = best_bpb_before - val_bpb  # positive = improvement

        # Mark the previous top stratum as superseded
        if self._strata:
            self._strata[-1]["superseded_at"] = iteration

        stratum = {
            "iteration": iteration,
            "changes": dict(changes),
            "val_bpb": val_bpb,
            "improvement_delta": improvement_delta,
            "is_true_improvement": is_improvement,
            "deposited_at": iteration,
            "superseded_at": None,  # None = still the current top layer
        }
        self._strata.append(stratum)
        logger.info(
            f"[Stratigraphy] New stratum #{len(self._strata)} deposited at "
            f"iter {iteration}: changes={list(changes.keys())}, "
            f"delta={improvement_delta:+.6f}"
        )

    def get_param_ages(self) -> dict[str, int]:
        """For each parameter, determine which stratum last changed it.

        Returns dict of param -> iterations_since_last_change.
        Higher = more ancient/stable.
        """
        ages: dict[str, int] = {}
        # Walk strata in reverse to find the most recent change for each param
        for stratum in reversed(self._strata):
            for param in stratum["changes"]:
                if param not in ages:
                    ages[param] = self._current_iteration - stratum["iteration"]
        return ages

    def get_foundation_params(self, min_age: int = 5) -> list[str]:
        """Identify 'foundation' parameters -- changed early and never modified since.

        These are the load-bearing pillars of the current config. Changing them
        is risky because the entire config was built on top of them.
        """
        ages = self.get_param_ages()
        return [
            param for param, age
            in sorted(ages.items(), key=lambda x: -x[1])
            if age >= min_age
        ]

    def get_recent_params(self, max_age: int = 2) -> list[str]:
        """Identify 'recent' parameters -- changed in the last few iterations.

        These are experimental and haven't been validated by subsequent
        improvements.
        """
        ages = self.get_param_ages()
        return [param for param, age in ages.items() if age <= max_age]

    def get_stratigraphy_text(self) -> str:
        """Generate a stratigraphic summary for the proposal prompt."""
        if not self._strata:
            return ""

        lines = [
            "## Stratigraphic Record "
            "(how the current config was built, layer by layer)"
        ]
        lines.append(
            "Each stratum below is an accepted change that built on the "
            "previous one. Parameters changed in early strata and never "
            "modified since are FOUNDATION parameters -- stable and "
            "load-bearing. Recent changes are EXPERIMENTAL -- not yet "
            "validated by further improvements.\n"
        )

        # Show the strata (limit to most recent 8 to avoid prompt bloat)
        display_strata = self._strata[-8:]
        if len(self._strata) > 8:
            lines.append(
                f"  (showing most recent 8 of {len(self._strata)} strata)\n"
            )

        for i, stratum in enumerate(display_strata):
            depth_label = len(self._strata) - len(display_strata) + i + 1
            age = self._current_iteration - stratum["iteration"]
            if stratum["superseded_at"] is not None:
                survived = stratum["superseded_at"] - stratum["deposited_at"]
                survival = f", survived {survived} iters"
            else:
                survival = ", CURRENT TOP LAYER"

            param_list = ", ".join(
                f"{p}={v}" for p, v in stratum["changes"].items()
            )
            imp_type = (
                "improvement" if stratum["is_true_improvement"]
                else "SA-accepted"
            )

            lines.append(
                f"  Stratum {depth_label} (iter {stratum['iteration']}, "
                f"age={age}{survival}): "
                f"{param_list} [{imp_type}, "
                f"delta={stratum['improvement_delta']:+.6f}]"
            )

        # Parameter age analysis
        ages = self.get_param_ages()
        if ages:
            lines.append("\n### Parameter Age Analysis")

            foundation = self.get_foundation_params(min_age=5)
            if foundation:
                lines.append(
                    f"  FOUNDATION (stable >=5 iters, risky to change): "
                    f"{', '.join(foundation)}"
                )

            recent = self.get_recent_params(max_age=2)
            if recent:
                lines.append(
                    f"  RECENT (changed <=2 iters ago, still experimental): "
                    f"{', '.join(recent)}"
                )

            # Show all params with ages
            sorted_ages = sorted(ages.items(), key=lambda x: -x[1])
            lines.append("  Age ranking: " + ", ".join(
                f"{p}({a} iters)" for p, a in sorted_ages
            ))

        # Foundation stability metric
        if len(self._strata) >= 3:
            true_improvements = sum(
                1 for s in self._strata if s["is_true_improvement"]
            )
            sa_accepts = len(self._strata) - true_improvements
            lines.append(
                f"\n  Config depth: {len(self._strata)} strata "
                f"({true_improvements} true improvements, "
                f"{sa_accepts} SA-accepted)"
            )

        return "\n".join(lines)
