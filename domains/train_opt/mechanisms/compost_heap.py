"""Compost heap — extracts value from failed experiments as nutrients for future proposals.

In permaculture, composting transforms waste (dead plants, food scraps) into rich
soil amendments. Nothing is truly waste — every failed crop contains nutrients that
can feed the next planting.

In this system, discarded and crashed experiments are currently thrown away. The LLM
sees a brief trace summary ("discard" / "crash") but never learns WHY something failed
in structured form. The compost heap collects failed experiments and distills them into
"anti-patterns" — structured lessons about what NOT to do and what regions of the
parameter space are barren.

Key insight: a failed experiment that moved MATRIX_LR from 0.04 to 0.08 and got
+0.005 bpb regression tells us that the gradient from 0.04 toward 0.08 is harmful.
This is valuable information that should constrain future proposals, not just
"MATRIX_LR increase: hurt" (which is all momentum tracks).

The compost heap maintains:
1. Barren zones — value ranges that consistently produce regressions per parameter
2. Failure frequency — which params are most often involved in regressions
3. Nutrient extraction — from failures, compute which parameters were probably to blame
   when multiple params changed simultaneously
"""
from __future__ import annotations

import ast
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class CompostHeap:
    """Transforms failed experiments into structured guidance for future proposals.

    Unlike CrashMemory (which tracks crashes) or Momentum (which tracks directions),
    the compost heap builds a spatial model of "barren zones" — regions of the
    parameter space that have been tested and found wanting. This prevents the LLM
    from revisiting known-bad territory.
    """

    def __init__(self, min_failures_for_zone: int = 2):
        self._min_failures_for_zone = min_failures_for_zone

        # param -> list of (old_value, new_value, delta_bpb)
        # Only records failures (delta_bpb > 0, i.e., regression)
        self._failure_records: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

        # param -> list of (new_value, delta_bpb) from multi-param failures
        # Used for blame attribution when multiple params change
        self._multi_param_failures: dict[str, list[tuple[float, float]]] = defaultdict(list)

        # Distilled barren zones: param -> list of (lo, hi, avg_regression)
        self._barren_zones: dict[str, list[tuple[float, float, float]]] = {}

        # Track total decomposed count for stats
        self._total_composted = 0

    def record(self, changes: dict, old_config: dict, val_bpb: float,
               best_bpb_before: float, status: str) -> None:
        """Compost a failed experiment — extract nutrients from failure.

        Only processes failures (status == "discard" with regression, or "crash").
        Improvements are not composted — they belong to the elite pool.
        """
        if status == "crash":
            # Crashes are composted with a large synthetic regression
            delta = 0.01  # treat crashes as severe failures
        elif status == "discard":
            delta = val_bpb - best_bpb_before
            if delta <= 0:
                return  # SA-accepted worse result, not a true failure in our model
        else:
            return  # "keep" results are not waste

        self._total_composted += 1
        params_changed = list(changes.keys())

        for param, new_val in changes.items():
            old_val = old_config.get(param)
            if old_val is None:
                continue

            try:
                old_num = float(ast.literal_eval(str(old_val)))
                new_num = float(ast.literal_eval(str(new_val)))
            except (ValueError, TypeError, SyntaxError, NameError):
                continue

            if len(params_changed) == 1:
                # Solo change — clear attribution
                self._failure_records[param].append((old_num, new_num, delta))
            else:
                # Multi-param change — partial attribution
                # Record with reduced confidence (will be used for zone estimation)
                self._multi_param_failures[param].append((new_num, delta))

        # Rebuild barren zones periodically
        if self._total_composted % 3 == 0:
            self._rebuild_barren_zones()

    def _rebuild_barren_zones(self) -> None:
        """Distill failure records into barren zones per parameter.

        A barren zone is a value range where failures cluster. Computed by
        finding overlapping failure intervals grouped by direction.
        """
        self._barren_zones.clear()

        for param, records in self._failure_records.items():
            if len(records) < self._min_failures_for_zone:
                continue

            # Each failure defines a "direction of harm": from old_val toward new_val
            # Group failures by whether they went up or down
            up_failures = [(old, new, d) for old, new, d in records if new > old]
            down_failures = [(old, new, d) for old, new, d in records if new < old]

            zones = []

            # Build zone for upward failures
            if len(up_failures) >= self._min_failures_for_zone:
                new_vals = [new for _, new, _ in up_failures]
                avg_delta = sum(d for _, _, d in up_failures) / len(up_failures)
                zones.append((min(new_vals), max(new_vals), avg_delta))

            # Build zone for downward failures
            if len(down_failures) >= self._min_failures_for_zone:
                new_vals = [new for _, new, _ in down_failures]
                avg_delta = sum(d for _, _, d in down_failures) / len(down_failures)
                zones.append((min(new_vals), max(new_vals), avg_delta))

            if zones:
                self._barren_zones[param] = zones

    def get_compost_text(self) -> str:
        """Generate compost-derived guidance for the proposal prompt.

        Reports:
        1. Barren zones — value ranges to avoid
        2. Failure frequency — params often involved in regressions
        """
        if self._total_composted == 0:
            return ""

        lines = ["## Composted Failure Analysis (learn from past failures)"]
        lines.append(
            "These lessons are distilled from failed experiments. "
            "Avoid the barren zones — these value ranges have been tested and produced regressions."
        )

        # Report barren zones
        if self._barren_zones:
            lines.append("\n### Barren Zones (value ranges that consistently failed)")
            for param in sorted(self._barren_zones.keys()):
                for lo, hi, avg_reg in self._barren_zones[param]:
                    if abs(lo - hi) < 1e-9:
                        lines.append(
                            f"  - {param}: value ~{lo:.6g} caused avg regression "
                            f"of {avg_reg:+.6f} bpb"
                        )
                    else:
                        lines.append(
                            f"  - {param}: range [{lo:.6g}, {hi:.6g}] is barren "
                            f"(avg regression {avg_reg:+.6f} bpb from "
                            f"{len(self._failure_records[param])} failures)"
                        )

        # Report parameters with most failures (even without formal zones)
        failure_counts = {
            p: len(recs) for p, recs in self._failure_records.items()
        }
        multi_counts = {
            p: len(recs) for p, recs in self._multi_param_failures.items()
        }

        # Merge counts
        all_params = set(failure_counts.keys()) | set(multi_counts.keys())
        if all_params:
            # Only report params not already covered by barren zones
            uncovered = all_params - set(self._barren_zones.keys())
            if uncovered:
                lines.append("\n### Failure Frequency (parameters often involved in regressions)")
                for param in sorted(uncovered,
                                    key=lambda p: -(failure_counts.get(p, 0)
                                                    + multi_counts.get(p, 0))):
                    solo = failure_counts.get(param, 0)
                    multi = multi_counts.get(param, 0)
                    if solo + multi >= 2:
                        lines.append(
                            f"  - {param}: involved in {solo} solo failure(s) and "
                            f"{multi} multi-param failure(s)"
                        )

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)
