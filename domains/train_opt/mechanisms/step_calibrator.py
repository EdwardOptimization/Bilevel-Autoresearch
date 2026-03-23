"""Step-size calibrator — tracks successful vs failed change magnitudes per parameter."""
from __future__ import annotations

import ast
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class StepSizeCalibrator:
    """Tracks successful vs failed change magnitudes per parameter.

    Builds a picture of what size changes work for each parameter, and
    generates recommendations that the LLM can use to calibrate its proposals.
    """

    def __init__(self):
        # param -> list of (relative_change_pct, was_improvement)
        self._history: dict[str, list[tuple[float, bool]]] = defaultdict(list)

    def record(self, changes: dict, old_config: dict, val_bpb: float,
               best_bpb_before: float, status: str) -> None:
        """Record the magnitude of a parameter change and its outcome."""
        if status == "crash":
            # Record crash as a failed change with magnitude info
            for param, new_val in changes.items():
                old_val = old_config.get(param)
                pct = self._compute_relative_change(old_val, new_val)
                if pct is not None:
                    self._history[param].append((pct, False))
            return

        is_improvement = val_bpb < best_bpb_before
        for param, new_val in changes.items():
            old_val = old_config.get(param)
            pct = self._compute_relative_change(old_val, new_val)
            if pct is not None:
                self._history[param].append((pct, is_improvement))

    def _compute_relative_change(self, old_val, new_val) -> float | None:
        """Compute relative change as a percentage."""
        try:
            old_num = float(ast.literal_eval(str(old_val)))
            new_num = float(ast.literal_eval(str(new_val)))
            if abs(old_num) < 1e-12:
                return None
            return abs(new_num - old_num) / abs(old_num) * 100
        except (ValueError, TypeError, SyntaxError, NameError):
            return None

    def get_step_size_text(self) -> str:
        """Generate step-size recommendations for the proposal prompt."""
        if not self._history:
            return ""

        lines = ["## Step-Size Recommendations (calibrated from past experiments)"]
        lines.append(
            "Based on past experiments, here are the change magnitudes that have "
            "worked vs failed for each parameter. Use these to calibrate your proposals."
        )

        for param in sorted(self._history.keys()):
            records = self._history[param]
            if len(records) < 2:
                continue

            good = [pct for pct, ok in records if ok]
            bad = [pct for pct, ok in records if not ok]

            parts = []
            if good:
                avg_good = sum(good) / len(good)
                min_good = min(good)
                max_good = max(good)
                parts.append(
                    f"successful changes: {min_good:.1f}%-{max_good:.1f}% "
                    f"(avg {avg_good:.1f}%, {len(good)} trial(s))"
                )
            if bad:
                avg_bad = sum(bad) / len(bad)
                parts.append(
                    f"failed changes: avg {avg_bad:.1f}% ({len(bad)} trial(s))"
                )

            if good:
                avg_good = sum(good) / len(good)
                # Recommend a range centered on successful magnitudes
                rec_lo = max(avg_good * 0.5, 0.5)
                rec_hi = avg_good * 1.5
                parts.append(f"RECOMMENDED: change by {rec_lo:.1f}%-{rec_hi:.1f}%")

            if parts:
                lines.append(f"- **{param}**: {'; '.join(parts)}")

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)
