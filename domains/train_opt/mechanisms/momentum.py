"""Momentum tracker — tracks which parameter change directions led to improvements."""
from __future__ import annotations

import ast
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MomentumTracker:
    """Tracks which parameter change directions have led to improvements.

    When a 'keep' result is recorded, it notes which parameters were changed
    and in which direction (increase/decrease). When a 'discard' happens,
    it records the opposite signal. This gives the LLM structured feedback
    about what directions are working in the search space.
    """

    def __init__(self):
        # param -> list of (direction, delta_bpb) tuples
        # direction: "increase" or "decrease"
        # delta_bpb: negative = improvement (lower bpb is better)
        self._signals: dict[str, list[tuple[str, float]]] = defaultdict(list)

    def record(self, changes: dict, old_config: dict, val_bpb: float,
               best_bpb_before: float, status: str) -> None:
        """Record the outcome of a parameter change."""
        if status == "crash":
            return  # Crash memory handles crashes separately

        delta_bpb = val_bpb - best_bpb_before  # negative = improvement

        for param, new_val in changes.items():
            old_val = old_config.get(param)
            if old_val is None:
                continue

            # Try to determine direction (increase/decrease) for numeric params
            direction = self._detect_direction(param, old_val, new_val)
            if direction:
                self._signals[param].append((direction, delta_bpb))
                logger.info(
                    f"[Momentum] {param}: {direction} -> delta_bpb={delta_bpb:+.6f} "
                    f"({'improvement' if delta_bpb < 0 else 'regression'})"
                )

    def _detect_direction(self, param: str, old_val: str, new_val) -> str | None:
        """Detect whether a parameter was increased or decreased."""
        try:
            old_num = float(ast.literal_eval(str(old_val)))
            new_num = float(ast.literal_eval(str(new_val)))
            if new_num > old_num:
                return "increase"
            elif new_num < old_num:
                return "decrease"
            return None
        except (ValueError, TypeError, SyntaxError, NameError):
            return None  # Non-numeric param (e.g. WINDOW_PATTERN)

    def get_momentum_text(self) -> str:
        """Generate a momentum summary to inject into the proposal prompt.

        Improvement 18: Includes confidence indicators based on sample size and
        consistency, so the LLM doesn't over-react to noisy early data.
        """
        if not self._signals:
            return ""

        lines = ["## Momentum Signals (what has worked / not worked)"]
        lines.append(
            "Based on past experiments, here are patterns about which parameter "
            "change directions led to improvements (lower bpb) vs regressions.\n"
            "Confidence: HIGH = 3+ consistent trials, MEDIUM = 2+ trials, LOW = 1 trial (tentative)."
        )

        for param, signals in sorted(self._signals.items()):
            if not signals:
                continue

            # Summarize by direction
            inc_signals = [s for s in signals if s[0] == "increase"]
            dec_signals = [s for s in signals if s[0] == "decrease"]

            parts = []
            if inc_signals:
                avg_delta = sum(d for _, d in inc_signals) / len(inc_signals)
                outcome = "helped" if avg_delta < 0 else "hurt"
                n = len(inc_signals)
                # Confidence based on sample size and consistency
                if n >= 3 and all((d < 0) == (avg_delta < 0) for _, d in inc_signals):
                    confidence = "HIGH"
                elif n >= 2:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
                parts.append(
                    f"increasing {outcome} [{confidence} confidence] ({n} trial(s), "
                    f"avg delta={avg_delta:+.6f})"
                )
            if dec_signals:
                avg_delta = sum(d for _, d in dec_signals) / len(dec_signals)
                outcome = "helped" if avg_delta < 0 else "hurt"
                n = len(dec_signals)
                if n >= 3 and all((d < 0) == (avg_delta < 0) for _, d in dec_signals):
                    confidence = "HIGH"
                elif n >= 2:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"
                parts.append(
                    f"decreasing {outcome} [{confidence} confidence] ({n} trial(s), "
                    f"avg delta={avg_delta:+.6f})"
                )

            if parts:
                lines.append(f"- **{param}**: {'; '.join(parts)}")

        if len(lines) <= 2:
            return ""  # No useful signals yet

        lines.append(
            "\nUse HIGH-confidence signals to strongly bias your proposals. "
            "Treat LOW-confidence signals as tentative — they may be noise."
        )
        return "\n".join(lines)
