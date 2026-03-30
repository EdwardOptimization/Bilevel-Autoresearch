"""Target of opportunity — interrupt the survey to follow up on surprises.

In observational astronomy, a Target of Opportunity (ToO) is an unexpected
transient event (supernova, GRB, tidal disruption event) that interrupts the
scheduled survey. When something surprising is detected, the telescope abandons
its tiling plan and points at the transient for follow-up observations.

In this system, a "transient" is an unexpectedly good result from a parameter
or direction that the momentum tracker says should NOT have worked. When the
LLM changes a parameter that has historically hurt performance — and it
IMPROVES — that's a surprise discovery worth investigating further.

This mechanism detects these surprises by comparing each result against the
momentum tracker's predictions. When a surprise is found, it generates a
"ToO alert" that tells the LLM to investigate this parameter further (take
more observations around the surprise) rather than returning to the survey.

Key insight: the current system treats every result equally. A surprising
improvement is just another "keep." But surprising improvements may indicate
that the search has found a new, better region of the landscape — this
deserves immediate follow-up, not a return to routine exploration.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TargetOfOpportunity:
    """Detects surprising improvements and generates follow-up directives.

    A "surprise" is defined as: a parameter change that contradicts the
    momentum signal (expected to hurt, but helped) AND the improvement is
    above the noise floor. When detected, the next 1-2 iterations should
    focus on this parameter to determine if it's a real discovery or a fluke.

    The ToO alert has a cooldown: after triggering, it won't trigger again
    for N iterations (to prevent the search from chasing noise).
    """

    def __init__(self, cooldown: int = 3, surprise_threshold: float = 0.001):
        self._cooldown = cooldown
        self._surprise_threshold = surprise_threshold
        # Active ToO alert: (param, direction, delta, iteration)
        self._active_too: dict | None = None
        # Iteration counter for cooldown
        self._last_trigger_iter: int = -100
        # History of surprises for reporting
        self._surprise_history: list[dict] = []

    def check_surprise(
        self,
        changes: dict,
        val_bpb: float,
        best_bpb_before: float,
        iteration: int,
        momentum_signals: dict[str, list[tuple[str, float]]],
    ) -> dict | None:
        """Check if the latest result is a surprising improvement.

        Args:
            changes: param -> new_value mapping from the proposal.
            val_bpb: the measured val_bpb.
            best_bpb_before: the best val_bpb before this iteration.
            iteration: current iteration number.
            momentum_signals: from MomentumTracker._signals — param -> [(dir, delta)].

        Returns:
            A surprise dict with 'param', 'direction', 'delta' if surprise
            detected, else None.
        """
        # Cooldown check
        if iteration - self._last_trigger_iter < self._cooldown:
            return None

        delta = val_bpb - best_bpb_before  # negative = improvement

        # Only care about genuine improvements
        if delta >= 0:
            return None

        improvement = -delta
        if improvement < self._surprise_threshold:
            return None

        # Check each changed parameter against its momentum prediction
        for param, new_val in changes.items():
            if param not in momentum_signals:
                continue

            signals = momentum_signals[param]
            if len(signals) < 2:
                continue  # Not enough history to identify a surprise

            # Determine the direction of this change
            # (We don't have old_val here, so we infer from the momentum
            #  tracker: if most recent signals for this param show "increase
            #  hurts" and the LLM just increased it successfully, that's a surprise)
            inc_signals = [d for dir_, d in signals if dir_ == "increase"]
            dec_signals = [d for dir_, d in signals if dir_ == "decrease"]

            # Check if the historical trend predicts this should have HURT
            # but it actually HELPED
            surprise = False
            direction = "unknown"

            if inc_signals and sum(inc_signals) / len(inc_signals) > 0:
                # Increasing has historically hurt. If we just got an improvement,
                # the LLM may have increased it (and been surprised).
                # We flag this as a potential surprise.
                surprise = True
                direction = "increase (historically hurt, now helped)"

            if dec_signals and sum(dec_signals) / len(dec_signals) > 0:
                # Decreasing has historically hurt
                surprise = True
                direction = "decrease (historically hurt, now helped)"

            if surprise:
                self._last_trigger_iter = iteration
                alert = {
                    "param": param,
                    "direction": direction,
                    "delta": delta,
                    "improvement": improvement,
                    "iteration": iteration,
                }
                self._active_too = alert
                self._surprise_history.append(alert)
                logger.info(
                    f"[ToO] SURPRISE detected: {param} {direction}, "
                    f"improvement={improvement:.6f} at iter {iteration}"
                )
                return alert

        return None

    def get_too_text(self) -> str:
        """Generate a Target of Opportunity alert for injection into the prompt.

        If a ToO is active (surprise detected recently), tells the LLM to
        follow up on the surprising parameter instead of returning to the
        normal survey plan.
        """
        if self._active_too is None:
            return ""

        alert = self._active_too

        lines = ["## TARGET OF OPPORTUNITY ALERT"]
        lines.append(
            f"A surprising discovery was made at iteration {alert['iteration']}: "
            f"**{alert['param']}** was changed in a direction that historically "
            f"hurt performance, but this time it IMPROVED val_bpb by "
            f"{alert['improvement']:.6f}."
        )
        lines.append(
            f"\nThis may indicate a new, better region of the search space. "
            f"**Follow up on this discovery** by:\n"
            f"1. Making a FURTHER change to {alert['param']} in the same direction\n"
            f"2. Testing nearby values of {alert['param']} to confirm this is real\n"
            f"3. DO NOT abandon this lead to go back to routine exploration yet"
        )

        if len(self._surprise_history) > 1:
            lines.append(
                f"\nPast surprises: {len(self._surprise_history)} total. "
                f"The search landscape may be more complex than the momentum "
                f"signals suggest."
            )

        return "\n".join(lines)

    def clear_alert(self) -> None:
        """Clear the active ToO alert (called after follow-up iterations)."""
        self._active_too = None
