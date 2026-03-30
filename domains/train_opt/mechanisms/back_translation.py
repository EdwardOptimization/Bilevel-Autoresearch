"""Back-translation — verify the LLM's understanding by asking it to reconstruct the trace.

Translation Theory Mechanism 36:
When the LLM proposes a change, it may have misunderstood the trace history. Before
committing to a GPU run, we compare the LLM's predicted outcome direction against the
actual val_bpb result. Persistent prediction errors reveal systematic misunderstandings
(e.g., the LLM thinks increasing LR always helps but the landscape is non-monotonic).
These calibration errors are fed back into subsequent proposals so the LLM can correct
its internal model of the loss landscape.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class BackTranslation:
    """Tracks LLM prediction accuracy to expose systematic misunderstandings.

    After each iteration, compares the LLM's predicted outcome against the
    actual val_bpb result. Accumulates calibration errors per parameter and
    per direction (increase/decrease). Generates a calibration report that
    warns the LLM about its blind spots.
    """

    def __init__(self, min_samples: int = 3, error_threshold: float = 0.002):
        """
        Args:
            min_samples: Minimum observations before reporting a calibration error.
            error_threshold: Minimum average |error| to flag a parameter as miscalibrated.
        """
        self._min_samples = min_samples
        self._error_threshold = error_threshold

        # param -> list of {predicted, hypothesis, actual_delta}
        # predicted: "lower" | "higher" | "uncertain"
        # actual_delta: val_bpb - best_before (negative = improvement)
        self._predictions: dict[str, list[dict]] = defaultdict(list)

        # Overall prediction record
        self._overall: list[dict] = []

    def record_prediction(self, changes: dict, expected_direction: str,
                          hypothesis: str) -> None:
        """Record the LLM's prediction before training runs.

        Called after proposal parsing, before the GPU run.

        Args:
            changes: Parameter changes dict from the proposal.
            expected_direction: "lower" | "higher" | "uncertain" from LLM.
            hypothesis: The LLM's reasoning string.
        """
        for param in changes:
            self._predictions[param].append({
                "predicted": expected_direction,
                "hypothesis": hypothesis,
                "actual_delta": None,  # filled in by record_outcome
            })
        self._overall.append({
            "predicted": expected_direction,
            "changes": list(changes.keys()),
            "hypothesis": hypothesis,
            "actual_delta": None,
        })

    def record_outcome(self, changes: dict, val_bpb: float,
                       best_bpb_before: float, status: str) -> None:
        """Record the actual outcome after training completes.

        Args:
            changes: The parameter changes that were applied.
            val_bpb: The observed validation bpb.
            best_bpb_before: The best bpb before this run.
            status: "keep" | "discard" | "crash".
        """
        if status == "crash":
            return

        actual_delta = val_bpb - best_bpb_before  # negative = improvement

        # Fill in the actual_delta for the most recent pending prediction per param
        for param in changes:
            if param in self._predictions:
                for entry in reversed(self._predictions[param]):
                    if entry["actual_delta"] is None:
                        entry["actual_delta"] = actual_delta
                        break

        # Fill in overall record
        for entry in reversed(self._overall):
            if entry["actual_delta"] is None:
                entry["actual_delta"] = actual_delta
                break

    def get_calibration_text(self) -> str:
        """Generate a calibration report for the proposal prompt.

        Identifies parameters where the LLM's predictions systematically
        diverge from actual outcomes.
        """
        param_errors: dict[str, list[tuple[str, float]]] = defaultdict(list)

        for param, entries in self._predictions.items():
            for entry in entries:
                if entry["actual_delta"] is None:
                    continue
                param_errors[param].append(
                    (entry["predicted"], entry["actual_delta"])
                )

        if not param_errors:
            return ""

        lines = ["## Back-Translation Calibration (prediction accuracy)"]
        lines.append(
            "Comparison of your predicted outcome vs actual result for each parameter. "
            "Use this to correct systematic biases in your reasoning.\n"
        )

        has_findings = False

        for param, errors in sorted(param_errors.items()):
            if len(errors) < self._min_samples:
                continue

            correct = 0
            wrong = 0
            overconfident_lower = 0
            overconfident_higher = 0

            for predicted, actual_delta in errors:
                if predicted == "lower" and actual_delta < 0:
                    correct += 1
                elif predicted == "higher" and actual_delta > 0:
                    correct += 1
                elif predicted == "uncertain":
                    correct += 1
                elif predicted == "lower" and actual_delta >= 0:
                    wrong += 1
                    overconfident_lower += 1
                elif predicted == "higher" and actual_delta <= 0:
                    wrong += 1
                    overconfident_higher += 1
                else:
                    wrong += 1

            total = correct + wrong
            if total == 0:
                continue

            accuracy = correct / total
            avg_abs_error = sum(abs(d) for _, d in errors) / len(errors)

            if accuracy < 0.6 or avg_abs_error > self._error_threshold:
                has_findings = True
                warning_parts = []
                if overconfident_lower > 0:
                    warning_parts.append(
                        f"predicted 'lower' but got WORSE {overconfident_lower}/{total} times"
                    )
                if overconfident_higher > 0:
                    warning_parts.append(
                        f"predicted 'higher' but got BETTER {overconfident_higher}/{total} times"
                    )
                lines.append(
                    f"- **{param}**: prediction accuracy={accuracy:.0%} "
                    f"({correct}/{total} correct). "
                    + "; ".join(warning_parts)
                )

        # Overall accuracy
        completed = [e for e in self._overall if e["actual_delta"] is not None]
        if len(completed) >= self._min_samples:
            overall_correct = sum(
                1 for e in completed
                if (e["predicted"] == "lower" and e["actual_delta"] < 0)
                or (e["predicted"] == "higher" and e["actual_delta"] > 0)
                or e["predicted"] == "uncertain"
            )
            overall_acc = overall_correct / len(completed)
            if overall_acc < 0.5:
                has_findings = True
                lines.append(
                    f"\nOVERALL prediction accuracy: {overall_acc:.0%} "
                    f"({overall_correct}/{len(completed)}). "
                    "Your internal model of the loss landscape may be significantly "
                    "miscalibrated. Consider making smaller, more cautious changes "
                    "and updating your assumptions."
                )

        if not has_findings:
            return ""

        lines.append(
            "\nWhen your predictions are systematically wrong for a parameter, your "
            "mental model of how that parameter affects training is likely incorrect. "
            "Consider the OPPOSITE of what you would normally predict."
        )
        return "\n".join(lines)
