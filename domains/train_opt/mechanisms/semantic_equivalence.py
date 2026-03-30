"""Semantic equivalence — detect when different wordings produce the same parameter change.

Translation Theory Mechanism 37:
The LLM can phrase the same idea many ways: "increase LR from 0.04 to 0.06",
"boost learning rate slightly", "try a higher LR". These are semantically
equivalent in effect (same parameter, same direction, similar magnitude) but
the system treats each as a novel proposal. This wastes iterations re-testing
the same region. The mechanism canonicalizes proposals by their actual effect
(param, direction, magnitude_bin) and detects when the LLM is proposing
something it has already tried under different wording. It warns the LLM
and suggests genuinely novel alternatives.
"""
from __future__ import annotations

import ast
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SemanticEquivalence:
    """Detects semantically equivalent proposals that differ only in wording.

    Canonicalizes each proposal to (param, direction, magnitude_bucket) and
    tracks how many times each canonical form has been tried. When the LLM
    proposes something already tested under a different hypothesis, it warns
    the LLM and reports the outcomes of prior equivalent attempts.
    """

    # Magnitude buckets: tiny (<5%), small (5-15%), medium (15-40%), large (>40%)
    _MAGNITUDE_BINS = [
        (0.05, "tiny"),
        (0.15, "small"),
        (0.40, "medium"),
        (float("inf"), "large"),
    ]

    def __init__(self, max_history: int = 50):
        """
        Args:
            max_history: Maximum number of canonical proposals to remember.
        """
        self._max_history = max_history

        # canonical_key -> list of {iteration, hypothesis, actual_delta, status}
        # canonical_key: (param, direction, magnitude_bin)
        self._canonical_history: dict[tuple, list[dict]] = defaultdict(list)

        # Track all raw proposals for deduplication
        self._raw_proposals: list[dict] = []

    def _magnitude_bin(self, old_val: str, new_val) -> str:
        """Classify the relative magnitude of a parameter change."""
        try:
            old_num = float(ast.literal_eval(str(old_val)))
            new_num = float(ast.literal_eval(str(new_val)))
            if old_num == 0:
                return "large"
            relative_change = abs(new_num - old_num) / abs(old_num)
            for threshold, label in self._MAGNITUDE_BINS:
                if relative_change < threshold:
                    return label
            return "large"
        except (ValueError, TypeError, SyntaxError, NameError):
            return "unknown"

    def _direction(self, old_val: str, new_val) -> str:
        """Determine direction of change."""
        try:
            old_num = float(ast.literal_eval(str(old_val)))
            new_num = float(ast.literal_eval(str(new_val)))
            if new_num > old_num:
                return "increase"
            elif new_num < old_num:
                return "decrease"
            return "same"
        except (ValueError, TypeError, SyntaxError, NameError):
            return "categorical"

    def canonicalize(self, changes: dict, old_config: dict) -> list[tuple]:
        """Convert a proposal to a list of canonical (param, direction, magnitude) keys."""
        keys = []
        for param, new_val in changes.items():
            old_val = old_config.get(param, "")
            direction = self._direction(old_val, new_val)
            mag_bin = self._magnitude_bin(old_val, new_val)
            keys.append((param, direction, mag_bin))
        return sorted(keys)

    def check_duplicate(self, changes: dict, old_config: dict) -> list[dict]:
        """Check if a proposal is semantically equivalent to a previous one.

        Returns a list of prior equivalent attempts with their outcomes.
        Returns empty list if this is genuinely novel.
        """
        canonical_keys = self.canonicalize(changes, old_config)
        duplicates = []

        for key in canonical_keys:
            if key in self._canonical_history:
                for prior in self._canonical_history[key]:
                    if prior["actual_delta"] is not None:
                        duplicates.append({
                            "canonical": key,
                            "prior_hypothesis": prior["hypothesis"],
                            "prior_delta": prior["actual_delta"],
                            "prior_status": prior["status"],
                            "prior_iteration": prior["iteration"],
                        })
        return duplicates

    def record(self, changes: dict, old_config: dict, iteration: int,
               hypothesis: str, val_bpb: float, best_bpb_before: float,
               status: str) -> None:
        """Record a proposal and its outcome for future deduplication.

        Args:
            changes: Parameter changes from the proposal.
            old_config: Config before changes were applied.
            iteration: Current iteration number.
            hypothesis: The LLM's reasoning string.
            val_bpb: Observed validation bpb.
            best_bpb_before: Best bpb before this run.
            status: "keep" | "discard" | "crash".
        """
        actual_delta = None if status == "crash" else (val_bpb - best_bpb_before)

        canonical_keys = self.canonicalize(changes, old_config)
        for key in canonical_keys:
            self._canonical_history[key].append({
                "iteration": iteration,
                "hypothesis": hypothesis,
                "actual_delta": actual_delta,
                "status": status,
            })

        self._raw_proposals.append({
            "iteration": iteration,
            "changes": changes,
            "canonical": canonical_keys,
            "hypothesis": hypothesis,
        })

        # Trim history if too large
        if len(self._raw_proposals) > self._max_history:
            self._raw_proposals = self._raw_proposals[-self._max_history:]

    def get_equivalence_text(self, changes: dict, old_config: dict) -> str:
        """Generate a warning if the proposed change is semantically equivalent
        to something already tried.

        Call this BEFORE committing to a GPU run (during candidate evaluation).
        """
        duplicates = self.check_duplicate(changes, old_config)
        if not duplicates:
            return ""

        lines = ["## Semantic Equivalence Warning"]
        lines.append(
            "The proposed change is semantically equivalent to changes already tried "
            "(same parameter, same direction, same magnitude range). Consider a "
            "genuinely different parameter or direction instead.\n"
        )

        seen = set()
        for dup in duplicates:
            param, direction, mag = dup["canonical"]
            key = (param, direction, mag)
            if key in seen:
                continue
            seen.add(key)

            delta_str = (
                f"delta={dup['prior_delta']:+.6f}"
                if dup["prior_delta"] is not None
                else "crashed"
            )
            lines.append(
                f"- **{param}** {direction} ({mag}): already tried at iter "
                f"{dup['prior_iteration']} ({delta_str}, {dup['prior_status']}). "
                f"Prior reasoning: \"{dup['prior_hypothesis'][:80]}...\""
            )

        lines.append(
            "\nDo not re-test the same region under different wording. "
            "Propose a change to a DIFFERENT parameter or a DIFFERENT magnitude."
        )
        return "\n".join(lines)

    def get_redundancy_summary(self) -> str:
        """Generate a summary of which canonical regions have been over-tested.

        Injected into the proposal prompt to steer toward novelty.
        """
        if not self._canonical_history:
            return ""

        # Find canonical keys tested 2+ times
        overtested = []
        for key, entries in sorted(self._canonical_history.items()):
            completed = [e for e in entries if e["actual_delta"] is not None]
            if len(completed) >= 2:
                avg_delta = sum(e["actual_delta"] for e in completed) / len(completed)
                overtested.append((key, len(completed), avg_delta))

        if not overtested:
            return ""

        lines = ["## Redundancy Map (over-tested regions)"]
        lines.append(
            "These parameter-direction-magnitude combinations have been tested "
            "multiple times. Their effect is well-characterized. Focus on UNTESTED "
            "combinations instead.\n"
        )

        for (param, direction, mag), count, avg_delta in overtested:
            effect = "helped" if avg_delta < 0 else "hurt"
            lines.append(
                f"- {param} {direction} ({mag}): tested {count}x, avg effect {effect} "
                f"(avg delta={avg_delta:+.6f}) — SKIP, already well-known"
            )

        return "\n".join(lines)
