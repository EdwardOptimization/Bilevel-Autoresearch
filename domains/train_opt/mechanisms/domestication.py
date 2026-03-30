"""Domestication vs foreignization — adapt the prompt frame to the LLM's comfort zone or force it out.

Translation Theory Mechanism 39:
In translation studies, "domestication" adapts a foreign text to the target
culture's norms, while "foreignization" preserves the source's strangeness.
Applied to LLM-driven search: the LLM has a "comfort zone" — parameter ranges
and change patterns it gravitates toward (e.g., always tweaking LR, always using
round numbers). Domestication lets the LLM think in its natural frame but risks
getting stuck. Foreignization forces the LLM to think in unfamiliar frames
(e.g., "think in terms of gradient noise ratio" or "reason about parameter
ratios instead of absolute values") to break out of ruts.

This mechanism tracks whether the LLM is stuck in comfort-zone patterns and
periodically injects foreignization frames to force novel reasoning.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# Foreignization frames: alternative ways to think about the parameter space
FOREIGNIZATION_FRAMES = [
    {
        "name": "ratio_thinking",
        "prompt": (
            "## FOREIGNIZATION: Ratio Thinking\n"
            "Instead of thinking about absolute parameter values, think about RATIOS:\n"
            "- What is the ratio of EMBEDDING_LR to MATRIX_LR? Should it be higher/lower?\n"
            "- What is WARMUP_RATIO / WARMDOWN_RATIO? What does that ratio imply?\n"
            "- What is WEIGHT_DECAY relative to the LRs? Is it proportional?\n"
            "Propose changes that adjust these RATIOS to theoretically optimal values."
        ),
    },
    {
        "name": "gradient_flow",
        "prompt": (
            "## FOREIGNIZATION: Gradient Flow Analysis\n"
            "Think about how gradients flow through the network:\n"
            "- DEPTH controls path length. More depth = more gradient attenuation.\n"
            "- LR values compensate for attenuation — deeper needs higher LR per layer.\n"
            "- WEIGHT_DECAY fights against LR — it's a damping term on the gradient.\n"
            "- ADAM_BETAS control gradient momentum — beta1 near 1.0 = heavy momentum.\n"
            "What change would improve gradient flow at the current DEPTH?"
        ),
    },
    {
        "name": "information_bottleneck",
        "prompt": (
            "## FOREIGNIZATION: Information Bottleneck\n"
            "Think about the model as an information bottleneck:\n"
            "- HEAD_DIM * num_heads = model width. Wider = more information bandwidth.\n"
            "- DEPTH = number of processing stages. Deeper = more compression.\n"
            "- TOTAL_BATCH_SIZE determines how many samples inform each gradient step.\n"
            "- The ASPECT_RATIO controls width-to-depth balance.\n"
            "What is the information bottleneck in the current config? How would you widen it?"
        ),
    },
    {
        "name": "loss_landscape_geometry",
        "prompt": (
            "## FOREIGNIZATION: Loss Landscape Geometry\n"
            "Think about the shape of the loss landscape:\n"
            "- Sharp minima (low LR, high weight decay) generalize poorly.\n"
            "- Flat minima (moderate LR, low weight decay) generalize better.\n"
            "- The warmup schedule affects which basin the optimizer falls into.\n"
            "- ADAM_BETAS control how quickly the optimizer adapts to curvature.\n"
            "What change would steer toward a FLATTER minimum?"
        ),
    },
    {
        "name": "scaling_law",
        "prompt": (
            "## FOREIGNIZATION: Scaling Law Reasoning\n"
            "Think about Chinchilla/compute-optimal scaling:\n"
            "- For a fixed compute budget (TIME_BUDGET), there's an optimal model size.\n"
            "- More DEPTH with same width = more params but same compute per token.\n"
            "- Larger TOTAL_BATCH_SIZE = fewer gradient steps in the same wall time.\n"
            "- Optimal LR scales with batch size (linear scaling rule).\n"
            "Is the current config compute-optimal? What would make it more so?"
        ),
    },
]


class Domestication:
    """Tracks LLM comfort-zone patterns and injects foreignization frames.

    Monitors which parameters the LLM gravitates toward and how often it uses
    round numbers or repeats similar patterns. When comfort-zone behavior is
    detected (same params changed N times in a row, or consistently round
    numbers), it triggers a foreignization frame to force novel reasoning.
    """

    def __init__(self, comfort_threshold: int = 3, foreignize_cooldown: int = 4):
        """
        Args:
            comfort_threshold: Number of consecutive comfort-zone proposals before
                               triggering foreignization.
            foreignize_cooldown: Minimum iterations between foreignization injections.
        """
        self._comfort_threshold = comfort_threshold
        self._foreignize_cooldown = foreignize_cooldown

        # Track which params were changed in recent proposals
        self._recent_params: list[set[str]] = []

        # Track whether values are "round" (multiples of 0.01, powers of 2, etc.)
        self._round_number_count: int = 0
        self._total_proposals: int = 0

        # Foreignization state
        self._last_foreignize_iter: int = -100
        self._foreignize_index: int = 0  # cycle through frames

        # param -> count of how many times it was the sole changed param
        self._solo_change_counts: dict[str, int] = defaultdict(int)

    def _is_round_number(self, value) -> bool:
        """Check if a value is suspiciously 'round' (LLM comfort zone)."""
        try:
            num = float(str(value))
            # Check if it's a "round" float: exactly N decimal places of 0 or 5
            # e.g., 0.05, 0.1, 0.001, 1.0 are round; 0.037, 0.128 are not
            s = f"{num:.10f}".rstrip("0")
            # Count significant digits after decimal point
            if "." in s:
                decimal_part = s.split(".")[1]
                if len(decimal_part) <= 2:
                    return True
                if decimal_part[-1] in ("5", "0"):
                    return True
            return False
        except (ValueError, TypeError):
            return False

    def record(self, changes: dict, iteration: int) -> None:
        """Record a proposal for comfort-zone analysis.

        Args:
            changes: Parameter changes from the proposal.
            iteration: Current iteration number.
        """
        params_changed = set(changes.keys())
        self._recent_params.append(params_changed)
        if len(self._recent_params) > self._comfort_threshold + 2:
            self._recent_params = self._recent_params[-(self._comfort_threshold + 2):]

        self._total_proposals += 1

        # Track round numbers
        for val in changes.values():
            if self._is_round_number(val):
                self._round_number_count += 1

        # Track solo changes
        if len(params_changed) == 1:
            param = next(iter(params_changed))
            self._solo_change_counts[param] += 1

    def _detect_comfort_zone(self) -> str | None:
        """Detect if the LLM is stuck in comfort-zone patterns.

        Returns a description of the detected pattern, or None.
        """
        if len(self._recent_params) < self._comfort_threshold:
            return None

        recent = self._recent_params[-self._comfort_threshold:]

        # Pattern 1: Same parameter changed N times in a row
        common_params = set.intersection(*recent) if recent else set()
        if common_params:
            return (
                f"same parameter(s) {common_params} changed in the last "
                f"{self._comfort_threshold} consecutive proposals"
            )

        # Pattern 2: Excessive round numbers (>80% of proposed values)
        if self._total_proposals >= self._comfort_threshold:
            total_values = sum(len(p) for p in self._recent_params)
            if total_values > 0:
                round_ratio = self._round_number_count / max(total_values, 1)
                if round_ratio > 0.8:
                    return (
                        f"round-number bias detected ({round_ratio:.0%} of proposed "
                        f"values are round numbers — try precise values like 0.037 "
                        f"instead of 0.04)"
                    )

        # Pattern 3: One parameter dominates all solo changes
        if self._solo_change_counts:
            top_param = max(self._solo_change_counts, key=self._solo_change_counts.get)
            top_count = self._solo_change_counts[top_param]
            if top_count >= self._comfort_threshold and self._total_proposals >= 5:
                ratio = top_count / self._total_proposals
                if ratio > 0.5:
                    return (
                        f"parameter fixation on {top_param} — changed alone in "
                        f"{top_count}/{self._total_proposals} proposals ({ratio:.0%})"
                    )

        return None

    def get_domestication_text(self, iteration: int) -> str:
        """Generate domestication/foreignization context for the prompt.

        Returns an empty string in normal (domestication) mode. Returns a
        foreignization frame when comfort-zone patterns are detected.
        """
        comfort_pattern = self._detect_comfort_zone()

        if comfort_pattern is None:
            return ""

        # Check cooldown
        if iteration - self._last_foreignize_iter < self._foreignize_cooldown:
            return ""

        # Trigger foreignization
        self._last_foreignize_iter = iteration
        frame = FOREIGNIZATION_FRAMES[self._foreignize_index % len(FOREIGNIZATION_FRAMES)]
        self._foreignize_index += 1

        logger.info(
            f"[Domestication] Comfort-zone detected: {comfort_pattern}. "
            f"Injecting foreignization frame: {frame['name']}"
        )

        lines = [
            "## Comfort-Zone Warning",
            f"Pattern detected: {comfort_pattern}.",
            "You are stuck in a familiar reasoning pattern. To break out, adopt the "
            "following alternative reasoning frame:\n",
            frame["prompt"],
        ]

        return "\n".join(lines)
