"""Search configuration for the training optimization inner loop.

The outer loop modifies this config to change HOW the inner loop searches,
not WHAT it searches for. This is the Level 1.5 control surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    """Controls the inner loop's search behavior. Modified by outer loop."""

    # Which hyperparameters the LLM is allowed to change
    editable_params: list[str] = field(default_factory=lambda: [
        "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN",
        "TOTAL_BATCH_SIZE", "EMBEDDING_LR", "UNEMBEDDING_LR",
        "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY", "ADAM_BETAS",
        "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC",
        "DEPTH", "DEVICE_BATCH_SIZE",
    ])

    # Parameters the outer loop has frozen (inner loop must not touch)
    frozen_params: list[str] = field(default_factory=list)

    # Search strategy description (injected into proposal prompt)
    strategy: str = "explore"  # e.g. "explore", "exploit", "focused_lr", "architecture_search"

    # Outer loop guidance text (injected into the LLM's proposal prompt)
    guidance: str = ""

    # How many inner iterations before outer loop intervenes
    inner_budget: int = 5

    # Training time budget per run in seconds (can be shortened for exploration)
    time_budget: int = 300

    @property
    def active_params(self) -> list[str]:
        """Parameters currently available for search."""
        return [p for p in self.editable_params if p not in self.frozen_params]


# The hyperparameter block in train.py that we parse and modify
HYPERPARAM_NAMES = {
    "ASPECT_RATIO": int,
    "HEAD_DIM": int,
    "WINDOW_PATTERN": str,
    "TOTAL_BATCH_SIZE": str,  # expressions like 2**19
    "EMBEDDING_LR": float,
    "UNEMBEDDING_LR": float,
    "MATRIX_LR": float,
    "SCALAR_LR": float,
    "WEIGHT_DECAY": float,
    "ADAM_BETAS": str,  # tuple literal
    "WARMUP_RATIO": float,
    "WARMDOWN_RATIO": float,
    "FINAL_LR_FRAC": float,
    "DEPTH": int,
    "DEVICE_BATCH_SIZE": int,
}
