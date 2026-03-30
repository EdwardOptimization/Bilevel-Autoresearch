"""Excavation grid -- systematic spatial sampling of under-explored parameter space.

In archaeology, an excavation grid divides a site into uniform squares (units).
Archaeologists systematically excavate each unit rather than digging randomly or
only where artifacts are visible. This ensures complete coverage and prevents
important areas from being overlooked just because they lack surface indicators.

The current search is LLM-guided, which creates coverage bias: the LLM tends to
revisit parameter regions it has seen improve, creating a "rich get richer" pattern.
Parameters or ranges that were never tried (or tried only once with bad luck) get
permanently ignored -- the LLM has no signal about them, so it never proposes them.

This mechanism divides the parameter space into a grid and tracks which "cells"
have been excavated (sampled). It identifies the most under-explored regions and
periodically forces the search to sample them, ensuring systematic coverage of
the full space rather than clustering around early successes.

For numeric parameters, the grid divides the plausible range into bins.
For categorical parameters, each possible value is a cell.
"""
from __future__ import annotations

import ast
import logging
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

# Plausible ranges for each parameter (used to define grid cells)
# These are rough bounds -- the grid is for coverage tracking, not hard limits
PARAM_RANGES = {
    "MATRIX_LR": (0.005, 0.15, 5),      # (min, max, n_bins)
    "SCALAR_LR": (0.005, 0.15, 5),
    "EMBEDDING_LR": (0.005, 0.15, 5),
    "UNEMBEDDING_LR": (0.005, 0.15, 5),
    "WEIGHT_DECAY": (0.0, 0.5, 5),
    "WARMUP_RATIO": (0.0, 0.3, 5),
    "WARMDOWN_RATIO": (0.3, 1.0, 5),
    "FINAL_LR_FRAC": (0.0, 0.3, 5),
    "DEPTH": (6, 18, 4),
    "ASPECT_RATIO": (4, 20, 4),
    "HEAD_DIM": (32, 128, 4),
}


class ExcavationGrid:
    """Systematic coverage tracker for parameter space exploration.

    Divides each parameter's plausible range into bins (grid cells) and
    tracks how many times each cell has been sampled. Identifies the
    most under-explored regions and suggests targeted excavations.

    This prevents the search from clustering around early successes and
    ensures that potentially good regions are not permanently ignored
    just because they were never sampled.
    """

    def __init__(self):
        # param -> {bin_index: sample_count}
        self._grid: dict[str, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # param -> {bin_index: best_bpb_in_bin}
        self._bin_best: dict[str, dict[int, float]] = defaultdict(
            lambda: defaultdict(lambda: float("inf"))
        )
        # param -> {bin_index: list of val_bpb results}
        self._bin_results: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def _get_bin(self, param: str, value) -> int | None:
        """Map a parameter value to its grid bin index."""
        if param not in PARAM_RANGES:
            return None
        lo, hi, n_bins = PARAM_RANGES[param]
        try:
            num = float(ast.literal_eval(str(value)))
        except (ValueError, TypeError, SyntaxError, NameError):
            return None
        if hi <= lo:
            return 0
        # Clamp to range
        num = max(lo, min(hi, num))
        bin_width = (hi - lo) / n_bins
        bin_idx = int((num - lo) / bin_width)
        return min(bin_idx, n_bins - 1)

    def _get_bin_range(self, param: str, bin_idx: int) -> tuple[float, float]:
        """Get the value range for a given bin."""
        if param not in PARAM_RANGES:
            return (0.0, 0.0)
        lo, hi, n_bins = PARAM_RANGES[param]
        bin_width = (hi - lo) / n_bins
        bin_lo = lo + bin_idx * bin_width
        bin_hi = bin_lo + bin_width
        return (bin_lo, bin_hi)

    def record(self, config: dict, val_bpb: float) -> None:
        """Record a configuration's position in the grid."""
        for param, value in config.items():
            bin_idx = self._get_bin(param, value)
            if bin_idx is not None:
                self._grid[param][bin_idx] += 1
                self._bin_results[param][bin_idx].append(val_bpb)
                if val_bpb < self._bin_best[param][bin_idx]:
                    self._bin_best[param][bin_idx] = val_bpb

    def get_underexplored(
        self, active_params: list[str], n: int = 3
    ) -> list[dict]:
        """Find the most under-explored parameter-bin combinations.

        Returns a list of dicts with {param, bin_idx, bin_range,
        sample_count, suggested_value}.
        """
        candidates = []
        for param in active_params:
            if param not in PARAM_RANGES:
                continue
            lo, hi, n_bins = PARAM_RANGES[param]
            for bin_idx in range(n_bins):
                count = self._grid[param].get(bin_idx, 0)
                bin_lo, bin_hi = self._get_bin_range(param, bin_idx)
                # Suggest the midpoint of the bin
                mid = (bin_lo + bin_hi) / 2
                # Integer params need rounding
                if param in ("DEPTH", "ASPECT_RATIO", "HEAD_DIM"):
                    mid = round(mid)
                else:
                    mid = round(mid, 6)
                candidates.append({
                    "param": param,
                    "bin_idx": bin_idx,
                    "bin_range": (round(bin_lo, 4), round(bin_hi, 4)),
                    "sample_count": count,
                    "suggested_value": mid,
                })

        # Sort by sample count (least explored first), break ties randomly
        candidates.sort(key=lambda x: (x["sample_count"], random.random()))
        return candidates[:n]

    def suggest_excavation(
        self, current_config: dict, active_params: list[str]
    ) -> dict | None:
        """Suggest a systematic excavation of an under-explored grid cell.

        Returns a proposal dict with 'changes' and 'hypothesis', or None.
        """
        underexplored = self.get_underexplored(active_params, n=5)

        for cell in underexplored:
            param = cell["param"]
            value = cell["suggested_value"]

            # Check if this would actually change the current config
            current_val = current_config.get(param)
            if current_val is not None:
                try:
                    current_num = float(ast.literal_eval(str(current_val)))
                    if abs(current_num - value) < 1e-9:
                        continue  # Already at this value
                except (ValueError, TypeError, SyntaxError, NameError):
                    continue

            bin_lo, bin_hi = cell["bin_range"]
            return {
                "changes": {param: value},
                "hypothesis": (
                    f"EXCAVATION: Systematically sampling under-explored "
                    f"region {param}=[{bin_lo:.4g}, {bin_hi:.4g}] "
                    f"(only {cell['sample_count']} prior sample(s)). "
                    f"This region has never been properly tested -- it "
                    f"may contain better configs than currently known."
                ),
                "expected_direction": "uncertain",
                "risk": "medium",
            }

        return None

    def get_excavation_text(self) -> str:
        """Generate coverage map text for the proposal prompt."""
        if not self._grid:
            return ""

        lines = ["## Excavation Grid (parameter space coverage map)"]
        lines.append(
            "Shows how thoroughly each parameter range has been explored. "
            "Empty cells are blind spots -- potentially good regions that "
            "have never been tested.\n"
        )

        for param in sorted(self._grid.keys()):
            if param not in PARAM_RANGES:
                continue
            lo, hi, n_bins = PARAM_RANGES[param]
            cells = []
            for bin_idx in range(n_bins):
                count = self._grid[param].get(bin_idx, 0)
                bin_lo, bin_hi = self._get_bin_range(param, bin_idx)
                best = self._bin_best[param].get(bin_idx, None)
                if count == 0:
                    cells.append(
                        f"[{bin_lo:.3g}-{bin_hi:.3g}]:EMPTY"
                    )
                else:
                    best_str = (
                        f",best={best:.4f}" if best and best < float("inf")
                        else ""
                    )
                    cells.append(
                        f"[{bin_lo:.3g}-{bin_hi:.3g}]:{count}x{best_str}"
                    )

            lines.append(f"  {param}: {' | '.join(cells)}")

        # Count total empty cells
        total_empty = 0
        total_cells = 0
        for param in PARAM_RANGES:
            _, _, n_bins = PARAM_RANGES[param]
            for bin_idx in range(n_bins):
                total_cells += 1
                if self._grid[param].get(bin_idx, 0) == 0:
                    total_empty += 1

        if total_cells > 0:
            coverage = (total_cells - total_empty) / total_cells * 100
            lines.append(
                f"\n  Coverage: {coverage:.0f}% ({total_cells - total_empty}/"
                f"{total_cells} cells explored, {total_empty} blind spots)"
            )

        return "\n".join(lines)
