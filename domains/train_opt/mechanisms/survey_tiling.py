"""Survey tiling — SDSS-style systematic coverage of the parameter sky.

In observational astronomy, the Sloan Digital Sky Survey (SDSS) tiles the sky
into overlapping fields so that every region is observed at least once. This
prevents the telescope from staring at the same patch repeatedly while leaving
huge swathes of sky unobserved.

This mechanism tracks which regions of the parameter space have been "observed"
(proposed) and which remain unexplored. It divides each parameter's range into
coarse bins and counts how many proposals have landed in each bin. The LLM is
then told which bins are over-observed and which are unexplored "dark sky."

The key insight: the current system has no survey strategy — the LLM repeatedly
tweaks the same parameters in the same range (the "bright patch" problem). Survey
tiling forces coverage of the full parameter sky before revisiting any tile.
"""
from __future__ import annotations

import ast
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SurveyTiling:
    """Tracks parameter-space coverage using binned observation counts.

    Divides each parameter's observed range into N_BINS tiles and counts how
    many proposals have landed in each tile. Generates a "coverage map" that
    tells the LLM which regions are over-observed and which are unexplored.
    """

    N_BINS = 5  # number of tiles per parameter dimension

    def __init__(self):
        # param -> list of observed numeric values
        self._observations: dict[str, list[float]] = defaultdict(list)
        # param -> (lo, hi) — the full range observed so far
        self._ranges: dict[str, tuple[float, float]] = {}
        # param -> list of bin counts (length N_BINS)
        self._bin_counts: dict[str, list[int]] = {}

    def record(self, changes: dict, old_config: dict) -> None:
        """Record a parameter observation (regardless of keep/discard/crash).

        Every proposal is an 'observation' — we record where it landed even if
        it crashed, because we now know something about that region.

        Args:
            changes: param -> new_value mapping from the proposal.
            old_config: the full config before changes were applied.
        """
        # Record the old config values for context (if first time seeing them)
        for param, val in old_config.items():
            num = self._try_numeric(val)
            if num is not None and not self._observations[param]:
                self._observations[param].append(num)

        # Record the new values from the proposal
        for param, val in changes.items():
            num = self._try_numeric(val)
            if num is not None:
                self._observations[param].append(num)

        # Rebuild bins after new observations
        self._rebuild_bins()

    def _try_numeric(self, val) -> float | None:
        """Try to parse a value as a float. Returns None for non-numeric."""
        try:
            return float(ast.literal_eval(str(val)))
        except (ValueError, TypeError, SyntaxError, NameError):
            return None

    def _rebuild_bins(self) -> None:
        """Rebuild the bin counts for all observed parameters."""
        self._bin_counts.clear()
        self._ranges.clear()

        for param, values in self._observations.items():
            if len(values) < 2:
                continue

            lo = min(values)
            hi = max(values)

            # Expand range by 20% on each side to include "unexplored border"
            margin = max((hi - lo) * 0.2, abs(lo) * 0.1 + 1e-9)
            lo -= margin
            hi += margin

            self._ranges[param] = (lo, hi)

            # Count observations per bin
            bin_width = (hi - lo) / self.N_BINS
            counts = [0] * self.N_BINS
            for v in values:
                idx = int((v - lo) / bin_width)
                idx = max(0, min(idx, self.N_BINS - 1))
                counts[idx] += 1

            self._bin_counts[param] = counts

    def get_coverage_text(self) -> str:
        """Generate a survey coverage map for injection into the proposal prompt.

        Reports:
        - Over-observed tiles (observed 3+ times with no improvement)
        - Dark-sky tiles (0 observations — completely unexplored regions)
        - Overall coverage fraction per parameter
        """
        if not self._bin_counts:
            return ""

        lines = ["## Survey Coverage Map (parameter sky tiling)"]
        lines.append(
            "Each parameter's range is divided into tiles. Tiles with 0 observations "
            "are 'dark sky' — completely unexplored. Tiles with many observations are "
            "'over-observed' — diminishing returns from further staring. Prioritize "
            "dark-sky tiles to systematically cover the parameter space."
        )

        dark_sky_suggestions = []

        for param in sorted(self._bin_counts.keys()):
            counts = self._bin_counts[param]
            lo, hi = self._ranges[param]
            covered = sum(1 for c in counts if c > 0)
            coverage_pct = covered / self.N_BINS * 100

            bin_width = (hi - lo) / self.N_BINS

            # Identify dark-sky and over-observed tiles
            dark_tiles = []
            over_tiles = []
            for i, c in enumerate(counts):
                tile_lo = lo + i * bin_width
                tile_hi = tile_lo + bin_width
                if c == 0:
                    dark_tiles.append(f"{tile_lo:.4g}-{tile_hi:.4g}")
                elif c >= 3:
                    over_tiles.append(f"{tile_lo:.4g}-{tile_hi:.4g} ({c}x)")

            status_parts = [f"coverage={coverage_pct:.0f}% ({covered}/{self.N_BINS} tiles)"]
            if dark_tiles:
                status_parts.append(f"DARK SKY: {', '.join(dark_tiles[:3])}")
                # Generate concrete suggestion for the first dark-sky tile
                first_dark_lo = lo + counts.index(0) * bin_width
                first_dark_mid = first_dark_lo + bin_width / 2
                dark_sky_suggestions.append((param, first_dark_mid))
            if over_tiles:
                status_parts.append(f"over-observed: {', '.join(over_tiles[:2])}")

            lines.append(f"- **{param}**: {'; '.join(status_parts)}")

        if dark_sky_suggestions:
            lines.append("\n### Suggested Dark-Sky Explorations")
            lines.append(
                "Try these unexplored regions (midpoint of first dark-sky tile):"
            )
            for param, suggested_val in dark_sky_suggestions[:4]:
                lines.append(f"  - {param} ~ {suggested_val:.4g}")

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)

    def get_least_covered_param(self, active_params: list[str]) -> str | None:
        """Return the active param with the lowest coverage fraction.

        Useful for the diversification directive — instead of random exploration,
        point the LLM at the parameter with the most unexplored sky.
        """
        worst_param = None
        worst_coverage = float("inf")

        for param in active_params:
            if param not in self._bin_counts:
                # Never observed at all — ultimate dark sky
                return param

            counts = self._bin_counts[param]
            covered = sum(1 for c in counts if c > 0)
            coverage = covered / self.N_BINS

            if coverage < worst_coverage:
                worst_coverage = coverage
                worst_param = param

        return worst_param
