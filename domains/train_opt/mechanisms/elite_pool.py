"""Elite pool — maintains a pool of the top-K configs seen during search."""
from __future__ import annotations

import ast
import random


class ElitePool:
    """Maintains a pool of the top-K configs seen during search.

    Instead of only tracking the single best, this pool records the K best
    configs (by val_bpb) with their full parameter settings. The LLM can
    then identify patterns across elite configs (e.g., all good configs have
    WEIGHT_DECAY between 0.12-0.17) and propose targeted interpolations.
    """

    def __init__(self, max_size: int = 5):
        self._max_size = max_size
        # List of (val_bpb, config_dict, iteration, description)
        self._pool: list[tuple[float, dict, int, str]] = []

    def add(self, val_bpb: float, config: dict, iteration: int, description: str) -> None:
        """Add a config to the pool if it qualifies."""
        entry = (val_bpb, config.copy(), iteration, description)
        self._pool.append(entry)
        # Keep only the top-K by val_bpb (lower is better)
        self._pool.sort(key=lambda x: x[0])
        if len(self._pool) > self._max_size:
            self._pool = self._pool[:self._max_size]

    @property
    def best_bpb(self) -> float:
        if not self._pool:
            return float("inf")
        return self._pool[0][0]

    def get_elite_text(self) -> str:
        """Generate a summary of elite configs for the proposal prompt."""
        if len(self._pool) < 2:
            return ""

        lines = ["## Elite Pool (top configs found so far — look for PATTERNS)"]
        lines.append(
            "These are the best configs found. Look for patterns across them "
            "to identify which parameter ranges work well, and propose changes "
            "that move toward the sweet spots."
        )

        for rank, (bpb, config, iteration, desc) in enumerate(self._pool):
            lines.append(f"\n### Rank {rank + 1} (val_bpb={bpb:.6f}, iter {iteration})")
            # Show only the params that differ from pool median or are interesting
            for param, val in sorted(config.items()):
                lines.append(f"  {param} = {val}")

        # Identify parameter convergence
        if len(self._pool) >= 3:
            lines.append("\n### Parameter Convergence Analysis")
            all_params = set()
            for _, config, _, _ in self._pool:
                all_params.update(config.keys())

            for param in sorted(all_params):
                values = []
                for _, config, _, _ in self._pool:
                    if param in config:
                        values.append(config[param])
                if len(set(values)) == 1:
                    lines.append(f"  {param}: ALL elites use {values[0]} — likely optimal")
                elif len(values) >= 2:
                    lines.append(f"  {param}: varies across elites: {', '.join(str(v) for v in values)}")

        return "\n".join(lines)

    # Parameters that should only take power-of-2 or expression values
    _EXPRESSION_PARAMS = {"TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE"}
    # Parameters that must be integers
    _INTEGER_PARAMS = {"ASPECT_RATIO", "HEAD_DIM", "DEPTH", "DEVICE_BATCH_SIZE"}

    def generate_crossover(self, current_config: dict, active_params: list[str]) -> dict | None:
        """Generate a crossover candidate by interpolating between top-2 elite configs.

        Returns a dict with 'changes' and 'hypothesis' keys, or None if crossover
        is not possible (e.g., fewer than 2 elites, or no numeric differences).

        Improvement 11: Validates crossover outputs — skips params that would generate
        clearly invalid values (non-power-of-2 batch sizes, non-integer depths, etc.).
        Only crosses over params with simple numeric values to avoid crashes.
        """
        if len(self._pool) < 2:
            return None

        best_config = self._pool[0][1]
        second_config = self._pool[1][1]

        changes = {}
        for param in active_params:
            if param not in best_config or param not in second_config:
                continue
            best_val = best_config[param]
            second_val = second_config[param]
            if best_val == second_val:
                continue

            # Skip expression-based params (TOTAL_BATCH_SIZE = 2**19) — interpolation
            # produces invalid values like 698524 that crash
            if param in self._EXPRESSION_PARAMS:
                continue

            # Skip non-numeric params (WINDOW_PATTERN, ADAM_BETAS)
            if param in ("WINDOW_PATTERN", "ADAM_BETAS"):
                continue

            # Try numeric interpolation
            try:
                best_num = float(ast.literal_eval(str(best_val)))
                second_num = float(ast.literal_eval(str(second_val)))
                # Weighted interpolation: 60% best, 40% second
                # With some random perturbation to avoid exact repeats
                alpha = 0.6 + random.uniform(-0.15, 0.15)
                interpolated = alpha * best_num + (1 - alpha) * second_num

                # Validation: integer params must stay integer
                if param in self._INTEGER_PARAMS:
                    interpolated = round(interpolated)
                else:
                    interpolated = round(interpolated, 6)

                # Only include if different from current
                try:
                    current_num = float(ast.literal_eval(str(current_config.get(param, ""))))
                    if abs(interpolated - current_num) > 1e-6:
                        changes[param] = interpolated
                except (ValueError, TypeError, SyntaxError, NameError):
                    pass
            except (ValueError, TypeError, SyntaxError, NameError):
                continue

        if not changes:
            return None

        return {
            "changes": changes,
            "hypothesis": (
                f"Crossover: interpolating between elite rank-1 (bpb={self._pool[0][0]:.6f}) "
                f"and rank-2 (bpb={self._pool[1][0]:.6f}) configs to explore the space "
                f"between proven good configurations."
            ),
            "expected_direction": "lower",
            "risk": "low",
        }
