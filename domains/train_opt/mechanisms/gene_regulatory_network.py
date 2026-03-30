"""Gene regulatory network — models compensatory parameter dependencies.

In molecular genetics, gene regulatory networks (GRNs) describe how genes
activate, inhibit, or modulate each other. Changing one gene's expression
often triggers compensatory changes in downstream genes to maintain homeostasis.

This mechanism learns parameter dependency rules from the search history:
when changing parameter A improved results, what was the state of parameter B?
It then generates "compensatory change" recommendations — if you increase LR,
you should also increase WEIGHT_DECAY to prevent divergence, etc.

Implementation: builds a conditional correlation table. For each param P,
tracks which OTHER params' values co-occur in elite configs vs bad configs.
Then recommends compensatory changes when the LLM proposes changing P.
"""
from __future__ import annotations

import ast
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class GeneRegulatoryNetwork:
    """Models learned parameter dependency rules (compensatory changes).

    Tracks co-occurrence patterns: when param A is at value X in elite configs,
    what values does param B tend to take? This builds a conditional dependency
    model that can recommend "if you change A, also adjust B accordingly."

    The network is built from two data sources:
    1. Elite configs — what parameter combinations work well together
    2. Improvement history — when changing A improved things, what was B's value
    """

    def __init__(self, min_elites: int = 3):
        self._min_elites = min_elites
        # Records: list of (config_dict, val_bpb, was_improvement)
        self._observations: list[tuple[dict, float, bool]] = []
        # Learned rules: param_trigger -> list of (param_target, direction_trigger,
        #                direction_target, strength)
        self._rules: list[dict] = []

    def record_config(self, config: dict, val_bpb: float,
                      was_improvement: bool) -> None:
        """Record a full config snapshot with its outcome."""
        self._observations.append((config.copy(), val_bpb, was_improvement))

    def build_rules(self, elite_configs: list[tuple[float, dict]]) -> None:
        """Build regulatory rules from elite config patterns.

        For each pair of numeric parameters (A, B), checks if they are
        correlated across elite configs (both high, both low, or inverse).
        A correlation indicates a dependency — changing one should trigger
        a compensatory change in the other.

        Args:
            elite_configs: list of (val_bpb, config_dict), sorted best-first.
        """
        if len(elite_configs) < self._min_elites:
            return

        self._rules.clear()

        # Extract numeric values for each param across elites
        param_values: dict[str, list[float]] = defaultdict(list)
        for _bpb, config in elite_configs:
            for param, val in config.items():
                try:
                    num = float(ast.literal_eval(str(val)))
                    param_values[param].append(num)
                except (ValueError, TypeError, SyntaxError, NameError):
                    pass

        # Only consider params with enough data points and variance
        valid_params: dict[str, list[float]] = {}
        for param, values in param_values.items():
            if len(values) >= self._min_elites:
                if max(values) != min(values):  # has variance
                    valid_params[param] = values

        if len(valid_params) < 2:
            return

        # Compute pairwise rank correlation (Spearman-like, stdlib only)
        params = sorted(valid_params.keys())
        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                p_a, p_b = params[i], params[j]
                vals_a = valid_params[p_a]
                vals_b = valid_params[p_b]

                # Use rank correlation
                n = min(len(vals_a), len(vals_b))
                if n < self._min_elites:
                    continue

                corr = self._rank_correlation(vals_a[:n], vals_b[:n])

                # Only create rules for strong correlations
                if abs(corr) < 0.6:
                    continue

                if corr > 0:
                    # Positive correlation: both move in same direction
                    rule = {
                        "trigger": p_a,
                        "target": p_b,
                        "relationship": "same_direction",
                        "strength": abs(corr),
                        "description": (
                            f"When {p_a} increases, {p_b} should also increase "
                            f"(corr={corr:.2f} across {n} elite configs)"
                        ),
                    }
                else:
                    # Negative correlation: move in opposite directions
                    rule = {
                        "trigger": p_a,
                        "target": p_b,
                        "relationship": "opposite_direction",
                        "strength": abs(corr),
                        "description": (
                            f"When {p_a} increases, {p_b} should decrease "
                            f"(corr={corr:.2f} across {n} elite configs)"
                        ),
                    }

                self._rules.append(rule)
                # Also add the reverse direction rule
                reverse = rule.copy()
                reverse["trigger"] = p_b
                reverse["target"] = p_a
                if rule["relationship"] == "same_direction":
                    reverse["description"] = (
                        f"When {p_b} increases, {p_a} should also increase "
                        f"(corr={corr:.2f} across {n} elite configs)"
                    )
                else:
                    reverse["description"] = (
                        f"When {p_b} increases, {p_a} should decrease "
                        f"(corr={corr:.2f} across {n} elite configs)"
                    )
                self._rules.append(reverse)

        logger.info(f"[GRN] Built {len(self._rules)} regulatory rules from {len(elite_configs)} elite configs")

    @staticmethod
    def _rank_correlation(x: list[float], y: list[float]) -> float:
        """Compute Spearman rank correlation (stdlib-only implementation).

        Returns a value in [-1, 1]. Positive = same direction, negative = opposite.
        """
        n = len(x)
        if n < 2:
            return 0.0

        # Compute ranks
        def _ranks(vals: list[float]) -> list[float]:
            indexed = sorted(enumerate(vals), key=lambda t: t[1])
            ranks = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1  # 1-based average rank for ties
                for k in range(i, j + 1):
                    ranks[indexed[k][0]] = avg_rank
                i = j + 1
            return ranks

        rx = _ranks(x)
        ry = _ranks(y)

        # Pearson correlation of ranks
        mean_rx = sum(rx) / n
        mean_ry = sum(ry) / n
        cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
        std_x = (sum((a - mean_rx) ** 2 for a in rx)) ** 0.5
        std_y = (sum((b - mean_ry) ** 2 for b in ry)) ** 0.5

        if std_x < 1e-12 or std_y < 1e-12:
            return 0.0

        return cov / (std_x * std_y)

    def get_grn_text(self, proposed_changes: dict | None = None) -> str:
        """Generate regulatory network recommendations for the proposal prompt.

        If proposed_changes is given, highlights rules relevant to those changes.
        Otherwise, shows all learned rules.
        """
        if not self._rules:
            return ""

        lines = ["## Parameter Dependency Rules (gene regulatory network)"]
        lines.append(
            "These rules were learned from elite config analysis. When you change "
            "one parameter, consider making the recommended compensatory change "
            "to its linked partner."
        )

        # Deduplicate: only show each (trigger, target) once, keep strongest
        seen = set()
        unique_rules = []
        for rule in sorted(self._rules, key=lambda r: -r["strength"]):
            key = (rule["trigger"], rule["target"])
            if key not in seen:
                seen.add(key)
                unique_rules.append(rule)

        # Show top rules (capped at 8 to avoid prompt bloat)
        for rule in unique_rules[:8]:
            strength_label = "STRONG" if rule["strength"] >= 0.8 else "moderate"
            lines.append(
                f"  - [{strength_label}] {rule['description']}"
            )

        return "\n".join(lines)
