"""Epistasis map — identifies non-additive parameter pair interactions.

In molecular genetics, epistasis is when the effect of one gene depends on the
presence of another gene. Two mutations that are each beneficial alone may be
harmful together (negative epistasis) or synergistic (positive epistasis).

This mechanism tracks which parameter PAIRS were changed together and whether
the joint effect was better or worse than expected from their individual effects.
It then warns the LLM about dangerous combinations and recommends synergistic ones.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EpistasisMap:
    """Tracks pairwise parameter interaction effects (epistasis).

    Records:
    - Individual parameter effects: param -> list of delta_bpb when changed alone
    - Pair effects: (param_a, param_b) -> list of delta_bpb when changed together

    After enough data, computes the epistasis coefficient:
        epistasis = observed_pair_effect - (expected_individual_a + expected_individual_b)
    Negative epistasis coefficient = synergistic (pair is better than sum of parts)
    Positive epistasis coefficient = antagonistic (pair is worse than sum of parts)
    """

    def __init__(self, min_samples_individual: int = 2, min_samples_pair: int = 1):
        self._min_samples_individual = min_samples_individual
        self._min_samples_pair = min_samples_pair
        # param -> list of delta_bpb when this param was the ONLY one changed
        self._solo_effects: dict[str, list[float]] = defaultdict(list)
        # frozenset({param_a, param_b}) -> list of delta_bpb when both changed together
        self._pair_effects: dict[frozenset, list[float]] = defaultdict(list)

    def record(self, changes: dict, val_bpb: float, best_bpb_before: float,
               status: str) -> None:
        """Record the outcome of a parameter change (solo or multi-param)."""
        if status == "crash":
            return

        delta = val_bpb - best_bpb_before  # negative = improvement
        params_changed = list(changes.keys())

        if len(params_changed) == 1:
            # Solo change — record individual effect
            self._solo_effects[params_changed[0]].append(delta)
        elif len(params_changed) >= 2:
            # Multi-param change — record all pairwise effects
            for i in range(len(params_changed)):
                for j in range(i + 1, len(params_changed)):
                    pair = frozenset({params_changed[i], params_changed[j]})
                    self._pair_effects[pair].append(delta)

    def get_epistasis_text(self) -> str:
        """Generate epistasis analysis for injection into the proposal prompt.

        Only reports pairs where we have enough data to compute epistasis
        coefficients, and where the interaction is meaningfully non-additive.
        """
        if not self._pair_effects:
            return ""

        # Compute average solo effects
        solo_avgs: dict[str, float] = {}
        for param, deltas in self._solo_effects.items():
            if len(deltas) >= self._min_samples_individual:
                solo_avgs[param] = sum(deltas) / len(deltas)

        if not solo_avgs:
            return ""

        lines = ["## Parameter Interaction Map (epistasis analysis)"]
        lines.append(
            "When two parameters are changed together, their joint effect may differ "
            "from the sum of their individual effects. Use this to decide whether to "
            "change parameters together (synergistic) or separately (antagonistic)."
        )

        synergistic = []
        antagonistic = []

        for pair, deltas in self._pair_effects.items():
            if len(deltas) < self._min_samples_pair:
                continue

            pair_list = sorted(pair)
            if len(pair_list) != 2:
                continue
            p_a, p_b = pair_list

            # Both params must have solo data to compute epistasis
            if p_a not in solo_avgs or p_b not in solo_avgs:
                continue

            observed_pair = sum(deltas) / len(deltas)
            expected_additive = solo_avgs[p_a] + solo_avgs[p_b]

            # Epistasis coefficient: how much the pair deviates from additivity
            epistasis_coeff = observed_pair - expected_additive

            # Only report if meaningfully non-additive (> 0.0005 bpb)
            if abs(epistasis_coeff) < 0.0005:
                continue

            entry = {
                "params": (p_a, p_b),
                "coeff": epistasis_coeff,
                "observed": observed_pair,
                "expected": expected_additive,
                "n_pair": len(deltas),
                "n_solo_a": len(self._solo_effects[p_a]),
                "n_solo_b": len(self._solo_effects[p_b]),
            }

            if epistasis_coeff < 0:
                synergistic.append(entry)
            else:
                antagonistic.append(entry)

        if not synergistic and not antagonistic:
            return ""

        if synergistic:
            synergistic.sort(key=lambda e: e["coeff"])
            lines.append("\n### Synergistic pairs (change TOGETHER for bonus effect)")
            for e in synergistic[:5]:
                p_a, p_b = e["params"]
                lines.append(
                    f"  - {p_a} + {p_b}: joint effect {e['observed']:+.6f} vs "
                    f"expected {e['expected']:+.6f} (synergy={-e['coeff']:.6f}, "
                    f"{e['n_pair']} joint trial(s))"
                )

        if antagonistic:
            antagonistic.sort(key=lambda e: -e["coeff"])
            lines.append("\n### Antagonistic pairs (change SEPARATELY — joint effect is worse)")
            for e in antagonistic[:5]:
                p_a, p_b = e["params"]
                lines.append(
                    f"  - {p_a} + {p_b}: joint effect {e['observed']:+.6f} vs "
                    f"expected {e['expected']:+.6f} (antagonism={e['coeff']:.6f}, "
                    f"{e['n_pair']} joint trial(s))"
                )

        return "\n".join(lines)
