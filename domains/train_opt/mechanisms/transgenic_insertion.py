"""Transgenic insertion — transplants proven gene blocks from elite configs.

In molecular genetics, transgenic insertion involves taking a functional gene
(or gene cassette) from one organism and inserting it into another. Unlike
crossover (which interpolates between two parents), transgenic insertion takes
an EXACT block of parameters from an elite config and inserts them wholesale.

This is more aggressive than crossover: instead of blending, it directly copies
proven-good parameter subsets. The key insight is that some parameters form
functional "operons" (co-regulated gene blocks) — they work as a unit and should
be transplanted together, not individually interpolated.

Examples of operons in this context:
- LR operon: MATRIX_LR + SCALAR_LR + EMBEDDING_LR + UNEMBEDDING_LR
- Schedule operon: WARMUP_RATIO + WARMDOWN_RATIO + FINAL_LR_FRAC
- Architecture operon: DEPTH + ASPECT_RATIO + HEAD_DIM
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Predefined parameter operons (functional gene blocks that should move together)
OPERONS = {
    "lr_operon": {
        "params": ["MATRIX_LR", "SCALAR_LR", "EMBEDDING_LR", "UNEMBEDDING_LR"],
        "description": "Learning rate block — these LRs are tuned relative to each other",
    },
    "schedule_operon": {
        "params": ["WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC"],
        "description": "LR schedule block — warmup/warmdown/final are interdependent",
    },
    "optimizer_operon": {
        "params": ["WEIGHT_DECAY", "ADAM_BETAS"],
        "description": "Optimizer regularization block — decay and momentum interact",
    },
    "architecture_operon": {
        "params": ["DEPTH", "ASPECT_RATIO", "HEAD_DIM"],
        "description": "Model architecture block — these define model shape together",
    },
}


class TransgenicInsertion:
    """Transplants exact parameter blocks from elite configs into the current config.

    Unlike crossover (which interpolates), transgenic insertion copies the exact
    values from a donor elite config for a whole "operon" (functional parameter block).
    This preserves the internal consistency of proven-good parameter groups.
    """

    def __init__(self):
        self._insertion_history: list[dict] = []
        # Track which operons have been tried (to cycle through them)
        self._operon_attempts: dict[str, int] = {name: 0 for name in OPERONS}

    def generate_transgenic_candidate(
        self,
        current_config: dict,
        elite_configs: list[tuple[float, dict]],
        active_params: list[str],
    ) -> dict | None:
        """Generate a transgenic insertion candidate.

        Picks the least-tried operon, finds the best elite that differs from
        current config on that operon, and transplants the elite's exact values.

        Args:
            current_config: Current hyperparameter config.
            elite_configs: List of (val_bpb, config_dict) sorted best-first.
            active_params: List of currently active (non-frozen) params.

        Returns:
            Proposal dict with 'changes' and 'hypothesis', or None.
        """
        if not elite_configs:
            return None

        active_set = set(active_params)

        # Sort operons by least-attempted first
        sorted_operons = sorted(
            OPERONS.items(),
            key=lambda x: self._operon_attempts[x[0]]
        )

        for operon_name, operon_def in sorted_operons:
            operon_params = operon_def["params"]

            # Filter to only active params in this operon
            available_params = [p for p in operon_params if p in active_set]
            if not available_params:
                continue

            # Find the best elite that has DIFFERENT values for at least one
            # param in this operon
            for elite_bpb, elite_config in elite_configs:
                changes = {}
                for param in available_params:
                    elite_val = elite_config.get(param)
                    current_val = current_config.get(param)
                    if elite_val is not None and str(elite_val) != str(current_val):
                        changes[param] = elite_val

                if changes:
                    self._operon_attempts[operon_name] += 1
                    self._insertion_history.append({
                        "operon": operon_name,
                        "donor_bpb": elite_bpb,
                        "changes": changes,
                    })

                    return {
                        "changes": changes,
                        "hypothesis": (
                            f"TRANSGENIC INSERTION: Transplant the exact "
                            f"'{operon_name}' block ({operon_def['description']}) "
                            f"from elite config (bpb={elite_bpb:.6f}). Unlike "
                            f"crossover interpolation, this preserves the proven "
                            f"internal consistency of the donor's parameter group."
                        ),
                        "expected_direction": "lower",
                        "risk": "low",
                    }

        return None

    def get_operon_text(self) -> str:
        """Generate operon information for the proposal prompt."""
        lines = ["## Parameter Operons (functional gene blocks)"]
        lines.append(
            "These parameter groups function as units — changing one member "
            "without adjusting the others may break their internal consistency. "
            "Consider changing entire operons together."
        )

        for name, operon_def in OPERONS.items():
            attempts = self._operon_attempts.get(name, 0)
            lines.append(
                f"  - {name}: {', '.join(operon_def['params'])} "
                f"({operon_def['description']}) "
                f"[tested {attempts}x]"
            )

        if self._insertion_history:
            lines.append("\n### Recent Transplant Results")
            for entry in self._insertion_history[-3:]:
                lines.append(
                    f"  - {entry['operon']} from donor (bpb={entry['donor_bpb']:.6f}): "
                    f"{entry['changes']}"
                )

        return "\n".join(lines)
