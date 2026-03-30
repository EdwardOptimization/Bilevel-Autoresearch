"""Seasonal cycling — enforces crop rotation through parameter groups.

In permaculture, crop rotation prevents soil depletion. Planting the same crop
in the same field year after year depletes specific nutrients and breeds pests.
Rotating through different crop families (legumes fix nitrogen, brassicas break
pest cycles, roots aerate soil) keeps the soil healthy and productive.

In this system, the LLM tends to "plant the same crop" — it finds that LR tweaks
improve val_bpb and then spends 10+ iterations making tiny LR adjustments while
ignoring schedule params, architecture, or optimizer settings. This is monoculture.

PlateauDetector catches one symptom (same params with diminishing returns) but only
triggers after the damage is done. Seasonal cycling is PREVENTIVE: it divides
iterations into "seasons" and assigns each season a preferred parameter family,
ensuring systematic rotation through the full parameter space.

Parameter families (seasonal crops):
- Spring: Learning rates (MATRIX_LR, SCALAR_LR, EMBEDDING_LR, UNEMBEDDING_LR)
- Summer: Schedule (WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC)
- Autumn: Optimizer (WEIGHT_DECAY, ADAM_BETAS, TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE)
- Winter: Architecture (DEPTH, ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN)

The seasonal signal is advisory, not mandatory: the LLM is told which family is
"in season" and encouraged to prioritize it, but can still change other params
if there is a strong reason.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Parameter families for rotation
SEASONS = {
    "spring": {
        "name": "Learning Rates",
        "params": {"MATRIX_LR", "SCALAR_LR", "EMBEDDING_LR", "UNEMBEDDING_LR"},
        "description": "Tune the learning rate magnitudes for different parameter groups",
    },
    "summer": {
        "name": "LR Schedule",
        "params": {"WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC"},
        "description": "Adjust the training schedule — warmup, warmdown, and final LR",
    },
    "autumn": {
        "name": "Optimizer & Batch",
        "params": {"WEIGHT_DECAY", "ADAM_BETAS", "TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE"},
        "description": "Tune optimizer regularization and batch size",
    },
    "winter": {
        "name": "Architecture",
        "params": {"DEPTH", "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN"},
        "description": "Explore model architecture changes (higher risk, higher potential)",
    },
}

SEASON_ORDER = ["spring", "summer", "autumn", "winter"]


class SeasonalCycling:
    """Enforces crop rotation through parameter families across iterations.

    Every N iterations, the "season" advances, changing which parameter family
    is prioritized. The LLM is told the current season and encouraged to focus
    on the in-season family, but is not forced to.

    Additionally tracks per-season harvest yields (improvements found) to learn
    which seasons are most productive and adjust season length accordingly.
    """

    def __init__(self, season_length: int = 3):
        self._season_length = season_length
        self._current_season_idx = 0
        self._iteration_in_season = 0

        # Track harvest per season: season_name -> list of delta_bpb for improvements
        self._season_harvests: dict[str, list[float]] = {s: [] for s in SEASON_ORDER}
        # Track total attempts per season
        self._season_attempts: dict[str, int] = {s: 0 for s in SEASON_ORDER}

    @property
    def current_season(self) -> str:
        return SEASON_ORDER[self._current_season_idx % len(SEASON_ORDER)]

    @property
    def season_info(self) -> dict:
        return SEASONS[self.current_season]

    def advance(self) -> None:
        """Advance one iteration within the current season, rotating if needed."""
        self._iteration_in_season += 1
        self._season_attempts[self.current_season] += 1

        if self._iteration_in_season >= self._season_length:
            old_season = self.current_season
            self._current_season_idx += 1
            self._iteration_in_season = 0
            new_season = self.current_season
            logger.info(
                f"[SeasonalCycling] Season change: {old_season} ({SEASONS[old_season]['name']}) "
                f"-> {new_season} ({SEASONS[new_season]['name']})"
            )

    def record_harvest(self, changes: dict, val_bpb: float,
                       best_bpb_before: float, status: str) -> None:
        """Record whether this iteration produced a harvest (improvement)."""
        if status == "crash":
            return

        delta = val_bpb - best_bpb_before
        if delta < 0:
            # Improvement! Record which season gets credit
            # Credit goes to the season whose params were actually changed
            params_changed = set(changes.keys())
            for season_name in SEASON_ORDER:
                season_params = SEASONS[season_name]["params"]
                if params_changed & season_params:
                    self._season_harvests[season_name].append(delta)

    def get_season_text(self, active_params: list[str]) -> str:
        """Generate seasonal guidance for the proposal prompt."""
        season = self.current_season
        info = self.season_info
        active_set = set(active_params)

        # Filter in-season params to only active ones
        in_season_active = sorted(info["params"] & active_set)
        iters_remaining = self._season_length - self._iteration_in_season

        lines = ["## Seasonal Focus (crop rotation)"]
        lines.append(
            f"Current season: {season.upper()} — {info['name']}"
        )
        lines.append(f"  {info['description']}")
        lines.append(f"  Iterations remaining this season: {iters_remaining}")

        if in_season_active:
            lines.append(
                f"  In-season parameters (PRIORITIZE these): {', '.join(in_season_active)}"
            )
        else:
            lines.append(
                "  No in-season parameters are currently active. "
                "Proceed with any active parameter."
            )

        # Show next season preview
        next_idx = (self._current_season_idx + 1) % len(SEASON_ORDER)
        next_season = SEASON_ORDER[next_idx]
        next_info = SEASONS[next_season]
        lines.append(
            f"  Next season: {next_season.upper()} — {next_info['name']}"
        )

        # Show season harvest history if we have data
        has_data = any(len(h) > 0 for h in self._season_harvests.values())
        if has_data:
            lines.append("\n### Season Harvest History")
            for s in SEASON_ORDER:
                harvests = self._season_harvests[s]
                attempts = self._season_attempts[s]
                if attempts > 0:
                    n_improvements = len(harvests)
                    hit_rate = n_improvements / attempts if attempts > 0 else 0
                    avg_improvement = (
                        sum(harvests) / len(harvests) if harvests else 0
                    )
                    lines.append(
                        f"  - {s} ({SEASONS[s]['name']}): "
                        f"{n_improvements}/{attempts} improvements "
                        f"(hit rate {hit_rate:.0%}"
                        f"{f', avg delta {avg_improvement:+.6f}' if harvests else ''})"
                    )

        lines.append(
            "\nFocus on in-season parameters unless you have strong evidence "
            "that another parameter change would be more impactful."
        )

        return "\n".join(lines)
