"""Fossil record -- preserve extinct configs that might be worth reviving (Lazarus taxa).

In paleontology, the fossil record preserves organisms that once thrived but went
extinct. Occasionally, a species thought to be extinct reappears after a long gap
in the record -- a "Lazarus taxon" (e.g., the coelacanth). The reappearance happens
because environmental conditions changed to once again favor that organism.

The elite pool only keeps the top-K configs by absolute val_bpb. But configs that
were discarded or superseded early in the search may deserve a second look:

1. A config discarded at iteration 3 (when baseline was 1.20) with val_bpb=1.18
   was "bad" then, but it explored a region of parameter space that the search
   has never returned to. Now at iteration 25 with best=1.08, the DIRECTION
   that config was exploring (not its absolute score) might be valuable.

2. A config that crashed early because of aggressive architectural changes might
   succeed now that other parameters (LR, schedule) have been better tuned to
   accommodate bold moves.

This mechanism maintains a "fossil record" of ALL configs ever tried (not just
elite survivors), and periodically suggests reviving the most promising extinct
configs -- those that showed the strongest improvement RELATIVE to their
predecessor, even if their absolute bpb was not competitive.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class FossilRecord:
    """Preserves extinct configs and suggests Lazarus revivals.

    Tracks every config ever evaluated (the full fossil record), not just
    the elite survivors. Periodically identifies promising "fossils" --
    configs that showed strong relative improvement or explored unique
    regions of parameter space -- and suggests reviving them.

    A fossil is considered promising for revival if:
    1. It achieved a strong RELATIVE improvement over its predecessor
       (even if its absolute bpb was poor)
    2. It explored parameters that haven't been touched since
    3. It was discarded or superseded, but the reason for extinction
       may no longer apply (e.g., the surrounding config has changed)
    """

    def __init__(self, max_fossils: int = 30):
        self._max_fossils = max_fossils
        # Full record: list of {iteration, changes, val_bpb, status,
        #              predecessor_bpb, relative_delta, config_snapshot}
        self._fossils: list[dict] = []
        # Track which fossils have been suggested for revival (avoid repeats)
        self._revival_suggested: set[int] = set()  # iteration numbers

    def record(self, iteration: int, changes: dict, val_bpb: float,
               best_bpb_before: float, status: str,
               config_snapshot: dict | None = None) -> None:
        """Record every evaluated config in the fossil record."""
        if status == "crash":
            # Crashes are recorded but marked specially
            relative_delta = 0.0
        else:
            relative_delta = best_bpb_before - val_bpb  # positive = improvement

        fossil = {
            "iteration": iteration,
            "changes": dict(changes),
            "val_bpb": val_bpb,
            "status": status,
            "predecessor_bpb": best_bpb_before,
            "relative_delta": relative_delta,
            "config_snapshot": dict(config_snapshot) if config_snapshot else {},
            "params_explored": set(changes.keys()),
        }
        self._fossils.append(fossil)

        # Trim to max size (keep most recent)
        if len(self._fossils) > self._max_fossils:
            self._fossils = self._fossils[-self._max_fossils:]

    def find_lazarus_candidates(
        self,
        current_best_bpb: float,
        active_params: list[str],
        n_candidates: int = 3,
    ) -> list[dict]:
        """Find the most promising fossils for revival.

        Scores each fossil on:
        1. Relative improvement (strong delta = was heading in a good direction)
        2. Exploration novelty (touched params not recently explored)
        3. Not already in elite pool or recently suggested
        """
        if len(self._fossils) < 3:
            return []

        active_set = set(active_params)
        # Determine which params have been touched recently (last 5 iterations)
        recent_params: set[str] = set()
        for fossil in self._fossils[-5:]:
            recent_params |= fossil["params_explored"]

        candidates = []
        for fossil in self._fossils:
            # Skip crashes, already-suggested, and the very recent
            if fossil["status"] == "crash":
                continue
            if fossil["iteration"] in self._revival_suggested:
                continue
            # Skip fossils from the last 3 iterations (too recent to be "extinct")
            if self._fossils and fossil["iteration"] >= self._fossils[-1]["iteration"] - 2:
                continue

            # Score 1: Relative improvement (positive = was improving)
            rel_score = fossil["relative_delta"]

            # Score 2: Exploration novelty -- params it explored that nobody
            # has touched recently
            novel_params = fossil["params_explored"] - recent_params
            novelty_score = len(novel_params & active_set) * 0.002

            # Score 3: Penalize fossils whose absolute bpb is very far from
            # current best (their region may be fundamentally bad)
            if fossil["val_bpb"] > 0:
                distance_penalty = max(
                    0, (fossil["val_bpb"] - current_best_bpb - 0.05) * 0.5
                )
            else:
                distance_penalty = 0.0

            total_score = rel_score + novelty_score - distance_penalty

            candidates.append({
                "fossil": fossil,
                "score": total_score,
                "novel_params": novel_params & active_set,
            })

        # Sort by score (best first) and take top N
        candidates.sort(key=lambda x: -x["score"])
        return candidates[:n_candidates]

    def suggest_revival(
        self,
        current_config: dict,
        current_best_bpb: float,
        active_params: list[str],
    ) -> dict | None:
        """Suggest a Lazarus revival -- re-apply the direction of a promising fossil.

        Instead of blindly replaying the fossil's exact changes, this extracts
        the DIRECTION of the fossil's changes and applies them to the current
        config. This accounts for the fact that the surrounding config has
        changed since the fossil was alive.

        Returns a proposal dict with 'changes' and 'hypothesis', or None.
        """
        candidates = self.find_lazarus_candidates(
            current_best_bpb, active_params, n_candidates=3
        )
        if not candidates:
            return None

        best = candidates[0]
        fossil = best["fossil"]

        # Re-apply the fossil's changes to the current config
        # (only params that are still active)
        active_set = set(active_params)
        changes = {}
        for param, value in fossil["changes"].items():
            if param in active_set:
                # Only include if it would actually change the current config
                current_val = str(current_config.get(param, ""))
                if str(value) != current_val:
                    changes[param] = value

        if not changes:
            # Try next candidate
            for cand in candidates[1:]:
                fossil = cand["fossil"]
                for param, value in fossil["changes"].items():
                    if param in active_set:
                        current_val = str(current_config.get(param, ""))
                        if str(value) != current_val:
                            changes[param] = value
                if changes:
                    break

        if not changes:
            return None

        self._revival_suggested.add(fossil["iteration"])

        novel_str = ""
        if best["novel_params"]:
            novel_str = (
                f" It explores {', '.join(best['novel_params'])} which "
                f"haven't been touched recently."
            )

        return {
            "changes": changes,
            "hypothesis": (
                f"LAZARUS REVIVAL: Re-applying changes from iter "
                f"{fossil['iteration']} (original delta="
                f"{fossil['relative_delta']:+.6f}, status={fossil['status']}). "
                f"This fossil explored a promising direction that was abandoned."
                f"{novel_str}"
            ),
            "expected_direction": "lower",
            "risk": "medium",
        }

    def get_fossil_text(self) -> str:
        """Generate a fossil record summary for the proposal prompt."""
        if len(self._fossils) < 5:
            return ""

        lines = [
            "## Fossil Record (extinct configs that may deserve revival)"
        ]
        lines.append(
            "These are past configs that were discarded or superseded. "
            "Some explored promising directions that were never followed up. "
            "Consider reviving directions that showed strong relative "
            "improvement, especially if they explored parameters not "
            "recently touched.\n"
        )

        # Show top 3 revival candidates
        current_best = min(
            (f["val_bpb"] for f in self._fossils if f["val_bpb"] > 0),
            default=float("inf"),
        )
        active_params = list({
            p for f in self._fossils for p in f["params_explored"]
        })
        candidates = self.find_lazarus_candidates(
            current_best, active_params, n_candidates=3
        )

        if candidates:
            lines.append("### Top Revival Candidates (Lazarus taxa)")
            for i, cand in enumerate(candidates):
                fossil = cand["fossil"]
                novel = (
                    f", novel params: {', '.join(cand['novel_params'])}"
                    if cand["novel_params"] else ""
                )
                lines.append(
                    f"  {i+1}. Iter {fossil['iteration']} "
                    f"[{fossil['status']}]: "
                    f"{fossil['changes']} "
                    f"(rel_delta={fossil['relative_delta']:+.6f}, "
                    f"score={cand['score']:.4f}{novel})"
                )

        # Summary stats
        n_discarded = sum(
            1 for f in self._fossils if f["status"] == "discard"
        )
        n_kept = sum(1 for f in self._fossils if f["status"] == "keep")
        n_crashed = sum(1 for f in self._fossils if f["status"] == "crash")
        lines.append(
            f"\n  Record: {len(self._fossils)} fossils "
            f"({n_kept} kept, {n_discarded} discarded, {n_crashed} crashed)"
        )

        return "\n".join(lines)
