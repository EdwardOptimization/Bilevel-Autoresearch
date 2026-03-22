"""Inner loop runner for training optimization.

One iteration = propose config change → modify train.py → run training → parse val_bpb → keep/discard.
The search behavior is controlled by SearchConfig, which the outer loop modifies.

Structural improvements (Level 2):
  1. Crash memory — track which params caused crashes and warn the LLM in proposals.
  2. Multi-candidate proposal — LLM generates 3 candidates, then picks best, before GPU time.
  3. Quick-test pre-filter — 15-second smoke test before committing to a full training run.

Structural improvements (Level 2 — Round 2):
  4. Fixed quick-test path bug — quick test file now written to work_dir (not iter_dir)
     so that `from prepare import ...` resolves correctly.
  5. Infrastructure vs training crash classification — if quick test fails due to an
     import error or syntax error (infrastructure bug), skip the quick test and go
     straight to full training instead of wrongly recording a crash.
  6. Momentum tracking — track which parameter change directions led to improvements
     and inject this signal into the proposal prompt so the LLM learns from patterns.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.llm_client import LLMClient

from .config import HYPERPARAM_NAMES, SearchConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    iteration: int
    val_bpb: float
    peak_vram_mb: float
    training_seconds: float
    num_params_m: float
    status: str  # "keep" | "discard" | "crash"
    changes: dict  # param_name -> new_value
    description: str
    depth: int = 0


@dataclass
class CrashRecord:
    """Records a single crash event for crash memory."""
    param: str
    value: str  # the value that caused the crash
    iteration: int
    error_hint: str  # short description of failure mode (OOM, timeout, etc.)


@dataclass
class TrainTrace:
    results: list[TrainResult] = field(default_factory=list)
    best_bpb: float = float("inf")
    best_iteration: int = 0

    def add(self, result: TrainResult) -> None:
        self.results.append(result)
        if result.status == "keep" and result.val_bpb < self.best_bpb:
            self.best_bpb = result.val_bpb
            self.best_iteration = result.iteration

    def summary(self, last_n: int = 10) -> str:
        lines = [f"Best: {self.best_bpb:.6f} (iter {self.best_iteration})"]
        for r in self.results[-last_n:]:
            lines.append(
                f"  iter {r.iteration}: bpb={r.val_bpb:.6f} [{r.status}] {r.description}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Crash Memory
# ---------------------------------------------------------------------------

class CrashMemory:
    """Tracks which parameter changes caused crashes so the LLM can avoid them.

    This is distinct from the outer loop's freeze mechanism: the outer loop
    freezes params entirely, while crash memory gives the LLM a warning and
    lets it decide (e.g. try a *smaller* change instead of the same crash).
    """

    def __init__(self):
        self._crashes: list[CrashRecord] = []
        # param -> count of crashes involving that param
        self._param_crash_counts: dict[str, int] = defaultdict(int)

    def record(self, changes: dict, iteration: int, error_hint: str = "timeout/OOM") -> None:
        """Record a crash caused by the given parameter changes."""
        for param, value in changes.items():
            rec = CrashRecord(
                param=param,
                value=str(value),
                iteration=iteration,
                error_hint=error_hint,
            )
            self._crashes.append(rec)
            self._param_crash_counts[param] += 1
            logger.info(f"[CrashMemory] Recorded crash for {param}={value} (total: {self._param_crash_counts[param]})")

    @property
    def crash_count(self) -> int:
        return len(self._crashes)

    def get_warning_text(self) -> str:
        """Generate a warning block to inject into the proposal prompt.

        Returns empty string if no crashes recorded.
        """
        if not self._crashes:
            return ""

        lines = ["## Crash History (IMPORTANT — read before proposing)"]
        lines.append(
            "The following parameter changes caused training crashes (OOM, timeout, divergence). "
            "Avoid repeating these mistakes. If you must change a crash-prone parameter, "
            "use a MUCH more conservative value."
        )

        # Group by param
        by_param: dict[str, list[CrashRecord]] = defaultdict(list)
        for rec in self._crashes:
            by_param[rec.param].append(rec)

        for param, records in by_param.items():
            count = len(records)
            values = [r.value for r in records]
            lines.append(
                f"- **{param}** crashed {count} time(s) with values: {', '.join(values)}. "
                f"Reason: {records[0].error_hint}."
            )
            if count >= 2:
                lines.append(
                    f"  WARNING: {param} has crashed {count}+ times. "
                    f"Strongly consider NOT changing this parameter."
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Momentum Tracker (Level 2 Round 2 — Improvement 6)
# ---------------------------------------------------------------------------

class MomentumTracker:
    """Tracks which parameter change directions have led to improvements.

    When a 'keep' result is recorded, it notes which parameters were changed
    and in which direction (increase/decrease). When a 'discard' happens,
    it records the opposite signal. This gives the LLM structured feedback
    about what directions are working in the search space.
    """

    def __init__(self):
        # param -> list of (direction, delta_bpb) tuples
        # direction: "increase" or "decrease"
        # delta_bpb: negative = improvement (lower bpb is better)
        self._signals: dict[str, list[tuple[str, float]]] = defaultdict(list)

    def record(self, changes: dict, old_config: dict, val_bpb: float,
               best_bpb_before: float, status: str) -> None:
        """Record the outcome of a parameter change."""
        if status == "crash":
            return  # Crash memory handles crashes separately

        delta_bpb = val_bpb - best_bpb_before  # negative = improvement

        for param, new_val in changes.items():
            old_val = old_config.get(param)
            if old_val is None:
                continue

            # Try to determine direction (increase/decrease) for numeric params
            direction = self._detect_direction(param, old_val, new_val)
            if direction:
                self._signals[param].append((direction, delta_bpb))
                logger.info(
                    f"[Momentum] {param}: {direction} -> delta_bpb={delta_bpb:+.6f} "
                    f"({'improvement' if delta_bpb < 0 else 'regression'})"
                )

    def _detect_direction(self, param: str, old_val: str, new_val) -> str | None:
        """Detect whether a parameter was increased or decreased."""
        try:
            old_num = float(eval(str(old_val)))
            new_num = float(eval(str(new_val)))
            if new_num > old_num:
                return "increase"
            elif new_num < old_num:
                return "decrease"
            return None
        except (ValueError, TypeError, SyntaxError, NameError):
            return None  # Non-numeric param (e.g. WINDOW_PATTERN)

    def get_momentum_text(self) -> str:
        """Generate a momentum summary to inject into the proposal prompt."""
        if not self._signals:
            return ""

        lines = ["## Momentum Signals (what has worked / not worked)"]
        lines.append(
            "Based on past experiments, here are patterns about which parameter "
            "change directions led to improvements (lower bpb) vs regressions:"
        )

        for param, signals in sorted(self._signals.items()):
            if not signals:
                continue

            # Summarize by direction
            inc_signals = [s for s in signals if s[0] == "increase"]
            dec_signals = [s for s in signals if s[0] == "decrease"]

            parts = []
            if inc_signals:
                avg_delta = sum(d for _, d in inc_signals) / len(inc_signals)
                outcome = "helped" if avg_delta < 0 else "hurt"
                parts.append(
                    f"increasing {outcome} ({len(inc_signals)} trial(s), "
                    f"avg delta={avg_delta:+.6f})"
                )
            if dec_signals:
                avg_delta = sum(d for _, d in dec_signals) / len(dec_signals)
                outcome = "helped" if avg_delta < 0 else "hurt"
                parts.append(
                    f"decreasing {outcome} ({len(dec_signals)} trial(s), "
                    f"avg delta={avg_delta:+.6f})"
                )

            if parts:
                lines.append(f"- **{param}**: {'; '.join(parts)}")

        if len(lines) <= 2:
            return ""  # No useful signals yet

        lines.append(
            "\nUse these signals to bias your proposals: continue in directions "
            "that have helped, and avoid directions that have hurt."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROPOSE_SYSTEM = """You are an autonomous ML researcher optimizing a GPT pretraining script.
Your goal: get the lowest val_bpb (validation bits per byte).
You modify hyperparameters in train.py. Each run trains for a fixed time budget.
Return valid JSON only."""

PROPOSE_PROMPT = """\
## Current Hyperparameters
{current_config}

## Experiment History
{trace_summary}

## Search Configuration
Strategy: {strategy}
Active parameters (you may change ONLY these): {active_params}
Frozen parameters (do NOT touch): {frozen_params}

{guidance}

{crash_warnings}

{momentum_signals}

## Task
Propose ONE change to the hyperparameters. Pick the single most promising modification.

Return JSON:
{{
  "changes": {{
    "PARAM_NAME": new_value
  }},
  "hypothesis": "Why this change should improve val_bpb",
  "expected_direction": "lower|higher|uncertain"
}}

Rules:
- Change 1-3 parameters at most
- Only change parameters listed in "Active parameters"
- Values must be valid Python literals (e.g. 0.04, 2**19, "(0.8, 0.95)", "SSSL")
- Be specific about WHY this change should help based on the trace
- Pay close attention to any crash history warnings above
- If momentum signals are available, bias your proposals toward directions that have worked
"""

# Multi-candidate prompt: generate 3 candidates at once
MULTI_CANDIDATE_PROMPT = """\
## Current Hyperparameters
{current_config}

## Experiment History
{trace_summary}

## Search Configuration
Strategy: {strategy}
Active parameters (you may change ONLY these): {active_params}
Frozen parameters (do NOT touch): {frozen_params}

{guidance}

{crash_warnings}

{momentum_signals}

## Task
Propose THREE different candidate hyperparameter changes. Each candidate should explore
a DIFFERENT parameter or direction. Rank them from most to least promising.

Return JSON:
{{
  "candidates": [
    {{
      "changes": {{"PARAM_NAME": new_value}},
      "hypothesis": "Why this change should improve val_bpb",
      "expected_direction": "lower|higher|uncertain",
      "risk": "low|medium|high"
    }},
    {{
      "changes": {{"PARAM_NAME": new_value}},
      "hypothesis": "...",
      "expected_direction": "...",
      "risk": "..."
    }},
    {{
      "changes": {{"PARAM_NAME": new_value}},
      "hypothesis": "...",
      "expected_direction": "...",
      "risk": "..."
    }}
  ]
}}

Rules:
- Each candidate should change 1-3 parameters at most
- Only change parameters listed in "Active parameters"
- Values must be valid Python literals (e.g. 0.04, 2**19, "(0.8, 0.95)", "SSSL")
- Make candidates genuinely different — explore different parts of the search space
- Pay close attention to any crash history warnings above
- If momentum signals are available, bias at least one candidate toward directions that have worked
"""

# Picker prompt: given 3 candidates, pick the best one
PICK_CANDIDATE_PROMPT = """\
You are selecting the best hyperparameter change to run next. Training takes several
minutes of GPU time, so pick carefully.

## Current Best val_bpb: {best_bpb}

## Candidates:
{candidates_text}

## Selection Criteria:
1. Most likely to IMPROVE val_bpb (lower is better)
2. Low risk of crashing (OOM, timeout, divergence)
3. Novel — not repeating something already tried in the history

Pick the index (0, 1, or 2) of the best candidate.

Return JSON:
{{
  "pick": 0,
  "reasoning": "Why this candidate is the best choice"
}}
"""


# ---------------------------------------------------------------------------
# Infrastructure error detection (Level 2 Round 2 — Improvement 5)
# ---------------------------------------------------------------------------

INFRA_ERROR_PATTERNS = [
    "ModuleNotFoundError",
    "ImportError",
    "SyntaxError",
    "IndentationError",
    "FileNotFoundError",
    "PermissionError",
    "No module named",
]


def _is_infrastructure_error(stderr: str) -> bool:
    """Check if a crash was caused by infrastructure (not training).

    Infrastructure errors are things like missing imports, syntax errors, etc.
    that indicate a bug in the runner, NOT a bad hyperparameter choice.
    These should NOT be recorded as crashes in crash memory.
    """
    for pattern in INFRA_ERROR_PATTERNS:
        if pattern in stderr:
            return True
    return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TrainRunner:
    """Runs the inner loop: propose → modify train.py → train → evaluate → keep/discard."""

    # Quick-test parameters
    QUICK_TEST_BUDGET = 15  # seconds for smoke test
    QUICK_TEST_LOSS_THRESHOLD = 20.0  # loss above this after 15s = likely diverging

    def __init__(
        self,
        train_py: Path,
        work_dir: Path,
        llm_client: LLMClient,
        search_config: SearchConfig | None = None,
        artifacts_dir: Path | None = None,
    ):
        self.train_py = Path(train_py)
        self.work_dir = Path(work_dir)
        self.client = llm_client
        self.search_config = search_config or SearchConfig()
        self.artifacts_dir = artifacts_dir or (self.work_dir / "artifacts" / "train_opt")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Read the original train.py
        self.original_code = self.train_py.read_text(encoding="utf-8")
        self.current_code = self.original_code
        self.trace = TrainTrace()

        # Improvement 1: Crash memory
        self.crash_memory = CrashMemory()

        # Improvement 6: Momentum tracking
        self.momentum = MomentumTracker()

    def run_iteration(self, iteration: int) -> TrainResult:
        """Run one inner loop iteration."""
        iter_dir = self.artifacts_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        current_config = self._extract_hyperparams(self.current_code)
        logger.info(f"[Iter {iteration}] Proposing changes...")

        # 1. LLM proposes changes (multi-candidate with picker)
        proposal = self._propose(current_config, iteration)
        (iter_dir / "proposal.json").write_text(
            json.dumps(proposal, indent=2), encoding="utf-8"
        )

        if not proposal.get("changes"):
            logger.warning(f"[Iter {iteration}] No changes proposed")
            return TrainResult(
                iteration=iteration, val_bpb=self.trace.best_bpb,
                peak_vram_mb=0, training_seconds=0, num_params_m=0,
                status="discard", changes={}, description="No changes proposed",
            )

        # 2. Apply changes to train.py
        new_code = self._apply_changes(self.current_code, proposal["changes"])
        new_code = self._inject_time_budget(new_code)
        modified_path = self.work_dir / "train.py"
        modified_path.write_text(new_code, encoding="utf-8")
        (iter_dir / "train.py").write_text(new_code, encoding="utf-8")

        # 3. Quick-test pre-filter (Improvement 3, fixed in Improvement 4)
        logger.info(f"[Iter {iteration}] Running quick test (15s)... (changes: {proposal['changes']})")
        quick_result = self._run_training(
            modified_path, iter_dir, quick_test=True
        )

        if quick_result is None:
            # Quick test crashed — check if it's an infrastructure error
            quicktest_err_path = iter_dir / "quicktest_error.txt"
            if quicktest_err_path.exists():
                err_text = quicktest_err_path.read_text(encoding="utf-8")
                if _is_infrastructure_error(err_text):
                    # Infrastructure error: skip quick test, go straight to full run
                    logger.warning(
                        f"[Iter {iteration}] Quick test had INFRASTRUCTURE error "
                        f"(not a training crash) — skipping quick test, trying full run"
                    )
                else:
                    # Real training crash
                    logger.warning(f"[Iter {iteration}] Quick test CRASHED — skipping full run")
                    self.crash_memory.record(
                        proposal["changes"], iteration, error_hint="crashed in 15s smoke test"
                    )
                    r = TrainResult(
                        iteration=iteration, val_bpb=0, peak_vram_mb=0,
                        training_seconds=0, num_params_m=0, status="crash",
                        changes=proposal["changes"],
                        description=f"CRASH (quick test): {proposal.get('hypothesis', '')}",
                    )
                    self.trace.add(r)
                    return r
            else:
                # No error file — could be timeout or other issue, skip quick test
                logger.warning(
                    f"[Iter {iteration}] Quick test failed without error file — "
                    f"skipping quick test, trying full run"
                )
        elif quick_result.get("quick_test_diverged"):
            # Loss was abnormally high — skip full run
            logger.warning(
                f"[Iter {iteration}] Quick test DIVERGING (loss={quick_result.get('val_bpb', '?')}) "
                f"— skipping full run"
            )
            self.crash_memory.record(
                proposal["changes"], iteration, error_hint="diverging loss in 15s smoke test"
            )
            r = TrainResult(
                iteration=iteration, val_bpb=0, peak_vram_mb=0,
                training_seconds=0, num_params_m=0, status="crash",
                changes=proposal["changes"],
                description=f"DIVERGED (quick test): {proposal.get('hypothesis', '')}",
            )
            self.trace.add(r)
            return r
        else:
            logger.info(f"[Iter {iteration}] Quick test passed — proceeding to full training")

        # 4. Full training run
        logger.info(f"[Iter {iteration}] Training... (changes: {proposal['changes']})")
        result = self._run_training(modified_path, iter_dir)

        if result is None:
            # Crash during full run
            logger.warning(f"[Iter {iteration}] Training crashed")
            # Record in crash memory (Improvement 1)
            self.crash_memory.record(
                proposal["changes"], iteration, error_hint="timeout/OOM in full training"
            )
            r = TrainResult(
                iteration=iteration, val_bpb=0, peak_vram_mb=0,
                training_seconds=0, num_params_m=0, status="crash",
                changes=proposal["changes"],
                description=f"CRASH: {proposal.get('hypothesis', '')}",
            )
            self.trace.add(r)
            return r

        # 5. Keep/discard
        best_before = self.trace.best_bpb
        is_better = result["val_bpb"] < self.trace.best_bpb
        status = "keep" if is_better else "discard"

        if is_better:
            self.current_code = new_code
            logger.info(
                f"[Iter {iteration}] KEEP: {result['val_bpb']:.6f} "
                f"(was {self.trace.best_bpb:.6f}, delta={self.trace.best_bpb - result['val_bpb']:.6f})"
            )
        else:
            logger.info(
                f"[Iter {iteration}] DISCARD: {result['val_bpb']:.6f} "
                f"(best={self.trace.best_bpb:.6f})"
            )

        r = TrainResult(
            iteration=iteration,
            val_bpb=result["val_bpb"],
            peak_vram_mb=result.get("peak_vram_mb", 0),
            training_seconds=result.get("training_seconds", 0),
            num_params_m=result.get("num_params_m", 0),
            status=status,
            changes=proposal["changes"],
            description=proposal.get("hypothesis", ""),
            depth=result.get("depth", 0),
        )
        self.trace.add(r)

        # Record momentum signal (Improvement 6)
        self.momentum.record(
            changes=proposal["changes"],
            old_config=current_config,
            val_bpb=result["val_bpb"],
            best_bpb_before=best_before,
            status=status,
        )

        # Save trace
        (iter_dir / "result.json").write_text(
            json.dumps({"val_bpb": r.val_bpb, "status": r.status,
                        "changes": r.changes, "description": r.description},
                       indent=2), encoding="utf-8"
        )

        return r

    def run_baseline(self) -> TrainResult:
        """Run the unmodified train.py to establish baseline."""
        logger.info("[Baseline] Running unmodified train.py...")
        iter_dir = self.artifacts_dir / "baseline"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Write current (possibly SDPA-patched) train.py with time budget override
        code = self._inject_time_budget(self.current_code)
        modified_path = self.work_dir / "train.py"
        modified_path.write_text(code, encoding="utf-8")

        result = self._run_training(modified_path, iter_dir)
        if result is None:
            raise RuntimeError("Baseline training crashed!")

        r = TrainResult(
            iteration=0, val_bpb=result["val_bpb"],
            peak_vram_mb=result.get("peak_vram_mb", 0),
            training_seconds=result.get("training_seconds", 0),
            num_params_m=result.get("num_params_m", 0),
            status="keep", changes={}, description="baseline",
            depth=result.get("depth", 0),
        )
        self.trace.add(r)
        logger.info(f"[Baseline] val_bpb={r.val_bpb:.6f}, params={r.num_params_m:.1f}M")
        return r

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _propose(self, current_config: dict, iteration: int) -> dict:
        """Ask LLM to propose hyperparameter changes.

        Uses multi-candidate proposal (Improvement 2):
        1. LLM generates 3 candidates in one call.
        2. LLM picks the best candidate in a second call.
        This filters bad ideas with ~1 extra LLM call instead of wasting GPU time.

        Falls back to single-proposal mode if multi-candidate parsing fails.
        """
        from src.llm_client import parse_json_response

        crash_warnings = self.crash_memory.get_warning_text()
        momentum_signals = self.momentum.get_momentum_text()

        # --- Attempt multi-candidate proposal ---
        multi_prompt = MULTI_CANDIDATE_PROMPT.format(
            current_config=json.dumps(current_config, indent=2),
            trace_summary=self.trace.summary(),
            strategy=self.search_config.strategy,
            active_params=", ".join(self.search_config.active_params),
            frozen_params=", ".join(self.search_config.frozen_params) or "(none)",
            guidance=self.search_config.guidance,
            crash_warnings=crash_warnings,
            momentum_signals=momentum_signals,
        )

        raw = self.client.call(multi_prompt, system=PROPOSE_SYSTEM, max_tokens=2000)
        multi_result = parse_json_response(raw)

        candidates = None
        if isinstance(multi_result, dict) and "candidates" in multi_result:
            raw_candidates = multi_result["candidates"]
            if isinstance(raw_candidates, list) and len(raw_candidates) >= 2:
                # Validate each candidate has changes
                valid = [c for c in raw_candidates if isinstance(c, dict) and c.get("changes")]
                if len(valid) >= 2:
                    candidates = valid[:3]  # cap at 3

        if candidates:
            logger.info(f"[Iter {iteration}] Got {len(candidates)} candidates, asking LLM to pick best")

            # Build picker prompt
            candidates_text_lines = []
            for i, c in enumerate(candidates):
                candidates_text_lines.append(
                    f"### Candidate {i}\n"
                    f"Changes: {json.dumps(c['changes'])}\n"
                    f"Hypothesis: {c.get('hypothesis', 'N/A')}\n"
                    f"Risk: {c.get('risk', 'unknown')}\n"
                )
            candidates_text = "\n".join(candidates_text_lines)

            pick_prompt = PICK_CANDIDATE_PROMPT.format(
                best_bpb=f"{self.trace.best_bpb:.6f}",
                candidates_text=candidates_text,
            )

            pick_raw = self.client.call(pick_prompt, system=PROPOSE_SYSTEM, max_tokens=500)
            pick_result = parse_json_response(pick_raw)

            picked_idx = 0  # default to first (most promising) if parsing fails
            if isinstance(pick_result, dict) and "pick" in pick_result:
                try:
                    idx = int(pick_result["pick"])
                    if 0 <= idx < len(candidates):
                        picked_idx = idx
                except (ValueError, TypeError):
                    pass

            chosen = candidates[picked_idx]
            logger.info(
                f"[Iter {iteration}] Picked candidate {picked_idx}: {chosen['changes']} "
                f"(reason: {pick_result.get('reasoning', 'N/A')[:100]})"
            )

            # Validate: only active params
            changes = chosen.get("changes", {})
            active = set(self.search_config.active_params)
            filtered = {k: v for k, v in changes.items() if k in active}
            if len(filtered) < len(changes):
                dropped = set(changes) - active
                logger.warning(f"[Iter {iteration}] Dropped frozen params from proposal: {dropped}")
            chosen["changes"] = filtered

            return chosen

        # --- Fallback: single proposal (original behavior) ---
        logger.info(f"[Iter {iteration}] Multi-candidate failed, falling back to single proposal")

        prompt = PROPOSE_PROMPT.format(
            current_config=json.dumps(current_config, indent=2),
            trace_summary=self.trace.summary(),
            strategy=self.search_config.strategy,
            active_params=", ".join(self.search_config.active_params),
            frozen_params=", ".join(self.search_config.frozen_params) or "(none)",
            guidance=self.search_config.guidance,
            crash_warnings=crash_warnings,
            momentum_signals=momentum_signals,
        )

        raw = self.client.call(prompt, system=PROPOSE_SYSTEM, max_tokens=1000)
        result = parse_json_response(raw)

        if not isinstance(result, dict) or "raw_content" in result:
            logger.warning(f"[Iter {iteration}] Failed to parse proposal: {raw[:200]}")
            return {"changes": {}, "hypothesis": "parse failure"}

        # Validate: only active params
        changes = result.get("changes", {})
        active = set(self.search_config.active_params)
        filtered = {k: v for k, v in changes.items() if k in active}
        if len(filtered) < len(changes):
            dropped = set(changes) - active
            logger.warning(f"[Iter {iteration}] Dropped frozen params from proposal: {dropped}")
        result["changes"] = filtered

        return result

    def _apply_changes(self, code: str, changes: dict) -> str:
        """Replace hyperparameter values in train.py code."""
        for param, value in changes.items():
            # Match lines like: PARAM_NAME = value  # comment
            pattern = rf'^({re.escape(param)}\s*=\s*)(.+?)(\s*#.*)?$'
            replacement = rf'\g<1>{value}\3'
            code = re.sub(pattern, replacement, code, count=1, flags=re.MULTILINE)
        return code

    def _inject_time_budget(self, code: str, override_budget: int | None = None) -> str:
        """Override TIME_BUDGET if search_config specifies a non-default value.

        Args:
            override_budget: If set, use this value instead of search_config.time_budget.
                             Used by quick-test to inject a 15-second budget.
        """
        budget = override_budget if override_budget is not None else self.search_config.time_budget
        if budget != 300:
            # Insert TIME_BUDGET override right after the prepare import line
            code = code.replace(
                "from prepare import MAX_SEQ_LEN, TIME_BUDGET,",
                f"from prepare import MAX_SEQ_LEN, TIME_BUDGET as _TIME_BUDGET,",
                1,
            )
            # Add override after the import block
            code = code.replace(
                "# ---------------------------------------------------------------------------\n# GPT Model",
                f"TIME_BUDGET = {budget}  # overridden by bilevel runner\n\n"
                "# ---------------------------------------------------------------------------\n# GPT Model",
                1,
            )
        return code

    def _extract_hyperparams(self, code: str) -> dict:
        """Parse current hyperparameter values from train.py code."""
        config = {}
        for param in HYPERPARAM_NAMES:
            pattern = rf'^{re.escape(param)}\s*=\s*(.+?)(?:\s*#.*)?$'
            match = re.search(pattern, code, re.MULTILINE)
            if match:
                config[param] = match.group(1).strip()
        return config

    def _run_training(
        self, train_path: Path, iter_dir: Path, quick_test: bool = False
    ) -> dict | None:
        """Execute train.py and parse results. Returns None on crash.

        Args:
            quick_test: If True, run a 15-second smoke test instead of full training.
                        Returns dict with 'quick_test_diverged' key if loss is abnormally high.
        """
        if quick_test:
            # Improvement 4: Write quick test to work_dir (not iter_dir) so that
            # `from prepare import ...` resolves correctly. The previous bug wrote
            # the file to iter_dir (inside artifacts/) where prepare.py doesn't exist.
            code = self._apply_changes(
                self.current_code,
                self._extract_changes_from_file(train_path),
            )
            code = self._inject_time_budget(code, override_budget=self.QUICK_TEST_BUDGET)
            # Write to work_dir so imports resolve correctly
            quick_path = self.work_dir / "train_quicktest.py"
            quick_path.write_text(code, encoding="utf-8")
            # Also save a copy in iter_dir for debugging
            (iter_dir / "train_quicktest.py").write_text(code, encoding="utf-8")
            run_path = quick_path
            log_name = "quicktest.log"
            timeout = self.QUICK_TEST_BUDGET * 6  # 90s max for 15s test (compile overhead)
        else:
            run_path = train_path
            log_name = "run.log"
            timeout = self.search_config.time_budget * 3

        log_path = iter_dir / log_name

        try:
            result = subprocess.run(
                ["uv", "run", str(run_path)],
                capture_output=True, text=True,
                timeout=timeout,
                cwd=str(self.work_dir),
            )
            log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

            if result.returncode != 0:
                logger.warning(
                    f"Training failed (exit {result.returncode})"
                    f"{' [quick test]' if quick_test else ''}"
                )
                (iter_dir / ("quicktest_error.txt" if quick_test else "error.txt")).write_text(
                    result.stderr[-2000:], encoding="utf-8"
                )
                return None

            parsed = self._parse_results(result.stdout)

            if quick_test and parsed is not None:
                # Check for divergence: abnormally high loss or NaN
                val_bpb = parsed.get("val_bpb", 0)
                if val_bpb != val_bpb:  # NaN check
                    parsed["quick_test_diverged"] = True
                    logger.warning(f"[Quick test] NaN detected in val_bpb")
                elif val_bpb > self.QUICK_TEST_LOSS_THRESHOLD:
                    parsed["quick_test_diverged"] = True
                    logger.warning(f"[Quick test] Loss {val_bpb:.4f} > threshold {self.QUICK_TEST_LOSS_THRESHOLD}")
                else:
                    parsed["quick_test_diverged"] = False

            # For quick test, if parsing fails entirely, check stderr for FAIL/OOM signals
            if quick_test and parsed is None:
                output = result.stdout + result.stderr
                if any(kw in output.upper() for kw in ["FAIL", "OOM", "OUT OF MEMORY", "NAN", "CUDA ERROR"]):
                    return None
                # No val_bpb but no obvious error — could be too short to produce output
                # Treat as passing (not enough data to judge)
                return {"val_bpb": 0, "quick_test_diverged": False}

            return parsed

        except subprocess.TimeoutExpired:
            logger.warning(f"Training timed out{' [quick test]' if quick_test else ''}")
            return None
        except Exception as e:
            logger.warning(f"Training error: {e}")
            return None

    def _extract_changes_from_file(self, train_path: Path) -> dict:
        """Extract the hyperparameter changes applied to a modified train.py file.

        Compares the modified file against self.current_code to find what changed.
        """
        modified_code = train_path.read_text(encoding="utf-8")
        current_config = self._extract_hyperparams(self.current_code)
        modified_config = self._extract_hyperparams(modified_code)
        changes = {}
        for param in modified_config:
            if param in current_config and modified_config[param] != current_config[param]:
                changes[param] = modified_config[param]
        return changes

    def _parse_results(self, stdout: str) -> dict | None:
        """Parse val_bpb and other metrics from training output."""
        metrics = {}
        for line in stdout.splitlines():
            for key in ["val_bpb", "peak_vram_mb", "training_seconds",
                        "num_params_M", "depth", "total_seconds"]:
                if line.startswith(f"{key}:"):
                    try:
                        val = float(line.split(":")[1].strip())
                        # Normalize key names
                        norm_key = key.lower().replace("_m", "_m")
                        if key == "num_params_M":
                            norm_key = "num_params_m"
                        metrics[norm_key] = val
                    except (ValueError, IndexError):
                        pass

        if "val_bpb" not in metrics:
            return None
        return metrics
