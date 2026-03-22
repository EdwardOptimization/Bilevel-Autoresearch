"""Inner loop runner for training optimization.

One iteration = propose config change → modify train.py → run training → parse val_bpb → keep/discard.
The search behavior is controlled by SearchConfig, which the outer loop modifies.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
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
"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TrainRunner:
    """Runs the inner loop: propose → modify train.py → train → evaluate → keep/discard."""

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

    def run_iteration(self, iteration: int) -> TrainResult:
        """Run one inner loop iteration."""
        iter_dir = self.artifacts_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        current_config = self._extract_hyperparams(self.current_code)
        logger.info(f"[Iter {iteration}] Proposing changes...")

        # 1. LLM proposes changes
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

        # 3. Run training
        logger.info(f"[Iter {iteration}] Training... (changes: {proposal['changes']})")
        result = self._run_training(modified_path, iter_dir)

        if result is None:
            # Crash
            logger.warning(f"[Iter {iteration}] Training crashed")
            r = TrainResult(
                iteration=iteration, val_bpb=0, peak_vram_mb=0,
                training_seconds=0, num_params_m=0, status="crash",
                changes=proposal["changes"],
                description=f"CRASH: {proposal.get('hypothesis', '')}",
            )
            self.trace.add(r)
            return r

        # 4. Keep/discard
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
        """Ask LLM to propose hyperparameter changes."""
        from src.llm_client import parse_json_response

        prompt = PROPOSE_PROMPT.format(
            current_config=json.dumps(current_config, indent=2),
            trace_summary=self.trace.summary(),
            strategy=self.search_config.strategy,
            active_params=", ".join(self.search_config.active_params),
            frozen_params=", ".join(self.search_config.frozen_params) or "(none)",
            guidance=self.search_config.guidance,
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

    def _inject_time_budget(self, code: str) -> str:
        """Override TIME_BUDGET if search_config specifies a non-default value."""
        budget = self.search_config.time_budget
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

    def _run_training(self, train_path: Path, iter_dir: Path) -> dict | None:
        """Execute train.py and parse results. Returns None on crash."""
        log_path = iter_dir / "run.log"

        try:
            result = subprocess.run(
                ["uv", "run", str(train_path)],
                capture_output=True, text=True,
                timeout=self.search_config.time_budget * 3,  # 3x budget for compile + eval overhead
                cwd=str(self.work_dir),
            )
            log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

            if result.returncode != 0:
                logger.warning(f"Training failed (exit {result.returncode})")
                (iter_dir / "error.txt").write_text(result.stderr[-2000:], encoding="utf-8")
                return None

            return self._parse_results(result.stdout)

        except subprocess.TimeoutExpired:
            logger.warning("Training timed out")
            return None
        except Exception as e:
            logger.warning(f"Training error: {e}")
            return None

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
